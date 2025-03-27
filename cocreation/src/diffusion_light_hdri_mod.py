from json import load
from typing import List, Tuple

import torch
import cv2
from skimage import transform


from src.diffusionlight.relighting.inpainter import BallInpainter
from src.diffusionlight.relighting.mask_utils import MaskGenerator
from src.diffusionlight.relighting.hdr_map_generator import HDREnvMapGenerator
from src.diffusionlight.relighting.ball_processor import get_ideal_normal_ball
from src.diffusionlight.relighting.argument import SD_MODELS, CONTROLNET_MODELS
from src.diffusionlight.relighting.utils import *
from types import SimpleNamespace

def dict_to_namespace(d):
    """
    Recursively convert a dictionary into a SimpleNamespace.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

class DiffusionLightPipeline:
    """
    Pipeline for generating shadows using diffusion light models.

    This class initializes and manages the pipeline for HDRI generation
    using diffusion light models. It handles model loading, configuration,
    and processing of images to generate environment maps.
    """

    MODEL_CONFIG_PATH = "/root/Ram/Repo/flamTeam/cocreation/src/diffusionlight/config.json"
    MODEL_PATH = './models'

    def __init__(self):
        """
        Initialize the pipeline with required models and settings.

        Args:
            device (torch.device): The device to run the models on (e.g., 'cpu' or 'cuda').
            model_path (str): Path to the model files.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model configuration from JSON file
        with open(self.MODEL_CONFIG_PATH, "r") as file:
            model_config = load(file)

        # Convert configuration dictionary to a namespace for easy access
        self.config = dict_to_namespace(model_config)

        # Initialize the pipeline and LoRA settings
        self.pipe = self._initialize_pipeline(self.config.model_option, use_controlnet=self.config.use_controlnet, offload=self.config.offload)
        self.enabled_lora = self._initialize_lora()
        
        # Generate mask and ideal ball for processing
        self.hdr_generator = HDREnvMapGenerator()
        self.mask_generator = MaskGenerator()
        self.normal_ball, self.mask_ball = get_ideal_normal_ball(size=self.config.ball_size + self.config.ball_dilate)

    def _initialize_pipeline(self, model_option: str, use_controlnet: bool, offload: bool) -> BallInpainter:
        """Initialize the BallInpainter pipeline based on model options.

        Args:
            model_option (str): The model option to use (e.g., 'sdxl').
            use_controlnet (bool): Whether to use ControlNet models.
            offload (bool): Whether to offload models to save memory.

        Returns:
            BallInpainter: An instance of the BallInpainter pipeline.

        Raises:
            ValueError: If the model option is not supported.
        """
        # Determine the appropriate torch data type based on the device
        torch_dtype = torch.float16 if "cuda" in str(self.device) else torch.float32

        # Initialize the model and pipeline based on the model option
        if model_option in ['sdxl', 'sdxl_turbo', 'sdxl_fast']:
            model = SD_MODELS[model_option]
            controlnet = CONTROLNET_MODELS.get(model_option) if use_controlnet else None
            pipe = BallInpainter.from_sdxl(
                model=model,
                controlnet=controlnet,
                device=self.device,
                torch_dtype=torch_dtype,
                offload=offload
            )
        else:
            raise ValueError(f"Current Diffusion Light pipeline does not support {model_option} model...")
        
        # Optionally compile the UNet model for performance optimization
        if self.config.use_torch_compile:
            try:
                print("compiling unet model")
                pipe.pipeline.unet = torch.compile(pipe.pipeline.unet, mode="reduce-overhead", fullgraph=True)
            except:
                pass
        
        return pipe

    def _initialize_lora(self) -> bool:
        """Initialize LoRA settings for the pipeline.

        Returns:
            bool: True if LoRA is enabled, False otherwise.

        Raises:
            ValueError: If LoRA scale is set but the model path is not provided.
        """
        # Check if LoRA parameters are set correctly
        if self.config.lora_params.scale > 0 and self.config.lora_params.model_path is None:
            raise ValueError("lora scale is not 0 but lora path is not set")
        
        # Load and fuse LoRA weights if enabled
        if (self.config.lora_params.model_path is not None) and (self.config.lora_params.use_lora):
            print(f"using lora path {self.config.lora_params.model_path}")
            print(f"using lora scale {self.config.lora_params.scale}")
            self.pipe.pipeline.load_lora_weights(self.config.lora_params.model_path)
            self.pipe.pipeline.fuse_lora(lora_scale=self.config.lora_params.scale)  # Fuse LoRA weights
            enabled_lora = True
        else:
            enabled_lora = False
            
        return enabled_lora
    
    def _interpolate_embedding(self, prompt: str, prompt_dark: str) -> dict:
        """
        Interpolate prompt embeddings for a given EV value.

        This method interpolates between normal and dark prompt embeddings
        based on exposure values (EV) to generate a range of embeddings for
        different lighting conditions.

        Args:
            prompt (str): The normal prompt text.
            prompt_dark (str): The dark prompt text.

        Returns:
            dict: A dictionary mapping EV values to interpolated embeddings.
        """
        print("Interpolate embedding...")

        # Normalize the EV value and calculate interpolants
        interpolants = [ev / self.config.max_negative_ev for ev in self.config.ev_values]
        
        print("EV : ", self.config.ev_values)
        print("Interpolants : ", interpolants)

        # Calculate prompt embeddings for normal and dark prompts
        prompt_embeds_normal, _, pooled_prompt_embeds_normal, _ = self.pipe.pipeline.encode_prompt(prompt)
        prompt_embeds_dark, _, pooled_prompt_embeds_dark, _ = self.pipe.pipeline.encode_prompt(prompt_dark)

        # Interpolate embeddings for each EV value
        interpolate_embeds = []
        for t in interpolants:
            int_prompt_embeds = prompt_embeds_normal + t * (prompt_embeds_dark - prompt_embeds_normal)
            int_pooled_prompt_embeds = pooled_prompt_embeds_normal + t * (pooled_prompt_embeds_dark - pooled_prompt_embeds_normal)
            interpolate_embeds.append((int_prompt_embeds, int_pooled_prompt_embeds))

        return dict(zip(self.config.ev_values, interpolate_embeds))
        
    def process_ball_to_envmap(self, ball_image: np.ndarray) -> np.ndarray:
        """
        Convert a ball image to an environment map in latitude-longitude format.

        This method processes a ball image to generate an environment map
        that can be used for lighting simulations.

        Args:
            ball_image (np.ndarray): Input ball image as a numpy array.

        Returns:
            np.ndarray: Generated environment map as a numpy array.
        """
        I = np.array([1, 0, 0])  # Camera direction vector
        env_grid = create_envmap_grid(self.config.envmap_gen_consts.height * self.config.envmap_gen_consts.scale)
        reflect_vec = get_cartesian_from_spherical(env_grid[..., 1], env_grid[..., 0])
        normal = get_normal_vector(I[None, None], reflect_vec)

        # Convert normal map to lookup position (range: [0, 1])
        pos = (normal + 1.0) / 2
        pos = 1.0 - pos
        pos = pos[..., 1:]

        # Using PyTorch for bilinear interpolation
        with torch.no_grad():
            grid = torch.from_numpy(pos[None].astype(np.float32)) * 2 - 1  # Convert to range [-1, 1]
            ball_image_tensor = torch.from_numpy(ball_image[None].astype(np.float32)).permute(0, 3, 1, 2)
            env_map = torch.nn.functional.grid_sample(
                ball_image_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True
            )
            env_map = env_map[0].permute(1, 2, 0).numpy()

        # Resize to final environment map dimensions
        env_map_resized = transform.resize(env_map, (self.config.envmap_gen_consts.height, self.config.envmap_gen_consts.height * 2), anti_aliasing=True)
        return env_map_resized

    def infer(self, input_image: np.ndarray, hdri_path: str) -> Tuple[List[np.ndarray]]:
        """
        Perform inpainting on the input image and return the inpainted image.

        This method processes the input image to generate inpainted images
        for different exposure values and creates an HDR environment map.

        Args:
            input_image (np.ndarray): Input image as a numpy array.
            hdri_path (str): Path to save the generated HDR environment map.

        Returns:
            Tuple[List[np.ndarray]]: A tuple containing the path to the HDR environment map.
        """
        prompt = self.config.prompt_config.prompt
        prompt_dark = self.config.prompt_config.prompt_dark
        
        # Convert input image to PIL format
        input_image = cv2.resize(input_image, (self.config.inpainting_args.resize_height, self.config.inpainting_args.resize_width))
        input_image = Image.fromarray(input_image)
        
        # Calculate mask position and size
        x, y, r = self.config.inpainting_args.resize_height // 2 - self.config.ball_size // 2, self.config.inpainting_args.resize_width // 2 - self.config.ball_size // 2, self.config.ball_size
        mask = self.mask_generator.generate_single(input_image, self.mask_ball, x, y, r + self.config.ball_dilate)
        
        # Interpolate embeddings for different EV values
        embedding_dict = self._interpolate_embedding(prompt=prompt, prompt_dark=prompt_dark)
       
        env_map_list = []
        ev_env_map_dict = {}
        for ev_value, (prompt_embeds, pooled_prompt_embeds) in embedding_dict.items():
            print(f"Inpainting for EV Value: {ev_value}....")
            generator = torch.Generator().manual_seed(0)
            inpaint_args = {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt": self.config.inpainting_args.negative_prompt,
                "num_inference_steps": self.config.inpainting_args.denoising_step,
                "generator": generator,
                "image": input_image,
                "mask_image": mask,
                "strength": self.config.inpainting_args.strength,
                "controlnet_conditioning_scale": self.config.inpainting_args.control_scale,
                "height": self.config.inpainting_args.resize_height,
                "width": self.config.inpainting_args.resize_width,
                "normal_ball": self.normal_ball,
                "mask_ball": self.mask_ball,
                "x": int(x),
                "y": int(y),
                "r": int(r),
                "guidance_scale": self.config.inpainting_args.guidance_scale,
            }

            # Add cross-attention kwargs if LoRA is enabled
            if self.enabled_lora:
                inpaint_args["cross_attention_kwargs"] = {"scale": self.config.lora_params.scale}

            # Perform inpainting and process the output image
            output_image = self.pipe.inpaint(**inpaint_args).images[0]
            cropped_image = output_image.crop((x, y, x + r, y + r))
            square_image = np.array(cropped_image)
            env_map_image = self.process_ball_to_envmap(square_image)
            env_map_list.append(env_map_image)
            ev_env_map_dict[str(ev_value)] = env_map_image
        
        # Generate HDR environment map
        self.hdr_generator.infer(ev_env_map_dict, hdri_path)
        
        return hdri_path

#example usage
def main():
    hdri_path = "cocreation/hdri/myNewHDRI.exr"
    input_image = "cocreation/assets/bg1.png"
    input_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    diffusion_light_hdri = DiffusionLightPipeline()
    diffusion_light_hdri.infer(input_image, hdri_path)

if __name__ == "__main__":
    main()