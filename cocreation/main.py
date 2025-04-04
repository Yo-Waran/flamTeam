import cv2
import numpy as np
import os
from src.diffusion_light_hdri_mod import DiffusionLightPipeline  # Import HDRI extraction module
from src.switch_light_pipeline_mod import SwitchLightPipeline  # Import Relighting module
from src.image_blending_mod import ImageBlending  # Import Alpha Blending module
from src.blender_relight_mod import BlenderRelight  # Import Relighting Module
from src.blender_cubemap_mod import HDRIExposureMatcher  # Import HDRI exposure Matcher module

class ImageProcessor2D:
    """
    A class to process 2D images including HDRI extraction, relighting, and alpha blending.
    """
    
    def __init__(self, bg_img_path, fg_img_path, cubemap_path=None):
        """
        Initialize ImageProcessor2D with input and output paths.
        
        Args:
            bg_img_path (str): Path to the background image.
            fg_img_path (str): Path to the foreground image.
            cubemap_path (str, optional): Path to the CubeMap HDRI. Defaults to None.
            output_dir (str, optional): Directory to save output images. Defaults to "output".
        """
        self.bg_img_path = bg_img_path
        self.fg_img_path = fg_img_path
        self.cubemap_path = cubemap_path
        
        # Define directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.maps_dir = os.path.join(self.base_dir, "maps")
        self.hdri_dir = os.path.join(self.base_dir, "hdri")
        self.output_dir = os.path.join(self.base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.maps_dir, exist_ok=True)
        os.makedirs(self.hdri_dir, exist_ok=True)
        os.makedirs(self.output_dir,exist_ok= True)
        # Output paths
        self.hdri_output_path = os.path.join(self.hdri_dir, "extracted_hdri.exr")
        self.relit_subject_path = os.path.join(self.output_dir, "relit_subject.png")
        self.final_output_path = os.path.join(self.output_dir, "final_output.png")
        
        # PBR Maps paths
        self.albedo_path = os.path.join(self.maps_dir, "albedo.png")
        self.normal_path = os.path.join(self.maps_dir, "normal.png")
        self.roughness_path = os.path.join(self.maps_dir, "roughness.png")
        self.alpha_path = os.path.join(self.maps_dir, "alpha.png")
    
    def extract_hdri(self):
        """
        Extracts HDRI from the background image using DiffusionLightPipeline().
        If a CubeMap is available, it adjusts the exposure using HDRIExposureMatcher().

        Raises:
            ValueError: If the background image cannot be loaded.
        """
        if not self.cubemap_path:
            print("Extracting HDRI...")
            bg_img = cv2.imread(self.bg_img_path, cv2.IMREAD_UNCHANGED)
            if bg_img is None:
                raise ValueError("Failed to load background image")
            hdr_extractor = DiffusionLightPipeline()
            hdr_extractor.infer(bg_img, self.hdri_output_path)
        else:
            print("Using existing CubeMap...")
            hdri_exposure_matcher = HDRIExposureMatcher()
            adjustment_factor, analysis_data = hdri_exposure_matcher.analyze_images(self.bg_img_path, self.cubemap_path)
            adjusted_hdri = hdri_exposure_matcher.apply_hdri_exposure_adjustment(analysis_data['hdri'], adjustment_factor)
            hdri_exposure_matcher.save_hdri(adjusted_hdri, self.cubemap_path)
    
    def extract_pbr_maps(self):
        """
        Extracts PBR maps from the foreground image using SwitchLightPipeline().
        """
        print("Extracting PBR Maps...")
        pbr_extractor = SwitchLightPipeline()
        pbr_extractor.infer(self.fg_img_path, self.maps_dir)
    
    def relight_subject(self):
        """
        Relights the foreground image using BlenderRelight().
        If a CubeMap is available, It uses the extracted HDRI or the adjusted CubeMap for relighting().
        """
        relighter = BlenderRelight()
        if not self.cubemap_path:
            print("Relighting subject using Extracted HDRI...")
            relighter.setup_relighting_scene(self.albedo_path, self.normal_path, self.roughness_path, self.alpha_path, self.hdri_output_path, self.relit_subject_path)
        else:
            print("Relighting subject using Adjusted CubeMap...")
            relighter.setup_cubemap_relighting_scene(self.albedo_path, self.normal_path, self.roughness_path, self.alpha_path, self.cubemap_path, self.cubemap_path, self.relit_subject_path)
    
    def blend_images(self):
        """
        Performs alpha blending between the relit foreground and the background image using ImageBlending().
        
        Raises:
            ValueError: If either the foreground or background image cannot be loaded.
        """
        print("Blending images...")
        fg_image = cv2.imread(self.relit_subject_path, cv2.IMREAD_UNCHANGED)
        bg_image = cv2.imread(self.bg_img_path, cv2.IMREAD_UNCHANGED)
        
        if fg_image is None or bg_image is None:
            raise ValueError("Error loading images. Check file paths!")
        
        # Ensure foreground has an alpha channel
        if fg_image.shape[2] == 3:
            print("Foreground image lacks an alpha channel, adding one...")
            alpha_channel = np.ones((fg_image.shape[0], fg_image.shape[1], 1), dtype=np.uint8) * 255
            fg_image = np.concatenate((fg_image, alpha_channel), axis=-1)
        
        # Resize foreground to match background dimensions
        fg_resized = cv2.resize(fg_image, (bg_image.shape[1], bg_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Blend images
        overlapper = ImageBlending()
        final_composite = overlapper.blend_images(fg_resized, bg_image)
        
        # Save the final blended image
        cv2.imwrite(self.final_output_path, final_composite)
        print(f"Saved Final composited image at {self.final_output_path}")
    
    def process(self):
        """
        Runs the full image processing pipeline: extracting HDRI, extracting PBR maps, relighting, and blending.
        """
        self.extract_hdri()
        self.extract_pbr_maps()
        self.relight_subject()
        self.blend_images()

if __name__ == "__main__":
    processor = ImageProcessor2D(
        bg_img_path="/root/Ram/Repo/flamTeam/cocreation/assets/bg3.png", #bg path here
        fg_img_path="/root/Ram/Repo/flamTeam/cocreation/assets/fg3.png", #fg path here
        cubemap_path=None #cubemap if any
    )
    processor.process()
