import os
import math
from PIL import Image
from types import SimpleNamespace

import numpy as np
import cv2
import bpy
import logging

from skimage import img_as_float

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HDRCubeMapPipeline:
    def __init__(self):
        print("Initialized HDR CubeMap Generator.....")
        self.faces = {}
        self.hdri_exposure_matcher = HDRIExposureMatcher()

    def load_faces(self, cube_image_path):
        """Loads cubemap faces from a list of file paths."""
        ext_list = ('jpg', 'jpeg', 'png')
        faces_list = [os.path.join(cube_image_path, cube_image) for cube_image in os.listdir(cube_image_path) if cube_image.endswith(ext_list)]
        face_mapping = {
            0: "right",
            1: "left",
            2: "top",
            3: "bottom",
            4: "front",
            5: "back",
        }
        
        for path in faces_list:
            for idx, face_name in face_mapping.items():
                if f"_face_{idx}" in path:
                    try:
                        img = Image.open(path).convert("RGB")
                        img_data = np.array(img, dtype=np.float32) / 255.0
                        self.faces[idx] = img_data
                        logger.info(f"Loaded face {idx} ({face_name}) from {path}")
                    except Exception as e:
                        logger.error(f"Error loading {path}: {e}")
                    break

    def cubemap_to_equirectangular(self, width, height):
        """Converts a cubemap to an equirectangular projection."""
        equirectangular = np.zeros((height, width, 3), dtype=np.float32)
        face_size = self.faces[0].shape[0]

        for j in range(height):
            for i in range(width):
                theta = 2 * math.pi * (i / width - 0.5)
                phi = math.pi * (j / height - 0.5)
                
                x = math.cos(phi) * math.sin(theta)
                y = math.sin(phi)
                z = math.cos(phi) * math.cos(theta)
                
                abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
                max_axis = max(abs_x, abs_y, abs_z)
                
                if max_axis == abs_x:
                    face_idx = 0 if x > 0 else 1
                    sc = face_size * (z / abs_x + 1) / 2
                    tc = face_size * (y / abs_x + 1) / 2
                elif max_axis == abs_y:
                    face_idx = 2 if y > 0 else 3
                    sc = face_size * (x / abs_y + 1) / 2
                    tc = face_size * (-z / abs_y + 1) / 2
                else:
                    face_idx = 4 if z > 0 else 5
                    sc = face_size * (x / abs_z + 1) / 2
                    tc = face_size * (y / abs_z + 1) / 2
                
                if face_idx in self.faces:
                    face = self.faces[face_idx]
                    sc = max(0, min(face_size - 1, sc))
                    tc = max(0, min(face_size - 1, tc))
                    
                    x0, y0 = int(sc), int(tc)
                    x1, y1 = min(x0 + 1, face_size - 1), min(y0 + 1, face_size - 1)
                    wx, wy = sc - x0, tc - y0
                    
                    p00, p01 = face[y0, x0], face[y0, x1]
                    p10, p11 = face[y1, x0], face[y1, x1]
                    
                    color = (1 - wx) * (1 - wy) * p00 + wx * (1 - wy) * p01 + (1 - wx) * wy * p10 + wx * wy * p11
                    equirectangular[j, i] = color
        
        return equirectangular

    def save_hdr(self, equirectangular, output_path):
        """Saves the generated HDRI as an .hdr file."""
        cv2.imwrite(output_path, cv2.cvtColor(equirectangular, cv2.COLOR_RGB2BGR))

    def infer(self, bg_image_path: str, face_image_paths: str, output_path: str = "output.hdr", adjusted_hdri_path: str = "adjusted_output.hdr", width: int = 2048, height: int = 1024) -> str:
        """
        Process the input cubemap face images and generate an equirectangular HDRI.
        
        :param image_paths: List of file paths to the cubemap face images
        :param output_path: Path to save the generated HDRI
        :param width: Width of the output HDRI
        :param height: Height of the output HDRI
        """
        self.load_faces(face_image_paths)
        if len(self.faces) != 6:
            logger.error("Invalid number of cubemap faces. Expected 6.")
            return
        
        equirectangular = self.cubemap_to_equirectangular(width, height)
        self.save_hdr(equirectangular, output_path)
        logger.info(f"HDRI saved at {output_path}")

        adjustment_factor, analysis_data = self.hdri_exposure_matcher.analyze_images(bg_image_path, output_path) #analyze the bg image and the HDRI

        adjusted_hdri = self.hdri_exposure_matcher.apply_hdri_exposure_adjustment(analysis_data['hdri'], adjustment_factor) #apply the HDRI adjustments in a new blender scene
        self.hdri_exposure_matcher.save_hdri(adjusted_hdri, adjusted_hdri_path) #apply the adjustments and save the new hdri

        return output_path

    def add_hdri_to_blender(self, blender_config: SimpleNamespace, adjusted_hdri_path: str):
        bpy.ops.wm.open_mainfile(filepath=blender_config.blender_output_path)
        
        # Setup HDRI environment based on the more complex node setup in the new screenshot
        if hasattr(blender_config, 'hdri_path') and os.path.exists(blender_config.hdri_path):
            # Set up world environment
            world = bpy.context.scene.world
            if world is None:
                world = bpy.data.worlds.new("World")
                bpy.context.scene.world = world
            
            # Enable nodes for the world
            world.use_nodes = True
            node_tree = world.node_tree
            
            # Clear existing nodes
            for node in node_tree.nodes:
                node_tree.nodes.remove(node)
            
            # Create texture coordinate node
            tex_coord = node_tree.nodes.new(type='ShaderNodeTexCoord')
            tex_coord.location = (-800, 300)
            
            # Create mapping node
            mapping = node_tree.nodes.new(type='ShaderNodeMapping')
            mapping.location = (-600, 300)
            # Set Y rotation to -90 degrees as shown in the screenshot
            mapping.inputs['Rotation'].default_value[1] = math.radians(-90)
            
            # Create environment texture node for the main cubemap
            env_node = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
            env_node.location = (-400, 400)
            env_node.name = os.path.basename(blender_config.hdri_path)
            
            # Load HDR image for the main cubemap
            main_hdr = bpy.data.images.load(blender_config.hdri_path, check_existing=True)
            env_node.image = main_hdr
            
            # Create environment texture node for the adjusted cubemap
            env_adjusted_node = node_tree.nodes.new(type='ShaderNodeTexEnvironment')
            env_adjusted_node.location = (-400, 100)
            env_adjusted_node.name = os.path.basename(adjusted_hdri_path)
            
            # Load HDR image for the adjusted cubemap
            adjusted_hdr = bpy.data.images.load(adjusted_hdri_path, check_existing=True)
            env_adjusted_node.image = adjusted_hdr
            
            # Create Mix node
            mix_node = node_tree.nodes.new(type='ShaderNodeMix')
            mix_node.location = (-200, 250)
            mix_node.data_type = 'RGBA'
            mix_node.blend_type = 'MIX'
            mix_node.clamp_factor = True
            mix_node.inputs['Factor'].default_value = 0.500
            
            # Create Background node
            background_node = node_tree.nodes.new(type='ShaderNodeBackground')
            background_node.location = (0, 250)
            background_node.inputs['Strength'].default_value = 1.0
            
            # Create output node
            output_node = node_tree.nodes.new(type='ShaderNodeOutputWorld')
            output_node.location = (200, 250)
            
            # Connect nodes
            node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            
            # Connect mapping to both environment nodes
            node_tree.links.new(mapping.outputs['Vector'], env_node.inputs['Vector'])
            node_tree.links.new(mapping.outputs['Vector'], env_adjusted_node.inputs['Vector'])
            
            # Connect environment nodes to mix (A and B inputs)
            node_tree.links.new(env_node.outputs['Color'], mix_node.inputs['A'])
            node_tree.links.new(env_adjusted_node.outputs['Color'], mix_node.inputs['B'])
            
            # Connect mix to background and background to output
            node_tree.links.new(mix_node.outputs['Result'], background_node.inputs['Color'])
            node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

        bpy.ops.wm.save_as_mainfile(filepath=blender_config.blender_output_path)

        
class HDRIExposureMatcher:
    def __init__(self, method='advanced'):
        self.method = method
        self.valid_methods = ['simple', 'histogram', 'advanced']
        if method not in self.valid_methods:
            logger.warning(f"Invalid method '{method}'. Using 'advanced' as default.")
            self.method = 'advanced'
    
    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_hdri(self, hdri_path):
        if not os.path.exists(hdri_path):
            raise FileNotFoundError(f"HDRI file not found: {hdri_path}")
        
        hdri = cv2.imread(hdri_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdri is None:
            raise ValueError(f"Could not load HDRI: {hdri_path}")
        
        hdri = cv2.cvtColor(hdri, cv2.COLOR_BGR2RGB)
        return hdri
    
    def save_hdri(self, hdri, output_path):
        hdri_bgr = cv2.cvtColor(hdri, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, hdri_bgr)
        logger.info(f"Saved adjusted HDRI to {output_path}")
    
    def calculate_luminance(self, image):
        lab_image = cv2.cvtColor(image.astype(np.float32) / 255.0 if image.dtype != np.float32 else image, 
                                cv2.COLOR_RGB2LAB)
        l_channel = lab_image[:,:,0]
        return l_channel
    
    def calculate_hdri_luminance(self, hdri):
        luminance = 0.2126 * hdri[:,:,0] + 0.7152 * hdri[:,:,1] + 0.0722 * hdri[:,:,2]
        return luminance
    
    def calculate_light_intensity(self, image, is_hdri=False):
        if is_hdri:
            luminance = self.calculate_hdri_luminance(image)
            
            intensity_metrics = {
                'mean_luminance': np.mean(luminance),
                'median_luminance': np.median(luminance),
                'luminance_percentile_90': np.percentile(luminance, 90),
                'luminance_percentile_10': np.percentile(luminance, 10),
                'max_luminance': np.max(luminance),
                'dynamic_range': np.log2(np.max(luminance) / (np.percentile(luminance, 1) + 1e-6)),
                'relative_brightness': np.mean(luminance)
            }
        else:
            luminance = self.calculate_luminance(image)
            
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            value_channel = hsv_image[:,:,2]
            
            intensity_metrics = {
                'mean_luminance': np.mean(luminance),
                'median_luminance': np.median(luminance),
                'mean_brightness': np.mean(value_channel),
                'luminance_percentile_90': np.percentile(luminance, 90),
                'luminance_percentile_10': np.percentile(luminance, 10),
                'relative_brightness': np.mean(luminance) / 100.0 * 100
            }
        
        return intensity_metrics
    
    def analyze_exposure_simple(self, image, is_hdri=False):
        if is_hdri:
            luminance = self.calculate_hdri_luminance(image)
        else:
            luminance = self.calculate_luminance(image)
        return np.mean(luminance)
    
    def analyze_exposure_histogram(self, image, is_hdri=False):
        if is_hdri:
            luminance = self.calculate_hdri_luminance(image)
            if np.max(luminance) > 0:
                log_luminance = np.log1p(luminance)
                hist, bins = np.histogram(log_luminance.flatten(), bins=256)
            else:
                return 0
        else:
            luminance = self.calculate_luminance(image)
            hist, bins = np.histogram(luminance.flatten(), bins=256, range=[0, 256])
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        weighted_sum = np.sum(hist * bin_centers)
        total_pixels = np.sum(hist)
        
        return weighted_sum / total_pixels if total_pixels > 0 else 0
    
    def analyze_exposure_advanced(self, image, is_hdri=False):
        if is_hdri:
            luminance = self.calculate_hdri_luminance(image)
            
            metrics = {
                'mean': np.mean(luminance),
                'median': np.median(luminance),
                'std_dev': np.std(luminance),
                'percentile_25': np.percentile(luminance, 25),
                'percentile_75': np.percentile(luminance, 75),
                'highlights': np.percentile(luminance, 90),
                'shadows': np.percentile(luminance, 10),
                'contrast': np.percentile(luminance, 95) - np.percentile(luminance, 5),
                'max_value': np.max(luminance),
                'dynamic_range': np.log2(np.max(luminance) / (np.percentile(luminance, 1) + 1e-6)),
                'rgb_mean': np.mean(image)
            }
        else:
            luminance = self.calculate_luminance(image)
            
            img_float = img_as_float(image)
            
            metrics = {
                'mean': np.mean(luminance),
                'median': np.median(luminance),
                'std_dev': np.std(luminance),
                'percentile_25': np.percentile(luminance, 25),
                'percentile_75': np.percentile(luminance, 75),
                'highlights': np.percentile(luminance, 90),
                'shadows': np.percentile(luminance, 10),
                'contrast': np.percentile(luminance, 95) - np.percentile(luminance, 5),
                'hsv_value': np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:,:,2]),
                'rgb_mean': np.mean(img_float)
            }
        
        return metrics
    
    def calculate_adjustment_factor(self, bg_metrics, hdri_metrics):
        if self.method == 'simple':
            factor = bg_metrics / hdri_metrics if hdri_metrics > 0 else 1.0
        
        elif self.method == 'histogram':
            factor = bg_metrics / hdri_metrics if hdri_metrics > 0 else 1.0
        
        elif self.method == 'advanced':
            midtone_factor = bg_metrics['mean'] / hdri_metrics['mean'] if hdri_metrics['mean'] > 0 else 1.0
            highlight_factor = bg_metrics['highlights'] / hdri_metrics['highlights'] if hdri_metrics['highlights'] > 0 else 1.0
            shadow_factor = bg_metrics['shadows'] / hdri_metrics['shadows'] if hdri_metrics['shadows'] > 0 else 1.0
            
            factor = (0.6 * midtone_factor + 
                     0.25 * highlight_factor + 
                     0.15 * shadow_factor)
        
        factor = max(0.01, min(factor, 10.0))
        return factor
    
    def analyze_images(self, bg_image_path, hdri_path):
        logger.info(f"Loading background image: {bg_image_path}")
        bg_image = self.load_image(bg_image_path)
        
        logger.info(f"Loading HDRI: {hdri_path}")
        hdri = self.load_hdri(hdri_path)
        
        bg_intensity = self.calculate_light_intensity(bg_image, is_hdri=False)
        hdri_intensity = self.calculate_light_intensity(hdri, is_hdri=True)
        
        logger.info(f"Analyzing exposure using {self.method} method")
        if self.method == 'simple':
            bg_exposure = self.analyze_exposure_simple(bg_image, is_hdri=False)
            hdri_exposure = self.analyze_exposure_simple(hdri, is_hdri=True)
        elif self.method == 'histogram':
            bg_exposure = self.analyze_exposure_histogram(bg_image, is_hdri=False)
            hdri_exposure = self.analyze_exposure_histogram(hdri, is_hdri=True)
        else:
            bg_exposure = self.analyze_exposure_advanced(bg_image, is_hdri=False)
            hdri_exposure = self.analyze_exposure_advanced(hdri, is_hdri=True)
        
        adjustment_factor = self.calculate_adjustment_factor(bg_exposure, hdri_exposure)
        logger.info(f"Calculated adjustment factor: {adjustment_factor:.4f}")
        
        logger.info("Background light intensity:")
        for key, value in bg_intensity.items():
            logger.info(f"  {key}: {value:.4f}")
            
        logger.info("HDRI light intensity:")
        for key, value in hdri_intensity.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return adjustment_factor, {
            'background_metrics': bg_exposure,
            'hdri_metrics': hdri_exposure,
            'background_image': bg_image,
            'hdri': hdri,
            'background_intensity': bg_intensity,
            'hdri_intensity': hdri_intensity
        }
    
    def apply_hdri_exposure_adjustment(self, hdri, factor):
        adjusted_hdri = hdri * factor
        return adjusted_hdri
