import torch
import cv2
import numpy as np
from sapiens.sapiens_inference.normal import SapiensNormal, SapiensNormalType, draw_normal_map

class NormalMapGenerator:
    """
    A class to generate normal maps from images using the Sapiens Normal Estimation module.
    
    This class loads an image, processes it to estimate the normal map, and saves the output.
    """
    
    def __init__(self, dtype=torch.float32):
        """
        Initializes the normal map generator.
        
        Args:
            dtype (torch.dtype, optional): The data type for computations. Defaults to torch.float32.
        """
        self.dtype = dtype
        self.estimator = SapiensNormal(SapiensNormalType.NORMAL_1B, dtype=self.dtype)
    
    def load_image(self, img_path):
        """
        Loads an image and extracts its alpha channel if present.
        
        Args:
            img_path (str): Path to the image file.
        
        Returns:
            tuple: A tuple containing the RGB image as a NumPy array and the alpha channel (or None if not present).
        
        Raises:
            FileNotFoundError: If the image cannot be found or loaded.
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        if img.shape[2] == 4:
            alpha = img[:, :, 3]  # Extract alpha channel
            img = img[:, :, :3]  # Keep only RGB channels
        else:
            alpha = None  # No alpha channel present
        
        return img, alpha
    
    def generate_normal_map(self, img):
        """
        Generates a normal map from an input image.
        
        Args:
            img (numpy.ndarray): The input image as a NumPy array (RGB format).
        
        Returns:
            numpy.ndarray: The generated normal map as a NumPy array.
        """
        normal_map = self.estimator(img)
        return draw_normal_map(normal_map)
    
    def save_image(self, output_path, normal_map, alpha=None):
        """
        Saves the generated normal map to a file, preserving the alpha channel if available.
        
        Args:
            output_path (str): Path to save the normal map.
            normal_map (numpy.ndarray): The normal map image as a NumPy array.
            alpha (numpy.ndarray, optional): The alpha channel to be reapplied. Defaults to None.
        """
        if alpha is not None:
            normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2BGRA)  # Convert to BGRA format
            normal_map[:, :, 3] = alpha  # Reapply alpha channel
        
        cv2.imwrite(output_path, normal_map)
        print(f"Output saved to {output_path}")
    
    def process(self, img_path, output_path):
        """
        Processes an image to generate and save its normal map.
        
        This function loads an image, estimates its normal map, and saves the result.
        
        Args:
            img_path (str): Path to the input image file.
            output_path (str): Path to save the generated normal map.
        """
        img, alpha = self.load_image(img_path)
        normal_map = self.generate_normal_map(img)
        self.save_image(output_path, normal_map, alpha)

# Example usage
if __name__ == "__main__":
    img_path = "/root/Ram/Repo/flamTeam/cocreation/assets/fg3.png"
    output_path = "/root/Ram/Repo/flamTeam/cocreation/output/output_normal_bg3.png"
    
    normal_generator = NormalMapGenerator()
    normal_generator.process(img_path, output_path)
