import numpy as np

# from src.basic_image_processing.utils.debug_system import *

DEBUG_FOLDER_PATH = './debug/image_blending'

class ImageBlending:
    """A class used to blend a foreground image onto a background image using alpha blending."""

    
    def __init__(self):
        """Initializes the ImageBlending class."""
        print("Initializing Image Blender Class...")

    def __extract_foreground_mask(self, foreground_image: np.ndarray) -> np.ndarray:
        """
        Extracts the alpha channel mask from the foreground image.

        Args:
            foreground_image (np.ndarray): The foreground image with an alpha channel.

        Returns:
            np.ndarray: A normalized alpha mask (0 to 1).

        Raises:
            ValueError: If the foreground image does not have 4 channels (RGBA).
        """
        if foreground_image.shape[2] != 4:
            raise ValueError("Foreground image must have 4 channels (RGBA).")

        # Normalize the alpha mask to range [0,1]
        return foreground_image[:, :, 3] / 255.0

    # @debug_saver(folder_path=DEBUG_FOLDER_PATH)
    def blend_images(self, foreground_image: np.ndarray, background_image: np.ndarray, debug_mode: bool = False) -> np.ndarray:
        """
        Blends the foreground image onto the background image using the alpha mask.

        Args:
            foreground_image (np.ndarray): The foreground image with an alpha channel.
            background_image (np.ndarray): The background image.
            debug_mode (bool): Whether to save debug images.

        Returns:
            np.ndarray: The blended image.

        Raises:
            ValueError: If the foreground and background images do not have the same dimensions.
        """
        print("Blending Foreground Image onto the Background...")

        # Ensure images have the same dimensions
        if foreground_image.shape[:2] != background_image.shape[:2]:
            raise ValueError("Foreground and background images must have the same dimensions.")

        # Extract the alpha channel as a blending mask
        alpha_mask = self.__extract_foreground_mask(foreground_image)

        # Convert mask to shape (H, W, 1) for broadcasting
        alpha_mask = np.expand_dims(alpha_mask, axis=-1)

        # Alpha blending: blended = alpha * foreground + (1 - alpha) * background
        blended_image = (alpha_mask * foreground_image[:, :, :3] + (1 - alpha_mask) * background_image).astype(np.uint8)

        # if debug_mode:
        #     DebugData("fg_image", foreground_image, "png"),
        #     DebugData("composite_image", blended_image, "png"),
        #     DebugData("composite_mask", alpha_mask, "png")

        return blended_image