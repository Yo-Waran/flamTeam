
from typing import List

import cv2
import numpy as np
from skimage.util import img_as_float
import ezexr

from diffusionlight.relighting.tonemapper import TonemapHDR


class HDREnvMapGenerator:
    GAMMA = 2.4
    SCALER = np.array([0.212671, 0.715160, 0.072169])

    def __init__(self) -> None:
        """
        Initialize the HDR environment map generator.
        """
        self.gamma = self.GAMMA
        self.scaler = self.SCALER

    def process_images(self, images: List[np.ndarray], evs: List[float]) -> np.ndarray:
        """
        Processes a list of images and corresponding exposure values to create an HDR image.

        Args:
            images (list of np.array): List of images (in numpy array format) with different exposures.
            evs (list of float): List of exposure values corresponding to the input images.

        Returns:
            np.array: Generated HDR image.
        """
        if len(images) != len(evs):
            raise ValueError("Number of images must match the number of exposure values (EVs).")
        
        # Convert EVs to descending order and sort images accordingly
        evs, images = zip(*sorted(zip(evs, images), reverse=True))
        evs = list(evs)

        # Linearize the first image
        image0 = (img_as_float(images[0])[..., :3] / 255) # Ensure RGB channels only
        image0_linear = np.power(image0, self.gamma)

        # Calculate luminance for each image
        luminances = []
        for img, ev in zip(images, evs):
            linear_img = np.power((img_as_float(img)[..., :3]) / 255, self.gamma)
            linear_img *= 1 / (2 ** ev)  # Scale brightness by exposure value
            lumi = linear_img @ self.scaler  # Compute luminance
            luminances.append(lumi)

        # Start HDR generation from the darkest image
        out_luminance = luminances[len(evs) - 1]
        for i in range(len(evs) - 1, 0, -1):
            maxval = 1 / (2 ** evs[i - 1])
            p1 = np.clip((luminances[i - 1] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
            p2 = out_luminance > luminances[i - 1]
            mask = (p1 * p2).astype(np.float32)
            out_luminance = luminances[i - 1] * (1 - mask) + out_luminance * mask

        # Combine luminance with the first image
        hdr_rgb = image0_linear * (out_luminance / (luminances[0] + 1e-10))[..., np.newaxis]

        return hdr_rgb

    def tonemap(self, hdr_image, percentile=99, max_mapping=0.9):
        """
        Tonemaps the HDR image for visualization.

        Args:
            hdr_image (np.array): HDR image.
            percentile (float): Percentile for tone mapping (default: 99).
            max_mapping (float): Maximum mapping value for tone mapping (default: 0.9).

        Returns:
            np.array: Tonemapped LDR image.
        """
        hdr2ldr = TonemapHDR(gamma=self.gamma, percentile=percentile, max_mapping=max_mapping)
        ldr_rgb, _, _ = hdr2ldr(hdr_image)
        return ldr_rgb

    def generate_hdr(self, images: List[np.ndarray], evs: List[float], tonemap_output: bool = False):
        """
        Generates an HDR image and optionally tonemaps it.

        Args:
            images (list of np.array): List of images (in numpy array format) with different exposures.
            evs (list of float): List of exposure values corresponding to the input images.
            tonemap_output (bool): Whether to tonemap the HDR image for visualization (default: True).

        Returns:
            tuple: HDR image and optionally tonemapped image.
        """
        hdr_image = self.process_images(images, evs)
        if tonemap_output:
            ldr_image = self.tonemap(hdr_image)
            return hdr_image, ldr_image
        return hdr_image, None

    def infer(self, ev_env_map_dict: dict, output_path: str) -> None:
        """
        Generates an HDR map from exposure-bracketed images and writes it to an EXR file.

        Args:
            images (list of np.ndarray): List of input images as NumPy arrays.
            evs (list of float): List of exposure values corresponding to the images.
            output_path (str): Path to save the generated EXR file.
        """
        
        evs = [eval(ev_value) for ev_value in ev_env_map_dict.keys()]
        images = list(ev_env_map_dict.values())
        hdr_image, _ = self.generate_hdr(images, evs)

        ezexr.imwrite(output_path, hdr_image.clip(0, 1).astype(np.float32))
        print(f"HDR map saved to: {output_path}")
