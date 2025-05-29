import cv2
import numpy as np
from PIL import Image


MASK_PATH = '/workspace/PIH/Demo_hr/Harm03_mask.png' 
IMG_PATH = '/workspace/PIH/Demo_hr/Harm03_FG.png'
OUTPUT_PATH = '/workspace/PIH/results/feathered.png'

def feather_mask_and_apply(image_path, mask_path, feather_radius=5, output_path='output.png'):
    # Load original image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # Add alpha channel if not present

    # Load the binary mask (0 or 255)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Feather the mask using Gaussian blur
    feathered_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_radius, sigmaY=feather_radius)

    # Normalize to 0–255 and convert to uint8
    feathered_mask = cv2.normalize(feathered_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Set the feathered mask as the alpha channel
    image[:, :, 3] = feathered_mask

    # Convert from BGRA to RGBA (for PIL compatibility)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    # Save the result
    result = Image.fromarray(image)
    result.save(output_path)
    print(f"Saved anti-aliased PNG to {output_path}")


if __name__ == "__main__":
    feather_mask_and_apply(IMG_PATH,MASK_PATH,5,OUTPUT_PATH)