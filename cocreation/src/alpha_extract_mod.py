import cv2
import numpy as np

class AlphaMaskGenerator():
    """
        This Class is used to make alpha map from the input PNG file. 
    """
    def __init__(self):
        pass

    def create_alpha_mask(self, input_path, output_path):
        """
        Generates an alpha mask from a PNG image.
        The foreground (non-transparent areas) will be white (255),
        and the transparent areas will be black (0).
        
        :param input_path: Path to the input PNG image.
        :param output_path: Path to save the generated alpha mask.
        """
        # Load the image with alpha channel
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("Image not found or unsupported format")

        # Check if the image has an alpha channel
        if img.shape[2] < 4:
            raise ValueError("Image does not have an alpha channel")

        # Extract the alpha channel
        print("Extracting Alpha Map...")
        alpha_channel = img[:, :, 3]

        # Create a mask where alpha > 0 is white, and fully transparent areas are black
        alpha_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)

        # Save the alpha mask as a PNG
        cv2.imwrite(output_path, alpha_mask)
        print("Alpha map extracted to {0}".format(output_path))

# Example usage (if running this script directly):
if __name__ == "__main__":
    #example usage
    alpha_generator = AlphaMaskGenerator()
    alpha_generator.create_alpha_mask("/root/Ram/Repo/flamTeam/cocreation/assets/fg2.png", "/root/Ram/Repo/flamTeam/cocreation/output/my_alpha_2.png")