import requests
import os
import zipfile
import cv2
import numpy as np

class SwitchLightPipeline:
    """
    A pipeline for acquiring and processing Physically Based Rendering (PBR) materials 
    from an external API and processing them locally.
    """
    BASE_URL = "https://sdk.beeble.ai/v1/acquire/pbr-materials"

    def __init__(self) -> None:
        """
        Initializes the pipeline with an API key.

        Raises:
            ValueError: If the API key is not found.
        """
        api_key = "ca0c9af608f0e03191283e9907fc97dae5e31f93c5cc199d374148e8cb9f9892"
        if not api_key:
            raise ValueError("API key not found. Set 'BEEBLE_API_KEY' as an environment variable.")
        self.HEADERS = {'x-api-key': api_key}

    def infer(self, input_image: str, output_path: str) -> str:
        """
        Sends an input image to the API for PBR material generation, then downloads 
        and extracts the received files.

        Args:
            input_image (str): Path to the input image file.
            output_path (str): Directory where the extracted PBR materials will be saved.

        Returns:
            str: The path to the extracted materials directory.

        Raises:
            ValueError: If the API response is unsuccessful or contains an unknown content type.
        """
        with open(input_image, 'rb') as src_file:
            files = {'source_image': src_file}
            data = {'auto_key': True, 'preview': False}
            print("Requesting PBR materials from:", self.BASE_URL)
            response = requests.post(self.BASE_URL, headers=self.HEADERS, files=files, data=data)

            if response.status_code != 200:
                raise ValueError(f"Error: {response.status_code} - {response.text}")

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                print("Received JSON response:", response.json())
                return output_path
            elif "application/zip" in content_type or "application/octet-stream" in content_type:
                os.makedirs(output_path, exist_ok=True)
                zip_path = os.path.join(output_path, "pbr_materials.zip")
                with open(zip_path, "wb") as zip_file:
                    zip_file.write(response.content)
                print(f"PBR materials saved as '{zip_path}'")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path)
                print(f"PBR materials extracted to '{output_path}'")
                return output_path
            else:
                raise ValueError(f"Unknown content type received: {content_type}")

if __name__ == "__main__":
    #example usage
    pipeline = SwitchLightPipeline()
    output_dir = pipeline.infer(
        '/root/Ram/DiffusionLight/cocreation/assets/fg1.png',
        '/root/Ram/DiffusionLight/cocreation/maps'
    )
    print("Final Maps directory:", output_dir)
