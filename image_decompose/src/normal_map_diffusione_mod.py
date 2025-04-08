import subprocess
import os

class DiffusionRunner:
    def __init__(self):
        # Hardcoded paths (modify if needed)
        self.script_path = "/root/Ram/Repo/flamTeam/wall_repainting/src/diffusion-e2e-ft/Marigold/run.py"
        self.venv_path = "/root/Ram/Repo/flamTeam/wall_repainting/src/diffusion-e2e-ft/.venv"

    def run_inference(self, input_rgb_dir, output_dir):
        command = [
            os.path.join(self.venv_path, "bin/python"),
            self.script_path,
            "--checkpoint=GonzaloMG/marigold-e2e-ft-normals",
            "--modality=normals",
            "--input_rgb_dir=" + input_rgb_dir,
            "--output_dir=" + output_dir
        ]

        print("Running:", " ".join(command))  # Optional: for debugging
        subprocess.run(command, check=True)

"""
Usage in main.py

from src.normal_map_diffusione_mod import DiffusionRunner

input_path = "/root/Ram/Repo/flamTeam/wall_repainting/assets/"
output_path = "/root/Ram/Repo/flamTeam/wall_repainting/assets/img5_nrmal"

runner = DiffusionRunner()
runner.run_inference(input_path, output_path)
"""