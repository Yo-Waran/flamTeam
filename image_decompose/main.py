from src.normal_map_diffusione_mod import DiffusionRunner

class WallRecolorPipeline:
    def __init__(self):
        self.runner = DiffusionRunner()

    def generate_normal_map(self, input_image_dir, output_dir):
        self.runner.run_inference(input_image_dir, output_dir)

if __name__ == "__main__":
    input_path = ""
    output_path = ""

    pipeline = WallRecolorPipeline()
    pipeline.generate_normal_map(input_path, output_path)