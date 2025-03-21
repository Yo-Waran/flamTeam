import cv2
import numpy as np
from diffusion_light_hdri import DiffusionLightPipeline  # Import HDRI extraction module
from SwitchLightPipeline_mod import SwitchLightPipeline  # Import Relighting module
from ImageBlending_mod import ImageBlending  # Import Alpha Blending module
from blender_relight_mod import BlenderRelight


# Input Paths
BG_IMG_PATH = "cocreation/assets/bg3.png"
FG_IMG_PATH = "cocreation/assets/fg3.png"

#OTHER PATHS
MAPS_OUTPUT_PATH = 'cocreation/maps'
HDRI_OUTPUT_PATH = "cocreation/hdri/extracted_hdri.exr"
RELIT_SUBJECT_PATH = "cocreation/output/relit_subject.png"
FINAL_OUTPUT_PATH = "cocreation/output/final_output.png"
ALBEDO_PATH = MAPS_OUTPUT_PATH+"/albedo.png"
NORMAL_PATH = MAPS_OUTPUT_PATH+"/normal.png"
ROUGHNESS_PATH = MAPS_OUTPUT_PATH+"/roughness.png"
ALPHA_PATH = MAPS_OUTPUT_PATH+"/alpha.png"

    
# Step 1: Extract HDRI from the background image
print("Step 1: Extracting HDRI...")
BG_IMG = cv2.imread(BG_IMG_PATH,cv2.IMREAD_UNCHANGED)
hdr_extractor = DiffusionLightPipeline()
EXTRACTED_HDRI = hdr_extractor.infer(BG_IMG,HDRI_OUTPUT_PATH)  # Save output HDRI path

# Step 2: Relight the foreground image using the extracted HDRI
print("Step 2: Extracting PBR Maps ...")
pbr_extractor = SwitchLightPipeline()
pbr_extractor.infer(FG_IMG_PATH,MAPS_OUTPUT_PATH)  # Save relit image path

#Step 3:Relight the Subject
print("Step 3: Relighting Subject...")
relighter = BlenderRelight()
RELIT_SUBJECT = relighter.setup_relighting_scene(ALBEDO_PATH,NORMAL_PATH,ROUGHNESS_PATH,ALPHA_PATH,HDRI_OUTPUT_PATH,RELIT_SUBJECT_PATH)

# Step 4: Load images and perform alpha blending
print("Step 4: Blending Images...")
fg_image = cv2.imread(RELIT_SUBJECT, cv2.IMREAD_UNCHANGED)  # Load relit image (RGBA)
bg_image = cv2.imread(BG_IMG_PATH, cv2.IMREAD_UNCHANGED)  # Load background image (RGB)

if fg_image is None or bg_image is None:
    raise ValueError("Error loading images. Check file paths!")

# Ensure background image has an alpha channel (if missing, add one)
if fg_image.shape[2] == 3:  # If foreground doesn't have an alpha channel
    print("Foreground image lacks an alpha channel, adding one...")
    alpha_channel = np.ones((fg_image.shape[0], fg_image.shape[1], 1), dtype=np.uint8) * 255
    fg_image = np.concatenate((fg_image, alpha_channel), axis=-1)

# Blend images
overlapper = ImageBlending()

# Load background image to get its dimensions
bg_height, bg_width = bg_image.shape[:2]

# Resize foreground image to match background
fg_image_resized = cv2.resize(fg_image, (bg_width, bg_height), interpolation=cv2.INTER_LINEAR)

# Blend images
final_composite = overlapper.blend_images(fg_image_resized, bg_image)

# Save the final blended image
cv2.imwrite(FINAL_OUTPUT_PATH, final_composite)
print(f"Saved Final composited image at {FINAL_OUTPUT_PATH}")