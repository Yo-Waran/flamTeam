import cv2
import numpy as np
import os
from diffusion_light_hdri_mod import DiffusionLightPipeline  # Import HDRI extraction module
from switch_light_pipeline_mod import SwitchLightPipeline  # Import Relighting module
from image_blending_mod import ImageBlending  # Import Alpha Blending module
from blender_relight_mod import BlenderRelight #Import Relighting Module
from blender_cubemap_mod import HDRIExposureMatcher #Import HDRI exposure Matcher module


# Input Paths
BG_IMG_PATH = "cocreation/assets/bg/bg7.png"
FG_IMG_PATH = "cocreation/assets/fg1.png"
#input cubemap paths
#CUBEMAP_PATH = 'cocreation/assets/cubemaps/cb6.hdr'
CUBEMAP_PATH = None
CUBEMAP_ADJUSTED_PATH = "cocreation/assets/cubemaps/cb7_adjusted.hdr"

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #get parent directory of the current file
MAPS_DIR = os.path.join(BASE_DIR, "maps")
HDRI_DIR = os.path.join(BASE_DIR, "hdri")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Output Paths
HDRI_OUTPUT_PATH = os.path.join(HDRI_DIR, "extracted_hdri_7.exr")
RELIT_SUBJECT_PATH = os.path.join(OUTPUT_DIR, "di_relit_subject_7.png")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_output.png")

# PBR Maps
ALBEDO_PATH = os.path.join(MAPS_DIR, "albedo.png")
NORMAL_PATH = os.path.join(MAPS_DIR, "normal.png")
ROUGHNESS_PATH = os.path.join(MAPS_DIR, "roughness.png")
ALPHA_PATH = os.path.join(MAPS_DIR, "alpha.png")
    
# Ensure all required directories exist
for path in [ MAPS_DIR, HDRI_DIR, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)

##################################################################
# Step 1: Extract HDRI from the background image if there is no CubeMap available  . If CubeMap exists , then use that
##################################################################

if not CUBEMAP_PATH:
    print("Step 1: Extracting HDRI...")
    BG_IMG = cv2.imread(BG_IMG_PATH,cv2.IMREAD_UNCHANGED)
    hdr_extractor = DiffusionLightPipeline()
    EXTRACTED_HDRI = hdr_extractor.infer(BG_IMG,HDRI_OUTPUT_PATH)  # Save output HDRI path
else:
    hdri_exposure_matcher = HDRIExposureMatcher() #make an instance of the HDRI exposure Class
    print("Step 1: Using Existing Cube Map...")
    adjustment_factor, analysis_data = hdri_exposure_matcher.analyze_images(BG_IMG_PATH, CUBEMAP_PATH) #analyze the bg image and the HDRI
    adjusted_hdri = hdri_exposure_matcher.apply_hdri_exposure_adjustment(analysis_data['hdri'], adjustment_factor) #apply the HDRI adjustments in a new blender scene
    print("Adjusting Existing Cube Map according to BG...")
    hdri_exposure_matcher.save_hdri(adjusted_hdri, CUBEMAP_ADJUSTED_PATH) #apply the adjustments and save the new hdri
    print(f"Saved adjusted HDRI to {CUBEMAP_ADJUSTED_PATH}")

##################################################################
# Step 2: Relight the foreground image using the extracted HDRI if there is no Cubemap available. If CubeMap exists, then use that for relighting
##################################################################
'''
print("Step 2: Extracting PBR Maps ...")
pbr_extractor = SwitchLightPipeline()
pbr_extractor.infer(FG_IMG_PATH,MAPS_DIR)  # Save extracted PBR images
'''
##################################################################
#Step 3:Relight the Subject
##################################################################
relighter = BlenderRelight()

if not CUBEMAP_PATH:
    print("Step 3: Relighting Subject using Extracted HDRI...")
    RELIT_SUBJECT = relighter.setup_relighting_scene(ALBEDO_PATH,NORMAL_PATH,ROUGHNESS_PATH,ALPHA_PATH,HDRI_OUTPUT_PATH,RELIT_SUBJECT_PATH)
else:
    print("Step 3: Relighting Subject using Adjusted CubeMaps...")
    RELIT_SUBJECT = relighter.setup_cubemap_relighting_scene(ALBEDO_PATH,NORMAL_PATH,ROUGHNESS_PATH,ALPHA_PATH,CUBEMAP_PATH, CUBEMAP_ADJUSTED_PATH, RELIT_SUBJECT_PATH)

'''
##################################################################
# Step 4: Load images and perform alpha blending
##################################################################
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

##################################################################
# Save the final blended image
##################################################################
cv2.imwrite(FINAL_OUTPUT_PATH, final_composite)
print(f"Saved Final composited image at {FINAL_OUTPUT_PATH}")
'''

