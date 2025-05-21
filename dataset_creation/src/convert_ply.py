import bpy
import os

'''
# This script is designed to be run in Blender 4.x
# It imports a .ply file, converts it to a curve, and exports it as an .obj file.
# The script uses the new Blender 4.x API for importing and exporting.
# Ensure you have the necessary permissions to run this script.
# Usage:
# 1. Open Blender 4.x
# 2. Open the Scripting workspace
# 3. Copy and paste this script into the text editor
# 4. Adjust the input and output file paths as needed   
# 5. Run the script
'''
# Function to import .ply, convert to curve, and export as .obj

def import_convert_export(input_ply_path, output_path):
    # Clean up the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # ✅ Import the .ply using Blender 4.x correct method
    bpy.ops.wm.ply_import(filepath=input_ply_path)

    # Get the imported object
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    print("Converting to Curve...")
    # Convert to Curve
    bpy.ops.object.convert(target='CURVE')

    print("Changing to NURBS...")
    # Edit mode to select all and change spline type
    bpy.ops.object.editmode_toggle()
    bpy.ops.curve.select_all(action='SELECT')
    bpy.ops.curve.spline_type_set(type='NURBS')
    bpy.ops.object.editmode_toggle()

    print("Exporting to OBJ...")
    # Export curve as .obj with nurbs support
    bpy.ops.wm.obj_export(
        filepath=output_path,
        export_selected_objects=True,
        apply_modifiers=True,
        export_curves_as_nurbs=True
    )

    print(f"✅ Exported to: {output_path}")

# Example usage:

if __name__ == "__main__":
    ply_path = "/Users/ramyogeshwaran/Downloads/avijit.ply"
    obj_path = "/Users/ramyogeshwaran/Downloads/avijit.obj"
    # Ensure the file paths are correct
    if not os.path.exists(ply_path):
        print(f"❌ Input file does not exist: {ply_path}")
    else:
        # Call the function to import, convert, and export
        import_convert_export(ply_path, obj_path)

'''
# Run this script in MacOS using the command line:
/Applications/Blender.app/Contents/MacOS/Blender --background --python /Users/ramyogeshwaran/Documents/RamYogeshwaran/Repo/flamTeam/dataset_creation/src/convert_ply.py
'''