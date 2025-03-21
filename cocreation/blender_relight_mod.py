import bpy
import os
import cv2

class BlenderRelight():
    """
    A class for setting up and rendering a relighting scene in Blender using provided texture maps.
    """
    
    def __init__(self):
        """
        Initializes the BlenderRelight class.
        """
        pass

    def clear_scene(self):
        """
        Deletes all objects, materials, and lights in the Blender scene.
        """
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        bpy.ops.outliner.orphans_purge()

    def setup_relighting_scene(self, albedo_path, normal_path, roughness_path, opacity_map_path, hdri_path, output_path):
        """
        Sets up and renders a relighting scene in Blender using provided texture maps.
        
        Args:
            albedo_path (str): Path to the albedo texture image.
            normal_path (str): Path to the normal map image.
            roughness_path (str): Path to the roughness map image.
            opacity_map_path (str): Path to the opacity map image.
            hdri_path (str): Path to the HDRI environment map.
            output_path (str): Path to save the rendered output image.
        
        Returns:
            str: The output file path of the rendered image.
        
        Raises:
            ValueError: If the albedo image fails to load.
        """
        self.clear_scene()

        image = cv2.imread(albedo_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Error: Could not load image at {albedo_path}")

        height, width = image.shape[:2]
        
        # Set Blender render resolution to match the image dimensions
        scene = bpy.context.scene
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        print(f"Render resolution set to: {width}x{height}")
        
        # Load cropped albedo to get dimensions
        img = bpy.data.images.load(albedo_path)
        width, height = img.size
        
        # Create image plane
        bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
        plane = bpy.context.object
        
        # Scale the plane to match image aspect ratio
        plane.scale.x = width / max(width, height)
        plane.scale.y = height / max(width, height)
        
        # Create material
        mat = bpy.data.materials.new(name="RelightMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        
        # Load textures and assign to shader
        def add_texture(image_path, node_type, target_input, is_alpha=False):
            """
            Loads a texture into the Blender shader and assigns it to a material.
            
            Args:
                image_path (str): Path to the texture image.
                node_type (str): Type of shader node to create.
                target_input (str): Shader input to which the texture is connected.
                is_alpha (bool, optional): Whether the texture is an alpha map. Defaults to False.
            """
            if not os.path.exists(image_path):
                print(f"Warning: {image_path} not found.")
                return
            img_node = mat.node_tree.nodes.new(type=node_type)
            img_node.image = bpy.data.images.load(image_path)

            if is_alpha:
                img_node.image.colorspace_settings.name = 'Non-Color'
                mat.blend_method = 'BLEND'
                mat.shadow_method = 'HASHED'
                mat.use_backface_culling = True
                mat.node_tree.links.new(img_node.outputs[0], bsdf.inputs['Alpha'])
            else:
                mat.node_tree.links.new(img_node.outputs[0], bsdf.inputs[target_input])
        
        add_texture(albedo_path, 'ShaderNodeTexImage', 'Base Color')
        add_texture(normal_path, 'ShaderNodeTexImage', 'Normal')
        add_texture(roughness_path, 'ShaderNodeTexImage', 'Roughness')
        add_texture(opacity_map_path, 'ShaderNodeTexImage', 'Alpha', is_alpha=True)
        
        plane.data.materials.append(mat)
        
        # Set up camera
        bpy.ops.object.camera_add(location=(0, 0, 2))
        cam = bpy.context.object
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = 2
        cam.rotation_euler = (0, 0, 0)
        bpy.context.scene.camera = cam
        
        # Set up HDRI environment lighting
        if os.path.exists(hdri_path):
            bpy.context.scene.world.use_nodes = True
            env_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeTexEnvironment')
            env_node.image = bpy.data.images.load(hdri_path)
            bg_node = bpy.context.scene.world.node_tree.nodes.get('Background')
            bpy.context.scene.world.node_tree.links.new(env_node.outputs[0], bg_node.inputs[0])
        else:
            print(f"Warning: HDRI {hdri_path} not found.")
        
        # Render settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.filepath = output_path
        
        # Render the scene
        bpy.ops.render.render(write_still=True)
        print(f"Rendered image saved to {output_path}")
        
        return output_path

if __name__ == "__main__":
    religter = BlenderRelight()
    religter.setup_relighting_scene(
        albedo_path="/root/Ram/DiffusionLight/cocreation/maps/albedo.png",
        normal_path="/root/Ram/DiffusionLight/cocreation/maps/normal.png",
        roughness_path="/root/Ram/DiffusionLight/cocreation/maps/roughness.png",
        opacity_map_path="/root/Ram/DiffusionLight/cocreation/maps/alpha.png",
        hdri_path="/root/Ram/DiffusionLight/cocreation/hdri/extracted_hdri1.exr",
        output_path="/root/Ram/DiffusionLight/cocreation/output/relit_subject12.png"
    )
