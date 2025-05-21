import maya.cmds as cmds

# Replace "scalp_mesh" with the actual name of your scalp mesh

'''
 Ensure the curves are selected in Maya before running this script
 You can run this script in Maya's script editor or as a Python script
 Make sure to have the curves selected in the Maya scene
 and the scalp mesh named "scalp_mesh" in the scene.
 This script will process the selected curves, project them to the scalp mesh,
 and optionally rebuild them with the specified spans.
 Adjust the rebuild and spans parameters as needed.
 Note: This script assumes you have Maya installed and running.
 It won't run in a standard Python environment.
'''

# This script prepares curves for XGen by projecting them onto a scalp mesh.
def prepare_curves_for_xgen(curves, scalp_mesh, rebuild=False, spans=10):
    if not cmds.objExists(scalp_mesh):
        cmds.warning("Scalp mesh does not exist: {}".format(scalp_mesh))
        return
    
    print("Processing {} curves...".format(len(curves)))

    # Create closestPointOnMesh node
    cpom_node = cmds.createNode("closestPointOnMesh", name="temp_cpom")
    scalp_shape = cmds.listRelatives(scalp_mesh, shapes=True, fullPath=True)[0]
    cmds.connectAttr(scalp_shape + ".worldMesh[0]", cpom_node + ".inMesh", force=True)

    final_curves = []

    for curve in curves:
        if not cmds.objExists(curve):
            continue

        # Reverse the curve direction
        cmds.reverseCurve(curve, ch=False, rpo=True)

        # Project first CV to the scalp mesh
        first_cv = "{}.cv[0]".format(curve)
        root_pos = cmds.pointPosition(first_cv, world=True)
        cmds.setAttr(cpom_node + ".inPosition", *root_pos, type="double3")
        closest_pos = cmds.getAttr(cpom_node + ".position")[0]
        cmds.xform(first_cv, worldSpace=True, translation=closest_pos)

        # Rebuild if enabled
        if rebuild:
            rebuilt_curve = cmds.rebuildCurve(curve, 
                                              ch=False, 
                                              rpo=True, 
                                              spans=spans, 
                                              keepRange=0, 
                                              replaceOriginal=True)[0]
            final_curves.append(rebuilt_curve)
        else:
            final_curves.append(curve)

    cmds.delete(cpom_node)
    print("Done. {} curves processed.".format(len(final_curves)))
    cmds.select(final_curves)

# Example usage
if __name__=="__main__":      
    curves = cmds.ls(selection=True, type="transform")
    prepare_curves_for_xgen(curves, scalp_mesh="scalp_mesh", rebuild=False, spans=8)
