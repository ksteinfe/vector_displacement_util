#bpy requires having blender imported as a module
# https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule
import bpy
from mathutils import Vector #bpy mathutils

import pathlib, time

PTH_BASE = pathlib.Path(r'C:\tmp\solid_base.blend') 
PTH_SRC = pathlib.Path(r'C:\tmp') 
PTH_DST = pathlib.Path(r'C:\tmp')
SUFFIX = "sld"


def main():
    files = sorted([p.resolve() for p in PTH_SRC.glob("*") if p.suffix in [".obj"]])
    for f in files:
        start_time = time.time()
        print("\n\n------------------------------ {}\n-----".format(f.name))
        dst = f
        if PTH_DST: dst = PTH_DST / f.name
        process_path(f,dst)
        print("----- processed {} in {}s\n------------------------------".format(f.name, (time.time() - start_time)))
        
        


def process_path(pth_src, pth_dst):
    obj_base = False
    if PTH_BASE: obj_base = load_base(PTH_BASE)
    
    import_obj(pth_src)

    bpy.ops.object.mode_set(mode='EDIT') # into edit mode
    
    print("... cleaning up")
    initial_cleanup()
    
    print("... decimating and smoothing along open edge")
    for n in [4,3,2,1]:
        decimate_near_boundary(n, 0.5)
        smooth_near_boundary(n)
    decimate_near_boundary(0, 0.5) # one last big decimate
    smooth_near_boundary(0, 1.0, 10) # one last big one smooth
    
    
    print("... flattening edge")
    flatten_edge(do_fill=obj_base is False)

    if obj_base:
        print("... joining and bridging")
        join_all_meshes()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.region_to_loop() # select boundary loops
        bpy.ops.mesh.bridge_edge_loops() # bridge
        bpy.ops.mesh.normals_make_consistent(inside=False) # unify normals
    
    '''
    print("... remeshing")
    remesh()
    
    print("... dissolving mesh")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.dissolve_limited() # angle_limit=0.0872665 (radians), use_dissolve_boundaries=False, delimit={'NORMAL'}
    '''

    print("... unifying normals")
    bpy.ops.mesh.normals_make_consistent(inside=False) # unify normals
    
    bpy.ops.object.mode_set(mode='OBJECT') # back into object mode
    save_obj(pth_dst)


# assumes mesh object is selected, and Blender is in edit mode
def triangulate():
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()

# assumes mesh object is selected, and Blender is in edit mode
def remesh(oct_dpth=8):
    bpy.ops.object.mode_set(mode='OBJECT') # back into object mode
    
    target_obj = bpy.context.active_object # imported OBJ should be active here
    bool_mod = target_obj.modifiers.new(name='remsh', type='REMESH')
    bool_mod.mode = 'SHARP'
    bool_mod.octree_depth = oct_dpth
    #bool_mod.mode = 'VOXEL'
    #bool_mod.voxel_size = 0.01

    # apply all modifiers.
    for modifier in target_obj.modifiers:
        print("... applying {}".format(modifier.name))
        bpy.ops.object.modifier_apply(modifier=modifier.name)

    bpy.ops.object.mode_set(mode='EDIT') # back into edit mode

# assumes mesh object is selected, and Blender is in edit mode
def flatten_edge(pow=0.5, do_fill=False):
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.region_to_loop() # select boundary loops    
    
    bpy.ops.object.mode_set(mode = 'OBJECT') # Back to object mode so that we can select vertices
    verts = [i for i in bpy.context.active_object.data.vertices if i.select]
    print("... flattenting {} verts along edge (out of {} in mesh)".format(len(verts),len(bpy.context.active_object.data.vertices)))
    minz = min([v.co.z for v in verts])
    for v in verts: v.co.z = v.co.z - ((v.co.z - minz)*pow)
    bpy.ops.object.mode_set(mode='EDIT') # into edit mode
    
    # dissolve non-manifold geometry
    bpy.ops.mesh.select_non_manifold(extend=False, use_boundary=False)
    bpy.ops.mesh.dissolve_verts(use_face_split=True)
    bpy.ops.mesh.select_all(action='DESELECT')

    # delete non-manifold geometry
    bpy.ops.mesh.select_non_manifold(extend=False, use_boundary=False)
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')

    triangulate()

    if do_fill: 
        print("... capping flat edge")
        bpy.ops.mesh.fill()
        

# assumes mesh object is selected, and Blender is in edit mode
def decimate_near_boundary(selstp, rat):
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.region_to_loop() # select boundary loops    
    for n in range(selstp): bpy.ops.mesh.select_more(use_face_step = False)
    bpy.ops.mesh.decimate(ratio=rat)

# assumes mesh object is selected, and Blender is in edit mode
def smooth_near_boundary(selstp, fac=1.0, rep=1):
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.region_to_loop() # select boundary loops    
    for n in range(selstp): bpy.ops.mesh.select_more(use_face_step = False)
    bpy.ops.mesh.vertices_smooth(factor=fac, repeat=rep)

# assumes mesh object is selected, and Blender is in edit mode
def initial_cleanup(iter=5):

    for n in range(iter):
        # delete loose geometry
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_loose()
        bpy.ops.mesh.delete(type='VERT')

        # dissolve degenerate geometry
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.dissolve_degenerate(threshold=0.0002)

        # dissolve non-manifold geometry
        bpy.ops.mesh.select_non_manifold(extend=False, use_boundary=False)
        bpy.ops.mesh.dissolve_verts(use_face_split=True)
        bpy.ops.mesh.select_all(action='DESELECT')

# assumes mesh object is selected, and Blender is in edit mode
def join_all_meshes():
    bpy.ops.object.mode_set(mode='OBJECT') # back into object mode
    names = []
    for ob in bpy.context.scene.objects:
        if ob.type == 'MESH':
            names.append(ob.name)
            ob.select_set(state=True)
            bpy.context.view_layer.objects.active = ob
        else:
            ob.select = False
    bpy.ops.object.join()
    print("... joined {} meshes: {}".format(len(names),names))
    bpy.ops.object.mode_set(mode='EDIT') # back into edit mode



def save_obj(pth):
    pth_obj = pathlib.Path(pth.parent / "{}_{}".format(pth.stem, SUFFIX) ).with_suffix('.obj')

    print("... saving obj to {}".format(pth_obj))
    bpy.ops.export_scene.obj(
        filepath=str(pth_obj), 
        check_existing=False, 
        axis_forward='-X', 
        axis_up='Z', 
        use_selection=True,
        use_materials=False, 
        global_scale=1, 
        path_mode='AUTO'
        )


def load_base(pth):
    bpy.ops.wm.open_mainfile(filepath=str(pth))
    print("... base file loaded from {}".format(pth))

    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if "Base" not in [m.name for m in objs]:
        raise Exception("No mesh objects named 'Base' in the base file!")
    
    obj = [m for m in objs if m.name=="Base"][0]
    print("... base obj loaded as {}".format(obj))
    return obj


def import_obj(pth):
    try:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()
    except KeyError:
        pass

    bpy.ops.import_scene.obj(filepath=str(pth), axis_forward='-X', axis_up='Z')
    print("... obj loaded from {}".format(pth))
    #print("selected after import: {}".format(bpy.context.selected_objects))
    obj = bpy.context.selected_objects[-1] # most recently selected thing
    print("... obj loaded as {}".format(obj))

    bpy.context.view_layer.objects.active = bpy.context.scene.objects.get(obj.name)
    bpy.data.objects[obj.name].select_set(True) # select imported object
    return obj

if __name__== "__main__":
  main()