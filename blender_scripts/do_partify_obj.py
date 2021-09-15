#bpy requires having blender imported as a module
# https://wiki.blender.org/wiki/Building_Blender/Other/BlenderAsPyModule
import bpy
from mathutils import Vector #bpy mathutils

import pathlib, time

PTH_CUTR = pathlib.Path(r'C:\tmp\cutter.blend')
PTH_SRC = pathlib.Path(r'C:\tmp')
PTH_DST = pathlib.Path(r'C:\tmp')

MSH_NAME_DIF = "SUBTRACT" # name of object in cutter file to use as a subtraction

def main():

    # load cutter once to find names of cutting objects
    objs_xsec_names, obj_dif_name = load_cutter(PTH_CUTR, verbose=True)
    cfg = {
        'objs_xsec_names':objs_xsec_names,
        'obj_dif_name':obj_dif_name,
    }

    files = sorted([p.resolve() for p in PTH_SRC.glob("*") if p.suffix in [".obj"]])
    for f in files:
        start_time = time.time()
        print("\n\n------------------------------ {}\n-----".format(f.name))
        dst = f
        if PTH_DST: dst = PTH_DST / f.name
        process_path(cfg,f,dst)
        print("----- processed {} in {}s\n------------------------------".format(f.name, (time.time() - start_time)))
        


def process_path(cfg,pth_src, pth_dst):
    
    for obj_xsec_name in cfg['objs_xsec_names']:
        do_cutting(obj_xsec_name,cfg['obj_dif_name'],pth_src, pth_dst)
        

def do_cutting(obj_xsec_name, obj_dif_name,pth_src, pth_dst):
        print("----- {}".format(obj_xsec_name))
        load_cutter(PTH_CUTR)
        target_obj = import_obj(pth_src)

        bpy.ops.object.mode_set(mode='EDIT') # into edit mode
        bpy.ops.mesh.normals_make_consistent(inside=False) # unify normals
        bpy.ops.object.mode_set(mode='OBJECT') # into object mode
        
        
        print("... intersecting {} with {}".format(target_obj.name, obj_xsec_name))
        # intersect with xsec
        xsec_mod = target_obj.modifiers.new(name='xsec_' + obj_xsec_name, type='BOOLEAN')
        xsec_mod.operation = 'INTERSECT'
        xsec_mod.solver = 'EXACT'
        xsec_mod.object = bpy.data.objects[obj_xsec_name]
        # subtract with dif
        
        print("... subtracting {} from {}".format(obj_dif_name, target_obj.name))
        diff_mod = target_obj.modifiers.new(name='diff_' + obj_dif_name, type='BOOLEAN')
        diff_mod.operation = 'DIFFERENCE'
        diff_mod.solver = 'EXACT'
        diff_mod.object = bpy.data.objects[obj_dif_name]
        

        # apply all modifiers.
        for modifier in target_obj.modifiers:
            print("... applying modifier {}".format(modifier.name))
            bpy.ops.object.modifier_apply(modifier=modifier.name)

        '''
        # delete cutting objects
        print(bpy.context.selected_objects)
        for o in bpy.context.scene.objects:
            print("{}\t{}".format(o.name,o.select_get()))
        bpy.data.objects[target_obj.name].select_set(False) # deselect imported object
        bpy.data.objects[obj_xsec_name].select_set(True)
        bpy.data.objects[obj_dif_name].select_set(True)
        bpy.ops.object.delete()
        '''

        bpy.data.objects[target_obj.name].select_set(True) # select imported object
        bpy.ops.object.mode_set(mode='EDIT') # into edit mode
        triangulate()
        bpy.ops.object.mode_set(mode='OBJECT') # into object mode

        save_obj(pth_dst, obj_xsec_name.lower())


# assumes mesh object is selected, and Blender is in edit mode
def triangulate():
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()


def save_obj(pth, suffix):
    pth_obj = pathlib.Path(pth.parent / "{}_{}".format(pth.stem, suffix) ).with_suffix('.obj')

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


def load_cutter(pth, verbose=False):
    bpy.ops.wm.open_mainfile(filepath=str(pth))
    if verbose: print("... cutter file loaded from {}".format(pth))

    obj_dif = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.name == MSH_NAME_DIF][0]
    obj_dif_name = obj_dif.name

    objs_xsec = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.name != MSH_NAME_DIF]
    objs_xsec_names = [m.name for m in objs_xsec]
    
    if verbose: print("... cuting objs_xsec loaded as {}".format(objs_xsec_names))
    if verbose: print("... cuting obj_dif loaded as {}".format(obj_dif_name))
    return objs_xsec_names, obj_dif_name


def import_obj(pth):
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