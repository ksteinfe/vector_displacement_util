import time, os, pathlib, math
from vdisp.imgop import flip_y, flip_z
import numpy as np
import pymeshlab




def remesh_path(pth_src, pth_dst):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(pth_src)
    print("loaded mesh")
    ms.surface_reconstruction_screened_poisson()
    ms.save_current_mesh(pth_dst)


def merge_directory(pth_src, pth_dst, fname):
    pths_obj = sorted([p.resolve() for p in (pth_src).glob("*") if p.suffix in [".obj", ".ply"]])
    if len(pths_obj) < 2: raise FileNotFoundError("Not enough mesh files in {}.\nI found {}, but need at least 2".format(pth_src, len(pths_obj)))
    print("Discovered {} source mesh files.".format(len(pths_obj)))
        
    ms = pymeshlab.MeshSet()        
    for pth in pths_obj:
        if pth_src == pth_dst and pth.name == fname: 
            print("...skipping {}".format(pth.name))
            continue
        ms.load_new_mesh(str(pth))
        print("... loaded {}".format(pth.stem))
    
    pth_out = pth_dst / fname
    print("... flattening and saving to {}".format(pth_out))
    ms.flatten_visible_layers(mergevertices=False)
    ms.save_current_mesh(str(pth_out))



def do_bool(pth_a, pth_b, typ, save_as=False):
    ms = pymeshlab.MeshSet()

    start_time = time.time()
    ms.load_new_mesh(str(pth_a))
    print("... loaded base mesh in Meshlab.")
    ms.load_new_mesh(str(pth_b))
    print("... loaded other mesh. now there are {} meshes.".format(len(ms)))
    if typ=="intersection":
        ms.mesh_boolean_intersection(first_mesh=0, second_mesh=1)
    elif typ=="union":
        ms.mesh_boolean_union(first_mesh=0, second_mesh=1)     
    elif typ=="difference":
        ms.mesh_boolean_difference(first_mesh=0, second_mesh=1)           
    else:
        raise ValueError("{} is an unknown boolean operation".format(typ)) 

    print("... completed boolean {} operation in {:.1f}s. now there are {} meshes.".format(typ,(time.time() - start_time),len(ms)))
    ms.remove_duplicate_vertices()
    ms.remove_duplicate_faces()
    ms.merge_close_vertices()

    pth_save = str(save_as) if save_as else str(pth_a)
    ms.save_current_mesh(pth_save)


def do_thicken_and_cut_window(pth_a, pth_b, offset_dist, disjoint_size=False, save_as=False):
    ms = pymeshlab.MeshSet()

    start_time = time.time()
    ms.load_new_mesh(str(pth_a))
    print("... loaded base mesh in Meshlab.")

    ms.duplicate_current_layer()
    print("... duplicated layer. Now there are {} meshes".format(len(ms)))
    #thrsh = np.abs(offset_dist/5)
    #ms.simplification_clustering_decimation(threshold=thrsh)
    #print("... simplification_clustering_decimation with a threshold of {}. Now there are {} meshes".format(thrsh, len(ms)))\
    perc = 0.01
    ms.simplification_quadric_edge_collapse_decimation(targetperc=perc)
    print("... simplification_quadric_edge_collapse_decimation with a targetperc of {}. Now there are {} meshes".format(perc, len(ms)))

    # I don't understand the 'offset' parameter
    # documentation describes as a percentage (where 50% is on the surface), but this operates in absolute units
    print("... resampling with an offset of {}".format(offset_dist))
    ms.uniform_mesh_resampling(offset=offset_dist, mergeclosevert=True , multisample =True)
    print("... completed uniform_mesh_resampling operation in {:.1f}s. now there are {} meshes.".format((time.time() - start_time),len(ms)))

    if not disjoint_size: disjoint_size = np.abs(offset_dist * 10)
    ms.remove_isolated_pieces_wrt_diameter(mincomponentdiag=disjoint_size)
    print("... removed isolated pieces with a diameter less than {}".format(disjoint_size))


    ms.load_new_mesh(str(pth_b))
    print("... loaded other mesh. now there are {} meshes.".format(len(ms)))
    
    ms.mesh_boolean_union(first_mesh=2, second_mesh=3)
    print("... completed boolean union operation. now there are {} meshes.".format(len(ms)))
    ms.remove_duplicate_vertices()
    ms.remove_duplicate_faces()
    ms.merge_close_vertices()

    ms.mesh_boolean_difference(first_mesh=0, second_mesh=4)
    print("... completed boolean difference operation. now there are {} meshes.".format(len(ms)))

    pth_save = str(save_as) if save_as else str(pth_a)
    ms.save_current_mesh(pth_save)

def ERASE__do_thicken_and_cut_window(pth_a, pth_b, offset_dist, disjoint_size=False, save_as=False):
    ms = pymeshlab.MeshSet()

    start_time = time.time()
    ms.load_new_mesh(str(pth_a))
    print("... loaded base mesh in Meshlab.")

    # I don't understand the 'offset' parameter
    # documentation describes as a percentage (where 50% is on the surface), but this operates in absolute units
    print("... resampling with an offset of {}".format(offset_dist))
    ms.uniform_mesh_resampling(offset=offset_dist, mergeclosevert=True , multisample =True)
    print("... completed uniform_mesh_resampling operation in {:.1f}s. now there are {} meshes.".format((time.time() - start_time),len(ms)))

    if not disjoint_size: disjoint_size = np.abs(offset_dist * 10)
    ms.remove_isolated_pieces_wrt_diameter(mincomponentdiag=disjoint_size)
    print("... removed isolated pieces with a diameter less than {}".format(disjoint_size))


    ms.load_new_mesh(str(pth_b))
    print("... loaded other mesh. now there are {} meshes.".format(len(ms)))
    
    ms.mesh_boolean_union(first_mesh=1, second_mesh=2)     
    print("... completed boolean union operation. now there are {} meshes.".format(len(ms)))
    ms.remove_duplicate_vertices()
    ms.remove_duplicate_faces()
    ms.merge_close_vertices()

    ms.mesh_boolean_difference(first_mesh=0, second_mesh=3)     
    print("... completed boolean difference operation. now there are {} meshes.".format(len(ms)))

    pth_save = str(save_as) if save_as else str(pth_a)
    ms.save_current_mesh(pth_save)



def do_flip_and_or_swap_axis(pth_a, save_as=False, about_pt=False, flipx=False, flipy=False, flipz=False, swapxy=False, swapxz=False, swapyz=False, invert_faces=True):
    
    if about_pt: do_translate(pth_a, (-about_pt[0],-about_pt[1],-about_pt[2]) )

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(pth_a))
    ms.transform_flip_and_or_swap_axis(flipx=flipx, flipy=flipy, flipz=flipz, swapxy=swapxy, swapxz=swapxz, swapyz=swapyz)
    if invert_faces: ms.invert_faces_orientation()
    pth_save = str(save_as) if save_as else str(pth_a)
    ms.save_current_mesh(pth_save)

    if about_pt: do_translate(pth_a,about_pt)


def do_translate(pth_a, vec, save_as=False):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(pth_a))
    ms.transform_translate_center_set_origin(traslmethod=3, neworigin=[ vec[0], vec[1], vec[2] ])
    pth_save = str(save_as) if save_as else str(pth_a)
    ms.save_current_mesh(pth_save)


def do_scale(pth_a, scl, save_as=False):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(pth_a))
    ms.transform_scale_normalize(axisx=scl, uniformflag=True)
    pth_save = str(save_as) if save_as else str(pth_a)
    ms.save_current_mesh(pth_save)



# axis is 0=X 1=Y 2=Z
# angle is in degrees
def do_rotate(pth_a, axis, angle, cpt, save_as=False):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(pth_a))
    ms.transform_rotate(rotaxis=axis, angle=angle, customcenter=cpt, rotcenter=2 ) #rotcenter=2=rotate around custom point
    pth_save = str(save_as) if save_as else str(pth_a)
    ms.save_current_mesh(pth_save)


def validate_module_mesh(fname, pth_msh):
    files = [p.resolve() for p in pth_msh.glob("*") if p.suffix in [".ply", ".obj"]]
    pth_partbool = [p for p in files if p.name == fname]
    if len(pth_partbool)!=1: raise FileNotFoundError("Boolean mesh {} was not found in the module mesh directory {}".format(fname,pth_msh))
    return pth_partbool[0]