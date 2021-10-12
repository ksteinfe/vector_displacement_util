import time, os, pathlib, traceback
import tempfile
import vdisp
from .io import read_tif16
from tqdm import tqdm
import numpy as np
import pymeshlab


# config for smoothing and culling based on alpha of image
ALPHA_SIG_SMOOTH_K = 500 # displacement is scaled by a sigmoid function of alpha of image. higher values here create sharper falloff of alpha
ALPHA_SIG_SMOOTH_MID = 0.01 # displacement is scaled by a sigmoid function of alpha of image. this is the midpoint of sigmoid, effecively the 'center point' of the falloff


# pth_src and pth_dest expect pathlib.Path
def solidify_directory(pth_src, xydisp_full, zdisp_inout, reconstruction_depth=8, trim_to=False, pth_dst = None):
    if not isinstance(pth_src, pathlib.Path): pth_src = pathlib.Path(pth_src)
    if pth_dst is None: pth_dst = pth_src
    elif not isinstance(pth_dst, pathlib.Path): pth_dst = pathlib.Path(pth_dst)

    files = sorted([p.resolve() for p in pth_src.glob("*") if p.suffix in [".tif", ".tiff"]])
    print("Discovered {} TIFF files in {}".format(len(files),pth_src))
    for n,f in enumerate(files):
        #if n%4!=0: continue
        start_time = time.time()
        print("\n\n------------------------------ {}\n-----".format(f.name))
        try:
            solidify_path(f,pth_dst,xydisp_full, zdisp_inout, reconstruction_depth=reconstruction_depth, trim_to="unit_circle.ply")
            print("----- processed {} in {}s\n------------------------------".format(f.name, (time.time() - start_time)))
        except Exception as e:
            print("!!!!! FAILED {} in {}s\n!!!!!!!!!!!!!!!!!!!!!".format(f.name, (time.time() - start_time)))
            print(traceback.format_exc())

        


def solidify_path(pth_src, pth_dst, xydisp_full, zdisp_inout, reconstruction_depth=8, trim_to=False):
    pth_trim = False
    if trim_to:
        pth_msh = pathlib.Path(vdisp.__file__).parent / 'msh' # path to module mesh directory
        files = [p.resolve() for p in pth_msh.glob("*") if p.suffix in [".ply", ".obj"]]
        pth_trim = [p for p in files if p.name == trim_to]
        if len(pth_trim)!=1: raise Exception("Cannot trim to mesh {} because it was not found in the module mesh directory".format(trim_to))
        pth_trim = str(pth_trim[0])

    print("... constructing base mesh.")
    start_time = time.time()
    pth_dst = pathlib.Path(pth_dst / "{}".format(pth_src.stem) ).with_suffix('.obj')
    img = read_tif16(pth_src,xydisp_full, zdisp_inout)
    pgrid = _img_to_pointgrid(img) # (512, 512, 5) (x,y,z,a,idx)
    bgrid = _back_pointgrid(img) # (512, 512, 5) (x,y,z,a,idx)
    objstr = _pts_to_objstr(pgrid, bgrid)
    print("... completed base mesh construction in {:.1f}s.".format(time.time() - start_time))

    pth_obj = pathlib.Path(pth_dst.parent / "{}".format(pth_dst.stem) ).with_suffix('.obj')
    if not trim_to:
        with open(pth_obj, 'w') as f:
            f.writelines(objstr)
    else:
        try:
            tf = tempfile.NamedTemporaryFile(suffix='.obj', mode='w',delete=False)
            with tf:
                tf.writelines(objstr)

            ms = pymeshlab.MeshSet()
            
            ms.load_new_mesh(tf.name)
            print("... loaded base mesh in Meshlab.")

            ms.surface_reconstruction_screened_poisson(depth=reconstruction_depth) # default is 8
            print("... completed ms.surface_reconstruction_screened_poisson(). now there are {} meshes.".format(len(ms)))
            ms.load_new_mesh(pth_trim)
            print("... loaded trimming mesh. now there are {} meshes.".format(len(ms)))
            
            start_time = time.time()
            ms.mesh_boolean_intersection(first_mesh=1, second_mesh=2)
            print("... completed ms.mesh_boolean_intersection() in {:.1f}s. now there are {} meshes.".format((time.time() - start_time),len(ms)))

            start_time = time.time()
            ms.simplification_quadric_edge_collapse_decimation(preserveboundary=True, preservenormal=True, preservetopology=True)
            print("... completed ms.simplification_quadric_edge_collapse_decimation() in {:.1f}s. now there are {} meshes.".format((time.time() - start_time),len(ms)))

            ms.remove_duplicate_vertices()
            ms.remove_duplicate_faces()
            ms.merge_close_vertices()
            ms.save_current_mesh(str(pth_obj))
        except Exception as e:
            print("!!! PYMESHLAB ERROR")
            print(e)
        finally:
            print("... closing temporary mesh.")
            tf.close()
            os.unlink(tf.name)



def _pts_to_objstr(pgrid, bgrid):

    objstr = []
    verts = {}
    vstr = "v {0:f} {1:f} {2:f}\n"
    n = 0
    for yi in range(pgrid.shape[0]): # grid is indexed y,x
        for xi in range(pgrid.shape[1]):
            objstr.append(vstr.format(*pgrid[yi,xi]))  # individual points are in x,y,z,a order
            verts[pgrid[yi,xi][4]] = n+1 # track index of points in OBJ vertex list;  pt[4] is a unique integer for this point
            n+=1

    for yi in range(bgrid.shape[0]): # grid is indexed y,x
        for xi in range(bgrid.shape[1]):
            objstr.append(vstr.format(*bgrid[yi,xi]))  # individual points are in x,y,z,a order
            verts[bgrid[yi,xi][4]] = n+1 # track index of points in OBJ vertex list;  pt[4] is a unique integer for this point
            n+=1

    #print("... wrote {} vertices to OBJ".format(n))

    fstr = "f {0} {1} {2}\n"
    objstr.append("o {}\n".format("VDISP_OBJ"))
    
    def append_faces(fps):
        if all([_point_is_valid(p) for p in fps]):
            objstr.append(fstr.format(*[ verts[p[4]] for p in fps ]) )  # pt[4] is a unique integer for this point
            

    # pgrid surface
    for yi in range(pgrid.shape[0]-1): # grid is indexed y,x
        for xi in range(pgrid.shape[1]-1):
            facpts = [ pgrid[idx] for idx in [(yi,xi),(yi+1,xi),(yi+1,xi+1)] ]
            append_faces(facpts)
            facpts = [ pgrid[idx] for idx in [(yi+1,xi+1),(yi,xi+1),(yi,xi)] ]
            append_faces(facpts)            

    # back surface
    for yi in range(bgrid.shape[0]-1): # grid is indexed y,x
        for xi in range(bgrid.shape[1]-1):
            facpts = [ bgrid[idx] for idx in [(yi,xi),(yi,xi+1),(yi+1,xi+1)] ] # winding in reverse from ptgrid 
            append_faces(facpts)
            facpts = [ bgrid[idx] for idx in [(yi+1,xi+1),(yi+1,xi),(yi,xi)] ] # winding in reverse from ptgrid 
            append_faces(facpts)            

    # edge surfaces
    def do_edge(prow,brow):
        prow,brow = prow.squeeze(), brow.squeeze()
        for ni in range(prow.shape[0]-1):
            facpts = [ prow[ni], prow[ni+1], brow[ni+1] ]
            append_faces(facpts)
            facpts = [ brow[ni+1], brow[ni], prow[ni] ]
            append_faces(facpts)            
            
    do_edge( pgrid[:1,:],  bgrid[:1,:]  )
    do_edge( bgrid[-1:,:], pgrid[-1:,:] )
    do_edge( bgrid[:,:1],  pgrid[:,:1]  )
    do_edge( pgrid[:,-1:], bgrid[:,-1:] )
    #print("... wrote edge faces to OBJ".format())

    return objstr



'''
def _writeXYZ(pgrid, pth_src):
    xyzstr = []
    for yi in range(pgrid.shape[0]): # grid is indexed y,x
        for xi in range(pgrid.shape[1]):
            if _point_is_valid(pgrid[yi,xi]): xyzstr.append("{0},{1},{2}\n".format(*pgrid[yi,xi])) # individual points are in x,y,z,a order

    pth_xyz = pathlib.Path(pth_src.parent / "{}".format(pth_src.stem) ).with_suffix('.xyz')
    with open(pth_xyz, 'w') as f:
        f.writelines(xyzstr)
'''


def _img_to_pointgrid(img):
    bx,by,bz = _base_pointcoords(img.shape)
    alpha = img[:,:,3]

    def sigmoid(x):
        k = ALPHA_SIG_SMOOTH_K
        mid = ALPHA_SIG_SMOOTH_MID
        return 1 / (1 + np.exp(-k*(x-mid)))

    apow = sigmoid(alpha)
    apow[:3,:], apow[-3:,:], apow[:,:3], apow[:,-3:] = 0,0,0,0 # set edges to zero displacement
    
    dx = bx + img[:,:,2]*0.5*apow # X displacement
    dy = by + img[:,:,0]*0.5*apow # Y displacement
    dz = bz + img[:,:,1]*0.5*apow # Z displacement
    
    idx = np.reshape( np.arange(img.shape[0] * img.shape[1]).astype(np.int32), (img.shape[0],img.shape[1]) )
    pgrid = np.stack((dx,dy,dz,alpha,idx),axis=2)
    #pgrid[:1,:,2], pgrid[-1:,:,2], pgrid[:,:1,2], pgrid[:,-1:,2] = -push_edge,-push_edge,-push_edge,-push_edge # push down vertices at boundary

    return(pgrid)


def _back_pointgrid(img, depth=0.05):
    bx,by,bz = _base_pointcoords(img.shape)
    bz -= depth
    alpha = np.zeros((img.shape[0], img.shape[1])) # doesn't matter, but keeping to remain isomorphic with real point grid
    
    idx = np.reshape( np.arange(img.shape[0] * img.shape[1], img.shape[0] * img.shape[1] * 2).astype(np.int32), (img.shape[0],img.shape[1]) )
    pgrid = np.stack((bx,by,bz,alpha,idx),axis=2)
    return(pgrid)


def _base_pointcoords(shp):
    nx, ny = (shp[0],shp[1])
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(1, 0, ny, dtype=np.float32)
    
    xv, yv = np.meshgrid(x, y)
    zv = np.full( (shp[0],shp[1]), 0.0, np.float32)
    
    return xv,yv,zv

# pt is array of [x,y,z,a]
def _point_is_valid(pt, plot_transparent_points=True):
    if plot_transparent_points: return True
    return pt[3] > 0.0

