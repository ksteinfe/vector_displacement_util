import time, os, pathlib
import numpy as np
import pywavefront as pywv
import vdisp as vd

PTH_SRC = pathlib.Path(r'C:\tmp') 
PTH_DST = pathlib.Path(r'C:\tmp')
DO_DEBUG = False

# smoothing and culling config
PLOT_TRANSPARENT_PTS = True # plot fully transparent points?
ALPHA_SIG_SMOOTH_K = 500 # displacement is scaled by a sigmoid function of alpha of image. higher values here create sharper falloff of alpha
ALPHA_SIG_SMOOTH_MID = 0.01 # displacement is scaled by a sigmoid function of alpha of image. this is the midpoint of sigmoid, effecively the 'center point' of the falloff

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

def main():
    files = sorted([p.resolve() for p in PTH_SRC.glob("*") if p.suffix in [".tif"]])
    for f in files:
        start_time = time.time()
        print("\n\n------------------------------ {}\n-----".format(f.name))
        dst = f
        if PTH_DST: dst = PTH_DST / f.name
        process_path(f,dst)
        print("----- processed {} in {}s\n------------------------------".format(f.name, (time.time() - start_time)))
        
        
        

def process_path(pth_src, pth_dst):
    print("processing {} to {}".format(pth_src,pth_dst))

    img = vd.read_tif16(pth_src, XYDISP_FULL, ZDISP_INOUT) # img[:,:,1] is Z displacement 
    
    pgrid = img_to_pointgrid(img) # (512, 512, 5) (x,y,z,a,idx)

    #print(pgrid.shape)
    
    if DO_DEBUG: writeXYZ(pgrid, pth_src)
    writeOBJ(pgrid, pth_dst)


# pt is array of [x,y,z,a]
def point_is_valid(pt):
    if PLOT_TRANSPARENT_PTS: return True
    return pt[3] > 0.0


def writeOBJ(pgrid, pth_src):
    objstr = []
    verts = {}
    n = 0
    for yi in range(pgrid.shape[0]): # grid is indexed y,x
        for xi in range(pgrid.shape[1]):
            objstr.append("v {0:f} {1:f} {2:f}\n".format(*pgrid[yi,xi]))  # individual points are in x,y,z,a order
            verts[pgrid[yi,xi][4]] = n+1 # track index of points in OBJ vertex list;  pt[4] is a unique integer for this point
            n+=1
    print("... wrote {} vertices to OBJ".format(n))

    objstr.append("o {}\n".format("VDISP_OBJ"))
    #objstr.append("o {}\n".format(pth_src.stem))

    m=0
    for yi in range(pgrid.shape[0]-1): # grid is indexed y,x
        for xi in range(pgrid.shape[1]-1):
            fac = [(yi,xi),(yi,xi+1),(yi+1,xi+1),(yi+1,xi)]
            facpts = [ pgrid[idx] for idx in fac ]
            if all([point_is_valid(p) for p in facpts]):
                objstr.append("f {0} {1} {2} {3}\n".format(*[ verts[p[4]] for p in facpts ]) )  # pt[4] is a unique integer for this point
                m+=1

    print("... wrote {} faces to OBJ".format(m))

    pth_obj = pathlib.Path(pth_src.parent / "{}".format(pth_src.stem) ).with_suffix('.obj')
    with open(pth_obj, 'w') as f:
        f.writelines(objstr)



def writeXYZ(pgrid, pth_src):
    xyzstr = []
    for yi in range(pgrid.shape[0]): # grid is indexed y,x
        for xi in range(pgrid.shape[1]):
            if point_is_valid(pgrid[yi,xi]): xyzstr.append("{0},{1},{2}\n".format(*pgrid[yi,xi])) # individual points are in x,y,z,a order

    pth_xyz = pathlib.Path(pth_src.parent / "{}".format(pth_src.stem) ).with_suffix('.xyz')
    with open(pth_xyz, 'w') as f:
        f.writelines(xyzstr)



def img_to_pointgrid(img):
    bx,by,bz = base_pointcoords(img.shape)
    alpha = img[:,:,3]

    def sigmoid(x):
        k = ALPHA_SIG_SMOOTH_K
        mid = ALPHA_SIG_SMOOTH_MID
        return 1 / (1 + np.exp(-k*(x-mid)))

    apow = sigmoid(alpha)
    apow[:1,:], apow[-1:,:], apow[:,:1], apow[:,-1:] = 0,0,0,0 # set edges to zero displacement
    
    dx = bx + img[:,:,2]*0.5*apow # X displacement
    dy = by + img[:,:,0]*0.5*apow # Y displacement
    dz = bz + img[:,:,1]*0.5*apow # Z displacement
    
    idx = np.reshape( np.arange(img.shape[0] * img.shape[1]).astype(np.int32), (img.shape[0],img.shape[1]) )
    pgrid = np.stack((dx,dy,dz,alpha,idx),axis=2)

    return(pgrid)


def base_pointcoords(shp):
    nx, ny = (shp[0],shp[1])
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(1, 0, ny, dtype=np.float32)
    
    xv, yv = np.meshgrid(x, y)
    zv = np.full( (shp[0],shp[1]), 0.0, np.float32)
    
    return xv,yv,zv


def _base_pointgrid(shp):
    nx, ny = (shp[0],shp[1])
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(1, 0, ny, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    zv = np.full( (shp[0],shp[1]), 0.0, np.float32)

    grid = np.stack((xv,yv,zv),axis=2)
    #grid = np.flip(grid, 1) # and reversed order of y to conform to raster image convention
    return grid # size is height, width (y,x)


if __name__== "__main__":
  main()