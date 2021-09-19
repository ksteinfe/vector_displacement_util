import time, os, pathlib, copy, sys, traceback
import numpy as np
import pywavefront as pywv

from scipy import spatial
from tqdm import tqdm
import cv2

PTH_SRC = pathlib.Path(r'C:\tmp')
PTH_DST = pathlib.Path(r'C:\tmp')
IMG_SIZE = 512
DO_ALPHA = True
DO_DEBUG = False

XFORM_OBJ_BY = 2 # False does not scale OR CENTER object; value of 2 centers and scales only by XY coords; value of 3 centers and scales by XYZ
SCALE_OBJ_TO = 0.58 # 0.58 for hex base; 0.50 to unit sphere

DO_CIRCLEIFY = True # projects the UV square coordinates to a unit circle

ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)
ZDISPTAG = "b"
if not ZDISP_INOUT: ZDISPTAG = "o"

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
XYDISPTAG = "f"
if not XYDISP_FULL: XYDISPTAG = "h"

def main():
    files = sorted([p.resolve() for p in PTH_SRC.glob("*") if p.suffix in [".obj"]])

    for f in files:
        start_time = time.time()
        print("\n\n------------------------------ {}\n-----".format(f.name))
        try:
            process_path(f, PTH_DST)
            print("----- processed {} in {}s\n------------------------------".format(f.name, (time.time() - start_time)))
        except Exception as e:
            print("!!!!! FAILED {} in {}s\n!!!!!!!!!!!!!!!!!!!!!".format(f.name, (time.time() - start_time)))
            print(traceback.format_exc())
        
        

def process_path(pth_src, pth_dst):
    print("processing {}".format(pth_src))

    pywv_obj = load_obj(pth_src) # load OBJ file and return a pywavefront OBJ

    # extract per-face UV and XYZ information. expressed as a list of dicts (one dict per UV-XYZ face)    
    fac_info = extract_fac_info(pywv_obj)

    # uv unit square to uv unit circle?
    if DO_CIRCLEIFY: fac_info = circleify_uv_square(fac_info)

    # construct disp image that describes vectors from UV to XYZ. expresed as numpy array ( imgsiz x imgsiz x 3or4 )
    # also builds a rhino command file for debugging vector displacements  
    rhlog = False
    if DO_DEBUG: rhlog = pathlib.Path(pth_src.parent / "{}_{}".format(pth_src.stem,IMG_SIZE) ).with_suffix('.txt')
    dsparr = build_displacement_image(fac_info, rhlog)

    # For some reason, things look better in Blender when displacement vectors are scaled by 2. ??
    dsparr *= 2 # Scale displacement results by 2. 
    

    if DO_ALPHA:
        pxarr = np.dstack( ( dsparr, np.ones((IMG_SIZE, IMG_SIZE)) ) )  # add alpha channel
        indxs = np.where(np.all(pxarr == [0,0,0,1], axis=-1)) # indices of invalid pixels with 0 displacement
        pxarr[indxs] = [0,0,0,0] # set invalid pixels to (0,0,0,0)
    else:
        pxarr = dsparr

    pth_dst = pathlib.Path(pth_dst / "{}_{}{}{}".format(pth_src.stem,ZDISPTAG,XYDISPTAG,IMG_SIZE) ).with_suffix('.tif')
    writeTIF(pxarr, pth_dst)


# save to -1->1 RGB[A] TIFF
def writeTIF(img, file):
    print("writing TIFF of shape {} to file {}".format(img.shape, str(file)))
    sz = img.shape
    R,G,B,A = extract_img_channels(img)  # X is [0] is Red; Y is [1] is Blue; Z is [2] is Green
    # convert RGB below; A is already 0->1

    if XYDISP_FULL:
        # xy displacement is in range of -1->1
        R,B = (R+1)/2, (B+1)/2 # convert from -1->1 to 0->1
    else:
        # xy displacement is in range of -0.5->0.5
        if np.min(R)<-0.5 or np.max(R)>0.5 or np.min(R)<-0.5 or np.max(R)>0.5:
            print("!!! XY DISPLACEMENT WILL BE CLIPPED:\tR({}->{})\tB({}->{})".format(np.min(R), np.max(R),np.min(B), np.max(B)))
        
        R = np.clip(R, -0.5, 0.5) + 0.5 # convert from -0.5->0.5 to 0->1
        B = np.clip(B, -0.5, 0.5) + 0.5 # convert from -0.5->0.5 to 0->1

    if ZDISP_INOUT:
        # z displacement is in range of -1->1
        G = (G+1)/2  # convert from -1->1 to 0->1
    else:
        # z displacement is in range of 0->1
        G = np.clip(G, 0, 1) # G should already be at 0->1

    idta = np.stack([B,G,R],axis=2) # CV2 wants BGRA order
    if A is not None:
        idta = np.stack([B,G,R,A],axis=2)

    cv2.imwrite( str(file), np.uint16(idta*65535) ) # from https://newbedev.com/python-read-and-write-tiff-16-bit-three-channel-colour-images

    '''
    im = Image.fromarray(np.uint16(idta*65535))
    im.save(file)
    '''

    '''
    # from https://blog.itsayellow.com/technical/saving-16-bit-tiff-images-with-pillow-in-python/#
    idta = np.uint16(idta*65535)
    outpil = idta.astype(idta.dtype.newbyteorder("<")).tobytes()
    img_out = Image.new('I;16', (sz[0],sz[1]) )
    img_out.frombytes(outpil)
    img_out.save(file)
    '''
      
def extract_img_channels(img):
    img = np.squeeze(img)
    
    ''' prior to x,y flip in build_displacement_image() below
        dsparr[y][x] = disp # swapped x and y here
    
    B = (-1 * img[:,:,0]).astype(np.float16) # invert B channel
    R = (img[:,:,1]).astype(np.float16)
    G = (img[:,:,2]).astype(np.float16)
    '''

    # adjust order and flipping of xyz here as required by Blender
    # Blender tangent space maps seem to accept RBG order
    R = (img[:,:,0]).astype(np.float16) # X is [0] is Red
    B = (img[:,:,1]).astype(np.float16) # Y is [1] is Blue
    G = (img[:,:,2]).astype(np.float16) # Z is [2] is Green

    if img.shape[2] == 4:
        A = (img[:,:,3]).astype(np.float16) # include alpha channel
    elif img.shape[2] == 3:
        A = None
    else:
        raise Exception("inappropriate image shape for an image {}".format(img.shape))    
    
    return R,G,B,A

def build_displacement_image(fac_info, rhlog=False):
    debuglog = []
    
    #print("...building uvpoint-face info for {} uvpts.".format(IMG_SIZE*IMG_SIZE))
    #start_time_uvpoint = time.time()
    tree = spatial.KDTree([fac['ctr_uv'] for fac in fac_info])

    uvptis = {}
    for x in tqdm(range(IMG_SIZE), desc="uv-face calculation"):
        for y in range(IMG_SIZE):
            uvpt = x/(IMG_SIZE-1), y/(IMG_SIZE-1)
            fac, num_faces_searched = containing_face(uvpt,fac_info,tree)
            if DO_DEBUG and num_faces_searched > 32 and num_faces_searched < 64: print("!!! FACES SEARCHED: {}".format(num_faces_searched))
            uvptis[(x,y)] = (uvpt[0],uvpt[1],fac)

    #print("...built uvpoint-face info in {}s".format(time.time() - start_time_uvpoint))

    #print("...started constructing displacement image.")
    dsparr = np.zeros( (IMG_SIZE,IMG_SIZE,3), dtype=np.float64 )
    min_px_val, max_px_val = 999,0
    min_px_zval, max_px_zval = 999,0
    for x in tqdm(range(IMG_SIZE), desc="disp image construction"):
        for y in range(IMG_SIZE):
            uvpti = uvptis[(x,y)]
            disp = disp_at_uvpt( uvpti )
            dsparr[y][x] = disp # swapped x and y here

            min_px_val = min(min_px_val,disp[0],disp[1],disp[2])
            max_px_val = max(max_px_val,disp[0],disp[1],disp[2])
            min_px_zval = min(min_px_zval,disp[2])
            max_px_zval = max(max_px_zval,disp[2])            
            #print(x,y, disp)
            mag = np.sqrt(disp.dot(disp))
            if mag>0 and rhlog: debuglog.append("Line {},{} @{},{},{}".format(uvpti[0],uvpti[1],disp[0],disp[1],disp[2]))
        

    #print("...done constructing displacement image.")
    print("min/max pixel vals: \t{},\t{}".format(min_px_val,max_px_val))
    print("min/max pixel Z vals: \t{},\t{}".format(min_px_zval,max_px_zval))
        
    #msk_zero_disp = np.all(dsparr == [0, 0, 0], axis=-1) # mask for invalid disp

    dsparr = np.flip(dsparr, 0) # and reversed order of y to conform to raster image convention

    if rhlog: 
        with open( rhlog, 'w+') as f:
            f.write("\n".join(debuglog))

    return dsparr

def circleify_uv_square(fac_info):
    fac_info = copy.deepcopy(fac_info)

    def xf(pt):
        u = (pt[0] - 0.5)*2
        v = (pt[1] - 0.5)*2
        return np.array( ( u*np.sqrt(1-v*v/2) , v*np.sqrt(1-u*u/2)) ) / 2 + 0.5

    for fac in fac_info:
        fac['tri_uv'] = [xf(pt) for pt in fac['tri_uv']]
        fac['ctr_uv'] = xf(fac['ctr_uv'])

    return fac_info

# given pywavefront mesh data, returns a list of dicts containing per-face uv/xyz information, 
#   [tri_uv] is a list of the image-space location of each vertex of the face
#   [tri_xyz] is a list of the world-space location of each vertex of the face
#   [tri_idx] is a list of the index of each vertex of the face
#   [ctr_uv] is the image-space location of the center of the face
def extract_fac_info(obj):
    print(obj.file_name)
    msh = obj.mesh_list[0]
    mat = msh.materials[0]
    assert mat.vertex_format == 'T2F_V3F' # ensure we're working with xyz and uv only
    vd = {v:n for n,v in enumerate(obj.vertices)} # dictionary relating index with xyz coords
    pts_xyz = {} # a duplicate of above, but extracted from interleaved material data
    pts_uv = {} # dictionary relating index with uv coords extracted from interleaved material data
    n=0
    while n < len(mat.vertices):
        u,v,x,y,z = mat.vertices[n], mat.vertices[n+1], mat.vertices[n+2], mat.vertices[n+3], mat.vertices[n+4]
        if (x,y,z) in vd:
            idx = vd[(x,y,z)]
            if idx in pts_uv and pts_uv[idx] != (u,v): raise Exception("found two UV points that don't match: {} != {}".format(pts_uv[idx],(u,v)))
            if idx in pts_xyz and pts_xyz[idx] != (x,y,z): raise Exception("found two XYZ points that don't match: {} != {}".format(pts_xyz[idx],(x,y,z)))
            pts_uv[idx] = (u,v)
            pts_xyz[idx] = (x,y,z)
        else: raise Exception("could not find this xyz point in list of object vertices {}".format((x,y,z)))
        n+=mat.vertex_size # increment forward 5 numbers

    # convert to numpy lists, indices should match those referenced in msh.faces
    pts_xyz = [np.array(pts_xyz[k]) for k in sorted(pts_xyz)]
    pts_uv = [np.array(pts_uv[k]) for k in sorted(pts_uv)]

    if DO_DEBUG: print("found {} UV pts and {} XYZ pts in an OBJ with {} vertices.".format(len(pts_uv),len(pts_xyz),len(obj.vertices)))
    assert len(pts_uv) == len(pts_xyz) == len(obj.vertices)

    minpt_xyz, maxpt_xyz = np.array((9999.,9999.,9999.)) , np.array((0.,0.,0.))
    minpt_uv, maxpt_uv = np.array((9999.,9999.)) , np.array((0.,0.))
    for pt in pts_xyz:
        maxpt_xyz = np.maximum(maxpt_xyz, pt)
        minpt_xyz = np.minimum(minpt_xyz, pt)

    for pt in pts_uv:
        maxpt_uv = np.maximum(maxpt_uv, pt)
        minpt_uv = np.minimum(minpt_uv, pt)

    bbox_ctr = (maxpt_xyz + minpt_xyz)/2
    

    # XFORM
    # 
    if not XFORM_OBJ_BY:
        print("... by request, NO TRANSFORMATION APPLIED")
    else:
        print("... pre-centering XYZ minmax: {}\t{}".format(minpt_xyz, maxpt_xyz))
        #print("UV minmax: {}\t{}".format(minpt_uv, maxpt_uv))

        # move OBJ so that bbox center (x,y) is at (0,0) origin
        # collect max coord values
        max2d = 0 # maxcoord for scaling on xy (after centering on 0,0 origin)
        max3d = 0 # maxcoord for scaling on xyz (after centering on 0,0 origin)
        for n in range(len(pts_xyz)):
            pts_xyz[n] -= np.array((bbox_ctr[0], bbox_ctr[1], 0.))   
            max2d = max(max2d, max(abs(pts_xyz[n][0]),abs(pts_xyz[n][1]) ) )
            max3d = max(max3d,(max(abs(pts_xyz[n]))))

        print("... post-centering max2d: {} \tmax3d: {}".format(max2d, max3d))
        
        if XFORM_OBJ_BY==2: maxcoord = max2d
        elif XFORM_OBJ_BY==3: maxcoord = max3d
        else: raise Exception("invalid XFORM_OBJ_BY {}".format(maxcoord))

        print("... scaling OBJ mesh by {}".format(SCALE_OBJ_TO/maxcoord))
        for n in range(len(pts_xyz)):
            pts_xyz[n] *= (SCALE_OBJ_TO/maxcoord) # scale to given size


        # after scaling in whatever way is specified, check z-vals fit within limits
        # if z falls outside a -0.5 -> 0.5 range, scale ONLY IN Z direction to fit
        # this effectively squashes objects to fit within displacement frame
        maxz = 0
        for n in range(len(pts_xyz)): maxz = max(maxz,abs(pts_xyz[n][2]))
        if maxz > 0.5:
            print("!!! SQUASHING OBJ mesh along Z axis to {:02}%% original size to fit in -0.5->0.5 frame".format(int(0.5/maxz*100)))
            for n in range(len(pts_xyz)):
                pts_xyz[n][2] *= (0.5/maxz) # scale to given size
            

        # re-center object on (0.5,0.5,0.0)
        for n in range(len(pts_xyz)):
            pts_xyz[n] += (0.5,0.5,0.)  



    # Construct Return
    #
    msh_info = []
    for fac in msh.faces:
        tri_uv = [pts_uv[f] for f in fac]
        tri_xyz = [pts_xyz[f] for f in fac]
        ctr_uv = np.array( [(tri_uv[0][0]+tri_uv[1][0]+tri_uv[2][0])/3 , (tri_uv[0][1]+tri_uv[1][1]+tri_uv[2][1])/3] )
        msh_info.append({'tri_uv': tri_uv, 'tri_xyz': tri_xyz, 'tri_idx': fac, 'ctr_uv':ctr_uv})

    print("... extracted {} valid faces".format(len(msh_info)))
    return msh_info

def containing_face(uvpt, fac_info, tree):
    n = 8 # number of nearby faces to search initally; doubles with each pass
    while n<128:
        _, idxs = tree.query( uvpt, min(n,len(tree.data)) ) # indices of nearby faces
        for i in idxs:
            if is_uvpt_on_face(uvpt, fac_info[i]): 
                return fac_info[i], n
        n*=2

    #print("no containing face after searching nearest {}".format(n))
    return False, n

def is_uvpt_on_face(uvpt, face):
    
    def is_point_in_2d_tri(pt,tri):
        def sign(pa,pb,pc): return (pa[0] - pc[0]) * (pb[1] - pc[1]) - (pb[0] - pc[0]) * (pa[1] - pc[1])
        d1,d2,d3 = sign(pt, tri[0], tri[1]), sign(pt, tri[1], tri[2]), sign(pt, tri[2], tri[0])
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return not (has_neg and has_pos)

    return is_point_in_2d_tri(uvpt, face['tri_uv'])

def disp_at_uvpt(uvpt):
    fac = uvpt[2]
    if not fac: return np.array( (0,0,0) ) # point is not on a face
    
    # compute barycentric coordinates (u, v, w) for point p with respect to triangle (a, b, c)
    def barycentric(p, a, b, c):
        v0 = b-a
        v1 = c-a
        v2 = p-a
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return (u,v,w)

    p = np.array((uvpt[0],uvpt[1],0))
    sa,sb,sc = np.array((fac['tri_uv'][0][0],fac['tri_uv'][0][1],0)), np.array((fac['tri_uv'][1][0],fac['tri_uv'][1][1],0)), np.array((fac['tri_uv'][2][0],fac['tri_uv'][2][1],0))
    bc = barycentric(p,sa,sb,sc)
    xyz = fac['tri_xyz'][0]*bc[0] + fac['tri_xyz'][1]*bc[1] + fac['tri_xyz'][2]*bc[2]
    disp = xyz-np.array((uvpt[0],uvpt[1],0))
    
    if np.isnan(disp).any():
        print("nan - is this face a degenerate triangle??")
        print(uvpt)
        print(fac['tri_uv'])
        return np.array( (0,0,0) )

    return disp

def load_obj(pth):
    obj = pywv.Wavefront(pth, strict=True, collect_faces=True)
    print("... obj loaded")
    return obj


if __name__== "__main__":
  main()
