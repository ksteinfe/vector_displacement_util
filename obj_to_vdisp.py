import time, os, pathlib, copy
import numpy as np
import pywavefront as pywv

from scipy import spatial
from scipy.spatial import KDTree
from PIL import Image
import OpenEXR, Imath
from tqdm import tqdm
import cv2

PTH_SRC = pathlib.Path(r'C:\tmp') 
#PTH_SRC = r'C:\tmp'
IMG_SIZE = 512
DO_ALPHA = True
DO_DEBUG = False


def main():
    files = sorted([p.resolve() for p in PTH_SRC.glob("*") if p.suffix in [".obj"]])

    for f in files:
        start_time = time.time()
        print("\n\n------------------------------ {}\n-----".format(f.name))
        process_path(f)
        print("----- processed {} in {}s\n------------------------------".format(f.name, (time.time() - start_time)))

def process_path(pth_src):
    fname, ext = os.path.splitext(pth_src)
    print("processing {}".format(pth_src))

    pywv_obj = load_obj(pth_src) # load OBJ file and return a pywavefront OBJ

    # extract per-face UV and XYZ information. expressed as a list of dicts (one dict per UV-XYZ face)    
    fac_info = extract_fac_info(pywv_obj)

    # uv unit square to uv unit circle?
    fac_info = circleify_uv_square(fac_info)

    # construct disp image that describes vectors from UV to XYZ. expresed as numpy array ( imgsiz x imgsiz x 3or4 )
    # also builds a rhino command file for debugging vector displacements  
    dsparr = build_displacement_image(fac_info, DO_DEBUG)

    # For some reason, things look better in Blender when displacement vectors are scaled by 2. ??
    dsparr *= 2 # Scale displacement results by 2. 
    

    if DO_ALPHA:
        pxarr = np.dstack( ( dsparr, np.ones((IMG_SIZE, IMG_SIZE)) ) )  # add alpha channel
        indxs = np.where(np.all(pxarr == [0,0,0,1], axis=-1)) # indices of invalid pixels with 0 displacement
        pxarr[indxs] = [0,0,0,0] # set invalid pixels to (0,0,0,0)
    else:
        pxarr = dsparr

    writeTIF(pxarr, "{}_{}.tif".format(fname, IMG_SIZE))

# save to -1->1 RGBA EXR
def _SUPERSEDED_writeEXR(img, file):
    print("writing EXR of shape {} to file {}".format(img.shape, file))
    sz = img.shape
    R,G,B,A = extract_img_channels(img)

    header = OpenEXR.Header(sz[1], sz[0])
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))

    if A is not None:
        header['channels'] = dict([(c, half_chan) for c in "RGBA"])
        out = OpenEXR.OutputFile(file, header)
        out.writePixels({'R' : R.tobytes(), 'G' : G.tobytes(), 'B' : B.tobytes(), 'A' : A.tobytes()})
        out.close()
    else:
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)            
        out.writePixels({'R' : R.tobytes(), 'G' : G.tobytes(), 'B' : B.tobytes()})
        out.close()

# save to -1->1 RGB[A] TIFF
def writeTIF(img, file):
    print("writing TIFF of shape {} to file {}".format(img.shape, file))
    sz = img.shape
    R,G,B,A = extract_img_channels(img)
    R,G,B = (R+1)/2, (G+1)/2, (B+1)/2  # convert to 0->1 float for RGB; A is already 0->1

    idta = np.stack([B,G,R],axis=2) 
    if A is not None:
        idta = np.stack([B,G,R,A],axis=2)

    cv2.imwrite( file, np.uint16(idta*65535) ) # from https://newbedev.com/python-read-and-write-tiff-16-bit-three-channel-colour-images

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
    R = (img[:,:,0]).astype(np.float16)
    B = (img[:,:,1]).astype(np.float16)
    G = (img[:,:,2]).astype(np.float16)

    if img.shape[2] == 4:
        A = (img[:,:,3]).astype(np.float16) # include alpha channel
    elif img.shape[2] == 3:
        A = None
    else:
        raise Exception("inappropriate image shape for an image {}".format(img.shape))    
    
    return R,G,B,A

def build_displacement_image(fac_info, do_rhlog=False):
    rhlog = []
    
    #print("...building uvpoint-face info for {} uvpts.".format(IMG_SIZE*IMG_SIZE))
    #start_time_uvpoint = time.time()
    tree = spatial.KDTree([fac['ctr_uv'] for fac in fac_info])

    uvptis = {}
    for x in tqdm(range(IMG_SIZE), desc="uv-face calculation"):
        for y in range(IMG_SIZE):
            uvpt = x/(IMG_SIZE-1), y/(IMG_SIZE-1)
            fac, num_faces_searched = containing_face(uvpt,fac_info,tree)
            if num_faces_searched > 32 and num_faces_searched < 64: print(num_faces_searched)
            uvptis[(x,y)] = (uvpt[0],uvpt[1],fac)

    #print("...built uvpoint-face info in {}s".format(time.time() - start_time_uvpoint))

    #print("...started constructing displacement image.")
    dsparr = np.zeros( (IMG_SIZE,IMG_SIZE,3), dtype=np.float64 )
    lenarr = []
    min_px_val, max_px_val = 1000,0
    for x in tqdm(range(IMG_SIZE), desc="disp image construction"):
        start_time = time.time()
        for y in range(IMG_SIZE):
            uvpti = uvptis[(x,y)]
            disp = disp_at_uvpt( uvpti )
            dsparr[y][x] = disp # swapped x and y here

            min_px_val = min(min_px_val,disp[0],disp[1],disp[2])
            max_px_val = max(max_px_val,disp[0],disp[1],disp[2])
            #print(x,y, disp)
            mag = np.sqrt(disp.dot(disp))
            if mag>0 and do_rhlog: rhlog.append("Line {},{} @{},{},{}".format(uvpti[0],uvpti[1],disp[0],disp[1],disp[2]))
        
        #if x%40==0 or (IMG_SIZE<300 and x%20==0) or (IMG_SIZE<200 and x%10==0) or (IMG_SIZE<70 and x%5==0):  print("row {} of {} completed in {:.2f}s".format(x,IMG_SIZE, time.time() - start_time))

    #print("...done constructing displacement image.")
    print("min/max pixel vals: {},{}".format(min_px_val,max_px_val))
        
    #msk_zero_disp = np.all(dsparr == [0, 0, 0], axis=-1) # mask for invalid disp

    dsparr = np.flip(dsparr, 0) # and reversed order of y to conform to raster image convention

    if do_rhlog: 
        with open( PTH_SRC / 'rhlog.txt', 'w+') as f:
            f.write("\n".join(rhlog))

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
def extract_fac_info(obj, do_xform=True):
    print(obj.file_name )
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
    if do_xform:
        #print("XYZ minmax: {}\t{}".format(minpt_xyz, maxpt_xyz))
        #print("UV minmax: {}\t{}".format(minpt_uv, maxpt_uv))
        maxcoord = 0
        for n in range(len(pts_xyz)):
            pts_xyz[n] -= np.array((bbox_ctr[0], bbox_ctr[1], 0.))   # bbox center to to origin
            maxcoord = max(maxcoord,(max(abs(pts_xyz[n]))))

        print("scaling OBJ mesh by {}".format(0.5/maxcoord))
        for n in range(len(pts_xyz)):
            pts_xyz[n] *= 0.5/maxcoord # scale to unit sphere

        for n in range(len(pts_xyz)):
            pts_xyz[n] += (0.5,0.5,0.)  # center object on (0.5,0.5,0.0)

    # construct return
    msh_info = []
    for fac in msh.faces:
        tri_uv = [pts_uv[f] for f in fac]
        tri_xyz = [pts_xyz[f] for f in fac]
        ctr_uv = np.array( [(tri_uv[0][0]+tri_uv[1][0]+tri_uv[2][0])/3 , (tri_uv[0][1]+tri_uv[1][1]+tri_uv[2][1])/3] )
        msh_info.append({'tri_uv': tri_uv, 'tri_xyz': tri_xyz, 'tri_idx': fac, 'ctr_uv':ctr_uv})

    print("... extracted {} valid faces".format(len(msh_info)))
    return msh_info

def _SUPERSEDED_extract_fac_info(msh_data, do_xform=True):
    # given blender mesh data, returns a list of dicts containing per-face uv/xyz information, 
    #   [uv] is the image-space location of the vertex
    #   [disp] is the vector from uv->xyz
    # disp = (vec_xyz - np.array((vec_uv[0],vec_uv[1],0)) )

    print(msh_data.polygons)
    vtxfaces = []
    centroid = np.array((0.,0.,0.))
    minpt, maxpt = np.array((9999.,9999.,9999.)) , np.array((0.,0.,0.))

    # first pass to collect summary info and convert to numpy arrays
    for msh_face in msh_data.polygons:
        if len(msh_face.vertices) != 3: raise Exception("non-triangular face found on mesh")
        face_idx = msh_face.index
        centroid += msh_face.center
        vtxs = []
        for vert_idx, loop_idx in zip(msh_face.vertices, msh_face.loop_indices):
            vec_xyz = np.array(msh_data.vertices[vert_idx].co.to_tuple())
            vec_uv = np.array(msh_data.uv_layers.active.data[loop_idx].uv.to_tuple())
            maxpt = np.maximum(maxpt, vec_xyz)
            minpt = np.minimum(minpt, vec_xyz)

            vtxs.append( {'uv':vec_uv, 'xyz':vec_xyz, 'idx':vert_idx} )
        
        # check for degenerates 
        if np.logical_and( (vtxs[0]['uv']==vtxs[1]['uv']).all(), (vtxs[1]['uv']==vtxs[2]['uv']).all() ): # all three uv points are the same?
            print("face {} is degenerate - duplicate UV points:\t{}\t{}\t{}".format(face_idx,vtxs[0]['uv'],vtxs[1]['uv'],vtxs[2]['uv']))
            continue

        # todo: check for co-linear uv points
        # todo: check for degenerate triangles in 3d?

        vtxfaces.append( vtxs )
    
    # calculate translation and scale
    centroid /= len(msh_data.polygons)
    bbox_ctr = (maxpt + minpt)/2
    if do_xform:
        #print("centroid: {}".format(centroid))
        #print("minpt: {}\tmaxpt: {}".format(minpt, maxpt))
        maxcoord = 0
        for vtxs in vtxfaces:
            for vtx in vtxs:
                vtx['xyz'] -= np.array((bbox_ctr[0], bbox_ctr[1], 0.))   # bbox center to to origin
                maxcoord = max(maxcoord,(max(abs(vtx['xyz']))))

        print("scaling OBJ mesh by {}".format(0.5/maxcoord))
        for vtxs in vtxfaces:
            for vtx in vtxs:
                vtx['xyz'] *= 0.5/maxcoord # scale to unit sphere

        for vtxs in vtxfaces:
            for vtx in vtxs:
                vtx['xyz'] += (0.5,0.5,0.)  # center object on (0.5,0.5,0.0)
        

    # construct return
    msh_info = []
    for vtxs in vtxfaces:
        tri_uv = [vtx['uv'] for vtx in vtxs]
        tri_xyz = [vtx['xyz'] for vtx in vtxs]
        tri_idx = [vtx['idx'] for vtx in vtxs]
        ctr_uv = np.array( [(tri_uv[0][0]+tri_uv[1][0]+tri_uv[2][0])/3 , (tri_uv[0][1]+tri_uv[1][1]+tri_uv[2][1])/3] )
        #tri_disp = [xyz-np.array((uv[0],uv[1],0)) for uv,xyz in zip(tri_uv,tri_xyz)]        
        msh_info.append({'tri_uv': tri_uv, 'tri_xyz': tri_xyz, 'tri_idx': tri_idx, 'ctr_uv':ctr_uv})
        #print(tri_disp)
    
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
