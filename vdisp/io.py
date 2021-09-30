import numpy as np
import cv2
import copy
from .imgop import calculate_alpha_from_disp_thresh

#TODO: reading and writing TIFs must take into account value ranges for XY and DISP
# can read from whatever into standard -1->1 range for XY and Z; 0-1 range for alpha
# can write from standard -1->1 range for XY and Z; 0-1 range for alpha into whatever

# zdisp_inout # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)
# xydisp_full # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
def read_tif16(pth, xydisp_full, zdisp_inout):
    img = cv2.imread(str(pth), -1) # open image without changing 16 bit data type
    img = img.astype(np.float32)/65535.0

    if xydisp_full:
        # convert RB channels from 0->1 to -1.0->1.0
        img[:,:,0] = img[:,:,0]*2-1 # X or Y is [0] is Red
        img[:,:,2] = img[:,:,2]*2-1 # X or Y is [2] is Blue
    else:
        # convert RB channels from 0->1 to -0.5->0.5
        img[:,:,0] = img[:,:,0]-0.5 # X or Y is [0] is Red
        img[:,:,2] = img[:,:,2]-0.5 # X or Y is [2] is Blue

    if zdisp_inout:
        # convert G channels from 0->1 to -1.0->1.0
        img[:,:,1] = img[:,:,1]*2-1 # Z is [1] is Green
    else:
        # no need to convert G channels from 0->1 to 0->1
        pass   

    if img.shape[2] == 3:
        img = calculate_alpha_from_disp_thresh(img)
    
    #print("IN\tR({}->{})\tG({}->{})\tB({}->{})\tA({}->{})".format(np.min(img[:,:,0]), np.max(img[:,:,0]),np.min(img[:,:,1]), np.max(img[:,:,1]),np.min(img[:,:,2]), np.max(img[:,:,2]),np.min(img[:,:,3]), np.max(img[:,:,3])))


    #print("read TIF of shape {} from {}".format(img.shape, pth))
    return img

# zdisp_inout = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)
# xydisp_full = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
def write_tif16(img, pth, xydisp_full, zdisp_inout):
    img = copy.deepcopy(img)
    #print("writing TIF of shape {} to {}".format(img.shape, pth))    

    if xydisp_full:
        # convert RB channels from -1->1 to 0->1
        img[:,:,0] = (img[:,:,0]+1)*0.5 # X or Y is [0] is Red
        img[:,:,2] = (img[:,:,2]+1)*0.5 # X or Y is [2] is Blue
    else:
        # convert RB channels from -0.5->0.5 to 0->1
        img[:,:,0] = img[:,:,0]+0.5 # X or Y is [0] is Red
        img[:,:,2] = img[:,:,2]+0.5 # X or Y is [2] is Blue

    if zdisp_inout: 
        # convert G channels from -1.0->1.0 to 0->1
        img[:,:,1] = (img[:,:,1]+1)*0.5 # Z is [1] is Green
    else:
        # no need to convert G channels from 0->1 to 0->1
        pass
    
    img = np.clip(img, 0.0, 1.0)

    #print("OUT\tR({}->{})\tG({}->{})\tB({}->{})\tA({}->{})".format(np.min(img[:,:,0]), np.max(img[:,:,0]),np.min(img[:,:,1]), np.max(img[:,:,1]),np.min(img[:,:,2]), np.max(img[:,:,2]),np.min(img[:,:,3]), np.max(img[:,:,3])))
    cv2.imwrite( str(pth), np.uint16(img*65535) ) # from https://newbedev.com/python-read-and-write-tiff-16-bit-three-channel-colour-images


# loads a folder full of TIFF images into a sequence of a desired length.
# if not enough images are present, we loop unil full
def load_sequence(pth, frame_count, xydisp_full, zdisp_inout):
    print("loading sequence {}".format(pth.stem))
    files = sorted([p.resolve() for p in pth.glob("*") if p.suffix in [".tif", ".tiff"]])
    if len(files)==0: raise Exception("No TIFF files found in {}".format(pth))

    ret,n = [],0
    while len(ret)<frame_count:
        ret.append(read_tif16(files[n%len(files)], xydisp_full, zdisp_inout))
        n+=1
    return ret
