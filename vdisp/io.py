import numpy as np
import cv2
import copy

def read_tif16(pth):
    img = cv2.imread(str(pth), -1) # open image without changing 16 bit data type
    img = img.astype(np.float32)/65535.0

    # convert RGB channels from 0->1 to -1.0->1.0
    img[:,:,0] = img[:,:,0]*2-1
    img[:,:,1] = img[:,:,1]*2-1
    img[:,:,2] = img[:,:,2]*2-1
    
    if img.shape[2] == 3:
        img = np.dstack( ( img, np.zeros((img.shape[0], img.shape[1])) ) )
        img[:,:,3] = abs(img[:,:,2]) > 0.0
        #print("added alpha channel to shape {}".format(img.shape))
    
    #print("read TIF of shape {} from {}".format(img.shape, pth))
    return img

def write_tif16(img, pth):
    img = copy.deepcopy(img)
    #print("writing TIF of shape {} to {}".format(img.shape, pth))
    
    
    # convert RGB channels from -1->1 to 0->1
    img[:,:,0] = (img[:,:,0]+1)*0.5
    img[:,:,1] = (img[:,:,1]+1)*0.5
    img[:,:,2] = (img[:,:,2]+1)*0.5
    

    #print(np.min(img[:,:,2]), np.max(img[:,:,2]))
    cv2.imwrite( str(pth), np.uint16(img*65535) ) # from https://newbedev.com/python-read-and-write-tiff-16-bit-three-channel-colour-images

