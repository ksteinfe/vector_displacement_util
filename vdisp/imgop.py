import math
import numpy as np
import cv2

from scipy.ndimage.morphology import binary_erosion
from scipy.spatial.distance import cdist
from scipy import ndimage


def flip_x(img):
    img = np.copy(img)
    img = np.flip(img, 0)
    img[:,:,0] = img[:,:,0]*-1
    return img

def flip_y(img):
    img = np.copy(img)
    img = np.flip(img, 1)
    img[:,:,2] *= -1
    return img


# rotates a given image a given number of degrees
# returns an image of a different shape
def rotate(img, deg):
    #print("rotating image of shape {}".format(img.shape))
    img = np.copy(img)
    theta = np.deg2rad(deg)
    x = img[:,:,2] * np.cos(theta) - img[:,:,0] * np.sin(theta)
    y = img[:,:,2] * np.sin(theta) + img[:,:,0] * np.cos(theta)

    img[:,:,2] = x
    img[:,:,0] = y
    
    img = ndimage.rotate(img, deg)
    #print("returning image of shape {}".format(img.shape))
    return img


def rotate_90_cw(img):
    img = np.copy(img)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)    
    r,b = img[:,:,2].copy(), img[:,:,0].copy()

    img[:,:,0] = r*-1
    img[:,:,2] = b
    return img

def rotate_90_ccw(img):
    img = np.copy(img)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    r,b = img[:,:,2].copy(), img[:,:,0].copy()

    img[:,:,0] = r
    img[:,:,2] = b*-1
    return img


# from https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
def overlay_image_alpha(imbase, imover, x, y):
    dimbase, dimover = imbase.shape, imover.shape
    x,y = x%dimbase[1], y%dimbase[0]
    #print("overlay {} on base of {} at {},{}".format(dimover, dimbase, x, y))
    
    if x+dimover[1] > dimbase[1]:
        imover_a = imover[:, 0:dimbase[1]-x] 
        imover_b = imover[:, dimbase[1]-x:]
        overlay_image_alpha(imbase, imover_a, x, y )
        overlay_image_alpha(imbase, imover_b, 0, y )
        return
        '''
        if x<0:        
            print("x split 1")
            imover_a = imover[:, abs(x):dimover[1]] 
            imover_b = imover[:, 0:abs(x)]
            overlay_image_alpha(imbase, imover_a, 0, y )
            overlay_image_alpha(imbase, imover_b, dimbase[1]+x, y )
            return
        '''

    if y+dimover[0] > dimbase[0]:
        imover_a = imover[0:dimbase[0]-y, :] 
        imover_b = imover[dimbase[0]-y:, :]
        overlay_image_alpha(imbase, imover_a, x, y )
        overlay_image_alpha(imbase, imover_b, x, 0 )
        return

    # Image ranges
    y1, y2 = max(0, y), min(dimbase[0], y + dimover[0])
    x1, x2 = max(0, x), min(dimbase[1], x + dimover[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(dimover[0], dimbase[0] - y)
    x1o, x2o = max(0, -x), min(dimover[1], dimbase[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        print("nothing to do!")
        return

    # Blend overlay within the determined ranges
    imbase_crp = imbase[y1:y2, x1:x2]
    imover_crp = imover[y1o:y2o, x1o:x2o]

    asum = imover_crp[:,:,3] + imbase_crp[:,:,3] + np.finfo(float).eps # add a tiny bit to make sure sum is non-zero
    #a = np.squeeze(imover_crp[:,:,3] / asum)
    a = imover_crp[:,:,3] / asum
    #a = np.squeeze(imover_crp[:,:,3])
    ai = 1.0 - a

    # X and Y displacement channels take the average of two images weighted by their alpha
    imbase_crp[:,:,0] = a*imover_crp[:,:,0] + ai*imbase_crp[:,:,0] 
    imbase_crp[:,:,2] = a*imover_crp[:,:,2] + ai*imbase_crp[:,:,2]

    # overlaid alpha channels take the maximum alpha of two images
    imbase_crp[:,:,3] = np.maximum( imbase_crp[:,:,3] , imover_crp[:,:,3] )

    # Z displacement channel is halfway between weighted average and maximum
    d_max = np.maximum( imbase_crp[:,:,1] , imover_crp[:,:,1] )
    d_avg = a*imover_crp[:,:,1] + ai*imbase_crp[:,:,1]
    imbase_crp[:,:,1] = (d_max+d_avg)/2
    



    
# from https://stackoverflow.com/questions/40492159/find-distance-from-the-edge-of-a-numpy-array
# higher mult expands 'clear' area in center; higher pwr tightens gradient along boundary
def apply_cityblock_vignette(im, mlt=2, pwr=3, do_save_mask=False):

    def dist_from_edge(img):
        I = binary_erosion(img) # Interior mask
        C = img - I             # Contour mask
        out = C.astype(int)     # Setup o/p and assign cityblock distances
        out[I] = cdist(np.argwhere(C), np.argwhere(I), 'cityblock').min(0) + 1
        return out

    mask = (im[:,:,3] > 0.0).astype(int)
    mask = dist_from_edge(mask).astype(np.float32) / 255.0
    mask = pow( np.clip(mask*mlt,0,1), pwr )
    
    im[:,:,3] = mask #overwrites current alpha with cityblock alpha

    #print(mask)
    #print(mask.min(), mask.max())
    if do_save_mask: cv2.imwrite(r'C:\tmp\mask.jpg'.format(), mask*255)


def apply_gaussian_vignette(im, sig=100, pwr=0.5, do_save_mask=False):
    gkern = cv2.getGaussianKernel(im.shape[1],sig) * cv2.getGaussianKernel(im.shape[0],sig).T
    gkern -= gkern.min()
    mask = pow(gkern/gkern.max(),pwr)
    mask = np.minimum(mask,im[:,:,3]) # take the minimum of current alpha and gaussian
    #for i in range(3): im[:,:,i] = im[:,:,i] * mask # scale vectors down? NO!    
    
    im[:,:,3] = mask
    
    if do_save_mask: cv2.imwrite(r'C:\tmp\mask.jpg'.format(), mask*255)
    


