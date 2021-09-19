import vdisp as vd
import vdisp.imgop as iop
import pathlib, os
import numpy as np
import cv2
import copy
from tqdm import tqdm

PTH_SRC = r'C:\tmp\210918_faces'

#ROTS = (['','a','b','c','d'],[0,-60,-30,30,60])
#ROTS = (['a','b','c','d'],[30,45,60,90])
ROTS = (['a','b'],[-60,60]) # for hex base; no flipping for symmetrical things like faces.
DO_FLIP_X = True
DO_FLIP_Y = False

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

def main():
    files = sorted([p.resolve() for p in pathlib.Path(PTH_SRC).glob("*") if p.suffix in [".tif", ".tiff"]])
    pbar = tqdm(files)
    for f in pbar:
        #for f in files:
        pbar.set_description("augmenting {}".format(f.name))
        fname, ext = os.path.splitext(f)
        img = vd.read_tif16(f,XYDISP_FULL,ZDISP_INOUT)
        sz = (img.shape[0],img.shape[1])

        ximg = iop.flip_x(copy.deepcopy(img))
        yimg = iop.flip_y(copy.deepcopy(img))
        if DO_FLIP_X: vd.write_tif16(ximg, "{}{}.tif".format(fname,'x'),XYDISP_FULL,ZDISP_INOUT)
        if DO_FLIP_Y: vd.write_tif16(yimg, "{}{}.tif".format(fname,'y'),XYDISP_FULL,ZDISP_INOUT)

        #TODO: rotating and then flipping works, but not the other way around??!

        for d,deg in zip(ROTS[0], ROTS[1]):
            #vd.write_tif16( cv2.resize( iop.rotate(np.copy(img), deg), sz ) , "{}{}.tif".format(fname,d))
            #vd.write_tif16( cv2.resize( iop.flip_x(np.copy(iop.rotate(np.copy(img),deg))), sz ) , "{}{}{}.tif".format(fname,d,'x'))
            #vd.write_tif16( cv2.resize( iop.flip_y(np.copy(iop.rotate(np.copy(img),deg))), sz ) , "{}{}{}.tif".format(fname,d,'y'))
            rotimg = iop.rotate(copy.deepcopy(img), deg)
            x = (rotimg.shape[1] / 2) - img.shape[1]/2
            y = (rotimg.shape[0] / 2) - img.shape[0]/2
            rotimg = rotimg[int(y):int(y+img.shape[1]), int(x):int(x+img.shape[0])]

            vd.write_tif16( rotimg, "{}{}.tif".format(fname,d),XYDISP_FULL,ZDISP_INOUT)
            if DO_FLIP_X: vd.write_tif16( iop.flip_x(rotimg), "{}{}{}.tif".format(fname,d,'x'),XYDISP_FULL,ZDISP_INOUT)
            if DO_FLIP_Y: vd.write_tif16( iop.flip_y(rotimg), "{}{}{}.tif".format(fname,d,'y'),XYDISP_FULL,ZDISP_INOUT)

        


if __name__ == "__main__":
    main()