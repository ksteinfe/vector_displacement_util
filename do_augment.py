import vdisp as vd
import vdisp.imgop as iop
import pathlib, os
import numpy as np
import cv2


PTH_SRC = r'C:\tmp'

def main():
    #rotations = (['','a','b','c','d'],[0,-60,-30,30,60])
    rotations = (['','a','b'],[0,45,90])

    files = sorted([p.resolve() for p in pathlib.Path(PTH_SRC).glob("*") if p.suffix in [".tif", ".tiff"]])
    for f in files:
        print("{}".format(f.name))
        fname, ext = os.path.splitext(f)
        img = vd.read_tif16(f)
        sz = (img.shape[0],img.shape[1])

        ximg = iop.flip_x(np.copy(img))
        yimg = iop.flip_y(np.copy(img))
        vd.write_tif16(ximg, "{}{}.tif".format(fname,'x'))
        vd.write_tif16(yimg, "{}{}.tif".format(fname,'y'))        

        #TODO: rotating and then flipping works, but not the other way around??!

        for d,deg in zip(rotations[0], rotations[1]):
            #print(deg)
            vd.write_tif16( cv2.resize( iop.rotate(np.copy(img), deg), sz ) , "{}{}.tif".format(fname,d))
            vd.write_tif16( cv2.resize( iop.flip_x(np.copy(iop.rotate(np.copy(img),deg))), sz ) , "{}{}{}.tif".format(fname,d,'x'))
            vd.write_tif16( cv2.resize( iop.flip_y(np.copy(iop.rotate(np.copy(img),deg))), sz ) , "{}{}{}.tif".format(fname,d,'y'))


if __name__ == "__main__":
    main()