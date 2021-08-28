import vdisp as vd
import vdisp.imgop as iop
import pathlib

PTH_SRC = pathlib.Path(r'C:\tmp\pergamonpanel_01_512.tif')


def main():
    img = vd.read_tif16(PTH_SRC) 
    img = iop.flip_x(img)
    vd.write_tif16(img, PTH_SRC.parents[0] / (PTH_SRC.stem + "_refined.tif") )


if __name__ == "__main__":
    main()




