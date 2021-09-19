import pathlib, shutil, os
import vdisp as vd
from vdisp.solidify import solidify_directory

PTH_SRC = pathlib.Path(r'C:\tmp\cuttest\tif')

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

def main():
    pth_01 = PTH_SRC.parents[0] / "01"
    shutil.rmtree(pth_01, ignore_errors=True)
    os.mkdir(pth_01)
    solidify_directory(PTH_SRC, XYDISP_FULL, ZDISP_INOUT, pth_01)


if __name__ == "__main__":
    main()


