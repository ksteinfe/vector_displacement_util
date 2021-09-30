import pathlib, shutil, os
import vdisp as vd
from vdisp.solidify import solidify_directory
import vdisp.mshop as mshop

PTH_SRC = pathlib.Path(r'C:\tmp\210918_faces-000060-210919-circle-400_0-1\solids')
PTH_DST = pathlib.Path(r'C:\tmp\210918_faces-000060-210919-circle-400_0-1')

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

def main():
    pths_obj = sorted([p.resolve() for p in (PTH_SRC).glob("*") if p.suffix in [".obj"]])

    

    pth_out = PTH_DST/"test.obj"
    mshop.do_rotate(pths_obj[0], axis=2, angle=120, cpt=(0.5,0.5,0.0),save_as=pth_out) # axis 2 is Z axis
    mshop.do_flip_and_or_swap_axis(pth_out,about_pt=(-0.5,0.0,0.0), flipx=True,save_as=pth_out)
    mshop.do_rotate(pth_out, axis=2, angle=-120, cpt=(0.5,0.5,0.0),save_as=pth_out) # axis 2 is Z axis


if __name__ == "__main__":
    main()


