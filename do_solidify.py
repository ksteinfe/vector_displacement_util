import pathlib
import vdisp as vd
from vdisp.solidify import solidify_directory

PTH_SRC = pathlib.Path(r'C:\tmp\210918_facesoverfit-000020-210930-circle-500-4\tif_refined')
PTH_DST = pathlib.Path(r'C:\tmp\210918_facesoverfit-000020-210930-circle-500-4')

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

# depth for surface_reconstruction_screened_poisson
# 8 is default ~30s
# 9 is high-res ~90s
# 10 is too much ~130s 
# 11 is ridiculous ~210s
reconstruction_depth = 10

def main():
    solidify_directory(
        pth_src=PTH_SRC, 
        pth_dst=PTH_DST, 
        xydisp_full=XYDISP_FULL, 
        zdisp_inout=ZDISP_INOUT,
        reconstruction_depth=reconstruction_depth
        )

if __name__ == "__main__":
    main()