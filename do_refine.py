import vdisp as vd

PTH_SRC = r'C:\tmp\210918_faces'
PTH_DST = r'C:\tmp'

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

def main():
    vd.refine.refine_directory(
        pth_src=PTH_SRC, 
        pth_dst=PTH_DST, 
        xydisp_full=XYDISP_FULL, 
        zdisp_inout=ZDISP_INOUT
        )


if __name__ == "__main__":
    main()