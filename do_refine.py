import vdisp as vd

PTH_SRC = r'C:\tmp'
PTH_DST = r'C:\tmp'

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

def main():
    vd.refine.refine_directory(PTH_SRC, XYDISP_FULL, ZDISP_INOUT, PTH_DST)



if __name__ == "__main__":
    main()