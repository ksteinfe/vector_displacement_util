import sys  
from pathlib import Path
sys.path.append(str(Path(__file__). resolve().parents[1]))

import vdisp as vd
import vdisp.imgop as iop
import numpy as np
import cv2
from tqdm import tqdm
import pathlib, itertools, math

PTH_SRC = pathlib.Path(r'C:\tmp\tif_refined')
PTH_DST = pathlib.Path(r'C:\tmp\twins')

XYDISP_FULL = False # if True, xy displacements span the whole uv square (-1 to 1); if False, xy displacements span half the uv square (-0.5 to 0.5)
ZDISP_INOUT = False # both-direction "b" displacements might be positive or negative; out-only "o" displacements are only positive (out)

CNT_FRAMES = 200

def main():
    cfg = { # keep size to aspect ratio of unit cylinder (1.25 : (6.28*.75) == 0.26539)
        'wdth': 828,
        'hght': 512
    }
    simple_twin(cfg)


def simple_twin(cfg):
    seq = vd.load_sequence(PTH_SRC, CNT_FRAMES, XYDISP_FULL, ZDISP_INOUT)

    create_base_image(cfg['wdth'],cfg['hght'])

    pbar = tqdm(range(CNT_FRAMES))
    for n in pbar:
        pbar.set_description("frame {} of {}".format(n,CNT_FRAMES))
        simple_twin_frame(n, cfg, seq)


def simple_twin_frame(idx_frame, cfg, seq):
    bimg = create_base_image(cfg['wdth'],cfg['hght'])
    img_a = seq[idx_frame]
    img_b = iop.flip_y(seq[-idx_frame])

    pad = sinlerp(0,50,idx_frame/CNT_FRAMES)
    iop.overlay_image_alpha(bimg, img_b, cfg['wdth']-512-pad, 0)
    iop.overlay_image_alpha(bimg, img_a, pad, 0)
    
    vd.write_tif16(bimg, PTH_DST / "{:03}.tif".format(idx_frame), XYDISP_FULL, ZDISP_INOUT )


def sinlerp(a,b,t): return int( (b-a)*math.sin(t*math.pi)+a )

def create_base_image(w,h):
    return np.full( (h,w,4), 0.0, np.float32) # size is height, width (y,x)

if __name__ == "__main__":
    main()