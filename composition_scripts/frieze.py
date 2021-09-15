import sys  
from pathlib import Path
sys.path.append(str(Path(__file__). resolve().parents[1]))

import vdisp as vd
import vdisp.imgop as iop
import numpy as np
import cv2
from tqdm import tqdm
import pathlib, itertools, math

PTH_DST = pathlib.Path(r'C:\tmp\frieze')
CNT_FRAMES = 40


def main():
    cfg = { # keep size at aspect ratio of object in .blend file, check with offset values below
        'wdth': 5787,
        'hght': 1536,
        'ofst_frame': max(1,int(CNT_FRAMES/5)),
    }
    #deep_freize(cfg)

    cfg = { # keep size to aspect ratio of unit cylinder (1.25 : (6.28*.75) == 0.26539)
        'wdth': 1930,
        'hght': 512,
        'ofst_frame': max(1,int(CNT_FRAMES/5)),
    }
    strip_freize(cfg)



def strip_freize(cfg):
    seq_a = vd.load_sequence( pathlib.Path(r'C:\tmp\sequence_a'), CNT_FRAMES )
    seq_b = vd.load_sequence( pathlib.Path(r'C:\tmp\sequence_b'), CNT_FRAMES )
    pbar = tqdm(range(CNT_FRAMES))
    for idx_frame in pbar:
        pbar.set_description("frame {} of {}".format(idx_frame,CNT_FRAMES))
        t = idx_frame / CNT_FRAMES
        bimg = create_base_image(cfg['wdth'],cfg['hght'])

        cx = sinlerp(0,-cfg['wdth'],t)
        frieze_stills(cfg, bimg, seq_a, 512, (cx,0), overlap=0.9, style="hop"  )

        cx = sinlerp(-cfg['wdth'],0,t)
        frieze_stills(cfg, bimg, seq_b, 512, (cx,0), overlap=0.9, style="hop", do_invert=True )        

        vd.write_tif16(bimg, PTH_DST / "{:03}.tif".format(idx_frame) )

def deep_freize(cfg):
    seq_a = vd.load_sequence( pathlib.Path(r'G:\My Drive\Research and Professional Service\21 Venice Biennale\Biennale Production\Blender Farm\210812_mirrorwalk\tif_refined'),CNT_FRAMES )
    seq_b = vd.load_sequence( pathlib.Path(r'G:\My Drive\Research and Professional Service\21 Venice Biennale\Biennale Production\Blender Farm\210812_first\tif_b'),CNT_FRAMES )
    seq_c = vd.load_sequence( pathlib.Path(r'C:\tmp\sequence_a') )

    pbar = tqdm(range(CNT_FRAMES))
    for n in pbar:
        pbar.set_description("frame {} of {}".format(n,CNT_FRAMES))
        #for n in range(CNT_FRAMES):
        deep_freize_frame(cfg, n, seq_a, seq_b, seq_c)


def deep_freize_frame(cfg, idx_frame, seq_a, seq_b, seq_c):
    t = idx_frame / CNT_FRAMES
    bimg = create_base_image(cfg['wdth'],cfg['hght'])

    pt_a = sinlerp_pt( (-128,0), (-1024,0), t)
    sz_a = sinlerp( 512, 1024, t )
    frieze_walk(cfg, bimg, seq_a, sz_a, pt_a, idx_frame, overlap=0.6, style="sidle" )

    pt_b = sinlerp_pt( (-1024,512), (-128,1024), t )
    sz_b = sinlerp( 1024, 512, t )    
    frieze_walk(cfg, bimg, seq_b, sz_b, pt_b, idx_frame, overlap=0.6, style="sidle"  )

    cx = sinlerp(-275,-275,t)
    cy = sinlerp(384,640,t)
    #pt_c = sinlerp_pt( (-275,384), (-275,640), t )
    frieze_stills(cfg, bimg, seq_c, 512, (cx,cy), overlap=0.75, style="sidle"  )

    vd.write_tif16(bimg, PTH_DST / "{:03}.tif".format(idx_frame) )



def frieze_walk(cfg, bimg,seq,sz,pt_origin,idx_seq_start,overlap=1.0,style="hop",do_invert=False):
    if style not in ["hop","sidle"]: raise Exception("{} is not a frieze group that I know.".format(style))
    px,py = pt_origin
    idx_seq = idx_seq_start
    n = 0
    while px < cfg['wdth']:
        im = iop.resize(seq[idx_seq%CNT_FRAMES],(sz,sz))
        if n%2==0 and style == "sidle":
            im = iop.flip_x(im)
            if do_invert: im = iop.flip_z(im)

        iop.overlay_image_alpha(bimg, im, px, py, tile_horz=False, tile_vert=False)
        px += int(sz*overlap)
        idx_seq += cfg['ofst_frame']
        n+=1 

def frieze_stills(cfg, bimg,seq,sz,pt_origin,overlap=1.0,style="hop",do_invert=False):
    if style not in ["hop","sidle"]: raise Exception("{} is not a frieze group that I know.".format(style))
    px,py = pt_origin
    n = 0
    while px < cfg['wdth']:
        im = iop.resize(seq[n%len(seq)],(sz,sz))
        if n%2==0 and style == "hop":
            if do_invert: im = iop.flip_z(im)
            iop.overlay_image_alpha(bimg, im, px, py, tile_horz=False, tile_vert=False)   
            px += int(sz*overlap)    
        elif n%2==0 and style == "sidle":
            iop.overlay_image_alpha(bimg, im, px, py, tile_horz=False, tile_vert=False)
            px += int(sz*overlap)
            im = iop.flip_x(im)
            if do_invert: im = iop.flip_z(im)
            iop.overlay_image_alpha(bimg, im, px, py, tile_horz=False, tile_vert=False)
            px += int(sz*overlap)

        n+=1 



def lerp_pt(pa,pb,t): return lerp(pa[0],pb[0],t), lerp(pa[1],pb[1],t)
def lerp(a,b,t): return int( (b-a)*t+a )

def sinlerp_pt(pa,pb,t): return sinlerp(pa[0],pb[0],t), sinlerp(pa[1],pb[1],t)
def sinlerp(a,b,t): return int( (b-a)*math.sin(t*math.pi)+a )



def create_base_image(w,h):
    return np.full( (h,w,4), 0.0, np.float32) # size is height, width (y,x)




if __name__ == "__main__":
    main()