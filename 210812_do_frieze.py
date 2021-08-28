import vdisp as vd
import vdisp.imgop as iop
import numpy as np
import cv2
from tqdm import tqdm
import pathlib, itertools, math

PTH_DST = pathlib.Path(r'C:\tmp\frieze')
PTH_TST_IMG = pathlib.Path(r'C:\tmp\to_combine\bustofalady_03_512.tif')
WDTH = 2485
HGHT = 1536
CNT_FRAMES = 200
OFST_FRAME = max(1,int(CNT_FRAMES/5))

def main():
    seq_a = load_sequence( pathlib.Path(r'G:\My Drive\Research and Professional Service\21 Venice Biennale\Biennale Production\Blender Farm\210812_mirrorwalk\tif_refined') )
    seq_b = load_sequence( pathlib.Path(r'G:\My Drive\Research and Professional Service\21 Venice Biennale\Biennale Production\Blender Farm\210812_first\tif_b') )
    seq_c = load_sequence( pathlib.Path(r'C:\tmp\to_combine') )



    pbar = tqdm(range(CNT_FRAMES))
    for n in pbar:
        pbar.set_description("frame {} of {}".format(n,CNT_FRAMES))
        #for n in range(CNT_FRAMES):
        make_frame(n, seq_a, seq_b, seq_c)


def make_frame(idx_frame, seq_a, seq_b, seq_c):
    t = idx_frame / CNT_FRAMES
    bimg = create_base_image(WDTH,HGHT)

    pt_a = sinlerp_pt( (-128,0), (-1024,0), t)
    sz_a = sinlerp( 512, 1024, t )
    frieze_walk(bimg, seq_a, sz_a, pt_a, idx_frame, style="sidle" )

    pt_b = sinlerp_pt( (-1024,512), (-128,1024), t )
    sz_b = sinlerp( 1024, 512, t )    
    frieze_walk(bimg, seq_b, sz_b, pt_b, idx_frame, style="sidle"  )

    cx = sinlerp(-275,-275,t)
    cy = sinlerp(384,640,t)
    #pt_c = sinlerp_pt( (-275,384), (-275,640), t )
    frieze_stills(bimg, seq_c, 512, (cx,cy), overlap=0.5, style="sidle"  )

    vd.write_tif16(bimg, PTH_DST / "{:03}.tif".format(idx_frame) )



def frieze_walk(bimg,seq,sz,pt_origin,idx_seq_start,overlap=0.5,style="hop"):
    if style not in ["hop","sidle"]: raise Exception("{} is not a frieze group that I know.".format(style))
    px,py = pt_origin
    idx_seq = idx_seq_start
    n = 0
    while px < WDTH:
        im = iop.resize(seq[idx_seq%CNT_FRAMES],(sz,sz))
        if n%2==0 and style == "sidle":
            im = iop.flip_x(im)

        iop.overlay_image_alpha(bimg, im, px, py, tile_horz=False, tile_vert=False)
        px += int(sz*overlap)
        idx_seq += OFST_FRAME
        n+=1 

def frieze_stills(bimg,seq,sz,pt_origin,overlap=0.5,style="hop"):
    if style not in ["hop","sidle"]: raise Exception("{} is not a frieze group that I know.".format(style))
    px,py = pt_origin
    n = 0
    while px < WDTH:
        im = iop.resize(seq[n%len(seq)],(sz,sz))
        if n%2==0 and style == "sidle":
            iop.overlay_image_alpha(bimg, im, px, py, tile_horz=False, tile_vert=False)
            px += int(sz*overlap)
            im = iop.flip_x(im)
            iop.overlay_image_alpha(bimg, im, px, py, tile_horz=False, tile_vert=False)
            px += int(sz*overlap)

        n+=1 



def lerp_pt(pa,pb,t): return lerp(pa[0],pb[0],t), lerp(pa[1],pb[1],t)
def lerp(a,b,t): return int( (b-a)*t+a )

def sinlerp_pt(pa,pb,t): return sinlerp(pa[0],pb[0],t), sinlerp(pa[1],pb[1],t)
def sinlerp(a,b,t): return int( (b-a)*math.sin(t*math.pi)+a )



def fake_sequence():
    img = vd.read_tif16(PTH_TST_IMG)
    return list(itertools.repeat(img, CNT_FRAMES))


def load_sequence(pth):
    print("loading sequence {}".format(pth.stem))
    files = sorted([p.resolve() for p in pth.glob("*") if p.suffix in [".tif", ".tiff"]])
    if len(files)==0: raise Exception("No TIFF files found in {}".format(pth))

    ret,n = [],0
    while len(ret)<CNT_FRAMES:
        ret.append(vd.read_tif16(files[n%len(files)]))
        n+=1
    return ret

def create_base_image(w,h):
    return np.full( (h,w,4), 0.0, np.float32) # size is height, width (y,x)




if __name__ == "__main__":
    main()