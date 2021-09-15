import time, os, pathlib
from .io import read_tif16, write_tif16
from .imgop import apply_cityblock_vignette, apply_gaussian_vignette
from tqdm import tqdm

# pth_src and pth_dest expect pathlib.Path
def refine_directory(pth_src, xydisp_full, zdisp_inout, pth_dst = None):
    if not isinstance(pth_src, pathlib.Path): pth_src = pathlib.Path(pth_src)
    if pth_dst is None: pth_dst = pth_src
    elif not isinstance(pth_dst, pathlib.Path): pth_dst = pathlib.Path(pth_dst)

    files = sorted([p.resolve() for p in pth_src.glob("*") if p.suffix in [".tif", ".tiff"]])
    pbar = tqdm(files)
    for f in pbar:
        #if f.parts[-1].endswith("{}.tif".format(suffix)): continue
        #start_time = time.time()
        pbar.set_description("refining {}".format(f))
        #print("{}".format(f.name))
        _refine_path(f,pth_dst,xydisp_full, zdisp_inout)
        #print("processed {} in {}s".format(f.name, (time.time() - start_time)))


def _refine_path(pth_src, pth_dst, xydisp_full, zdisp_inout):
    #fname, ext = os.path.splitext(pth_src)
    #print("processing {}".format(pth_src.name))

    img = read_tif16(pth_src,xydisp_full, zdisp_inout) 

    apply_cityblock_vignette(img,do_save_mask=False)
    #apply_gaussian_vignette(img,do_save_mask=False)

    if pth_src.parents[0] == pth_dst:
        write_tif16(img, pth_src.parents[0] / (pth_src.stem + "_refined.tif"), xydisp_full, zdisp_inout )
    else:
        write_tif16(img, pth_dst / pth_src.name, xydisp_full, zdisp_inout )



