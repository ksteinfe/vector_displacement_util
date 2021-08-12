import time, os, pathlib
from .io import read_tif16, write_tif16
from .imgop import apply_cityblock_vignette

def refine_directory(pth_src):
    suffix = "refined"
    files = sorted([p.resolve() for p in pathlib.Path(pth_src).glob("*") if p.suffix in [".tif", ".tiff"]])
    for f in files:
        if f.parts[-1].endswith("{}.tif".format(suffix)): continue
        start_time = time.time()
        print("\n----- {}".format(f.name))
        refine_path(f,suffix)
        print("processed {} in {}s".format(f.name, (time.time() - start_time)))


def refine_path(pth_src,suffix):
    fname, ext = os.path.splitext(pth_src)
    print("processing {}".format(pth_src))

    img = read_tif16(pth_src) 

    apply_cityblock_vignette(img,do_save_mask=False)
    write_tif16(img, "{}_{}.tif".format(fname,suffix))



