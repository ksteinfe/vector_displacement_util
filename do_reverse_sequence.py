import vdisp as vd
import pathlib
import shutil

PTH_SRC = r'C:\Users\ksteinfe\Desktop\tif_a'
PTH_DST = r'C:\Users\ksteinfe\Desktop\tif_b'

def main():
    pth_src, pth_dst = pathlib.Path(PTH_SRC), pathlib.Path(PTH_DST)
    if pth_src == pth_dst:
        print("nope. paths are the same. {}".format(pth_src))
        exit()
    
    files = sorted([p.resolve() for p in pth_src.glob("*") if p.suffix in [".tif", ".tiff", ".png"]])
    for n,f in enumerate(files):
        print(f.name)
        shutil.copyfile( f, pth_dst / "{:03}{}".format(len(files)-n-1,f.suffix) )



if __name__ == "__main__":
    main()