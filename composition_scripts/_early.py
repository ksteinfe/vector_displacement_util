import vdisp as vd
import vdisp.imgop as iop
import pathlib, os
import numpy as np
import cv2


def main():
    pass


            
def animate_rotation(pth_src):
    
    files = sorted([p.resolve() for p in pathlib.Path(pth_src).glob("*") if p.suffix in [".tif", ".tiff"]])
    for f in files:
        print("{}".format(f.name))
        fname, ext = os.path.splitext(f)
        img = vd.read_tif16(f)
        sz = (img.shape[0],img.shape[1])

        steps = 200
        for n in range(steps):
            deg = int(n/steps * 360)
            print(deg)
            vd.write_tif16( cv2.resize( iop.rotate(np.copy(img), deg), sz ) , "{}_{:03}.tif".format(fname,n))




def _combine_210802():
    pth_a = r'C:\tmp\bustofalady_01_512.tif'
    pth_b = r'C:\tmp\pergamonpanel_01_512.tif'

    img_a = vd.read_tif16(pth_a)
    img_b = vd.read_tif16(pth_b)
    sze = img_a.shape[0]

    iop.apply_cityblock_vignette(img_a)
    iop.apply_cityblock_vignette(img_b)

    def func(ia,ib,pa,pb):
        img = np.full((sze,int(sze*1.5),4), 0.0, np.float32) # size is height, width (y,x)

        # Perform blending
        iop.overlay_image_alpha(img, ia, pa, 0)
        iop.overlay_image_alpha(img, ib, pb, 0)
        return(img)

    pos0,pos1,dlt = 0,sze*0.5,sze*0.5
    frames = 100
    for n in range(frames):
        img = func(img_a, img_b, int(pos0+dlt*n/frames), int(pos1-dlt*n/frames))
        vd.write_tif16(img, r"C:\tmp\combine\pergalady_{:03}.tif".format(n))

    for n in range(frames):
        img = func(img_b, img_a, int(pos0+dlt*n/frames), int(pos1-dlt*n/frames))
        vd.write_tif16(img, r"C:\tmp\combine\pergalady_{:03}.tif".format(n+frames)) 


def _combine():
    cnt_frames = 400
    iw = 2826 # target image width
    


    pth_a = r'C:\tmp\bustofalady_03_512.tif'
    pth_b = r'C:\tmp\headofdavid_02_512.tif'
    pth_c = r'C:\tmp\headofdavid_03_512.tif'
    pth_d = r'C:\tmp\headofdavid_04_512.tif'
    pth_e = r'C:\tmp\headofdavid_05_512.tif'

    img_a = vd.read_tif16(pth_a)
    img_b = vd.read_tif16(pth_b)
    img_c = vd.read_tif16(pth_c)
    img_d = vd.read_tif16(pth_d)
    img_e = vd.read_tif16(pth_e)

    iop.apply_cityblock_vignette(img_a)
    iop.apply_cityblock_vignette(img_b)
    iop.apply_cityblock_vignette(img_c)
    iop.apply_cityblock_vignette(img_d)
    iop.apply_cityblock_vignette(img_e)

    img_a = fade_sine(img_a, 0.5, 1.0, cnt_frames)
    img_b = fade_sine(img_b, 1.0, 0.5, cnt_frames)
    img_c = fade_sine(img_c, 0.5, 1.0, cnt_frames)
    img_d = fade_sine(img_d, 0.5, 1.0, cnt_frames)
    img_e = fade_sine(img_e, 1.0, 0.5, cnt_frames)

    arr = np.sin( np.linspace(0,np.pi,cnt_frames) )
    pas = (arr * iw).astype(np.int)
    pbs = (arr * -iw/2).astype(np.int)
    pcs = np.flip(np.linspace(0,iw,cnt_frames)+512).astype(np.int)
    pds = np.flip(np.linspace(0,iw,cnt_frames)+256).astype(np.int)
    pes = ((arr * -iw)+iw/2).astype(np.int)

    for n,ps in enumerate(zip(pas,pbs,pcs,pds,pes)):
        pa,pb,pc,pd,pe = ps
        #print(n,pa,pb,pc)
        img = np.full( (512,iw,4), 0.0, np.float32) # size is height, width (y,x)

        # Perform blending
        iop.overlay_image_alpha(img, img_a[n], pa, 0)
        iop.overlay_image_alpha(img, img_b[n], pb, 0)
        iop.overlay_image_alpha(img, img_c[n], pc, 0)
        iop.overlay_image_alpha(img, img_d[n], pd, 0)
        iop.overlay_image_alpha(img, img_e[n], pe, 0)
        vd.write_tif16(img, r"C:\tmp\combine\{:03}.tif".format(n)) 


# given an image, returns steps number of images with their vector lengths adjusted to fade from t0 -> t1 -> t0 using sine fading
def fade_sine(base_img, t0, t1, steps):
    imgs = []
    for n in range(steps):
        t = np.sin(np.pi * float(n) / steps)
        t = (t1-t0)*t+t0
        img = np.copy(base_img)
        img[:,:,0] *= t
        img[:,:,1] *= t
        img[:,:,2] *= t
        imgs.append(img)
    return imgs



if __name__ == "__main__":
    main()
