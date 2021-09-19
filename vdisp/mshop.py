import time, os, pathlib
import numpy as np
import pymeshlab




def remesh_path(pth_src, pth_dst):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(pth_src)
    print("loaded mesh")
    ms.surface_reconstruction_screened_poisson()
    ms.save_current_mesh(pth_dst)