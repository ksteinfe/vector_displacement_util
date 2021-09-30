import os, pathlib, time, traceback, json, shutil
import pymeshlab
import vdisp
import vdisp.mshop as mshop
import numpy as np

PTH_SRC = pathlib.Path(r'C:\tmp\210918_faces-000060-210919-circle-400_0-1\parts')
PTH_DST = pathlib.Path(r'C:\tmp\210918_faces-000060-210919-circle-400_0-1')
FNAME_OUT = "merged.obj"

def main():
    mshop.merge_directory(PTH_SRC, PTH_DST, FNAME_OUT)
    



if __name__ == "__main__":
    main()