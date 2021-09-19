import pathlib, time, traceback
import pymeshlab
import vdisp

PTH_SRC = pathlib.Path(r'C:\tmp\210912_dphheads-000201-210915-circle-400_0-2\solid')
PTH_DST = pathlib.Path(r'C:\tmp\210912_dphheads-000201-210915-circle-400_0-2\parts')

PARTBOOL = "hex.ply"

def main():
    pth_msh = pathlib.Path(vdisp.__file__).parent / 'msh' # path to module mesh directory
    files = [p.resolve() for p in pth_msh.glob("*") if p.suffix in [".ply", ".obj"]]
    pth_partbool = [p for p in files if p.name == PARTBOOL]
    if len(pth_partbool)!=1: raise Exception("Cannot trim to mesh {} because it was not found in the module mesh directory".format(PARTBOOL))
    pth_partbool = str(pth_partbool[0])


    files = sorted([p.resolve() for p in PTH_SRC.glob("*") if p.suffix in [".obj"]])
    for f in files:
        start_time = time.time()
        print("\n\n------------------------------ {}\n-----".format(f.name))
        try:
            process_path(f,PTH_DST,pth_partbool)
            print("----- processed {} in {}s\n------------------------------".format(f.name, (time.time() - start_time)))
        except Exception as e:
            print("!!!!! FAILED {} in {}s\n!!!!!!!!!!!!!!!!!!!!!".format(f.name, (time.time() - start_time)))
            print(traceback.format_exc())
        


def process_path(pth_src, pth_dst, pth_partbool):
    pth_dst = pathlib.Path(pth_dst / "{}".format(pth_src.stem) ).with_suffix('.obj')

    ms = pymeshlab.MeshSet()

    ms.load_new_mesh(str(pth_src))
    print("... loaded base mesh in Meshlab.")
    ms.load_new_mesh(str(pth_partbool))
    print("... loaded trimming mesh. now there are {} meshes.".format(len(ms)))
    
    start_time = time.time()
    ms.mesh_boolean_intersection(first_mesh=0, second_mesh=1)
    print("... completed ms.mesh_boolean_intersection() in {:.1f}s. now there are {} meshes.".format((time.time() - start_time),len(ms)))

    ms.remove_duplicate_vertices()
    ms.remove_duplicate_faces()
    ms.merge_close_vertices()
    ms.save_current_mesh(str(pth_dst))

if __name__ == "__main__":
    main()