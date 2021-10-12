import os, pathlib, time, traceback, json, shutil, random
import pymeshlab
import vdisp
import vdisp.mshop as mshop
import numpy as np

PTH_SRC = pathlib.Path(r'G:\My Drive\Research and Professional Service\21 Venice Biennale\Biennale Production\Latent Printable\210918_facesoverfit-000020-210930-circle-500-4')
FNAME_CFG = "aggregation_exp.json"
DIRNAME_OBJ = "solids_9" # subdirectory of PTH_SRC that contains source files
DIRNAME_DST = "parts" # subdirectory of PTH_SRC in which to save results. existing files will be overwritten.

DIRNAME_MOD_MSH = "msh" # subdirectory of VDISP module that contains meshes for booleans
SCALE = 100
def main():
    if not os.path.isfile(PTH_SRC / FNAME_CFG): raise FileExistsError("Could not find {} in {}".format(FNAME_CFG, PTH_SRC))
    cfg_agg = False
    with open(PTH_SRC / FNAME_CFG) as f: cfg_agg = json.load(f)
    print("Loaded a {} step aggregation with {} operations, with each sample cut into {} pieces.".format(len(cfg_agg['samples']), len(cfg_agg['ops']), len(cfg_agg['piecewise'])) )

    pths_obj = sorted([p.resolve() for p in (PTH_SRC / DIRNAME_OBJ).glob("*") if p.suffix in [".obj"]])
    if len(pths_obj) < len(cfg_agg['samples']): raise FileNotFoundError("Not enough OBJ files in {}.\nI found {}, but need at least {}".format(PTH_SRC / DIRNAME_OBJ, len(pths_obj), len(cfg_agg['samples'])))
    print("Discovered {} source OBJ files.".format(len(pths_obj)))

    # validate all Boolean meshes
    pth_module_msh = pathlib.Path(vdisp.__file__).parent / DIRNAME_MOD_MSH # path to module mesh directory 
    for b in cfg_agg['ops']: b['pth'] = mshop.validate_module_mesh(b['fname'], pth_module_msh)
    for p in cfg_agg['piecewise']: 
        for b in p['ops']:
            b['pth'] = mshop.validate_module_mesh(b['fname'], pth_module_msh)
    
    # match aggregation samples with evenly-spaced OBJ files from the latent walk
    fidxs = list(np.linspace( 0, len(pths_obj)-1, num=len(cfg_agg['samples']) ).astype(int))
    for d,fidx in zip(cfg_agg['samples'],fidxs):
        d['pth_file'] = pths_obj[fidx]

    # set up dest directory, remove existing files if present
    pth_dst = PTH_SRC / DIRNAME_DST
    os.makedirs(pth_dst, exist_ok=True)
    for f in os.listdir(pth_dst):
        try: shutil.rmtree(pth_dst / f)
        except OSError: os.remove(pth_dst / f)

    dirname_id_msh = False # subdirectory of DIRNAME_MOD_MSH that contains meshes for engraving IDs
    if 'engrave_dir' in cfg_agg and cfg_agg['engrave_dir']: dirname_id_msh = cfg_agg['engrave_dir']
    if not dirname_id_msh: print("engraving directory not defined, this part will not be engraved.")    

    #names = ["003","005","006","104","105","106"]

    for n,smpl in enumerate(cfg_agg['samples']):
        #if smpl['name'] not in names:  continue

        start_time = time.time()
        print("\n\n------------------------------ {}\t{}\n-----".format(smpl['name'], smpl['pth_file'].name))
        try:
            pth_id_msh = False
            if dirname_id_msh:
                try:
                    dirname_id_msh = cfg_agg['engrave_dir'] # subdirectory of DIRNAME_MOD_MSH that contains meshes for engraving IDs
                    pth_id_msh = mshop.validate_module_mesh("{}.obj".format(smpl['name']), pth_module_msh / dirname_id_msh) # integrate into piecewise actions
                except FileNotFoundError:
                    print("!!! INVALID ENGRAVING DIRECTORY")
            
            process_sample(smpl,cfg_agg['ops'], cfg_agg['piecewise'], pth_dst, SCALE, pth_id_msh)
            print("----- processed {} in {}s\n------------------------------".format(smpl['name'], (time.time() - start_time)))
        except Exception as e:
            print("!!!!! FAILED {} in {}s\n!!!!!!!!!!!!!!!!!!!!!".format(smpl['name'], (time.time() - start_time)))
            print(traceback.format_exc())

        n+=1
        #if n>3: break
        
    mshop.merge_directory(pth_dst, PTH_SRC, "merged parts.obj")
        


def process_sample(cfg, dct_ops, dct_piecewise, pth_dst, scale, pth_id_msh=False):
    pth_tmp = pathlib.Path(pth_dst.parent / "{}_tmp".format(cfg['name']) ).with_suffix('.obj')
    shutil.copyfile(cfg['pth_file'], pth_tmp)

    if cfg['do_mirror']:
        mshop.do_rotate(pth_tmp, axis=2, angle=cfg['mirror_rot'], cpt=(0.5,0.5,0.0)) # axis 2 is Z axis
        mshop.do_flip_and_or_swap_axis(pth_tmp,about_pt=(-0.5,0.0,0.0), flipx=True)
        mshop.do_rotate(pth_tmp, axis=2, angle=-cfg['mirror_rot'], cpt=(0.5,0.5,0.0)) # axis 2 is Z axis

    for n, spl in enumerate(dct_piecewise):
        print("final split {}".format(spl['name']))

        pth_obj = pathlib.Path(pth_dst / "{}{}".format(cfg['name'],spl['name']) ).with_suffix('.obj')
        shutil.copyfile(pth_tmp, pth_obj)

        
        for n, op in enumerate(spl['ops']):
            print("operation {}: {}".format(n, op['type']))
            if op['type'] == "bool":
                mshop.do_bool( pth_obj , op['pth'], op['btype'])
            elif op['type'] == "thicken_and_cut":
                #TODO:VAL
                mshop.do_thicken_and_cut_window( pth_obj , op['pth'], op['ofst'])
            else:
                raise ValueError("{} is an unknown operation".format(op['type']))

        if pth_id_msh: 
            print("engraving {}".format(pth_id_msh))
            mshop.do_bool( pth_obj , pth_id_msh, "difference")


        if random.choice([True,False]):
            #if False:
            x,y = spl['ctr'][0], spl['ctr'][1]
            mshop.do_rotate(pth_obj, axis=2, angle=180, cpt=(0.5+x,0.5+y,0.0)) # start from origin at 0.5,0.5 -> flip x&y axis; axis 2 is Z axis

        print("... translating to sample center at {} and part spacing at {}".format(cfg['vec'], spl['vec_spacing']))
        x,y = cfg['vec'][0] + spl['vec_spacing'][0], cfg['vec'][1] + spl['vec_spacing'][1] 
        mshop.do_translate(pth_obj, (0.5-x,0.5-y,0) ) # start from origin at 0.5,0.5 -> flip x&y axis

        mshop.do_scale(pth_obj, scale)
    
    
    pth_tmp.unlink()





if __name__ == "__main__":
    main()