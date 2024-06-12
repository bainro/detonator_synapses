import os
import tifffile
from imaris_ims_file_reader.ims import ims

results_dir = os.path.join(os.getcwd(), r"tiffs/")
os.makedirs(results_dir, exist_ok=True)

ims_files = os.listdir(os.getcwd())
ims_files = [os.path.join(os.getcwd(), f) for f in ims_files]

for ims_file in ims_files:
    if os.path.isfile(ims_file) and ims_file[-4:] == '.ims':
        root_name = ims_file.split(r'/')[-1][:-4]
        ims_data = ims(ims_file)
        converted_file = os.path.join(results_dir, root_name + ".tiff")
        # might be able to recover all pixels but a few,
        # but don't care enough right now. https://tinyurl.com/5xfpkyr6
        try:
            tifffile.imwrite(converted_file, ims_data[0,:,0,:,:])    
        except:
            print()
            print(f'{ims_file.split(r"/")[-1]} CONTAINS CORRUPTED PIXEL(S). SKIPPED!')
            print()
