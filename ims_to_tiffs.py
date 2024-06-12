import tifffile
from imaris_ims_file_reader.ims import ims

results_dir = os.path.join(os.getcwd(), r"/tiffs/")
os.makedirs(results_dir, exist_ok=True)

ims_files = os.listdir(os.getcwd())
ims_files = [os.path.join(os.getcwd(), f) for f in ims_files]

for ims_file in ims_files:
    root_name = ims_file.split(r'/')[-1][:-4]
    ims_data = ims(ims_file)
    converted_file = os.path.join(results_dir, root_name)
    tifffile.imwrite(converted_file, ims_data[0,::2,0,:,:])

# psd_ch = ims_data[0,0,0,:,:].copy()
# syn_ch = ims_data[0,1,0,:,:].copy()
# will keep them separate and just crop one because thresholding will functionally crop the other :)
# tifffile.imwrite(f'{r"/tmp/"}{root_name}_CH1.tiff', psd_ch)
# tifffile.imwrite(f'{r"/tmp/"}{root_name}_CH2.tiff', syn_ch)
