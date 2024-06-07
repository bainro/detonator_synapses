# if we do use this, make sure to add the cell counts as prefixes too!

import tifffile
from imaris_ims_file_reader.ims import ims

file_1 = r'/home/rbain/Desktop/ELA_soma_dets_imaris/right_CA3_6L_tiled_first_z_snap_t_right_CA3_6L_tiled_first_z_snap.ims'
file_2 = r'/home/rbain/Desktop/ELA_soma_dets_imaris/right_CA3_10R_tiled_first_z_snap_t_right_CA3_10R_tiled_first_z_snap.ims'
for ims_file in [file_1, file_2]:
    root_name = ims_file.split(r'/')[-1][:-4]
    ims_data = ims(ims_file)
    psd_ch = ims_data[0,0,0,:,:].copy()
    syn_ch = ims_data[0,1,0,:,:].copy()
    # will keep them separate and just crop one because thresholding will functionally crop the other :)
    tifffile.imwrite(f'{r"/tmp/"}{root_name}_CH1.tiff', psd_ch)
    tifffile.imwrite(f'{r"/tmp/"}{root_name}_CH2.tiff', syn_ch)

