import os
import cv2
import numpy as np


tif_dir = os.getcwd()
all_files = os.listdir()
tifs = [f for f in all_files if f.endswith(".tif")]

mask_ch = None
mask_H, mask_W = None, None
for tf in tifs:
    ch_path = os.path.join(tif_dir, tf)
    ch = cv2.imread(ch_path, cv2.IMREAD_UNCHANGED)
    # for one image we only cropped the 1st ch (red)
    # then use this bit of logic to crop the others
    if type(mask_H) == type(None):
        mask_H, mask_W = ch.shape
    ch = ch[:mask_H, -mask_W:]
    norm_ch = ch / ch.max()
    flat_ch = norm_ch.flatten()
    if type(mask_ch) == type(None):
        mask_ch = flat_ch
    flat_ch = flat_ch[mask_ch > 0]
    np.save(ch_path[:-4] + '.npy', flat_ch)