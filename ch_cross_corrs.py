# Note the color channels are mixed w.r.t the paper
# Green is actually SYN1, Blue is GCaMP.

import os
import csv
import cv2
import time
import datetime
import matplotlib
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage.segmentation import flood


tif_dir = os.getcwd()

# processes subsets of the images in parallel processes
def block_worker(b, c, h_offset, w_offset, size_thresh):
    # b == image block; c == clusters list, offsets wrt full image
    sq = square(3)
    H, W = b.shape
    
    # check the edge for clusters that might be split between blocks
    for h in range(H):
        for w in [0, -1]:
            if b[h, w] != 0:
                cluster_mask = flood(b, (h, w), footprint=sq)                    
                b[cluster_mask == True] = 0
                c.append(['edge', cluster_mask, h_offset, w_offset])
    for w in range(W):
        for h in [0, -1]:
            if b[h, w] != 0:
                cluster_mask = flood(b, (h, w), footprint=sq)                    
                b[cluster_mask == True] = 0
                c.append(['edge', cluster_mask, h_offset, w_offset])
    
    # cheeky little trick to 4X the speed
    for h in range(0, H, 4):
        # that's 16x savings for those keeping track at home!
        for w in range(0, W, 4):
            # flood fill if not zero
            if b[h, w] != 0:
                cluster_mask = flood(b, (h, w), footprint=sq)                    
                # set cluster's pixels to black
                b[cluster_mask == True] = 0
                cluster_size = cluster_mask.sum()
                # filter by size
                if cluster_size >= size_thresh:
                    # record the current cluster
                    c.append(['normal', cluster_mask, h_offset, w_offset])
    return c # end `def block_worker`

all_files = os.listdir(tif_dir)
tifs = [f for f in all_files if f.endswith(".tif") or f.endswith(".tiff")]
tifs = sorted(tifs)
err_txt = 'Your current directory contains no TIF images :('
assert len(tifs) > 0, err_txt

try:
    mp.set_start_method('spawn')
except:
    pass

r_threshold = 515
g_threshold = 350
results_dir = os.path.join(tif_dir)
ch1 = tifs[0]
ch2 = tifs[1]
ch3 = tifs[2]
    
# naming constraint: Begins w/ # of pyr cells followed by '_'
char_after_soma_count = ch1.find("_") + 1
name_root = ch1[char_after_soma_count:-7]

size_thresh = 155
block_size = 500 # for block tiling
edge_block_size = 990
# avoids hitting the edges of the image
assert edge_block_size < block_size * 2, ':('

ch1_fullpath = os.path.join(tif_dir, ch1)
og_red = cv2.imread(ch1_fullpath, cv2.IMREAD_UNCHANGED)
# I cropped only the red channel to pyramidal cell layer
og_H, og_W = og_red.shape
# We add a single px of black around the whole perimeter
# to avoid labeling true edge clusters as "edge" clusters
# that result from being split acorss tiled blocks
H = og_H + 2
W = og_W + 2
# Additionally, we pad to be a whole number of blocks
H = (H // block_size + 1) * block_size
W = (W // block_size + 1) * block_size
padded_version = np.zeros((H, W))
end_h, end_w = 1+og_red.shape[0], 1+og_red.shape[1]
padded_version[1:end_h, 1:end_w] = og_red
og_red = padded_version.copy()
red = og_red.copy()
ch2_fullpath = os.path.join(tif_dir, ch2)
og_green = cv2.imread(ch2_fullpath, cv2.IMREAD_UNCHANGED)
og_green = og_green[:og_H,-og_W:]
padded_version[1:end_h, 1:end_w] = og_green
og_green = padded_version.copy()
green = og_green.copy()
ch3_fullpath = os.path.join(tif_dir, ch3)
og_blue = cv2.imread(ch3_fullpath, cv2.IMREAD_UNCHANGED)
og_blue = og_blue[:og_H,-og_W:]
padded_version[1:end_h, 1:end_w] = og_blue
og_blue = padded_version.copy()
del padded_version
blue = og_blue.copy()

red[og_red < r_threshold] = 0
red[og_green < g_threshold] = 0
# green[og_green < g_threshold] = 0
# green[og_red < r_threshold] = 0
red = red / red.max() # normalize
# green = green / green.max() # normalize
binary_thresh = red.copy()
binary_thresh[binary_thresh > 0] = 1

clusters = []
H, W = red.shape 
# for refining clusters that were found on the edge of a main_block
for h in range(0, H, block_size):
    for w in range(0, W, block_size):
        block = binary_thresh[h:h+block_size, 
                              w:w+block_size].copy()
        # skip any that are blank (i.e. pure black, not signal)
        if block.sum() == 0: continue
        clusters = block_worker(block, clusters, h, w, size_thresh)

gain = 3
norm_red = og_red / og_red.max()
norm_green = og_green / og_green.max()
norm_blue = og_blue / og_blue.max()
original = np.dstack([norm_red, norm_green, norm_blue])
del norm_red, norm_green, norm_blue
clusters_removed = original.copy()
only_clusters = np.zeros_like(original)
viz_clusters = np.zeros_like(binary_thresh)
colors = [x / 8 for x in range(1, 9)]

# print(f"\n{len(clusters)} clusters unfiltered!\n")
edge_masks = []
i = -1 # allows incrementing at the beginning of loop
# save cluster results in individual csv files too
csv_name = os.path.join(results_dir, f"{name_root}.csv")
results = []
while clusters:
    i += 1
    c = clusters.pop()
    # print(f'putative cluster #{i}')
    c_type = c[0]
    c_mask = c[1]
    h_offset = c[2]
    w_offset = c[3]
    assert len(c_mask) > 0
    num_rows = len(results)
    # reapply flood centered on each edge cluster
    if c_type == 'normal':
        for h, w in np.argwhere(c_mask):
            pt = (h + h_offset, w + w_offset)
            viz_clusters[pt] = colors[num_rows % len(colors)]
            clusters_removed[pt] = 0
        h_end = h_offset + block_size
        w_end = w_offset + block_size
        size = c_mask.sum()
        red_block = og_red[h_offset:h_end, w_offset:w_end]
        cluster_psd = red_block[c_mask == True]
        avg_psd = cluster_psd.mean()
        green_block = og_green[h_offset:h_end, w_offset:w_end]
        cluster_syn = green_block[c_mask == True]
        avg_syn = cluster_syn.mean()
        results.append([num_rows, size, avg_psd, avg_syn])
    elif c_type == 'edge':            
        mask_p = np.argwhere(c_mask)[0]
        h = h_offset + mask_p[0] - edge_block_size // 2
        # Couldn't make all cases fast, but it's surely possible
        fast_way = True
        if h < 0:
            x = mask_p[0]
            h = 0
            h_end = edge_block_size
        else:
            x = edge_block_size // 2
            h_end = h + edge_block_size
            if h_end > H:
                fast_way = False
        w = w_offset + mask_p[1] - edge_block_size // 2
        if w < 0:
            y = mask_p[1]
            w = 0
            w_end = edge_block_size
        else:
            y = edge_block_size // 2
            w_end = w + edge_block_size
            if w_end > W:
                fast_way = False
        if fast_way:
            block = binary_thresh[h:h_end, w:w_end]
            flood_pt = (x, y)
            # simple sanity check that our indexing isn't off by 1
            assert block[flood_pt] == 1 
            c_mask = flood(block, flood_pt, footprint=square(3))
            _c_mask = np.argwhere(c_mask)
            c_mask = []
            for x,y in _c_mask:
                c_mask.append([x+h, y+w])
        else:
            flood_pt = (mask_p[0] + h_offset, mask_p[1] + w_offset)
            # simple sanity check that our indexing is correct
            assert binary_thresh[flood_pt] == 1 
            c_mask = flood(binary_thresh, flood_pt, footprint=square(3))
            c_mask = np.argwhere(c_mask)
        assert len(c_mask) > 0
        size = len(c_mask)
        if size >= size_thresh:
            # filter duplicate clusters
            is_new = True
            for em in edge_masks:
                if np.array_equal(c_mask, em):
                    is_new = False
                    break
            if is_new:
                edge_masks.append(c_mask)
                cluster_psd = 0
                cluster_syn = 0
                for x,y in c_mask:
                    viz_clusters[x,y] = colors[num_rows % len(colors)]
                    clusters_removed[x,y] = 0
                    only_clusters[x,y,:] = original[x,y,:]
                    cluster_psd += og_red[x,y]
                    cluster_syn += og_green[x,y]
                avg_psd = cluster_psd / size
                avg_syn = cluster_syn / size
                results.append([num_rows, size, avg_psd, avg_syn])
del edge_masks

fig, ax = plt.subplots(nrows=2, ncols=3)
fig.set_figheight(15)
fig.set_figwidth(22)
original[:,:,1][original[:,:,0] == 0] = 0
original[:,:,2][original[:,:,0] == 0] = 0
ax[0,0].imshow(original * gain, interpolation='none')
ax[0,0].set_title('original')
ax[0,1].imshow(only_clusters * gain, interpolation='none')
ax[0,1].set_title('only clusters')
clusters_removed[original[:,:,0] == 0] = 0
ax[0,2].imshow(clusters_removed * gain, interpolation='none')
ax[0,2].set_title('clusters removed')
# Note: Green and Blue are swapped relative to some of the paper's images
_original = original.copy() * gain
_original[:,:,1:] = 0
ax[1,0].imshow(_original, interpolation='none')
ax[1,0].set_title('PSD95')
_original = original.copy() * gain
_original[:,:,:2] = 0
ax[1,1].imshow(_original, interpolation='none')
ax[1,1].set_title('GCaMP6')
_original = original.copy() * gain
_original[:,:,0] = 0
_original[:,:,2] = 0
for i in range(2):
    for j in range(3):
        ax[i,j].set_axis_off()
ax[1,2].imshow(_original, interpolation='none')
ax[1,2].set_title('Syn-1')
chs_img_path = f'{name_root}.png'
chs_img_path = os.path.join(results_dir, chs_img_path)
plt.savefig(chs_img_path, dpi=300)
plt.close(fig) # prevent plotting huge figures inline

del _original

cluster_img_path = f'{name_root}.png'
cluster_img_path = os.path.join(results_dir, cluster_img_path)
plt.savefig(cluster_img_path, dpi=300)
plt.close(fig) # prevent plotting huge figures inline

def create_rmap(flat_ch1, flat_ch2):
    # create correlation map
    R_map = np.zeros((61,61))
    # this channel does not move
    _static_ch = flat_ch1[30:-31, 30:-31]
    static_ch = _static_ch[_static_ch > 0].flatten()
    for i in range(61):
        for j in range(61):
            # start top left
            _slide_ch = flat_ch2[i:i-61, j:j-61]
            slide_ch = _slide_ch[_static_ch > 0].flatten()
            R1 = np.corrcoef(static_ch, slide_ch)
            R_map[i,j] = R1[0,1]
            print(f'R_map[{i},{j}]: {R1[0,1]}')
  
    return R_map

for condition in ["only", "without"]:
    if condition == "without":
        img = clusters_removed
    else:
        img = only_clusters
        
    corr_file = os.path.join(results_dir, f"{name_root}_{condition}_corr.txt")
        
    _flat_red = img[:,:,0]
    flat_red = _flat_red[_flat_red > 0].flatten()
    _flat_green = img[:,:,1]
    flat_green = _flat_green[_flat_red > 0].flatten()
    _flat_blue = img[:,:,2]
    flat_blue = _flat_blue[_flat_red > 0].flatten()
    R1 = np.corrcoef([flat_red, flat_green, flat_blue])
    
    # correlation including only detected clusters
    with open(corr_file, 'w') as cf:
        # RG, RB, GB correlations
        # implicit RR == GG == BB == 1
        cf.write(f"{R1[0,1]}\n{R1[0,2]}\n{R1[1,2]}\n")
    
    pairwise_chs = []
    pairwise_chs.append([_flat_red, _flat_red, 'RR'])
    pairwise_chs.append([_flat_red, _flat_green, 'RG'])
    pairwise_chs.append([_flat_red, _flat_blue, 'RB'])
    pairwise_chs.append([_flat_green, _flat_green, 'GG'])
    pairwise_chs.append([_flat_green, _flat_blue, 'GB'])
    pairwise_chs.append([_flat_blue, _flat_blue, 'BB']) # GCaMP - GCaMP
    for ch1, ch2, name in pairwise_chs:
        R_map = create_rmap(ch1, ch2)
    
        fig = plt.figure()
        cmap = matplotlib.colormaps['jet']
        _R_map = plt.imshow(R_map, cmap=cmap, interpolation='none', vmax=1, vmin=-0.2)
        plt.colorbar(_R_map)
        
        R_map_path = os.path.join(results_dir, f'{condition}_{name}_Rmap.png')
        plt.savefig(R_map_path, dpi=300)
        plt.close(fig)
