# Note the color channels are mixed w.r.t the paper
# Green is actually SYN1, Blue is GCaMP.

import os
import cv2
import ray
import matplotlib
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from skimage.morphology import square
from contextlib import redirect_stdout
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
ch4 = tifs[3]
    
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
# SYN-1
ch2_fullpath = os.path.join(tif_dir, ch2)
og_green = cv2.imread(ch2_fullpath, cv2.IMREAD_UNCHANGED)
og_green = og_green[:og_H,-og_W:]
padded_version[1:end_h, 1:end_w] = og_green
og_green = padded_version.copy()
green = og_green.copy()
# GCaMP
ch3_fullpath = os.path.join(tif_dir, ch3)
og_blue = cv2.imread(ch3_fullpath, cv2.IMREAD_UNCHANGED)
og_blue = og_blue[:og_H,-og_W:]
padded_version[1:end_h, 1:end_w] = og_blue
og_blue = padded_version.copy()
blue = og_blue.copy()
# DAPI
ch4_fullpath = os.path.join(tif_dir, ch4)
og_dapi = cv2.imread(ch4_fullpath, cv2.IMREAD_UNCHANGED)
og_dapi = og_dapi[:og_H,-og_W:]
padded_version[1:end_h, 1:end_w] = og_dapi
og_dapi = padded_version.copy()
dapi = og_dapi.copy()
del padded_version

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
'''
# for refining clusters that were found on the edge of a main_block
for h in range(0, H, block_size):
    for w in range(0, W, block_size):
        block = binary_thresh[h:h+block_size, 
                              w:w+block_size].copy()
        # skip any that are blank (i.e. pure black, not signal)
        if block.sum() == 0: continue
        clusters = block_worker(block, clusters, h, w, size_thresh)
'''

gain = 3
norm_red = og_red / og_red.max()
norm_green = og_green / og_green.max()
norm_blue = og_blue / og_blue.max()
norm_dapi = og_dapi / og_dapi.max()
original = np.dstack([norm_red, norm_green, norm_blue, norm_dapi])
del norm_red, norm_green, norm_blue, norm_dapi
clusters_removed = original.copy()
only_clusters = np.zeros_like(original)
viz_clusters = np.zeros_like(binary_thresh)
colors = [x / 8 for x in range(1, 9)]

# print(f"\n{len(clusters)} clusters unfiltered!\n")
edge_masks = []
i = -1 # allows incrementing at the beginning of loop
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

'''
def create_rmap(ch1, ch2):
    # create correlation map
    R_map = np.zeros((121,121))
    # this channel does not move
    # this channel can have values equal to 0
    _static_ch = ch1[60:-61, 60:-61]
    for i in range(121):
        for j in range(121):
            # this channel should not have 0s
            _slide_ch = ch2[i:i-121, j:j-121]
            # make sure we're not including any 0s from the blank crop margins
            slide_ch = _slide_ch[np.logical_and(_static_ch > 0, _slide_ch > 0)]
            static_ch = _static_ch[np.logical_and(_static_ch > 0, _slide_ch > 0)]
            R1 = np.corrcoef(static_ch.flatten(), slide_ch.flatten())
            R_map[i,j] = R1[0,1]
            print(f'R_map[{i},{j}]: {R1[0,1]}')
  
    return R_map
'''

# for condition in ["only", "without", "all"]:
for condition in ["all"]:
    
    if condition == "without":
        img = clusters_removed
    elif condition == "only":
        img = only_clusters
    elif condition == "all":
        img = original
    else:
        assert False, "illegal case"
        
    corr_file = os.path.join(results_dir, f"{name_root}_{condition}_corr.txt")
        
    _red = img[:,:,0]
    flat_red = _red[_red > 0].flatten()
    _green = img[:,:,1]
    flat_green = _green[_red > 0].flatten()
    _blue = img[:,:,2]
    flat_blue = _blue[_red > 0].flatten()
    _dapi = img[:,:,3]
    flat_dapi = _dapi[_red > 0].flatten()
    R1 = np.corrcoef([flat_red, flat_green, flat_blue, flat_dapi])
    
    # correlation including only detected clusters
    with open(corr_file, 'w') as cf:
        # main diagonal should be all 1s. Weird 0.9999 values sometimes
        assert R1[0,0] + R1[1,1] + R1[2,2] + R1[3,3] > 3.98
        # RG, RB, RD, GB, GD, BD correlations
        cf.write(f"{R1[0,1]}\n{R1[0,2]}\n{R1[0,3]}\n{R1[1,2]}\n{R1[1,3]}\n{R1[2,3]}\n")
    
    pairwise_chs = []
    pairwise_chs.append([_red,   original[:,:,0], 'RR']) # PSD95 - PSD95
    pairwise_chs.append([_red,   original[:,:,1], 'RG'])
    pairwise_chs.append([_red,   original[:,:,2], 'RB'])
    pairwise_chs.append([_red,   original[:,:,3], 'RD'])
    pairwise_chs.append([_green, original[:,:,1], 'GG']) # SYN1 - SYN1
    pairwise_chs.append([_green, original[:,:,2], 'GB'])
    pairwise_chs.append([_green, original[:,:,3], 'GD'])
    pairwise_chs.append([_blue,  original[:,:,2], 'BB']) # GCaMP - GCaMP
    pairwise_chs.append([_blue,  original[:,:,3], 'BD'])
    pairwise_chs.append([_dapi,  original[:,:,3], 'DD']) # DAPI - DAPI
    
    NUM_WORKERS = 3 * mp.cpu_count() // 4 # i.e. 75%
    ray.shutdown()
    ray.init(num_cpus=NUM_WORKERS)
    for ch1, ch2, name in pairwise_chs:
        for n_offsets in [4, 8, 16, 32]:
            stride = 16
            R_map_dim = (2*n_offsets+1)
            R_map = np.zeros((R_map_dim, R_map_dim))
            with open('info.txt', 'w') as f:
                with redirect_stdout(f):
                    print(f"Calculating: {name}")
                    print(f"stride: {stride}, n_offset: {n_offsets}")
                    print("n_offsets is how many strides to take away from the unshifted image.")
                    print("This will be done for each direction (i.e. up, down, left, right).")
                    total_shift = stride * (2 * n_offsets) + 1
                    print(f"Shifted area in pixels: {total_shift} x {total_shift}")
                    print('"Shifted area" is the bounding box that encompasses all pairs of offsets')
                    micron_per_pixel = 0.05 # IMAGE SPECIFIC!!! i.e. this is HARD-CODED!
                    total_len = total_shift * micron_per_pixel
                    print(f"That area is {total_len:.2f} x {total_len} microns:.2f")
                    print(f"given that this specific image has {micron_per_pixel} microns pixel side lengths.")
                    print("Note that the usable width of the cropped image is ~70 microns, so going too far")
                    print("beyond that with shifted area is not recommended.")
                    print("========")
                    
            @ray.remote
            def rmap_worker(_ch1, _ch2, i, n_offsets, stride=1):
                # R_map ends up being (2 * n_offset + 1)**2 pixels
                R_map_portion = np.zeros(((2*n_offsets)+1))
                # this channel does not move
                # this channel can have values equal to 0
                _static_ch = _ch1[n_offsets*stride:(-1*n_offsets-1)*stride, n_offsets*stride:(-1*n_offsets-1)*stride]
                for j in range(0, (2*n_offsets+1)*stride, stride):
                    # this channel should not have 0s
                    _slide_ch = _ch2[i*stride:i*stride-(2*n_offsets+1)*stride, j:j-(2*n_offsets+1)*stride]
                    # make sure we're not including any 0s from the blank crop margins
                    slide_ch = _slide_ch[np.logical_and(_static_ch > 0, _slide_ch > 0)]
                    static_ch = _static_ch[np.logical_and(_static_ch > 0, _slide_ch > 0)]
                    R1 = np.corrcoef(static_ch.flatten(), slide_ch.flatten())
                    R_map_portion[j//stride] = R1[0,1]
                    print(f'R_map[{i},{j//stride}]: {R1[0,1]}')
                return (i, R_map_portion)
            
            ray_ch1  = ray.put(ch1)
            ray_ch2  = ray.put(ch2)
            futures = []
            for i in range(0, (2*n_offsets+1)*stride, stride):
                futures.append(rmap_worker.remote(ray_ch1, ray_ch2, i//stride, n_offsets, stride))
            results = ray.get(futures)
            for i, R_map_portion in results:
                R_map[i,:] = R_map_portion 
        
            fig = plt.figure()
            cmap = matplotlib.colormaps['jet']
            
            _R_map = plt.imshow(R_map, cmap=cmap, interpolation='none', vmax=1, vmin=-0.2)
            plt.colorbar(_R_map)
            
            R_map_path = os.path.join(results_dir, f'{condition}_{name}_{stride}stride_{n_offsets}offsets_Rmap')
            np.save(R_map_path + ".npy", R_map)
            plt.savefig(R_map_path + ".png", dpi=300)
            plt.close(fig)
            ray.shutdown()
