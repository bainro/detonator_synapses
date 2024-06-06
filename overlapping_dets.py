import os
import csv
import cv2
import time
import datetime
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from skimage.morphology import square
from skimage.segmentation import flood

tif_dir = os.getcwd()

# simple error callback for debugging processes
def ecb(e):
    assert False, print(e)

def combine_CSVs(results_dir):
    # results_dir should be 2 directories below the TIF images
    tif_dir = os.path.join(results_dir, os.pardir, os.pardir)
    tif_dir = os.path.abspath(tif_dir)
    all_files = os.listdir(tif_dir)
    tifs = [f for f in all_files if f.endswith(".tif")]
    tifs = sorted(tifs)
    err_txt = f'{tif_dir} contains no TIF images :('
    assert len(tifs) > 0, err_txt
    
    rows = []
    for ch1, ch2 in zip(tifs[::2], tifs[1::2]):
        err_txt = "filename structure violated"
        assert ch1[-5] == '1' and ch2[-5] == '2', err_txt
        assert ch1[:-5] == ch2[:-5], err_txt

        # naming constraint: Begins w/ # of pyr cells followed by '_'
        char_after_soma_count = ch1.find("_") + 1
        name_root = ch1[char_after_soma_count:-7]
        img_csv = os.path.join(results_dir, f'{name_root}.csv')
        if not os.path.isfile(img_csv):
            print(f"{img_csv} not found! Skipping.")
            continue
        rows.append([name_root])    
        
        ch1_fullpath = os.path.join(tif_dir, ch1)
        ch2_fullpath = os.path.join(tif_dir, ch2)
        og_red = cv2.imread(ch1_fullpath, cv2.IMREAD_UNCHANGED)
        og_green = cv2.imread(ch2_fullpath, cv2.IMREAD_UNCHANGED)
        
        # load individual csv files
        clusters = []
        with open(img_csv, mode ='r') as file:
            csv_file = csv.reader(file)
            for i, line in enumerate(csv_file):
                # omit header & empty lines
                if i == 0 or line == []:
                    continue
                clusters.append(line)
        
        n_clusters = len(clusters)
        rows[-1].append(n_clusters)
        
        total_psd = 0
        total_syn = 0
        sizes = []
        # c = [cluster id, size, psd, syn]
        for i, c in enumerate(clusters):
            cluster_size = int(c[1])
            sizes.append(cluster_size)
            total_psd += float(c[2]) * cluster_size
            total_syn += float(c[3]) * cluster_size
        
        # Avg pixel values over all clusters in a given image
        total_size = sum(sizes)
        avg_psd = total_psd / total_size
        median_psd = line[4] # every line is the same
        avg_syn = total_syn / total_size
        median_syn = line[5] # every line is the same
        avg_size = total_size / n_clusters
        median_size = np.median(np.array(sizes))
        rows[-1].append(avg_size)
        rows[-1].append(median_size)
        rows[-1].append(avg_psd)
        rows[-1].append(median_psd)
        rows[-1].append(avg_syn)
        rows[-1].append(median_syn)
        
        pyr_cell_count = int(ch1[:ch1.find("_")])
        rows[-1].append(pyr_cell_count)
        
        # calculate actual non-blank space in image
        combined = np.multiply(og_red, og_green)
        combined[combined != 0] = 1
        img_area = combined.sum()
        rows[-1].append(img_area)
        
    # save master csv file
    combined_csv = os.path.join(results_dir, "combined_results.csv")
    with open(combined_csv, 'w') as main_csv_file:
        main_writer = csv.writer(main_csv_file)
        header = [
            'Name', '# Clusters', 'Mean Size', 'Median Size', 
            'Mean PSD', 'Median PSD', 'Mean SYN', 'Median SYN', 
            '# Cells', 'Entire curved area (pixels)'
        ]
        main_writer.writerow(header)
        main_writer.writerows(rows)

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

def img_worker(ch1, ch2, results_dir, r_threshold, g_threshold):
    # Separate red (PSD95) and green (synaptotagmin 1) channels.
    # Saves images in results_dir.
    
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
    H, W = og_red.shape
    # We add a single px of black around the whole perimeter
    # to avoid labeling true edge clusters as "edge" clusters
    # that result from being split acorss tiled blocks
    H += 2
    W += 2
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
    padded_version[1:end_h, 1:end_w] = og_green
    og_green = padded_version.copy()
    del padded_version
    green = og_green.copy()
    
    # for ch in [og_red, og_green]:
    #     print(f'\ndtype: {ch.dtype}, shape: {ch.shape}')
    #     print(f'min: {np.min(ch)}, max: {np.max(ch)}')
    
    red[og_red < r_threshold] = 0
    red[og_green < g_threshold] = 0
    green[og_green < g_threshold] = 0
    green[og_red < r_threshold] = 0
    red = red / red.max() # normalize
    green = green / green.max() # normalize
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
    original = np.dstack([norm_red, norm_green, np.zeros_like(og_green)])
    del norm_red, norm_green
    clusters_removed = original.copy()
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
            pts = np.argwhere(c_mask)
            c_x = pts[:,0].mean() + h_offset
            c_y = pts[:,1].mean() + w_offset
            # viz_clusters[int(c_x)-7:int(c_x)+8, int(c_y)-7:int(c_y)+8] = 1
            for h, w in pts:
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
            results.append([num_rows, size, avg_psd, avg_syn, c_x, c_y])
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
                    c_x = np.array(c_mask)[:,0].mean()
                    c_y = np.array(c_mask)[:,1].mean()
                    '''
                    try:
                        viz_clusters[int(c_x)-7:int(c_x)+8, int(c_y)-7:int(c_y)+8] = 1
                        pass
                    except:
                        print('probably an indexing error at the edge :)')
                    '''
                    cluster_psd = 0
                    cluster_syn = 0
                    for x,y in c_mask:
                        viz_clusters[x,y] = colors[num_rows % len(colors)]
                        clusters_removed[x,y] = 0
                        cluster_psd += og_red[x,y]
                        cluster_syn += og_green[x,y]
                    avg_psd = cluster_psd / size
                    avg_syn = cluster_syn / size
                    results.append([num_rows, size, avg_psd, avg_syn, c_x, c_y])
    del edge_masks
    
    median_psd = np.median(og_red[viz_clusters > 0])
    median_syn = np.median(og_green[viz_clusters > 0])
    # append the GLOBAL median PSD and SYN values to each row
    # not efficient, but convenient for this research code ;)
    for r in results:
        r.append(median_psd)
        r.append(median_syn)
        
    # write the csv file
    with open(csv_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        header = ['Cluster ID', 'Size', 'Mean PSD', 'Mean SYN', 'Median PSD', 'Median SYN']
        writer.writerow(header)
        writer.writerows(results)
    del results

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax[0,0].imshow(original * gain)
    ax[0,0].set_title('original')
    ax[1,0].imshow(binary_thresh, cmap='gray')
    ax[1,0].set_title('binary thresholding')
    cmap = plt.get_cmap('gist_rainbow')
    cmap.set_under('black')
    ax[1,1].imshow(viz_clusters, cmap=cmap, interpolation='none', vmin=0.1)
    ax[1,1].set_title('clusters')
    ax[0,1].imshow(clusters_removed * gain)
    ax[0,1].set_title('clusters removed')
    
    cluster_img_path = f'{name_root}.png'
    cluster_img_path = os.path.join(results_dir, cluster_img_path)
    plt.savefig(cluster_img_path, dpi=300)
    plt.close(fig) # prevent plotting huge figures inline
    
    viz_clusters[viz_clusters > 0] = 256
    path = os.path.join(results_dir, f'{name_root}_fullres_cluster_mask.png')
    cv2.imwrite(path, viz_clusters)

    return # end `def img_worker`


if __name__ == "__main__":
    all_files = os.listdir(tif_dir)
    tifs = [f for f in all_files if f.endswith(".tif")]
    tifs = sorted(tifs)
    err_txt = 'Your current directory contains no TIF images :('
    assert len(tifs) > 0, err_txt
    
    processing_start = time.time()
    
    try:
        mp.set_start_method('spawn')
    except:
        pass
    
    n_proc = 3
    img_pool = mp.Pool(n_proc)
    
    thresh_search = [[515, 350]]
    for r_threshold, g_threshold in thresh_search:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamp += f'_{r_threshold}_{g_threshold}'
        results_dir = os.path.join(tif_dir, 'results', timestamp)
        os.makedirs(results_dir, exist_ok=True)
        for ch1, ch2 in zip(tifs[::2], tifs[1::2]):
            err_txt = "filename structure violated"
            assert ch1[-5] == '1' and ch2[-5] == '2', err_txt
            assert ch1[:-5] == ch2[:-5], err_txt
            args = [ch1, ch2, results_dir, r_threshold, g_threshold]
            img_pool.apply_async(func=img_worker, args=args, error_callback=ecb)
    
    img_pool.close()
    img_pool.join()
    del img_pool
    
    # save master csv file
    combine_CSVs(results_dir)
    
    print(f'\nCompleted in {(time.time() - processing_start) / 60:.2f} minutes.')
    print(f'\nResults saved in {results_dir}')