import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

max_k = 40

if __name__ == "__main__":
    print() # just looks better :)
    csv_dir = os.getcwd()
    all_files = os.listdir(csv_dir)
    csv_files = [f for f in all_files if f.endswith(".csv")]
    csv_files = sorted(csv_files)
    err_txt = f'{csv_dir} contains no csv files :('
    assert len(csv_files) > 0, err_txt
    
    EE_ids = ['7RL_', '12R_', '6L_', '13R_', '9R_']
    # for each experimental group's kNN avg distances
    EE, CT = [], []
    for csv_f in csv_files:
        # skip over combined_results.csv
        if 'combined' in csv_f:
            continue

        fullpath = os.path.join(csv_dir, csv_f)
        assert os.path.isfile(fullpath), f'{fullpath} not found!'
        
        # load individual csv files
        centers = []
        with open(fullpath, mode ='r') as file:
            csv_file = csv.reader(file)
            for i, line in enumerate(csv_file):
                # omit header & empty lines
                if i == 0 or line == []:
                    continue
                c_x, c_y = float(line[4]), float(line[5])
                cluster_size = int(line[1])
                n_subclusters = cluster_size // 200
                for _ in range(n_subclusters):
                    centers.append([c_x, c_y])
            
        mean_distances = []
        k_s = range(1, max_k)
        for k in k_s:
            kgraph = kneighbors_graph(centers, k, mode='distance', n_jobs=4)
            kgraph = kgraph.toarray()
            mean_distances.append(kgraph.sum() / np.count_nonzero(kgraph))
        
        is_EE = False
        for EE_id in EE_ids:
            if EE_id in csv_f:
                is_EE = True
                
        if is_EE:
            EE.append(mean_distances)
        else:
            CT.append(mean_distances)
    
    n_EE, n_CT = len(EE), len(CT)
    EE = np.array(EE).sum(axis=0)
    CT = np.array(CT).sum(axis=0)
    EE = np.divide(EE, n_EE)
    CT = np.divide(CT, n_CT)
    
    plt.figure()
    plt.plot(EE, label='EE')
    plt.plot(CT, label='CT')
    xticks = list(range(0, max_k, 4))
    plt.xticks(xticks, xticks)
    plt.legend()
    plt.show()
    