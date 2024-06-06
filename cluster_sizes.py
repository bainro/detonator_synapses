import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print() # just looks better :)
    csv_dir = os.getcwd()
    all_files = os.listdir(csv_dir)
    csv_files = [f for f in all_files if f.endswith(".csv")]
    csv_files = sorted(csv_files)
    err_txt = f'{csv_dir} contains no csv files :('
    assert len(csv_files) > 0, err_txt
    
    EE_ids = ['7RL_', '12R_', '6L_', '13R_', '9R_']
    # for each experimental group's cluster sizes
    EE, CT = [], []
    for csv_f in csv_files:
        # skip over combined_results.csv
        if 'combined' in csv_f:
            continue

        fullpath = os.path.join(csv_dir, csv_f)
        assert os.path.isfile(fullpath), f'{fullpath} not found!'
        
        # load individual csv files
        clusters = []
        with open(fullpath, mode ='r') as file:
            csv_file = csv.reader(file)
            for i, line in enumerate(csv_file):
                # omit header & empty lines
                if i == 0 or line == []:
                    continue
                clusters.append(line)

        tmp_sizes = []
        # c = [cluster id, size, psd, syn]
        for i, c in enumerate(clusters):
            assert len(c) == 4, 'combined_results.csv changed names?'
            cluster_size = int(c[1])
            # print(f'cluster #{i} has a size of {cluster_size} pixels.')
            tmp_sizes.append(cluster_size)
            
        is_EE = False
        for EE_id in EE_ids:
            if EE_id in csv_f:
                is_EE = True
                
        if is_EE:
            EE += tmp_sizes
        else:
            CT += tmp_sizes
    
    
    # below 10,000
    b = 10
    # plt.xscale('log', base=b) 
    # plt.yscale('log', base=b) 
    plt.hist(CT, label='CT', bins=range(155, 10000, 10))
    plt.hist(EE, label='EE', bins=range(155, 10000, 10))
    plt.title("EE vs CT Cluster Size Distribution")
    plt.xlabel("Cluster Size")
    plt.ylabel("Counts")
    xticks = [155, 500, 1000, 2000, 4000, 8000]
    plt.xticks(xticks, xticks)
    plt.legend()
    
    import statistics
    EE_mean = statistics.mean(EE)
    CT_mean = statistics.mean(CT)
    print(EE_mean, CT_mean)
    print(abs(EE_mean - CT_mean))
    plt.axvline(x=EE_mean, color='orange', linewidth=0.5)
    plt.axvline(x=EE_mean, color='blue', linewidth=0.5)
    
    plt.show()

    assert False, ":("

    '''
    b = 10
    ax[0].set_xscale('log', base=b) 
    ax[0].set_yscale('log', base=b)
    ax[1].set_xscale('log', base=b) 
    ax[1].set_yscale('log', base=b)
    ax[0].hist(CT, label='CT', color='blue',   bins=range(155, 10000, 100))
    ax[1].hist(EE, label='EE', color='orange', bins=range(155, 10000, 100))
    ax[0].set_title("EE vs CT Cluster Size Distribution")
    ax[1].set_xlabel("Cluster Size")
    ax[0].set_ylabel("Counts")
    ax[1].set_ylabel("Counts")
    xticks = [155, 500, 1000, 2000, 4000, 8000]
    ax[0].set_xticks(xticks, xticks)
    ax[1].set_xticks(xticks, xticks)
    ax[0].legend()
    ax[1].legend()
    plt.show()
    '''

    # above 10,000
    plt.figure()
    #plt.xscale('log', base=b) 
    #plt.yscale('log', base=b) 
    CT = [val for val in CT if val > 10000]
    EE = [val for val in EE if val > 10000]
    plt.hist(CT, label='CT', bins=range(10000, 30000, 500))
    plt.hist(EE, label='EE', bins=range(10000, 30000, 500))
    plt.title("EE vs CT Cluster Size Distribution")
    plt.xlabel("Cluster Size")
    plt.ylabel("Counts")
    xticks = [155, 500, 1000, 2000, 4000, 8000]
    #plt.xticks(xticks, xticks)
    plt.legend()
    plt.show()
    
    print(f'Number of EE clusters: {len(EE)}')
    print(f'Number of CT clusters: {len(CT)}')
    print(f'Largest EE cluster: {max(EE)}')
    print(f'Largest CT cluster: {max(CT)}')
    print('\nDONE!')
