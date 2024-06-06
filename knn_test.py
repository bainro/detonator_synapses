import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

n = 2 # number of tight and loose images
max_k = 40 # we run variable size neighborhoods for kNN
n_pts = 50 # number of points per cluster type
img_size = 200 # image's side length
tight, loose, both = [], [], []
for i in range(2):
    for j in range(n):
        img = np.zeros((img_size, img_size))
        if i == 0:
            stdev = 5
        else:
            stdev = 30
        # sample from 2 gaussians for x and y
        # a value of 0 means the center of the image.
        x_s = np.random.normal(loc=0, scale=stdev, size=n_pts) 
        y_s = np.random.normal(loc=0, scale=stdev, size=n_pts)
        x_s = np.round(x_s + (img_size // 2)).astype(int)
        y_s = np.round(y_s + (img_size // 2)).astype(int)
        centers = [[x_s[_], y_s[_]] for _ in range(len(x_s))]
        # draw all the points on the image
        for x, y in centers:
            img[x, y] = 1
        
        plt.figure()
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.show() # display the image
        plt.savefig(f'{int(i * n + j)}.png')
        
        # running the variable sized kNN neighborhoods
        mean_distances = []
        k_s = range(1, max_k)
        for k in k_s:
            kgraph = kneighbors_graph(centers, k, mode='distance', n_jobs=4)
            kgraph = kgraph.toarray()
            mean_distances.append(kgraph.sum() / np.count_nonzero(kgraph))
                
        if i == 0:
            tight.append(mean_distances)
        else:
            loose.append(mean_distances)

# draw combined images with n_pts each of loose & tight clustering
for j in range(n):
    img = np.zeros((img_size, img_size))
    
    stdev = 5
    x_s = np.random.normal(loc=0, scale=stdev, size=n_pts) 
    y_s = np.random.normal(loc=0, scale=stdev, size=n_pts)
    x_s = np.round(x_s + (img_size // 2)).astype(int)
    y_s = np.round(y_s + (img_size // 2)).astype(int)
    centers = [[x_s[_], y_s[_]] for _ in range(len(x_s))]
    # draw all the points on the image
    for x, y in centers:
        img[x, y] = 1
    
    stdev = 30
    x_s = np.random.normal(loc=0, scale=stdev, size=n_pts) 
    y_s = np.random.normal(loc=0, scale=stdev, size=n_pts)
    x_s = np.round(x_s + (img_size // 2)).astype(int)
    y_s = np.round(y_s + (img_size // 2)).astype(int)
    centers += [[x_s[_], y_s[_]] for _ in range(len(x_s))]
    # draw all the points on the image
    for x, y in centers:
        img[x, y] = 1
    
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.show() # display the image
    plt.savefig(f'both_{int(j)}.png')
    
    # running the variable sized kNN neighborhoods
    mean_distances = []
    k_s = range(1, max_k)
    for k in k_s:
        kgraph = kneighbors_graph(centers, k, mode='distance', n_jobs=4)
        kgraph = kgraph.toarray()
        mean_distances.append(kgraph.sum() / np.count_nonzero(kgraph))
    both.append(mean_distances)

tight = np.array(tight).sum(axis=0)
loose = np.array(loose).sum(axis=0)
both = np.array(both).sum(axis=0)
tight = np.divide(tight, n)
loose = np.divide(loose, n)
both = np.divide(both, n)

plt.figure()
plt.plot(tight, label='tight')
plt.plot(loose, label='loose')
plt.plot(both, label='both')
xticks = list(range(0, max_k, 4))
plt.xticks(xticks, xticks)
plt.legend()
plt.show()
plt.savefig("results.png")
    