import os
import matplotlib.pyplot as plt


all_files = os.listdir(os.getcwd())
corr_files = [f for f in all_files if f.endswith("_corr.txt")]
err_txt = 'Your current directory contains no *_corr.txt files :('
assert len(corr_files) > 0, err_txt

#results_dir = os.path.join(tif_dir, 'results', timestamp)
# os.makedirs(results_dir, exist_ok=True)

clusters_Rs = []
# clusters removed
no_clusters_Rs = []
for cf in corr_files:
    with open(cf) as f:
        lines = [float(line.rstrip()) for line in f]
        no_clusters_Rs.append(lines[0])
        clusters_Rs.append(lines[1])

f = plt.figure() # figsize=(8,6)
n_bins = 20
plt.hist(no_clusters_Rs, alpha=0.5, label="clusters removed")
plt.hist(clusters_Rs, alpha=0.5, label="only clusters")
no_avg = sum(no_clusters_Rs) / len(no_clusters_Rs)
only_avg = sum(clusters_Rs) / len(clusters_Rs)
print(f'avg R without clusters: {no_avg:.2f}')
print(f'avg R with only clusters: {only_avg:.2f}')
plt.axvline(x=no_avg, linestyle='--')
plt.axvline(x=only_avg, color='orange', linestyle='--')
plt.xlabel("Pearson Correlation Coefficients") # size=14
plt.ylabel("Count") # size=14
axes = f.gca()
axes.yaxis.get_major_locator().set_params(integer=True)
plt.title("\nPSD95 and SYN-1 Cross-Correlations\nWithout Clusters and With Only Clusters\n")
plt.legend() # loc='upper right')
plt.savefig("R_histograms.png", bbox_inches='tight', dpi=300)
