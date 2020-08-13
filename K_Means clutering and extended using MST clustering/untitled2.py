from mst_clustering import MSTClustering
import matplotlib.pyplot as plt 
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans


def plot_mst(model,marker='s', cmap='rainbow'):
    X = model.X_fit_
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axi, full_graph, colors in zip(ax, [True, False], ['lightblue', model.labels_]):
        segments = model.get_graph_segments(full_graph=full_graph)
        axi.plot(segments[0], segments[1], '-ok', zorder=1, lw=1)
        axi.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, zorder=2)
        axi.axis('tight')
        
    
    ax[0].set_title('Full Minimum Spanning Tree', size=16)
    ax[1].set_title('Trimmed Minimum Spanning Tree', size=16);
    
    
# create some data
X, y = make_blobs(100, centers=5, cluster_std=0.90)
print(X)


# predict the labels with the MST algorithm
model = MSTClustering(cutoff_scale=1.5, approximate=True, n_neighbors=100)
labels = model.fit_predict(X)
counts = np.bincount(labels)
print("No. of clusters: ")
clusters=len(counts)
print(len(counts))
print("No. of elements in each Clusters: ")
print(counts)

# plot the results
plt.scatter(X[0:, 0], X[0:, 1],marker='o', c=labels, cmap='rainbow');
plt.show()
# plot the brief model
plot_mst(model)



wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

plt.show()

kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
counts = np.bincount(pred_y)
print("No. of clusters: ")
print(len(counts))
print("No. of elements in each Clusters: ")
print(counts)
plt.scatter(X[:,0], X[:,1],marker='o',c=pred_y, cmap='rainbow')
plt.show()
