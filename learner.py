#%%
import math
import numpy as np
import pandas as pd
import learner.prediction_strength as ps
import learner.export as le
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#%%
training_csv = './training.csv'

df = pd.read_csv(training_csv, header=None)
X = df.copy()
X.head()
X = X.dropna(axis=0, how='any')
y = X.iloc[:, len(X.columns) - 1]
X = X.drop(X.columns[-1], axis=1)
voters = X.iloc[:,0:len(X.columns)].sum(axis=1)
X = X.div(y, axis=0)
X.insert(loc=len(X.columns), column=int(len(X.columns)), value=(y-voters) / y)
X.head()

#%%
K = 7
clusters = range(1, K + 1)
wss_list = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=73, n_init=20)
    model.fit(X)
    wss_list.append(model.inertia_)

# plotting
_, ax = plt.subplots()
ax.plot(clusters, wss_list, '-o', color='black')
ax.set(title='Elbow plot', 
       xlabel='number of clusters', 
       ylabel='WSS');

#%%
results = ps.prediction_strength_of_clusters(X, K)
threshold = 0.70
ry = list(map(lambda x : x[1], results))
_, ax = plt.subplots()
ax.plot(clusters, ry, '-o', color='black')
ax.axhline(y=threshold, c='red');
ax.set(title='Determining the optimal number of clusters', 
       xlabel='number of clusters', 
       ylabel='prediction strength');

#%%        
k_optimal = -math.inf
s_optimal = -math.inf
centroids = []

for k, s, c in results:
    if s > threshold:
        k_optimal, s_optimal, centroids = k, s, c

le.export(k_optimal, centroids, './param.out')

# %%
# reduce the data in order to see 
if centroids is None:
    print("No centroids defined")

PCA_model = PCA(n_components=2).fit(X.to_numpy())

labels = KMeans(n_clusters=k_optimal, random_state=73, n_init=20).fit(X.to_numpy()).labels_

# Function to plot current state of the algorithm.
# For visualisation purposes, only the first two PC are shown.
PC = PCA_model.transform(X.to_numpy())
C2 = PCA_model.transform(centroids)

ax = plt.scatter(PC[:,0], PC[:,1], c=labels, alpha=0.5)
ax = plt.scatter(C2[:,0], C2[:,1], c='#82cfff', s=100, edgecolors = 'black')
ax = plt.title("Clustering representation")
ax = plt.show()
ax = plt.clf()
# %%
