#%%
import math
import pandas as pd
import learner.compute as lc
import learner.export as le
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

training_csv = './training.csv'

X = pd.read_csv(training_csv, header=None)
X = X.dropna(axis=0, how='any')

pob = X[X.columns[-1]]
X = X.drop(X.columns[-1], axis=1)
X = X.div(pob, axis=0)
#%%
K = 7
clusters = range(1, K + 1)
wss_list = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=73, n_init='auto')
    model.fit(X)
    wss_list.append(model.inertia_)

# plotting
_, ax = plt.subplots()
ax.plot(clusters, wss_list, '-o', color='black')
ax.set(title='Elbow plot', 
       xlabel='number of clusters', 
       ylabel='WSS');

#%%
results = lc.prediction_strength_of_clusters(X, K)
# %%
threshold = 0.8
ps = list(map(lambda x : x[1], results))
_, ax = plt.subplots()
ax.plot(clusters, ps, '-o', color='black')
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
