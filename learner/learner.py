#%%
import math
import pandas as pd
import kmeans_util as ku
import export_util as eu
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

training_csv = '../data/training.csv'

X = pd.read_csv(training_csv, header=None)
X = X.dropna(axis=0, how='any')

#%%
clusters = range(1, 8)
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
ss = []

for k in clusters:
    try:
        X_train, X_test = ku.train_test_split(k, X)
        train_model = KMeans(n_clusters=k, random_state=73, n_init='auto').fit(X_train)
        test_model = KMeans(n_clusters=k, random_state=73, n_init='auto').fit(X_test)
        train_centroids = train_model.cluster_centers_
        ps = ku.prediction_strength(k, train_centroids, X_test.to_numpy(), test_model.labels_)
        ss.append((k, ps, train_centroids))
    except Exception as e:
        print('Maybe too few samples...', e)
#%%
pss = list(map(lambda s : s[1], ss))
pss
#%%        
k_optimal, s_optimal, centroids = -math.inf, -math.inf, []

for k, s, c in ss:
    if s >= s_optimal:
        k_optimal, s_optimal, centroids = k, s, c

eu.export(k_optimal, centroids, '../data/param.out')

