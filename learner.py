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
        X_train, X_valid = lc.train_test_split(k, X)
        train_model = KMeans(n_clusters=k, random_state=73, n_init='auto').fit(X_train)
        valid_model = KMeans(n_clusters=k, random_state=73, n_init='auto').fit(X_valid)
        C_train, C_valid = train_model.predict(X_valid), valid_model.labels_
        train_centroids = train_model.cluster_centers_
        ps = lc.classic_prediction_strength(k, C_train, C_valid)
        ss.append((k, ps, train_centroids))
    except Exception as e:
        print('Unexpected error...', e)
#%%
pss = list(map(lambda s : s[1], ss))
pss
#%%        
k_optimal, s_optimal = -math.inf, -math.inf
centroids, threshold = [], 0.8

for k, s, c in ss:
    if s > threshold:
        k_optimal, s_optimal, centroids = k, s, c

le.export(k_optimal, centroids, './param.out')
# %%
