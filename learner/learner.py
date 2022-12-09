import math
import pandas as pd
import sklearn.cluster as sklc 
import kmeans_util as ku
import export_util as eu

training_csv = '../data/training.csv'

X = pd.read_csv(training_csv, header=None)
X = X.dropna(axis=0, how='any')
ss = []

for n in range(1, 11):
    try:
        X_train, X_test = ku.train_test_split(n, X)
        train_model = sklc.KMeans(n_clusters=n, random_state=73, n_init='auto').fit(X_train)
        test_model = sklc.KMeans(n_clusters=n, random_state=73, n_init='auto').fit(X_test)
        train_centroids = train_model.cluster_centers_
        s = ku.prediction_strength(n, train_centroids, X_test.to_numpy(), test_model.labels_)
        ss.append((n, s, train_centroids))
    except Exception as e:
        print('Maybe too few samples...', e)
        
k_optimal, s_optimal, centroids = -math.inf, -math.inf, []

for k, s, c in ss:
    if s >= s_optimal:
        k_optimal, s_optimal, centroids = k, s, c

eu.export(k_optimal, centroids, '../data/param.out')

