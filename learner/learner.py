import math
import pandas as pd
import sklearn.cluster as sklc 
import from kmeans_util as ku
import from export_util as eu

training_csv = '../data/training.csv'

X = pd.read_csv(training_csv)
X_train, X_test = train_test_split(X, test_size = .15, random_state = 73)
ss = []

for n in range(1, 11):
    train_model = sklc.KMeans(n_clusters = n, random_state = 73).fit(X_train)
    test_model = sklc.KMeans(n_clusters = n, random_state = 73).fit(X_test)
    train_centroids = train_model.cluster_centers_
    s = ku.prediction_strength(train_centroids, test_model.to_numpy(), test_model.lables_)
    ss.append((n, s, train_centroids))

k_optimal = -math.inf
s_optimal = -math.inf
centroids = []

for k, s, c in ss:
    if s >= s_optimal:
        k_optimal, s_optimal, centroids = s, k, c

eu.export(k_optimal, centroids, '../data/param.out')

