---
> # Learner and predictor for a clustering task - Unsupervised M.L
>
> 💻 Authors: Wilber B. Quito, Andrea Ramirez
>
> 🗓️ Date: 11/12/2022
>
> ✍🏼 Machine Learning - Master in Data Science - *Universitat de Girona*

___

# Clustering learner with K-means

The aim of this project is to group samples linked by an optimal number of clusters using a clustering algorithm ⚙️.

We decided to use the K-means as clustering algorith because one of its adventages is of scaling to large data sets, and it can easily adapt to new examples. Since one disadventage of using this algorithm is that we must know *k*. There are some options to find the optimal number of clusters with K-means, for example GAP. However, to evaluate the number of clusters we've used the Prediction Strength algorithm to find the *optimal number of centroids*, hence the **optimal number of clusters (k)**. 

![Clustering](./img/portada.png)

## Implementation

The implementation is made in two different scripts. We have the scripts *learner.py* and *preditor.R* ✍🏼.

### learner.py

Since we decided to use the K-means algorithm ⚙️, we import it from the *sklearn.cluster* library, along with other libraries to transform and explore the data.

The script expects to find a file named *training.csv* 💽 which should have the samples to clusterize. 

```
training_csv = './training.csv'
```

<details>
	<summary>Part of the <i>training.csv</i></summary>

```
20133,33265,26126,11062,135762
10823,43384,23416,5508,144598
16662,20697,15018,5846,94977
11403,19607,16463,6698,78397
1527,4241,4764,1124,18294
17829,17880,16174,8980,92602
2459,7833,5932,1202,27599
1745,755,676,1036,5696
31335,23294,22324,11377,119782
6406,11948,14016,3816,54139
14622,24403,16031,6791,87705
21047,28386,18187,9282,110165
17829,17880,16174,8980,83626
1981,7543,4008,1159,21065
1805,4011,3073,1117,14081
2824,8172,6667,1512,29064
1981,7543,4008,1159,21747
10823,43384,23416,5508,119165
6155,8207,13180,3669,48472
2441,11005,7111,1584,35514
```

</details>

Before finding the optimal number of clusters, our script reads the data set as a *data frame* 🗂️; and drops all the rows that contain N/A. 

Taking into consideration that our training data comes from the **voters' turnout** 📮, we decided to transform the total population into all the people who didn't vote. Then normalize the data to standardize it, to reduce data redundancy, and improve and protect the model's integrity. 

```
y = X.iloc[:, len(X.columns) - 1]
X = X.drop(X.columns[-1], axis=1)
voters = X.iloc[:,0:len(X.columns)].sum(axis=1)
X = X.div(y, axis=0)
X.insert(loc=len(X.columns), column=int(len(X.columns)), value=(y-voters) / y)
X.head()
```

The implementation tries to find the *optimal number of clusters* ⚙️ between 1 and 7 included. 

```
K = 7
clusters = range(1, K + 1)
wss_list = []

for k in clusters:
    model = KMeans(n_clusters=k, random_state=73)
    model.fit(X)
    wss_list.append(model.inertia_)
```

To have a first look of the *clusters* in our data set, we perform an **Elbow plot** 📊:

```
# plotting
_, ax = plt.subplots()
ax.plot(clusters, wss_list, '-o', color='black')
ax.set(title='Elbow plot', 
       xlabel='number of clusters', 
       ylabel='WSS');
 ```
 
![Elbow plot](./img/elbow_plot.png)

From the Elbow plot, we suspect that the ideal number of clusters for the training sample is 2 or 3. 
 
We had not used any library to compute the Prediction Strength. Instead, we implemented the algorithm from scratch ⚙️ by using the following equation. The implementation is in the file *compute.py* in the *learner* module. We wanted to use the recommended threshold between 0.8 and 0.9. However, with the training data that we were using, it could only determine *1 optimal cluster*; therefore, we lowered the threshold to 0.70.

![Prediction Strength](./img/ps-equation.png)

```
results = ps.prediction_strength_of_clusters(X, K)
threshold = 0.70
ry = list(map(lambda x : x[1], results))
_, ax = plt.subplots()
ax.plot(clusters, ry, '-o', color='black')
ax.axhline(y=threshold, c='red');
ax.set(title='Determining the optimal number of clusters', 
       xlabel='number of clusters', 
       ylabel='prediction strength');
```

![Optimal number of clusters](./img/optimal_number.png)

Once the algorithm has found the **optimal number** ⚙️, it exports the number of centroids found in the *training.csv* distribution and its centroids into a file named *param.out* 💾. The exported centroids are normalized on a scale from 0 to 1.

```
k_optimal = -math.inf
s_optimal = -math.inf
centroids = []

for k, s, c in results:
    if s > threshold:
        k_optimal, s_optimal, centroids = k, s, c

le.export(k_optimal, centroids, './param.out')
```

💾 *param.out*

```
3
0.14,0.24,0.21,0.08,0.33
0.24,0.16,0.16,0.12,0.32
0.08,0.32,0.22,0.05,0.32
```

Still, to verify that our model works, we plot the clusters. 📊

```
if centroids is None:
    print("No centroids defined")

PCA_model = PCA(n_components=2).fit(X.to_numpy())

labels = KMeans(n_clusters=k_optimal, random_state=73).fit(X.to_numpy()).labels_

# Function to plot current state of the algorithm.
# For visualisation purposes, only the first two PC are shown.
PC = PCA_model.transform(X.to_numpy())
C2 = PCA_model.transform(centroids)

ax = plt.scatter(PC[:,0], PC[:,1], c=labels, alpha=0.5)
ax = plt.scatter(C2[:,0], C2[:,1], c='#82cfff', s=100, edgecolors = 'black')
ax = plt.title("title")
ax = plt.show()
ax = plt.clf()
```

![Cluster using k-means](./img/k-means.png)

### predictor.R

The script gets the output of *learner.py*, picks the normalized centroids ⚙️, and reads the file *testing.csv* 💽 which applies normalization and assigns each sample of the testing into a cluster by computing the minimum Euclidean distance between each sample and the centroids. 

```
centroids <- read.table(params.filename, header=F, sep=",", dec=".", skip=1)
n <- read.table(params.filename, header=F, sep=",", dec=".", nrows=1)[1,1]

if (n != nrow(centroids)) {
  stop('The number of centroids are grown, check the file: ', params.filename)
}

euclidean <- function(a, b) sqrt(sum((a - b)^2))

closest_centroid <- function (x, centroids) {
  K = nrow(centroids)
  i.min = sapply(1:K, function(i) {
    x <- euclidean(x, centroids[i,])
    x
  }) |> which.min()
  i.min
}

df <- read.csv(testing.filename, header=F, sep=',')
X <- df
y <- X[ncol(X)]
voters <- rowSums(X[,1:ncol(X)-1])
X[,ncol(X)] <- y - voters
X <- X / y[,1]
```

Finally, it exports the clusterization into a file named *clustering.out* 💾 where each *i* row of this file is the clustering assignation of the *i* sample of *testing.csv*.

```
xs <- apply(X, 1, function(x) closest_centroid(x, centroids))
xs <- list(xs)
data.table::fwrite(xs, clustering.filename)
```

💽 *testing.csv* 

```
21047,28386,18187,9282,118598
11403,19607,16463,6698,78341
24335,30318,28390,12302,138566
1745,755,676,1036,5810
890,3347,2538,502,11271
1543,9718,4477,785,22955
6406,11948,14016,3816,51474
2459,7833,5932,1202,26121
10823,43384,23416,5508,118923
14622,24403,16031,6791,87753
23632,15432,18256,12063,95067
3342,16252,7773,1980,47424
23632,15432,18256,12063,100631
17829,17880,16174,8980,92372
1527,4241,4764,1124,18005
5172,8083,6566,2517,38056
17785,9106,10677,8275,68889
18971,39516,21040,8887,131706
1527,4241,4764,1124,16632
```

💾 *clustering.out*

```
1
1
1
2
3
3
1
3
3
1
2
3
2
2
1
1
2
1
1
1
```
