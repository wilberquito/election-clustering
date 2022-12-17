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

The aim of this project is to group samples linked by an optimal number of clusters using a clustering algorithm.

We decided to use the K-means as clustering algorith because one of its adventages is of scaling to large data sets, and can easily adapt to new examles. Since one disadventage of using this algorithm is that we must know *k*. There are some options to find the optimal number of clusters with K-means, for example GAP. However, to evaluate the number of clusters we've used the Prediction Strength algorithm to find the *optimal number of centroids*, hence the **optimal number of clusters (k)**. 

![Clustering](./img/portada.png)

## Implementation

The implementation is made in two different scripts. We have the scripts *learner.py* and *preditor.R*.

### learner.py

Since we decided to use the K-means algorithm, we import it from the *sklearn.cluster* library, along with pandas, numpy, and math libraries.

The script expects to find a file named *training.csv* which should have the samples to clusterize. 
```
training_csv = './training.csv'
```
The implementation tries to find the optimal number of cluster between 1 and 7 included. Once the algorithm has found the optimal number, it exports the number of centroids found in the *training.csv* distribution and it's centroids into a file named *param.out*.  

We had not used any library to compute the Prediction Strength, instead, we implemented from scratch the algorithm using the following equation. The implementation is in the file *compute.py* in the *learner* module.

![Prediction Strength](./img/ps-equation.png)

### predictor.R

The script get's the output of *learner.py*, picks the centroids and reads the file *testing.csv* and assign each sample of the testing into a cluster by computing the minimun Euclidean distance between each sample and the centroids. Finally, exports the clusterization into a file named *clustering.out* where each *i* column of this file is the clustering assignation of the *i* sample of *testing.csv*.
