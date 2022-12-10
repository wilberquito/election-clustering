
params.filename <- '../data/param.out'
testing.filename <- '../data/testing.csv'
clustering.filename <- '../clustering.out'

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
    print(x)
    x
  }) |> which.min()
  print('---')
  i.min
}

testing.df <- read.csv(testing.filename, header=F, sep=',')

clusterization <- sapply(testing.df, closest_centroid, centroids=centroids)
clusterization
