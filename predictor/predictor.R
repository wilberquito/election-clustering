
params.filename = '../data/param.out'
testing.filename = '../data/testing.out'
clustering.filename = '../clustering.out'

centroids <- read.table(params.filename, header=F, sep=",", dec=".", skip=1)
n <- read.table(params.filename, header=F, sep=",", dec=".", nrows=1)[1,1]

if (n != nrow(centroids)) {
  stop('The number of centroids are grown, check the file: ', params.filename)
}

euclidean <- function(a, b) sqrt(sum((a - b)^2))

closest_centroid <- function (x, centroids) {
  K = nrows(centroids)
  i.min = sapply(1:K, function(i) {
    euclidian(x, centroids[,i])
  }) |> which.min()
}

