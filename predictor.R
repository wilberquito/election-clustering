params.filename <- './param.out'
testing.filename <- './testing.csv'
clustering.filename <- './clustering.out'

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

xs <- apply(X, 1, function(x) closest_centroid(x, centroids))
xs <- list(xs)
data.table::fwrite(xs, clustering.filename)

