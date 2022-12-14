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
y <- df[ncol(df)]
df <- df[1:ncol(df) - 1] / y[,1]

ks <- sapply(testing.df, closest_centroid, centroids=centroids)
ks <- list(ks)

data.table::fwrite(ks, clustering.filename)
