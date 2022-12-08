import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import sys
from scipy.spatial import distance

class Kmeans:
    centers = None,
    cluster = None,
    K = 0,
    X = None,
    PC = None,

    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.cluster = np.zeros(self.X.shape[0])
        self.PC = PCA(n_components=2).fit(self.X)
    
    def set_centers(self, centers):
        self.centers = centers
        return self
    
    def __get_closest_centroid(self, obs, centroids):
        '''
        Function for retrieving the closest centroid to the given observation 
        in terms of the Euclidean distance.

        Parameters
        ----------
        obs : array
            An array containing the observation to be matched to the nearest centroid
        centroids : array
            An array containing the centroids. len(centroids) >= 1

        Returns
        -------
        (colour, min_centroid) : (int, array)
            The color assigned by order found in the array
            The centroid closes to the obs 
        '''
        min_distance = sys.float_info.max
        min_centroid = 0
        colour = 0

        for i, c in enumerate(centroids):
            dist = distance.euclidean(obs, c)
            if dist < min_distance:
                colour = i + 1
                min_distance = dist
                min_centroid = c

        return colour, min_centroid
    
    def assignment_step(self):
        for i, sample in enumerate(self.X):
            colour, _ = self.__get_closest_centroid(sample, self.centers)
            self.cluster[i] = colour
        return self
    
    def updating_step(self):
        colours = range(1, self.K + 1)
        for c in colours:
            S = np.zeros(len(self.centers[c - 1]))
            n = 0
            for i, _ in enumerate(self.X):
                if self.cluster[i] == c:
                    S = np.add(S, self.X[i,:])
                    n = n + 1
            
            if n == 0:
                print(f'UPS! none item in the cluster: {c}')
                continue
            
            self.centers[c - 1] = S / n
        return self
    
    def sequential_step(self, x, alpha):
        # Write your sequential_step here
        #### -- next line updates the closest centroid to x using alpha learning rate --
        
        colour, centroid = self.__get_closest_centroid(x, self.centers)
        mean = centroid + alpha * np.subtract(x, centroid)
        self.centers[colour - 1] = mean
        
        return self

    def plot(self, title = ""):
        if self.centers is None:
          print("No centroids defined")

        # Function to plot current state of the algorithm.
        # For visualisation purposes, only the first two PC are shown.
        PC = self.PC.transform(self.X)
        C2 = self.PC.transform(self.centers)

        if self.cluster[0] is None:
          plt.scatter(PC[:,0], PC[:,1], alpha=0.5)
        else:
          plt.scatter(PC[:,0], PC[:,1], c=self.cluster, alpha=0.5)

        plt.scatter(C2[:,0], C2[:,1], s=100, c=np.arange(self.K), edgecolors = 'black')
        plt.title(title)
        plt.show()
        plt.clf()
