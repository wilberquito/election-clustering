import math
from scipy.spatial import distance

def closest_centroid(x, centroids):
    '''
    Function for retrieving the closest centroid to the given observation 
    in terms of the Euclidean distance.
    
    Parameters
    ----------
    x : array
        An array containing the observation to be matched to the nearest centroid
    centroids : array
        An array containing the centroids
    
    Returns
    -------
    min_centroid : array
        The centroid closes to the obs 
    '''
    min_distance = math.inf
    min_centroid = 0
    
    for c in centroids:
        dist = distance.euclidean(x, c)
        if dist < min_distance:
            min_distance = dist
            min_centroid = c
            
    return min_centroid


def prediction_strength(k, train_centroids, X_test, test_labels):
    '''
    Function for calculating the prediction strength of clustering for
    a given number of clusters
    
    Parameters
    ----------
    k : int
        The number of clusters
    train_centroids : array
        Centroids from the clustering on the training set
    X_test : array
        Test set observations
    test_labels : array
        Labels predicted for the test set
        
    Returns
    -------
    prediction_strength : float
        Calculated prediction strength
    '''
    n_test = len(X_test)
    
    # populate the co-membership matrix
    D = np.zeros(shape=(n_test, n_test))
    for x1, c1 in zip(X_test, list(range(n_test))):
        for x2, c2 in zip(X_test, list(range(n_test))):
            # checks not to be the same sample
            if tuple(x1) != tuple(x2):
                # when 2 samples are part of the same centroid in the matrix of samples they are assigned to 1.
                if tuple(get_closest_centroid(x1, train_centroids)) == tuple(get_closest_centroid(x2, train_centroids)):
                    D[c1, c2] = 1.0
    
    # calculate the prediction strengths for each cluster
    ss = []
    for j in range(k):
        s = 0
        for x1, l1, c1 in zip(X_test, test_labels, list(range(n_test))):
            # focuses on labels in the cluster j
            if l1 != j:
                continue
            for x2, l2, c2 in zip(X_test, test_labels, list(range(n_test))):
                # checks if 2 differents samples were marked as part of the
                # same cluster. If they were, consults the co-membership matrix
                # to add 1 point if they were part of the same cluster thanks to the
                # euclidian vector, otherwise the sum do not modify the cluster prediction strengths.
                if tuple(x1) != tuple(x2) and l1 == l2:
                    s += D[c1,c2]
               
        examples_j = X_test[test_labels == j, :].tolist()
        n_examples_j = len(examples_j)
        examples_product = n_examples_j * (n_examples_j - 1)
        
        if examples_product == 0:
            ss.append(math.inf)
        else:
            ss.append(s / examples_product) 

    prediction_strength = min(ss)
    return prediction_strength

