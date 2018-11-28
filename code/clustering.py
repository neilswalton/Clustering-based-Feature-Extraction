##########################################################
#                                                        #
# General clustering abstract class with specific        #
# implementations for k-means, DBSCAN and biclustering   #
# (from Cheng and Church's "Biclustering Expression      #
# Data")                                                 #
#                                                        #
# Authors: Amy Peerlinck and Neil Walton                 #
#                                                        #
##########################################################

from sklearn.cluster import KMeans, DBSCAN
import skfuzzy
from dataprocessing import DataReader
from abc import ABC, abstractmethod
import numpy as np


class Cluster(ABC):
    '''
    Abstract class for the various clustering techniques
    '''

    def __init__(self, data):
        self.data = data
        super().__init__()

    @abstractmethod
    def assign_clusters(self):
        '''
        Abstract method to assign data points to clusters
        implementation varies by method. Returns the cluster
        assignments as an array.
        '''

        pass

class Kmeans(Cluster):
    '''
    Implementation of the k-means clustering algorithm using
    the scikit learn implementation
    '''

    def __init__(self, data, k=3):
        self.k = k
        super().__init__(data)

    def assign_clusters(self):
        '''
        Use the fit and predict methods to assign data
        to the k clusters
        '''

        km = KMeans(n_clusters=self.k).fit(self.data)
        return np.array(km.predict(self.data))

    def assign_fuzzy_clusters(self):
        cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(self.data.T,c=self.k, m=2, error=0.005, maxiter=1000, init=None)
        u_pred, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans_predict(self.data.T,cntr, m=2, error=0.005, maxiter=1000)
        return np.array(u_pred)

class Dbscan(Cluster):
    '''
    Wrapper for the scitkit learn implementation of DBSCAN
    '''

    def __init__(self, data, min_points=4, e=0.5):
        self.min_points = min_points
        self.e = e
        super().__init__(data)

    def assign_clusters(self):
        '''
        Assign the datapoints to clusters using DBSCAN
        and return an array of the cluster assignments
        '''

        db = DBSCAN(eps=self.e, min_samples=self.min_points)
        return db.fit_predict(self.data)

class Bicluster(Cluster):
    '''
    Biclustering implementation according to the algorithm
    provided in Cheng and Church's "Biclustering Expression
    Data"
    '''

    def __init__(self, data):
        super().__init__(data)

    def assign_clusters(self):
        '''
        Using biclustering, cluster the data points into
        various clusters and return and array of arrays
        of those resulting cluster assignments
        '''

        pass

def load_iris():
    path = '../data/iris.txt'
    iris_text = open(path, 'r');
    data_matrix = []
    labels = []

    for line in iris_text:

        temp_list = line.strip().split(',')
        features = np.array([float(x) for x in temp_list[:4]])
        data_matrix.append(features)
        if temp_list[-1] == 'Iris-setosa':
            labels.append(0)
        elif temp_list[-1] == 'Iris-versicolor':
            labels.append(1)
        elif temp_list[-1] == 'Iris-virginica':
            labels.append(2)

    return (np.array(data_matrix), np.array(labels))

if __name__ == '__main__':
    iris= DataReader("../data/iris.txt").run()
    in_ = iris[0]
    out = iris[1]

    k = Kmeans(in_, k=3)
    d = Dbscan(in_, min_points=4, e=.5)
    b = Bicluster(in_)
    clusts = k.assign_fuzzy_clusters()

    print (clusts)
