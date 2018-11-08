############################################################
#                                                          #
# General feature extraction abstract class with specific  #
# implementations for the biclustering based method, the   #
# cluster/pca method, and the feature clustering method.   #
#                                                          #
# Authors: Amy Peerlinck and Neil Walton                   #
#                                                          #
############################################################

from abc import ABC, abstractmethod
import numpy as np
from clustering import Kmeans, Bicluster, Dbscan

class FeatureExtractor(ABC):
    '''
    Abstract class for the feature extractors implemented
    below
    '''

    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    @abstractmethod
    def extract_features(self):
        '''
        Abstract method for extracting the features relevant
        to the specific feature extraction technique
        '''

        pass

class BiclusterExtractor(FeatureExtractor):
    '''
    Implementation of the biclustering based feature extractor
    '''

    def __init__(self, input, labels):
        super().__init__(input, labels)

    def extract_features(self):
        '''
        Using the biclustering technique of Cheng and Church,
        extract features as binary arrays that indicate of
        which clusters each data point is a member and return
        these vectors as the extracted features
        '''

        pass

class FeatureCluster(FeatureExtractor):
    '''
    Implementation of the feature cluster feature extractor
    '''

    def __init__(self, input, labels):
        super().__init__(input, labels)

    def extract_features(self):
        '''
        DESCRIPTION HERE
        '''

        pass

class ClusterPCA(FeatureExtractor):
    '''
    Implementation of the cluster pca feature extractor
    '''

    def __init__(self, input, labels, method='kmeans'):
        super().__init__(input, labels)
        self.method = method
        self.clustering = self._get_clustering()

    def _get_clustering(self):
        '''
        Return the selected clustering method. Vald methods
        are "kmeans" and "dbscan"
        '''

        if self.method == 'kmeans':
            return Kmeans(self.input, k=3)
        elif self.method == 'dbscan':
            return Dbscan(self.input, min_points=4, e=0.5)
        else:
            raise CantClusterLikeThat('Invalid clustering method selected.')

    def extract_features(self):
        '''
        Cluster the features of the data set, then use PCA to
        extract new features from each of the resulting clusters.
        Valid clustering techniques include DBSCAN and kmeans
        '''

        clusters = self.clustering.assign_clusters()
        return clusters

class CantClusterLikeThat(Exception):
    def __init__(self, message):
        self.message = message

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
    iris = load_iris()
    in_ = iris[0]
    out = iris[1]

    bc = BiclusterExtractor(in_, out)
    fc = FeatureCluster(in_, out)
    cpca = ClusterPCA(in_, out, method='c')
    print (cpca.extract_features())
