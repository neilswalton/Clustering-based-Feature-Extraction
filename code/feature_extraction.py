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
from utils import Pca
from classifiers import Knn, NaiveBayes
from matplotlib import pyplot as plt

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

    def __init__(self, input, labels, method='kmeans', num_clusters=2, feats_per_cluster=1):
        super().__init__(input, labels)
        self.num_clusters = num_clusters
        self.feats_per_cluster = feats_per_cluster
        self.method = method
        self.clustering = self._get_clustering()

    def _get_clustering(self):
        '''
        Return the selected clustering method. Vald methods
        are "kmeans" and "dbscan"
        '''

        if self.method == 'kmeans':
            return Kmeans(self.input.T, k=self.num_clusters)
        elif self.method == 'dbscan':
            return Dbscan(self.input.T, min_points=4, e=0.5)
        else:
            raise CantClusterLikeThat('Invalid clustering method selected "' +self.method+ '".')

    def extract_features(self):
        '''
        Cluster the features of the data set, then use PCA to
        extract new features from each of the resulting clusters.
        Valid clustering techniques include DBSCAN and kmeans
        '''

        features = self.input.T #Transpose so we're clustering features
        clusters = self.clustering.assign_clusters()
        new_features = np.array([])

        #For each cluster, run PCA on the columns in the cluster to reduce dimension
        for c in set(clusters):
            columns = []
            for i in range(len(clusters)):
                if clusters[i] == c:
                    columns.append(features[i])
            columns =  np.array(columns).T
            p = Pca(columns, n=self.feats_per_cluster)
            extracted_features = p.get_components()
            if new_features.shape[0] == 0:
                new_features = extracted_features
            else:
                new_features = np.hstack((new_features, extracted_features))

        return new_features

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
    cpca = ClusterPCA(in_, out, method='kmeans')
    feats = cpca.extract_features()
    #print (feats)

    p = Pca(in_, n=2)
    pca_feats = p.get_components()

    k=8
    knn1 = Knn(in_, out, k=k)
    knn2 = Knn(feats, out, k=k)
    knn3 = Knn(pca_feats, out, k=k)
    print ('Knn score with original features: ', knn1.k_fold_score()[0])
    print ('Knn score with extracted features: ', knn2.k_fold_score()[0])
    print ('Knn score with PCA features: ', knn3.k_fold_score()[0])

    nb1 = NaiveBayes(in_, out)
    nb2 = NaiveBayes(feats, out)
    nb3 = NaiveBayes(pca_feats, out)
    print ('Naive Bayes score with original features: ', nb1.k_fold_score()[0])
    print ('Naive Bayes score with extracted features: ', nb2.k_fold_score()[0])
    print ('Naive Bayes score with PCA features: ', nb3.k_fold_score()[0])

    #plt.figure(1)
    plt.subplot(121, rasterized=True)
    plt.scatter(pca_feats[:,0], pca_feats[:,1], c=out)
    #plt.show()

    #plt.figure(2)
    plt.subplot(122)
    plt.scatter(feats[:,0], feats[:,1], c=out)
    plt.show()
