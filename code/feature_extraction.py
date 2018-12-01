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
from dataprocessing import DataReader
from utils import Pca, silhouette
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

    def _get_clustering(self, method):
        '''
        Return the selected clustering method. Valid methods
        are "kmeans" and "dbscan"
        '''

        pass

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

        bc = Bicluster(self.input)
        delta = 0.15
        alpha = 1.1
        n = 4
        clusters = bc.assign_clusters(delta, alpha, n)

        #Create a binary vector where a 1 indicates
        #the ith data point belongs to the jth cluster
        num_rows = self.input.shape[0]
        num_columns = n
        binary_features = np.zeros((num_rows, num_columns))

        for j, c in enumerate(clusters):
            for i in c[0]:
                binary_features[i][j] = 1

        return binary_features

class FeatureCluster(FeatureExtractor):
    '''
    Implementation of the feature cluster feature extractor
    '''

    def __init__(self, input, labels, method="kmeans", num_clusters=2, _type="hard"):
        super().__init__(input, labels)
        self.method = method
        self.num_clusters = num_clusters
        self._type = _type
        if self._type == "soft" or self.type=="mixed":
            self.method ="kmeans"
        self.clustering = self._get_clustering(self.method)
        self.cluster_labels = np.array([])

    def _get_clustering(self, method):
        '''
        Return the selected clustering method. Valid methods
        are "kmeans" and "dbscan"
        '''

        if self.method == 'kmeans':
            return Kmeans(self.input.T, k=2)
        elif self.method == 'dbscan':
            return Dbscan(self.input.T, min_points=10, e=0.0015)
        else:
            raise CantClusterLikeThat('Invalid clustering method selected "' +self.method+ '".')


    def hard_combination(self):
        '''
        Dot product between hard weighted matrix and original feature matrix.
        Hard weights consist of 0 if feature does not belong to cluster, 1 otherwise.
        This means the new feature is a sum of all old feature values in the 
        corresponding cluster.
        '''
        nr_datapoints = self.input.shape[0]
        nr_features = self.input.shape[1]
        weight_matrix = np.zeros([self.num_clusters,nr_features])
        combined_clusters = np.zeros([nr_datapoints, self.num_clusters])

        for i,c in enumerate(self.cluster_labels):
            weight_matrix[c,i] = 1
        for k, cluster in enumerate(weight_matrix):
            to_sum = self.input[:,[i for i, weight in enumerate(cluster) if weight == 1]] #150,2
            combined_clusters[:,k] = np.sum(to_sum, axis=1)

        return combined_clusters

    def soft_combination(self):
        '''
        Dot product between soft weighted matrix and original feature matrix.
        Soft weights depend on the probability of a feature belonging to the cluster.
        This means the new feature is a weighted combinations of all old feature values.
        '''
        nr_datapoints = self.input.shape[0]
        nr_features = self.input.shape[1]
        weight_matrix = np.zeros([self.num_clusters,nr_features])
        combined_clusters = np.zeros([nr_datapoints, self.num_clusters])

        for i,c in enumerate(self.cluster_labels):
            for j, prob in enumerate(c):
                weight_matrix[i,j] = prob
        for k, cluster_prob in enumerate(weight_matrix):
            to_sum = np.asarray([np.multiply(self.input[:,i],prob) for i, prob in enumerate(cluster_prob)]).T 
            combined_clusters[:,k] = np.sum(to_sum,axis=1)

        return combined_clusters

    def mixed_combination(self):
        nr_datapoints = self.input.shape[0]
        nr_features = self.input.shape[1]
        hard_weight_matrix = np.zeros([self.num_clusters,nr_features])
        soft_weight_matrix = np.zeros([self.num_clusters,nr_features])     
        for i,c in enumerate(self.cluster_labels[0]):
            hard_weight_matrix[c,i] = 1
        for i,c in enumerate(self.cluster_labels[1]):
            for j, prob in enumerate(c):
                soft_weight_matrix[i,j] = prob
        hard_weight_matrix = np.multiply(hard_weight_matrix, 0.2)
        soft_weight_matrix = np.multiply(soft_weight_matrix, 0.8)
        weight_matrix = [soft_weight_matrix[i]+h for i,h in enumerate(hard_weight_matrix)]
        for k, cluster_prob in enumerate(weight_matrix):
            to_sum = np.asarray([np.multiply(self.input[:,i],prob) for i, prob in enumerate(cluster_prob)]).T 
            combined_clusters[:,k] = np.sum(to_sum,axis=1)

        return combined_clusters

    def weighted_combination(self):
        '''
        Calls correct weighting function
        '''
        combined_clusters = np.zeros([nr_datapoints, self.num_clusters])
        if self._type=="hard":
            combined_clusters = self.hard_combination()
        elif self._type=="soft":
            combined_clusters = self.soft_combination()
        elif self._type=="mixed":
            combined_clusters = self.mixed_combination()

        print(combined_clusters)

        return combined_clusters

    def extract_features(self):
        '''
        Combine features using K-Means or DBSCAN for hard weighted clustering 
        and Fuzzy C-Means for soft weighted clustering.
        Perform soft, hard or mixed weighted combination of clustered features.
        '''
        features = self.input.T #Transpose so we're clustering features
        if self._type =="hard":
            self.cluster_labels = self.clustering.assign_clusters()
        elif self._type=="soft":
            self.cluster_labels = self.clustering.assign_fuzzy_clusters()
        elif self._type=="mixed":
            self.cluster_labels = [self.clustering.assign_clusters() , self.clustering.assign_fuzzy_clusters()]
        new_features = self.weighted_combination()

        return new_features

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
        Return the selected clustering method. Valid methods
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

        sc = silhouette(features, clusters) #silhouette coefficient

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
        print(features)
        data_matrix.append(features)
        if temp_list[-1] == 'Iris-setosa':
            labels.append(0)
        elif temp_list[-1] == 'Iris-versicolor':
            labels.append(1)
        elif temp_list[-1] == 'Iris-virginica':
            labels.append(2)

    return (np.array(data_matrix), np.array(labels))

if __name__ == '__main__':
    #iris = load_iris()
    iris= DataReader("../data/iris.txt").run()
    in_ = iris[0]
    out = iris[1]

    bc = BiclusterExtractor(in_, out)
    fc = FeatureCluster(in_, out, _type="hard")
    cpca = ClusterPCA(in_, out, method='kmeans')
    feats = fc.extract_features()
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
