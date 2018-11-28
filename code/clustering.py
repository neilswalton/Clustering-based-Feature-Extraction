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
from abc import ABC, abstractmethod
import numpy as np
import time


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

    def assign_clusters(self, delta=300, n=10):
        '''
        Using biclustering, cluster the data points into
        n various biclusters and return and array of arrays
        of those resulting cluster assignments.
        Steps:
            1) Remove multiple rows/columns until cluster converges
            2) Remove individual rows/columns until msr < delta
            3) Add rows/columns back in to get maximal cluster
            4) Randomize elements that have been assigned to cluster
            5) Repeat n times to create n biclusters
        '''

        bicluster = self.data
        #bicluster = np.array([np.array([2,2,3]),np.array([4,5,6])])
        msr = self._mean_squared_residual(bicluster)
        print (msr)

        next_cluster = self._multiple_node_deletion(bicluster, delta, alpha=1.2)

        #Use _multiple_node_deletion until the cluster converges
        while (not np.array_equal(bicluster, next_cluster)):
            bicluster = next_cluster
            next_cluster = self._multiple_node_deletion(bicluster, delta, alpha=1.2)

        msr = self._mean_squared_residual(bicluster)
        print (msr)

        #Fine tune the bicluster until the threshold msr (delta) is met
        while (msr > delta):
            bicluster = self._single_node_deletion(bicluster)
            msr = self._mean_squared_residual(bicluster)
            print (msr)

        print (bicluster)
        print (bicluster.shape)

    def _single_node_deletion(self, submatrix):
        '''
        Single node deletion algorithm (Algorithm 1 from
        Cheng and Church's paper). Deletes the single row
        or column with the highest mean squared residue
        '''

        max_score = -1
        ind = -1
        row_col = None

        submatrix_mean = np.mean(submatrix)
        column_means = {}
        for j in range(submatrix.shape[1]):
            column_means[j] = np.mean(submatrix[:,j])

        row_means = {}
        for i in range(submatrix.shape[0]):
            row_means[i] = np.mean(submatrix[i,:])

        for i in range(submatrix.shape[0]):
            temp_score = self._node_score(i, submatrix, 'row', row_means,
                column_means, submatrix_mean)
            if temp_score > max_score:
                max_score = temp_score
                ind = i
                row_col = 'row'

        for j in range(submatrix.shape[1]):
            temp_score = self._node_score(j, submatrix, 'column', row_means,
                column_means, submatrix_mean)
            if temp_score > max_score:
                max_score = temp_score
                ind = j
                row_col = 'column'

        if row_col == 'row':
            new_matrix = np.delete(submatrix, ind, 0)

        if row_col == 'column':
            new_matrix = np.delete(submatrix, ind, 1)

        return new_matrix

    def _multiple_node_deletion(self, submatrix, delta, alpha=1.2):
        '''
        Delete all rows and columns whose mean squared residues
        are greater than alpha times the MSR of the entire
        submatrix
        '''

        submatrix_mean = np.mean(submatrix)
        msr = self._mean_squared_residual(submatrix)

        #MSR is already below the threshold amount delta
        if msr < delta:
            return submatrix

        threshold = alpha * msr
        rows_to_delete = []
        cols_to_delete = []

        #Calculate row and column means
        column_means = {}
        for j in range(submatrix.shape[1]):
            column_means[j] = np.mean(submatrix[:,j])

        row_means = {}
        for i in range(submatrix.shape[0]):
            row_means[i] = np.mean(submatrix[i,:])

        #Keep track of all rows whose MSR is greater than the threshold
        for i in range(submatrix.shape[0]):
            temp_score = self._node_score(i, submatrix, 'row', row_means,
                column_means, submatrix_mean)
            if temp_score > threshold:
                rows_to_delete.append(i)

        #Delete the rows selected above
        temp_matrix = np.delete(submatrix, rows_to_delete, 0)

        #Recalculate mean and msr for the new matrix
        temp_matrix_mean = np.mean(temp_matrix)
        new_msr = self._mean_squared_residual(temp_matrix)
        threshold = alpha * new_msr

        #Return if removing rows made msr < delta
        if new_msr < delta:
            return temp_matrix

        #Recalculate column and row means
        column_means = {}
        for j in range(temp_matrix.shape[1]):
            column_means[j] = np.mean(temp_matrix[:,j])

        row_means = {}
        for i in range(temp_matrix.shape[0]):
            row_means[i] = np.mean(temp_matrix[i,:])

        #Keep track of columns with msr greater than the new threshold
        for j in range(temp_matrix.shape[1]):
            temp_score = self._node_score(j, temp_matrix, 'column', row_means,
                column_means, temp_matrix_mean)
            if temp_score > threshold:
                cols_to_delete.append(j)

        #Delete the columns identified above
        final_matrix = np.delete(temp_matrix, cols_to_delete, 1)

        return final_matrix


    def _node_score(self, ind, submatrix, row_or_column, row_means,
                    column_means, submatrix_mean):
        '''
        Calculate the mean squared residual of the submatrix
        for the specified row/column index. The row_or_column
        parameter specifies whether the index refers to a row
        or a column
        '''

        score = 0

        if row_or_column == 'row':
            size = submatrix.shape[1]
            row_mean = row_means[ind]
            for j in range(size):
                column_mean = column_means[j]
                residue = submatrix[ind,j] - row_mean - column_mean + submatrix_mean
                squared_residue = np.power(residue, 2)
                score += (squared_residue/float(size))
        if row_or_column == 'column':
            size = submatrix.shape[0]
            column_mean = column_means[ind]
            for i in range(size):
                row_mean = row_means[i]
                residue = submatrix[i,ind] - row_mean - column_mean + submatrix_mean
                squared_residue = np.power(residue, 2)
                score += (squared_residue/float(size))

        return score


    def _mean_squared_residual(self, submatrix):
        '''
        Calculate the mean squared residual for the
        specified submatrix
        '''

        submatrix_mean = np.mean(submatrix)
        size = submatrix.size
        msr = 0
        column_means = {}
        for j in range(submatrix.shape[1]):
            column_means[j] = np.mean(submatrix[:,j])

        for i, row in enumerate(submatrix):
            row_mean = np.mean(submatrix[i,:])
            for j, column in enumerate(row):
                column_mean = column_means[j]
                residue = submatrix[i,j] - column_mean - row_mean + submatrix_mean
                squared_residue = np.power(residue, 2)
                msr += (squared_residue/float(size))

        return msr

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

    k = Kmeans(in_, k=3)
    d = Dbscan(in_, min_points=4, e=.5)
    b = Bicluster(in_)
    clusts = d.assign_clusters()

    print (clusts)
