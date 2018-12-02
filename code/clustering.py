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
import time, skfuzzy


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
        cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(self.data.T, self.k, 2, error=0.005, maxiter=1000, init=None)

        u_pred, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans_predict(self.data.T, cntr, 2, error=0.005, maxiter=1000)
        return u_pred

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
        self.remaining_rows = np.arange(data.shape[0])
        self.remaining_cols = np.arange(data.shape[1])

    def assign_clusters(self, delta=0.1, alpha=1.2, n=10):
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

        self.alpha = alpha
        input_data = np.copy(self.data)
        clusters = []
        min_val = np.min(self.data)
        max_val = np.max(self.data)

        for i in range(n):
            print ('Finding bicluster ', i+1, ' of ',n)
            self.remaining_rows = np.arange(self.data.shape[0])
            self.remaining_cols = np.arange(self.data.shape[1])
            cluster = self._get_bicluster(input_data, delta)
            clusters.append(cluster)

            #Randomize the data that has been assigned to n_clusters
            for i in cluster[0]:
                for j in cluster[1]:
                    input_data[i][j] = np.random.uniform(low=min_val, high=max_val)

        return clusters

    def _get_bicluster(self, input, delta):
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

        bicluster = input
        #bicluster = np.array([np.array([2,2,3]),np.array([4,5,6])])
        msr = self._mean_squared_residual(bicluster)

        next_cluster = self._multiple_node_deletion(bicluster, delta, alpha=self.alpha)

        #Use _multiple_node_deletion until the cluster converges
        while (not np.array_equal(bicluster, next_cluster)):
            bicluster = next_cluster
            next_cluster = self._multiple_node_deletion(bicluster, delta, alpha=self.alpha)


        msr = self._mean_squared_residual(bicluster)

        #Fine tune the bicluster until the threshold msr (delta) is met
        while (msr > delta):
            bicluster = self._single_node_deletion(bicluster)
            msr = self._mean_squared_residual(bicluster)

        #Cheng and Church state they run the addition step only once
        bicluster = self._node_addition(bicluster)

        print (bicluster.shape)

        return (self.remaining_rows, self.remaining_cols)

    def _node_addition(self, submatrix):
        '''
        Add rows and columns back in that do
        not increase the MSR back to above
        the threshold
        '''

        deleted_rows = np.setdiff1d(np.arange(self.data.shape[0]),
                        self.remaining_rows)
        deleted_cols = np.setdiff1d(np.arange(self.data.shape[1]),
                        self.remaining_cols)

        msr = self._mean_squared_residual(submatrix)

        #Means of all the original columns, but only
        #the selected rows of the bicluster
        col_scores = self._add_col_scores(submatrix)
        temp_cols = np.argwhere(col_scores < msr).flatten()

        cols_to_add = np.intersect1d(temp_cols,deleted_cols)

        #Add the columns selected above
        if len(cols_to_add) != 0:
            new_cols = np.take(self.data,
                cols_to_add, axis=1)[self.remaining_rows,:]
            submatrix = np.append(submatrix, new_cols, axis=1)
            self.remaining_cols = np.hstack((self.remaining_cols, cols_to_add))

        #Recalculate variables and add rows
        msr = self._mean_squared_residual(submatrix)

        row_scores = self._add_row_scores(submatrix)
        temp_rows = np.argwhere(row_scores < msr).flatten()

        rows_to_add = np.intersect1d(temp_rows,deleted_rows)

        #Add the row selected above
        if len(rows_to_add) != 0:
            new_rows = np.take(self.data,
                rows_to_add, axis=0)[:, self.remaining_cols]
            submatrix = np.append(submatrix, new_rows, axis=0)
            self.remaining_rows = np.hstack((self.remaining_rows, rows_to_add))

        return submatrix

    def _single_node_deletion(self, submatrix):
        '''
        Single node deletion algorithm (Algorithm 1 from
        Cheng and Church's paper). Deletes the single row
        or column with the highest mean squared residue
        '''

        row_scores = self._row_scores(submatrix)
        col_scores = self._col_scores(submatrix)

        if np.max(row_scores) >= np.max(col_scores):
            ind = np.argmax(row_scores)
            new_matrix = np.delete(submatrix, ind, 0)
            self.remaining_rows = np.delete(self.remaining_rows, ind, 0)

        elif np.max(row_scores) < np.max(col_scores):
            ind = np.argmax(row_scores)
            new_matrix = np.delete(submatrix, ind, 1)
            self.remaining_cols = np.delete(self.remaining_cols, ind, 0)

        return new_matrix

    def _multiple_node_deletion(self, submatrix, delta, alpha=1.2):
        '''
        Delete all rows and columns whose mean squared residues
        are greater than alpha times the MSR of the entire
        submatrix
        '''

        #submatrix_mean = np.mean(submatrix)
        msr = self._mean_squared_residual(submatrix)
        threshold = alpha * msr

        #MSR is already below the threshold amount delta
        if msr < delta:
            return submatrix

        row_scores = self._row_scores(submatrix)
        #
        rows_to_delete = np.argwhere(row_scores > threshold).flatten()

        #Delete the rows selected above
        temp_matrix = np.delete(submatrix, rows_to_delete, 0)
        self.remaining_rows = np.delete(self.remaining_rows, rows_to_delete, 0)

        #Recalculate mean and msr for the new matrix
        new_msr = self._mean_squared_residual(temp_matrix)
        threshold = alpha * new_msr

        #Return if removing rows made msr < delta
        if new_msr < delta:
            return temp_matrix

        col_scores = self._col_scores(temp_matrix)
        cols_to_delete = np.argwhere(col_scores > threshold).flatten()

        #Delete the columns identified above
        final_matrix = np.delete(temp_matrix, cols_to_delete, 1)
        self.remaining_cols = np.delete(self.remaining_cols, cols_to_delete, 0)

        return final_matrix

    def _row_scores(self, submatrix):
        '''
        Return the scores for each row in the submatrix
        '''

        submatrix_mean = np.mean(submatrix)
        col_means = np.mean(submatrix,axis=0).T
        row_means = np.mean(submatrix,axis=1)
        col_means = np.reshape(col_means, (1, len(col_means)))
        row_means = np.reshape(row_means, (len(row_means), 1))

        residues = np.add(np.subtract(np.subtract(submatrix,
            row_means), col_means), submatrix_mean)

        squared_residues = np.power(residues, 2)
        msr = np.mean(squared_residues, axis=1)

        return msr
    def _col_scores(self, submatrix):
        '''
        Return the scores for each column in the submatrix
        '''

        submatrix_mean = np.mean(submatrix)
        col_means = np.mean(submatrix,axis=0).T
        row_means = np.mean(submatrix,axis=1)
        col_means = np.reshape(col_means, (1, len(col_means)))
        row_means = np.reshape(row_means, (len(row_means), 1))

        residues = np.add(np.subtract(np.subtract(submatrix,
            row_means), col_means), submatrix_mean)

        squared_residues = np.power(residues, 2)
        msr = np.mean(squared_residues, axis=0)

        return msr

    def _add_col_scores(self, submatrix):
        '''
        Return the scores for each column in the submatrix
        for the column addition step
        '''

        submatrix_mean = np.mean(submatrix)
        row_subset = np.take(self.data, self.remaining_rows, axis=0)
        col_means = np.mean(row_subset,axis=0)
        row_means = np.mean(submatrix,axis=1).T
        col_means = np.reshape(col_means, (1, len(col_means)))
        row_means = np.reshape(row_means, (len(row_means), 1))

        residues = np.add(np.subtract(np.subtract(row_subset,
            row_means), col_means), submatrix_mean)

        squared_residues = np.power(residues, 2)
        msr = np.mean(squared_residues, axis=0)

        return msr

    def _add_row_scores(self, submatrix):
        '''
        Return the scores for each column in the submatrix
        for the row addition step
        '''

        submatrix_mean = np.mean(submatrix)
        col_subset = np.take(self.data, self.remaining_cols, axis=1)
        col_means = np.mean(submatrix,axis=0)
        row_means = np.mean(col_subset,axis=1).T
        col_means = np.reshape(col_means, (1, len(col_means)))
        row_means = np.reshape(row_means, (len(row_means), 1))

        residues = np.add(np.subtract(np.subtract(col_subset,
            row_means), col_means), submatrix_mean)

        squared_residues = np.power(residues, 2)
        msr = np.mean(squared_residues, axis=1)

        return msr

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

    def _inverse_node_score(self, ind, submatrix, row_means,
                    column_means, submatrix_mean):
        '''
        Calculate the mean squared residual of the submatrix
        for the specified row/column index. The row_or_column
        parameter specifies whether the index refers to a row
        or a column
        '''

        score = 0

        size = submatrix.shape[1]
        row_mean = row_means[ind]
        for j in range(size):
            column_mean = column_means[j]
            residue =  row_mean - submatrix[ind,j] - column_mean + submatrix_mean
            squared_residue = np.power(residue, 2)
            score += (squared_residue/float(size))

        return score

    def _mean_squared_residual(self, submatrix):
        '''
        Calculate the mean squared residual for the
        specified submatrix
        '''

        submatrix_mean = np.mean(submatrix)
        row_means = np.mean(submatrix,axis=0).T
        col_means = np.mean(submatrix,axis=1)

        row_means = np.reshape(row_means, (1, len(row_means)))
        col_means = np.reshape(col_means, (len(col_means), 1))

        residues = np.add(np.subtract(np.subtract(submatrix,
            row_means), col_means), submatrix_mean)

        squared_residues = np.power(residues, 2)
        msr = np.mean(squared_residues)

        return msr

    def _msr(self, submatrix):
        '''
        Calculate the mean squared residual for the
        specified submatrix
        '''

        submatrix_mean = np.mean(submatrix)
        row_means = np.mean(submatrix,axis=0).T
        col_means = np.mean(submatrix,axis=1)

        row_means = np.reshape(row_means, (1, len(row_means)))
        col_means = np.reshape(col_means, (len(col_means), 1))

        residues = np.add(np.subtract(np.subtract(submatrix,
            row_means), col_means), submatrix_mean)

        squared_residues = np.power(residues, 2)
        msr = np.mean(squared_residues)

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
