###################################################
#                                                 #
# General utility file for scoring methods, pca,  #
# etc.                                            #
#                                                 #
# Authors: Amy Peerlinck and Neil Walton          #
#                                                 #
###################################################

import numpy as np
from sklearn.decomposition import PCA
from math import ceil
from sklearn.metrics import silhouette_score as sc, f1_score as fscore

class Pca:
    '''
    The PCA class. Can fit be fit to data and transform
    a specified data point to the number of specified
    principal components
    '''

    def __init__(self, data, n=1):
        self.data = data
        self.n = n
        self.pca = self._get_pca()

    def _get_pca(self):
        '''
        Return the pca model for the specified n. If n<1 then
        the number of components should be the size of the
        cluster times n
        '''

        if self.n<=0:
            raise ValueError('Number of components cannot be negative.')
        elif self.n<1:
            n = ceil(len(self.data.T)*self.n)
        else:
            n = self.n

        n = min(n, len(self.data.T)) #Can't have fewer components than features
        return PCA(n_components=n).fit(self.data)

    def get_components(self):
        '''
        Return the top n principal components of the data.
        If n<1, Reduce the dimension of each cluster by size of
        cluster times n (rounded up to nearest integer)
        '''

        return self.pca.transform(self.data)


def silhouette(data, clusters):
    score = sc(data, clusters)
    return score

def f_score(true, pred):
    score = fscore(true, pred, average='weighted')
    return score
