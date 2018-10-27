################################################################
#                                                              #
# The classifier class that implements the general classifer   #
# superclass, the k-nearest neighbor, Naive Bayes, and         #
# feedforward neural network classifiers.                      #
#                                                              #
# Authors: Amy Peerlinck and Neil Walton                       #
#                                                              #
################################################################

import numpy as np
from abc import ABC, abstractmethod
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

class Classifier(ABC):
    '''
    Classifier abstract class. This is the superclass of
    the specific classification algorithms implemented below
    '''

    def __init__(self, input, labels):
        self.input = input
        self.labels = labels
        self.model = None
        self.fit_yet = False
        super().__init__()

    @abstractmethod
    def fit(self, in_, out):
        '''
        The different algorithms have different calls to fit
        the models to the data
        '''

        pass

    @abstractmethod
    def score(self, in_, out):
        '''
        Output the accuracy of the classifier obtained on the
        specified input and output (labels)
        '''

        pass

    def k_fold_score(self, k=10):
        '''
        Use stratified k fold crossvalidation on the data and
        return the mean, standard deviation, and raw accuracies
        obtained from each fold
        '''

        scores = []
        final_score = None
        stdev = None
        skf = StratifiedKFold(n_splits=k, shuffle=True)
        skf.get_n_splits(self.input, self.labels)

        for train_ind, test_ind in skf.split(self.input, self.labels):
            train_in = np.take(self.input, train_ind, axis=0)
            test_in = np.take(self.input, test_ind, axis=0)
            train_out = np.take(self.labels, train_ind)
            test_out = np.take(self.labels, test_ind)

            self.fit(train_in, train_out)
            score = self.score(test_in, test_out)
            scores.append(score)

        final_score = np.mean(scores)
        stdev = np.std(scores)

        return final_score, stdev, scores

class Knn(Classifier):
    '''
    The k-nearest neighbors classification algorithm
    '''

    def __init__(self, input, labels, k=5):

        super().__init__(input, labels)
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, in_, out):
        '''
        Fit the k nearest neighbors algorithm using scikit
        learn's default parameters
        '''

        self.model.fit(in_, out)
        self.fit_yet = True

    def score(self, in_, out):
        '''
        Using the fit knn model, return the accuracy of the
        model on the specified input data
        '''

        if not self.fit_yet:
            raise ModelNotFit('Must fit the Knn model before scoring.')

        else:
            return self.model.score(in_, out)

    def plot_k_scores(self, max_k=10):
        '''
        For k = 1..max_k, calculate the 10 fold crossvalidated score
        of the k-means algorithm using the separate values of k and
        plot the resulting scores
        '''

        scores = [None] #Shift the plot so it starts at k=1
        ks_to_plot = np.arange(1, max_k+1, int(max_k/10))

        for i in range(1,max_k+1):
            self.model = KNeighborsClassifier(n_neighbors=i)
            score, _, _ = self.k_fold_score(k=10)
            scores.append(score)

        #Plot the scores
        plt.plot(scores)
        plt.xticks(ks_to_plot)
        plt.title('Accuracy achieved for varied values of k')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.show()

        #Reset the model to the original value of k
        self.model = KNeighborsClassifier(n_neighbors=self.k)

class ModelNotFit(Exception):
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
    k = Knn(in_, out, 1)

    score, stdev, scores = k.k_fold_score()

    print(score)
    print(stdev)
    print(scores)

    k.plot_k_scores(10)
