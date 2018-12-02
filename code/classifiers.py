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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from utils import f_score

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
    def score(self, in_, out, method):
        '''
        Output the accuracy of the classifier obtained on the
        specified input and output (labels)
        '''

        pass

    def k_fold_score(self, k=10, scoring_method='accuracy'):
        '''
        Use stratified k fold crossvalidation on the data and
        return the mean, standard deviation, and raw accuracies
        obtained from each fold. Valid scoring methods are
        'accuracy', 'fscore', and 'both'
        '''

        if scoring_method == 'accuracy' or scoring_method == 'fscore':
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
                score = self.score(test_in, test_out, scoring_method)
                scores.append(score)

            final_score = np.mean(scores)
            stdev = np.std(scores)

            return final_score, stdev, scores

        elif scoring_method == 'both':
            accuracies = []
            fscores = []
            final_accuracy = None
            accuracy_stdev = None
            final_fscore = None
            fscore_stdev = None
            skf = StratifiedKFold(n_splits=k, shuffle=True)
            skf.get_n_splits(self.input, self.labels)

            for train_ind, test_ind in skf.split(self.input, self.labels):
                train_in = np.take(self.input, train_ind, axis=0)
                test_in = np.take(self.input, test_ind, axis=0)
                train_out = np.take(self.labels, train_ind)
                test_out = np.take(self.labels, test_ind)

                self.fit(train_in, train_out)
                accuracy, fscore = self.score(test_in, test_out, scoring_method)
                accuracies.append(accuracy)
                fscores.append(fscore)

            final_accuracy = np.mean(accuracies)
            accuracy_stdev = np.std(accuracies)
            final_fscore = np.mean(fscores)
            fscore_stdev = np.std(fscores)

            return final_accuracy, accuracy_stdev, accuracies, final_fscore, fscore_stdev, fscores

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

    def score(self, in_, out, method):
        '''
        Using the fit knn model, return the accuracy of the
        model on the specified input data. Valid scoring methods
        are 'accuracy', 'fscore', and 'both'
        '''

        if not self.fit_yet:
            raise ModelNotFit('Must fit the Knn model before scoring.')

        else:
            if method == 'accuracy':
                return self.model.score(in_, out)
            elif method == 'fscore':
                pred = self.model.predict(in_)
                score = f_score(out, pred)
                return score
            elif method == 'both':
                pred = self.model.predict(in_)
                fscore = f_score(out, pred)
                accuracy = self.model.score(in_, out)
                return accuracy, fscore

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

class NaiveBayes(Classifier):
    '''
    The Naive Bayes classifier. Both the Gaussian and
    multinomial implementations are provided as options
    (mode='gaussian' and mode='multinomial')
    '''

    def __init__(self, input, labels, method='gaussian'):
        super().__init__(input, labels)
        self.method = method
        self.model = self._get_model()

    def _get_model(self):
        '''
        Create the Naive Bayes model specified by the
        method parameter
        '''

        if self.method == 'gaussian':
            return GaussianNB()
        elif self.method == 'multinomial':
            return MultinomialNB()
        else:
            raise NaiveBayesNotDefined('Naive Bayes method \'' +self.method+
                '\' does not exist.')

    def fit(self, in_, out):
        '''
        Fit the naive bayes model to the specified data_matrix
        '''

        self.model.fit(in_, out)
        self.fit_yet = True

    def score(self, in_, out, method):
        '''
        Score the model on the provided input
        '''

        if not self.fit_yet:
            raise ModelNotFit('Must fit the Naive Bayes model before scoring.')

        else:
            if method == 'accuracy':
                return self.model.score(in_, out)
            elif method == 'fscore':
                pred = self.model.predict(in_)
                score = f_score(out, pred)
                return score
            elif method == 'both':
                pred = self.model.predict(in_)
                fscore = f_score(out, pred)
                accuracy = self.model.score(in_, out)
                return accuracy, fscore

class FFNN(Classifier):
    '''
    The feedforward neural network class. User specifies the
    number of layers and hidden units per layer, relu activation
    function, adam optimizer, and softmax classifier
    '''

    def __init__(self, input, labels, layers=[10,10], dropout=0.0):
        super().__init__(input, labels)
        self.layers = layers
        self.dropout = dropout
        self.num_classes = len(set(self.labels))
        self.model = self._create_network()

    def _create_network(self):
        '''
        Given the number of layers and number of hidden nodes per
        layer (specified in self.layers), as well as the dropout
        rate, create the neural network
        '''

        if len(self.layers) == 0:
            raise ValueError('Must specify at least one hidden layer for FFNN '+
                '(layers list cannot be empty).')

        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError('Dropout rate must be between 0 and 1.')

        #Create the model and add the first layer
        model = Sequential()
        model.add(Dense(self.layers[0], activation='relu', input_dim=len(self.input[0])))
        if self.dropout != 0.0:
            model.add(Dropout(self.dropout))

        #Add any additional hidden layers
        for i in range(1, len(self.layers)):
            model.add(Dense(self.layers[i], activation='relu'))
            if self.dropout != 0.0:
                model.add(Dropout(self.dropout))

        #Create the softmax classifier with one node per class label
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def fit(self, in_, out):
        '''
        Train the neural network on the specified data using
        the adam optimizer
        '''

        #Reset the model in between runs
        self.model = self._create_network()
        cat_out = to_categorical(out, num_classes=self.num_classes)
        adam = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

        #Set verbose=0 for no output during training, 2 for one line per epoch
        #Need to adjust steps_per_epoch depending on the dataset
        self.model.fit(in_, cat_out, epochs=10, verbose=2, steps_per_epoch=500)
        self.fit_yet = True

    def score(self, in_, out, method):
        '''
        Score the neural network on the provided input
        '''

        if not self.fit_yet:
            raise ModelNotFit('Must train the neural network before scoring.')

        else:

            if method == 'accuracy':
                cat_out = to_categorical(out, num_classes=self.num_classes)
                return self.model.evaluate(in_, cat_out, batch_size=128)[1]
            elif method == 'fscore':
                pred = self.model.predict(in_)
                pred = np.array([np.argmax(x) for x in pred])
                score = f_score(out, pred)
                return score
            elif method == 'both':
                cat_out = to_categorical(out, num_classes=self.num_classes)
                accuracy = self.model.evaluate(in_, cat_out, batch_size=128)[1]
                pred = self.model.predict(in_)
                pred = np.array([np.argmax(x) for x in pred])
                fscore = f_score(out, pred)

                return accuracy, fscore

class ModelNotFit(Exception):
    def __init__(self, message):
        self.message = message

class NaiveBayesNotDefined(Exception):
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
    layers = [4]
    drop = 0.0
    k = FFNN(in_, out, layers, drop)

    score, stdev, scores = k.k_fold_score()

    print(score)
    print(stdev)
    print(scores)
