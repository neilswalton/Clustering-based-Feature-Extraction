from dataprocessing import DataReader
from classifiers import Knn, FFNN, NaiveBayes
from feature_extraction import ClusterPCA, BiclusterExtractor, FeatureCluster
from utils import Pca
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import time
import csv

#Set dataset parameters
dataset = 'nursery'
num_clusters = 2

#Set parameters for classifiers
layers = [4] #List of number of nodes per hidden layer
list_of_ks = [9,10,2,19,19] #If k param needs to change between different extracted features

dr = DataReader('../data/'+dataset+'.csv')
in_, out = dr.run()

#Extract the features
cpca = ClusterPCA(in_, out, method='kmeans', num_clusters=2, feats_per_cluster=2)
cpca_feats = cpca.extract_features()

bc = BiclusterExtractor(in_, out, n=num_clusters)
bc_feats = bc.extract_features()

fc = FeatureCluster(in_, out, method="kmeans", num_clusters=num_clusters, _type="soft")
fc_soft_feats = fc.extract_features()

fc2 = FeatureCluster(in_, out, method="kmeans", num_clusters=num_clusters, _type="mixed")
fc_mixed_feats = fc2.extract_features()

features = [in_, cpca_feats, bc_feats, fc_soft_feats, fc_mixed_feats]
feature_labels = ['All', 'CPCA', 'BC', 'FC_Soft', 'FC_Mixed']

file_out_name = dataset + '_results.csv'
file_out = open(file_out_name, 'w')
fieldnames = ['method', 'classifier', 'score_type', 'mean', 'stdev', 'time', 'all_scores']
writer = csv.writer(file_out)
writer.writerow(fieldnames)

for i, feats in enumerate(features):
    k = list_of_ks[i]
    method = feature_labels[i]
    knn = Knn(feats, out, k)
    #ffnn = FFNN(in_, out, layers)
    nb = NaiveBayes(in_, out)

    knn_start = time.time()
    k_acc_mean, k_acc_std, k_accs, k_f_mean, k_f_std, k_fs = knn.k_fold_score(10, 'both')
    knn_stop = time.time()
    ffnn_start = time.time()
    #f_acc_mean, f_acc_std, f_accs, f_f_mean, f_f_std, f_fs = ffnn.k_fold_score(10, 'both')
    ffnn_stop = time.time()
    nb_start = time.time()
    n_acc_mean, n_acc_std, n_accs, n_f_mean, n_f_std, n_fs = nb.k_fold_score(10, 'both')
    nb_stop = time.time()

    knn_time = knn_stop - knn_start
    ffnn_time = ffnn_stop - ffnn_start
    nb_time = nb_stop - nb_start

    writer.writerow([method, 'knn', 'accuracy', k_acc_mean,
        k_acc_std, knn_time, k_accs])
    writer.writerow([method, 'knn', 'fscore', k_f_mean,
        k_f_std, knn_time, k_fs])

    '''writer.writerow([method, 'ffnn', 'accuracy', f_acc_mean,
        f_acc_std, ffnn_time, f_accs])
    writer.writerow([method, 'ffnn', 'fscore', f_f_mean,
        f_f_std, ffnn_time, f_fs])'''

    writer.writerow([method, 'nb', 'accuracy', n_acc_mean,
        n_acc_std, nb_time, n_accs])
    writer.writerow([method, 'nb', 'fscore', n_f_mean,
        n_f_std, nb_time, n_fs])








#
