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
tune_k = False
dataset = 'mushroom'
num_clusters = 6
dr = DataReader('../data/'+dataset+'.csv')
in_, out = dr.run()
hidden_size = 100#round((num_clusters + len(set(out)))/2)

#Set parameters for classifiers
layers = [hidden_size] #List of number of nodes per hidden layer
list_of_ks = [9,10,2,19,19] #If k param needs to change between different extracted features

#Extract the features
cpca = ClusterPCA(in_, out, method='kmeans', num_clusters=num_clusters, feats_per_cluster=2)
cpca_feats = cpca.extract_features()

bc = BiclusterExtractor(in_, out, n=num_clusters)
bc_feats = bc.extract_features()

fc = FeatureCluster(in_, out, method="kmeans", num_clusters=num_clusters, _type="soft")
fc_soft_feats = fc.extract_features()

fc2 = FeatureCluster(in_, out, method="kmeans", num_clusters=num_clusters, _type="mixed")
fc_mixed_feats = fc2.extract_features()

#Tune k for knn
if tune_k:
    all_knn = Knn(in_, out, k=1)
    cpca_knn = Knn(cpca_feats, out, k=1)
    bc_knn = Knn(bc_feats, out, k=1)
    fc_soft_knn = Knn(fc_soft_feats, out, k=1)
    fc_mixed_knn = Knn(fc_mixed_feats, out, k=1)

    all_knn.plot_k_scores(20)
    cpca_knn.plot_k_scores(20)
    bc_knn.plot_k_scores(20)
    fc_soft_knn.plot_k_scores(20)
    fc_mixed_knn.plot_k_scores(20)

else:

    features = [in_, cpca_feats, bc_feats, fc_soft_feats, fc_mixed_feats]
    feature_labels = ['All', 'CPCA', 'BC', 'FC_Soft', 'FC_Mixed']

    file_out_name = dataset + '_results.csv'
    file_out = open(file_out_name, 'w')
    fieldnames = ['method', 'classifier', 'score_type', 'mean', 'stdev', 'time', 'all_scores']
    writer = csv.writer(file_out)
    writer.writerow([num_clusters])
    writer.writerow(fieldnames)

    for i, feats in enumerate(features):
        k = list_of_ks[i]
        method = feature_labels[i]
        knn = Knn(feats, out, k)
        ffnn = FFNN(in_, out, layers)
        nb = NaiveBayes(in_, out)

        knn_start = time.time()
        print("knn")
        k_acc_mean, k_acc_std, k_accs, k_f_mean, k_f_std, k_fs = knn.k_fold_score(10, 'both')
        knn_stop = time.time()
        ffnn_start = time.time()
        print("ffnn")
        f_acc_mean, f_acc_std, f_accs, f_f_mean, f_f_std, f_fs = ffnn.k_fold_score(10, 'both')
        ffnn_stop = time.time()
        nb_start = time.time()
        print("nb")
        n_acc_mean, n_acc_std, n_accs, n_f_mean, n_f_std, n_fs = nb.k_fold_score(10, 'both')
        nb_stop = time.time()

        knn_time = knn_stop - knn_start
        ffnn_time = ffnn_stop - ffnn_start
        nb_time = nb_stop - nb_start

        writer.writerow([method, 'knn', 'accuracy', k_acc_mean,
            k_acc_std, knn_time, k_accs])
        writer.writerow([method, 'knn', 'fscore', k_f_mean,
            k_f_std, knn_time, k_fs])

        writer.writerow([method, 'ffnn', 'accuracy', f_acc_mean,
            f_acc_std, ffnn_time, f_accs])
        writer.writerow([method, 'ffnn', 'fscore', f_f_mean,
            f_f_std, ffnn_time, f_fs])

        writer.writerow([method, 'nb', 'accuracy', n_acc_mean,
            n_acc_std, nb_time, n_accs])
        writer.writerow([method, 'nb', 'fscore', n_f_mean,
            n_f_std, nb_time, n_fs])








#
