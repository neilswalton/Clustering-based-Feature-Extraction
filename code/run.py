from dataprocessing import DataReader
from classifiers import Knn, FFNN, NaiveBayes
from feature_extraction import ClusterPCA
from utils import Pca
import numpy as np

dr = DataReader("../data/nursery.csv")
in_, out = dr.run()

#Extract the features
cpca = ClusterPCA(in_, out, method='kmeans', num_clusters=4, feats_per_cluster=2)
feats = cpca.extract_features()

#Set parameters for classifiers
layers = [4,3]

#Create and run classifier
#ffnn = FFNN(in_, out, layers)
#nb = NaiveBayes(in_out)
knn = FFNN(feats, out, layers)
final_score, stdev, scores = knn.k_fold_score(scoring_method='fscore')


print(final_score)
print(stdev)
print(scores)
