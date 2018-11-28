from dataprocessing import DataReader
from classifiers import Knn, FFNN, NaiveBayes
from feature_extraction import ClusterPCA, BiclusterExtractor
from utils import Pca
import numpy as np

dr = DataReader("../data/nursery.csv")
in_, out = dr.run()

#Extract the features
#cpca = ClusterPCA(in_, out, method='kmeans', num_clusters=4, feats_per_cluster=2)
#feats = cpca.extract_features()

bc = BiclusterExtractor(in_, out)
bc_feats = bc.extract_features()

#Set parameters for classifiers
layers = [4,3]

#Create and run classifier
#ffnn = FFNN(in_, out, layers)
#nb = NaiveBayes(in_, out)
knn = Knn(bc_feats, out, k=9)
knn.plot_k_scores(20)
final_score, stdev, scores = knn.k_fold_score(scoring_method='fscore')


print(final_score)
print(stdev)
print(scores)
