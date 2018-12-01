from dataprocessing import DataReader
from classifiers import Knn, FFNN, NaiveBayes
from feature_extraction import ClusterPCA, BiclusterExtractor
from utils import Pca
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

dr = DataReader("../data/nursery.csv")
in_, out = dr.run()

#Extract the features
#cpca = ClusterPCA(in_, out, method='kmeans', num_clusters=4, feats_per_cluster=2)
#feats = cpca.extract_features()

#bc = BiclusterExtractor(in_, out)
#bc_feats = bc.extract_features()
sample_size = 100
x1 = np.random.normal(0.1, 0.01, sample_size)
z1 = np.random.normal(0.5, 0.01, sample_size)
#y1 = np.random.normal(0.1, 0.01, sample_size)
y1 = np.random.uniform(0.09, 0.11, sample_size)

x2 = np.random.normal(0.1, 0.01, sample_size)
z2 = np.random.normal(10, 0.01, sample_size)
#y2 = np.random.normal(0.9, 0.25, sample_size)
y2 = np.random.uniform(0.1, 0.11, sample_size)

x3 = np.random.normal(0.1, 0.01, sample_size)
z3 = np.random.normal(10, 0.01, sample_size)
#y3 = np.random.normal(0.6, 0.25, sample_size)
y3 = np.random.uniform(0.09, 0.1, sample_size)

group_1 = np.vstack((x1,y1,z1)).T
group_2 = np.vstack((x2,y2,z2)).T
group_3 = np.vstack((x3,y3,z3)).T

data = np.vstack((group_1, group_2, group_3))
label_1 = np.empty(sample_size)
label_1.fill(1)
label_2 = np.empty(sample_size)
label_2.fill(2)
label_3 = np.empty(sample_size)
label_3.fill(3)
labels = np.hstack((label_1, label_2, label_3))

fig = plt.figure()
fig.suptitle('3D Synthetic Dataset')
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')
ax.scatter(data[:,0], data[:,1], data[:,2], c=labels)
plt.show()

cpca = ClusterPCA(data, labels, method='kmeans', num_clusters=2, feats_per_cluster=1)
feats = cpca.extract_features()

p = Pca(data, 2)
pca_feats = p.get_components()

f = plt.figure(1)
plt.scatter(feats[:,0], feats[:, 1], c=labels)

g = plt.figure(2)
plt.scatter(pca_feats[:,0], pca_feats[:, 1], c=labels)
plt.show()
#Set parameters for classifiers
layers = [4,3]

#Create and run classifier
#ffnn = FFNN(in_, out, layers)
#nb = NaiveBayes(in_, out)
'''knn = Knn(bc_feats, out, k=9)
knn.plot_k_scores(20)
final_score, stdev, scores = knn.k_fold_score(scoring_method='fscore')


print(final_score)
print(stdev)
print(scores)'''
