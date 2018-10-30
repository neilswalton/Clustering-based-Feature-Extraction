from dataprocessing import DataReader
from classifiers import Knn, FFNN
import numpy as np

dr = DataReader("../data/nursery.csv")
in_, out = dr.run()

classes = set(out)

layers = [4,3]

knn = FFNN(in_, out, layers)
final_score, stdev, scores = knn.k_fold_score()


print(final_score)
print(stdev)
print(scores)
