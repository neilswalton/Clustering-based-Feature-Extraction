from dataprocessing import DataReader
from classifiers import Knn, FFNN
import numpy as np

dr = DataReader("../data/nursery.csv")
in_, out = dr.run()

classes = set(out)


for i, lab in enumerate(out):
    if lab==2:
        out = np.delete(out, i, axis=0)
        in_ = np.delete(in_, i, axis=0)

'''for c in classes:
    count = 0
    for item in out:
        if item == c:
            count += 1

    print (c, count)'''

for i, item in enumerate(out):
    if item == 3:
        out[i]=2
    elif item==4:
        out[i]=3

layers = [4,3]

knn = FFNN(in_, out, layers)
final_score, stdev, scores = knn.k_fold_score()


print(final_score)
print(stdev)
print(scores)
