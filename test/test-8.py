import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
#
# iris = load_iris()
# y = iris.target
# X = iris.data
# #作为数组，X是几维？
# print(X.shape)
# #作为数据表或特征矩阵，X是几维？
#
# pd.DataFrame(X)
#
# f = np.load('../data/feature.npy')
# print("1")
# print(f)

m = 40
n = 66

# a = np.arange(1,m*n+1).reshape(m,n)
a = np.random.randint(-10,11,(m, n))
print(a)
a=np.maximum(a,-a)
print(a)