import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

f = np.load("../data/feature_done.npy")

pca = PCA(0.9)
pca.fit()


W = np.load("../data/abs_W.npy")
f_70 = np.load("../data/feature_70_done.npy")

print(f_70.shape)
print(W.shape)
# F = f_70 @ W
# print(F.shape)

def sklearn_pca(f):
    pca = PCA()
    resu_all = pca.fit_transform(f)
    print("返回模型的各个特征向量", pca.components_)  # 返回模型的各个特征向量（原始数据）
    print("返回各个成分各自的方差百分比（贡献率）", pca.explained_variance_ratio_)  # 返回各个成分各自的方差百分比（贡献率）
    print(pca.components_.shape)
    print(pca.explained_variance_ratio_.shape)
    print(resu_all.shape)
    pca_22 = PCA(22)
    resu_22 = pca_22.fit_transform(f)
    print(resu_22.shape)
    pca_auto = PCA(0.9)
    resu_auto = pca_auto.fit_transform(f)
    print(pca_auto.n_components_)
    print(resu_all.shape)
    print(resu_22.shape)
    print(resu_auto.shape)
    print(np.allclose(resu_all[:,:22],resu_22))
    print(np.allclose(resu_all[:,:7],resu_auto))

sklearn_pca(f)
