import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

t = 0.65

f_40 = np.load("../data/feature_done.npy")
f_70 = np.load("../data/feature_70_done.npy")


def svd(f):
    F = f - np.mean(f, axis=0)
    U, S, V = np.linalg.svd(F, full_matrices=False)
    S = np.mat(np.diag(S))
    print(U.shape)
    print(S.shape)
    print(V.shape)
    W = U @ S
    return np.maximum(W, -W), W


def selectk(W):
    k_result = 0
    sum_1top = np.zeros((40, 1))
    for i in range(40):
        sum_1top += W[:, i]
    # print(sum_1top)
    #
    ava = sum_1top.mean()
    # for i in range(40):
    #     sum_1top[i] = ava

    # print(sum_1top)

    for k in range(2, 40):
        sum_jtok = np.zeros((40, 1))
        for i in range(2, k + 1):
            # sum_jtok += W[:, i] / sum_1top
            sum_jtok += W[:, i] / ava
        # print(k,sum_jtok.mean())
        if sum_jtok.mean() < t:
            k_result = k
    return k_result


# abs_W, W = svd(f)
# # print("-"*40 +'abs_W' +"-"*40)
# k = selectk(abs_W)
# print(k)
# W_result = W[:, :k + 1]
# print(W_result.shape)
#
# np.save("../data/abs_W", abs_W)
# np.save("../data/W_result", W_result)

def sklearn_pca(f_40,f_70,a):
    pca = PCA(0.95)
    pca.fit(f_40)
    resu_f = pca.transform(f_40)
    resu_f_70 = pca.transform(f_70)
    s = pca.transform(a)
    s1 = pca.transform(a[:2,:])
    print(pca.n_components)
    print(resu_f.shape)
    print(resu_f_70.shape)
    print(s.shape)
    print(s1.shape)
    print(np.allclose(s[:2,:], s1))

a = np.random.random((10,66))
print(a[:1,:].shape)
sklearn_pca(f_40,f_70,a)



