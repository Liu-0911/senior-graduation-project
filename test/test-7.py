# U, S, Vt =np.linalg.svd(A) 其中U Vt都为正交矩阵，最后的主成分即为U*S 或者A*Vt.T


from sklearn.utils.extmath import svd_flip
from  sklearn.decomposition import PCA
import numpy as np

m = 40
n = 66
k = 3
A = np.arange(1, m * n + 1).reshape(m, n)
# A = np.random.rand(m, n)
# print(A)

#sklearn中的PCA就是调用np.linalg.svd(a_new,full_matrices=False)得到USV三个矩阵 sklearn在处理数据之前会进行中心化


def sklearn_pca(A,k):
    print("-----------------sklearn------------------")
    # print(A)
    pca = PCA(k)
    A1 = pca.fit_transform(A)
    # print(pca.n_samples_)    #n_samples_ = m
    # print(pca.n_features_)   #n_features_ = n
    # print(pca.components_)
    # print(pca.n_components)

    print(A1)
    print("-----------------sklearn------------------")
    return A1

def numpy_svd(a,k):
    # print(a)
    a_new = a - np.mean(a, axis=0)
    # print(a_new)
    U, S, V = np.linalg.svd(a_new)
    S = np.mat(np.diag(S))
    print(f'A.shape={a_new.shape}')
    print(f'U.shape={U.shape}')
    print(f'S.shape={S.shape}')
    print(f'V.shape={V.shape}')
    # print(U.shape)
    # print(S.shape)
    # print(V)
    # print(S)
    Vh = V.T
    A1 = np.dot(a_new, Vh[:, :k])
    # A11 = np.dot(a_new,(V[:k,:]).T)
    print(A1)
    print("-----------------------------------")
    # print(V@Vh)
    print("-----------------------------------")
    return A1

def numpy_svd_F(a,k):
    # print(a)

    a_new = a - np.mean(a, axis=0)
    # print(a_new)
    U, S, V = np.linalg.svd(a_new,full_matrices=False)
    S = np.mat(np.diag(S))
    U1,v1 = svd_flip(U,V)
    W = U@S
    print(f'A.shape={a_new.shape}')
    print(f'U.shape={U.shape}')
    print(f'S.shape={S.shape}')
    print(f'V.shape={V.shape}')
    print(f'W.shape={W.shape}')
    # print(S)
    print("------------US---------------------")
    Vh = V.T
    US = np.dot(U,S[:,:k])
    # print(US)
    print("------------A---------------------")
    A = np.dot(a_new, Vh[:, :k])

    # print(A)
    print(f'US和A是否相等{np.allclose(A,US)}')
    # print(U@S@V)
    print("------------US1---------------------")
    US1 = np.dot(U, S)
    # print(US1)
    print("------------A1---------------------")
    A1 = np.dot(a_new, Vh)
    # print(A1)
    print(f'US1和A1是否相等{np.allclose(A1, US1)}')
    # print(A11)
    print("-----------------------------------")
    resu = U @ S @ V
    print(np.allclose(resu,a_new))
    # print(V@V.T)
    # print(resu)
    # print(a_new)
    return US

# def numpy_ori(a):
#     X_new = a - np.mean(a, axis=0)
#     C =  np.cov(X_new.T,rowvar=False)
#     eig_vals, eig_vecs = np.linalg.eigh(C)
#     idx = np.argsort(eig_vals)[::-1]
#     eig_vecs = eig_vecs[:, idx]
#     eig_vals = eig_vals[idx]
#     # X_pca = np.dot(X_new, eig_vecs)
#     # print(X_new.shape)
#     # print(C.shape)
#     # print(eig_vecs.shape)
#     print(X_pca)
#     print("-----------------------------------")

def comS(a):
    a_new = a - np.mean(a, axis=0)
    U, S, V = np.linalg.svd(a_new)
    U1, S1, V1 = np.linalg.svd(a_new,full_matrices=False)
    print(S)
    print(S1)

# sklearn_pca_a1 = sklearn_pca(A,k)
# numpy_svd_a1 = numpy_svd(A,k)
numpy_svd_F(A,k)
# print(sklearn_pca_a1 - numpy_svd_a1)
# comS(A)
# numpy_ori(A)

