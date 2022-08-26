# 测试SVD分解和PCA分析

import numpy as np

m = 3
n = 5

a = np.arange(1,m*n+1).reshape(m,n)
# print(a)
# a_new = a - np.mean(a, axis=0)
a_new = a
print(a_new)
U, S, V = np.linalg.svd(a_new,full_matrices=False)
S = np.mat(np.diag(S))
print('-'*20+"U"+'-'*20)
print(U,U.shape)
print('-'*20+"S"+'-'*20)
print(S,S.shape)
print('-'*20+"V"+'-'*20)
print(V,V.shape)
# print(U@S@V)
# print(U.shape,type(U))
# print(S.shape,type(S))
# print(V.shape,type(V))
# print(a_new @ V.T)

U1, S1, V1 = np.linalg.svd(a_new)
S1 = np.mat(np.diag(S1))
print('-'*20+"U1"+'-'*20)
print(U1,U1.shape)
print('-'*20+"S1"+'-'*20)
print(S1,S1.shape)
print('-'*20+"V1[:m,:]"+'-'*20)
print(V1[:m,:],V1[:m,:].shape)
print('-'*20+"V1"+'-'*20)
print(V1,V1.shape)






