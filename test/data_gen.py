import random

import numpy as np
from sklearn.decomposition import PCA
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

feature_lxy_ori = np.load('../data/feature_70_done.npy')
feature_lxy_ori = np.around(feature_lxy_ori,5)
# print(feature_lxy_ori)
# print(feature_lxy_ori.shape)


def showposof1(feature_lxy_ori):  #用于显示原特征中1的位置
    for i in range(feature_lxy_ori.shape[0]):
        tmp = feature_lxy_ori[i]
        index_1 = np.where(tmp == 1)[0]
        print(i,index_1,index_1.shape)

def set1(feature):   #恢复gen的特征数组中的1
    feature[:,20] = 1
    feature[:,35] = 1
    feature[:, 41] = 1
    feature[:, 53] = 1
    feature[:, 59] = 1
    return feature

def gen(feature_ori):
    b = 0.05
    a = -0.05
    diff1 = (b-a) * np.random.random(feature_lxy_ori.shape) + a  #产生混淆矩阵
    feature_gen = feature_ori + diff1
    return feature_gen
# diff = np.zeros(feature_lxy_ori.shape)
# showposof1(feature_lxy_ori)


feature_lxy_gen1 = gen(feature_lxy_ori)
feature_lxy_gen1 = set1(feature_lxy_gen1)
illegal_feature = np.load("../data_illegal/feature_25_done.npy")

def sklearn_pca(legal_feature,legal_feature_70,illegal_feature):
    # pca_1 = PCA()
    # pca_1.fit(legal_feature)
    # plt.plot(np.arange(1,41), np.cumsum(pca_1.explained_variance_ratio_))
    # plt.xlabel('主成分数量')
    # plt.ylabel('累计可解释方差贡献率')
    # plt.show()
    pca = PCA(0.95)
    pca.fit(legal_feature)
    resu_lf = pca.transform(legal_feature)
    resu_lf_70 = pca.transform(legal_feature_70)
    resu_ill = pca.transform(illegal_feature)
    return resu_lf,resu_lf_70,resu_ill,resu_ill.shape[1]

def CalLegalDist(resu_lf_70,n,num_of_features):
    resu_legal = resu_lf_70[:n,:]
    print(resu_legal.shape[0])
    list_dist_legal = np.zeros(resu_legal.shape[0])
    for i in range(resu_legal.shape[0]):
        resu = np.zeros(num_of_features)
        for j in range(resu_lf_70.shape[0]):
            resu += resu_lf_70[j, :] - resu_legal[i, :]
            # print(resu_lf_70[j, :])
            # print(resu_legal[i, :])
        # print(resu)
        # print('-'*100)
        list_dist_legal[i] = np.linalg.norm(resu) / resu_lf_70.shape[0]
    return list_dist_legal

def CalIllegalDist(resu_lf_70,resu_ill,num_of_features):
    print(resu_ill.shape[0])
    list_dist_illegal = np.zeros(resu_ill.shape[0])
    for i in range(resu_ill.shape[0]):
        resu = np.zeros(num_of_features)
        for j in range(resu_lf_70.shape[0]):
            resu += resu_lf_70[j, :] - resu_ill[i, :]
        list_dist_illegal[i] = np.linalg.norm(resu) / resu_lf_70.shape[0]
    return list_dist_illegal

def RecuGenYouDenJ(list_dist_legal,list_dist_illegal):
    maxYouDenJ = max_eit = 0
    for eit in np.arange(1,15,0.5):
        TP = TN = FP = FN =  0
        for i in range(len(list_dist_legal)):
            if list_dist_legal[i] <= eit:
                TP += 1             # 合法用户验证通过（实际正确被预测正确）
            else:
                FN += 1             # 合法用户验证未通过（实际正确被预测错误）
        for i in range(len(list_dist_illegal)):
            if list_dist_illegal[i] <= eit:
                FP += 1             # 非法用户验证通过（实际错误被预测正确）
            else:
                TN += 1             # 非法用户验证未通过（实际错误被预测错误）
        Se = TP/(TP+FN)
        Sp = TN/(FP+TN)
        YouDenJ = Se + Sp -1
        print(eit, YouDenJ)
        if YouDenJ > maxYouDenJ:
            maxYouDenJ = YouDenJ
            max_eit = eit
    print(f'max_eit={max_eit}, maxYouDenJ={maxYouDenJ}')
    return max_eit,maxYouDenJ

resu_lf,resu_lf_70,resu_ill,num_of_features = sklearn_pca(feature_lxy_ori[:40,:],feature_lxy_ori,illegal_feature)
list_dist_legal = CalLegalDist(resu_lf_70,10,num_of_features)
list_dist_illegal = CalIllegalDist(resu_lf_70,resu_ill,num_of_features)