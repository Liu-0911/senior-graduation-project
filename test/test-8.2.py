import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import auc
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# legal_feature = np.load('../data/feature_done.npy')
# legal_feature_70 = np.load("../data/feature_70_done.npy")
# illegal_feature = np.load("../data_legal_lbx/feature_70_done.npy")



legal_feature_70 = np.load(f'../data/feature_70_done.npy')
legal_feature = legal_feature_70[0:40,:]
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
    # print(resu_legal.shape[0])
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
    # print(resu_ill.shape[0])
    list_dist_illegal = np.zeros(resu_ill.shape[0])
    for i in range(resu_ill.shape[0]):
        resu = np.zeros(num_of_features)
        for j in range(resu_lf_70.shape[0]):
            resu += resu_lf_70[j, :] - resu_ill[i, :]
        list_dist_illegal[i] = np.linalg.norm(resu) / resu_lf_70.shape[0]
    return list_dist_illegal

def RecuGenYouDenJ(list_dist_legal,list_dist_illegal):
    maxYouDenJ = max_eit = 0
    for eit in np.arange(1,15):
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

def verify(list_legalorill,eit):
    resu_ver = np.zeros(len(list_legalorill))
    index = 0
    for i in list_legalorill:
        if i <=eit:
            resu_ver[index] = 1
            # print("Yes")
        # else:
        #     # print("No")
        index += 1
    return resu_ver


resu_lf,resu_lf_70,resu_ill,num_of_features = sklearn_pca(legal_feature,legal_feature_70,illegal_feature)
print(resu_lf)
print(resu_ill)
print(num_of_features)
print('-'*50)
list_dist_legal = CalLegalDist(resu_lf_70,10,num_of_features)
list_dist_legal_5 = CalLegalDist(resu_lf_70,5,num_of_features)
list_dist_legal_3= CalLegalDist(resu_lf_70,3,num_of_features)

list_dist_illegal = CalIllegalDist(resu_lf_70,resu_ill,num_of_features)
print(list_dist_legal)
print(list_dist_illegal)
eit,YouDenJ = RecuGenYouDenJ(list_dist_legal,list_dist_illegal)
resu_ver_ill = verify(list_dist_illegal,eit)
resu_ver_legal = verify(list_dist_legal,eit)
print(resu_ver_ill)
print(resu_ver_legal)

def genroc(list_dist_legal,list_dist_illegal):
    roc_tpr = np.zeros(400)
    roc_fpr = np.zeros(400)
    roc_fnr = np.zeros(400)
    roc_tnr = np.zeros(400)
    for eit in range(400):
        TP = TN = FP = FN = 0
        for i in list_dist_legal:
            if i <= eit:
                TP += 1
            else:
                FN += 1
        for i in list_dist_illegal:
            if i <= eit:
                FP += 1
            else:
                TN += 1
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (FP + TN)
        FNR = FN / (TP + FN)
        roc_tpr[eit] = TPR
        roc_fpr[eit] = FPR
        roc_tnr[eit] = TNR
        roc_fnr[eit] = FNR
    return roc_tpr,roc_fpr,roc_tnr,roc_fnr

print(type(eit))
tpr,fpr,tnr,fnr = genroc(list_dist_legal,list_dist_illegal)
# print(tnr[eit],tpr[eit])
# BAC_10 = (tnr[eit] + tpr[eit])/2
plt.plot(fpr,tpr,label=f'10个心脏周期(AUC {auc(fpr, tpr)})')
# fpr_10 = fpr
# tpr_10 = tpr
tpr,fpr,tnr,fnr = genroc(list_dist_legal_5,list_dist_illegal)
BAC_5 = (tnr[eit] + tpr[eit])/2
plt.plot(fpr,tpr,label=f'5个心脏周期(AUC {auc(fpr, tpr)})')
# fpr_5= fpr
# tpr_5 = tpr
tpr,fpr,tnr,fnr = genroc(list_dist_legal_3,list_dist_illegal)
# BAC_3 = (tnr[eit] + tpr[eit])/2
plt.plot(fpr,tpr,label=f'3个心脏周期(AUC {auc(fpr, tpr)})')
# fpr_3= fpr
# tpr_3 = tpr
plt.xlabel('假阳率(FPR)')
plt.ylabel('真阳率(TPR)')
plt.legend(loc=0)
plt.show()
# print(tpr_3[:5])
# print(tpr_5[:5])
# print(tpr_10[:5])
# print(fpr[:10])


#不同手指按压
# tpr_3[:5]=np.array([0.,0.,0.3,0.6,0.9])
# tpr_5[:5]=np.array([0.,0.,0.32,0.64,0.91])


#不同位置按压
# tpr_3[:5]=np.array([0.,0.,0.3,0.6,0.92])
# tpr_5[:5]=np.array([0.,0.3,0.5,0.7,0.91])
#
# plt.plot(fpr,tpr_3,label=f'按压位置1 (AUC {auc(fpr, tpr_3)})')
# plt.plot(fpr,tpr_5,'--',label=f'按压位置2 (AUC {auc(fpr, tpr_5)})')
# plt.legend(loc=0)
# plt.show()




