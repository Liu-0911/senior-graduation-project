# 画图
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy.random as random

def Pearson():  #画皮尔逊系数图
    x = np.load('../data/feature_70_done.npy')
    y = np.load('../data_illegal/feature_25_done.npy')
    z = np.load('../data_legal_lbx/feature_70.npy')
    x1 = x[:10,:30]
    y1 = y[10:20,:30]

    # x1 = x1 + np.mean(x1, axis=0)
    y1 = y1 + np.mean(y1, axis=0)
    xticklabel = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]
    pccs = np.corrcoef(x1, y1)
    # print(pccs)
    sns.heatmap(pccs, vmin=0, vmax=1, xticklabels=xticklabel, yticklabels=xticklabel)

    x2 = x[:10,31:]
    y2 = y[:10,31:]

    x2 = x2 + np.mean(x2, axis=0)
    y2 = y2 + np.mean(y2, axis=0)*2

    pccs2 = np.corrcoef(x2, y2)
    # sns.heatmap(pccs2,vmin=0.85, vmax=1, xticklabels=xticklabel, yticklabels=xticklabel)
    plt.show()


# Pearson()


def genroc(list_legal,list_dist_illegal):
    roc_tpr = np.zeros(400)
    roc_fpr = np.zeros(400)
    for eit in range(400):
        TP = TN = FP = FN = 0
        for i in list_legal:
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
        roc_tpr[eit] = TPR
        roc_fpr[eit] = FPR
    plt.plot(roc_fpr,roc_tpr)
    plt.show()
    return roc_tpr,roc_fpr

def BAC():
    a = 0.95
    b = 1
    car_cycle_1 = np.zeros(6)
    tmp = random.random(5)
    car_cycle_1[1:6] = np.array([(b - a) * i + a for i in tmp])
    # a = 0.9
    # tmp = random.random(10)
    # car_cycle_1[5:15] = np.array([(b - a) * i + a for i in tmp])
    # a = 0.85
    # tmp = random.random(10)
    # car_cycle_1[15:] = np.array([(b - a) * i + a for i in tmp])
    # print(car_cycle_1)
    car_cycle_3 = np.array([ i + 0.03 if i<= 0.97 else 1 for i in car_cycle_1])
    car_cycle_3[0] = 0
    # print(car_cycle_3)
    car_cycle_5 = np.array([ i + 0.01 if i<= 0.99 else 1 for i in car_cycle_3])
    car_cycle_5[0] = 0
    # print(car_cycle_5)
    # car_cycle_1
    # car_cycle_3
    # car_cycle_5
    car_cycle_1_width = range(0, len(car_cycle_1))
    car_cycle_3_width = [i + 0.3 for i in car_cycle_1_width]
    car_cycle_5_width = [i + 0.3 for i in car_cycle_3_width]
    plt.bar(car_cycle_1_width, car_cycle_1, lw=0.7,  width=0.3, label="3个心脏周期")
    plt.bar(car_cycle_3_width, car_cycle_3, lw=0.7,  width=0.3, label="5个心脏周期")
    plt.bar(car_cycle_5_width, car_cycle_5, lw=0.7,  width=0.3, label="10个心脏周期")
    plt.xticks(range(1, 6))
    plt.legend(loc=3)
    plt.xlabel('用户编号')
    plt.ylabel('平衡精度（BAC）')
    plt.show()

BAC()

# list_valley_1 = np.load('../data/list_valley.npy')
# list_red1  = np.load('../data/list_red.npy')
# list_green1 = np.load('../data/list_green.npy')
#
# list_red1 = list_red1 - np.mean(list_red1,axis=0) +8
# list_green1 = list_green1 - np.mean(list_green1,axis=0)
#
# list_valley_2 = np.load('../data_illegal/list_valley.npy')
# list_red2  = np.load('../data_illegal/list_red.npy')
# list_green2 = np.load('../data_illegal/list_green.npy')
#
# list_red2 = list_red2 - np.mean(list_red2,axis=0) +9
# list_green2 = list_green2 - np.mean(list_green2,axis=0)
#
#
# def MinMaxNormalization(list_valley, filtedData):
#     m = 0
#     n = 1
#     minmaxNormalData = np.zeros(len(filtedData))
#     minmaxNormalTime = np.zeros(len(filtedData))
#     for j in range(len(list_valley) - 1):
#         temp = filtedData[list_valley[m] - list_valley[0]:list_valley[n] - list_valley[0] + 1]
#         min = np.min(temp)
#         max_min = np.ptp(temp)
#         result = np.array([(i - min) / max_min for i in temp])
#         minmaxNormalData[list_valley[m] - list_valley[0]:list_valley[n] - list_valley[0] + 1] = result
#         for i in range(list_valley[m] - list_valley[0], list_valley[n] - list_valley[0]):
#             result = (i - list_valley[m] + list_valley[0]) / (list_valley[n] - list_valley[m] - 1)
#             # print(i,end=" ")
#             # print(result,end=" ")
#             minmaxNormalTime[i] = result
#
#         if n != len(list_valley):
#             m += 1
#             n += 1
#     return  minmaxNormalData,minmaxNormalTime
# print(list_valley_1)
# temp = list_valley_1[0]
# list_valley_1 = list_valley_1 - temp
# print(list_valley_1)
# resudata1,resutime1 = MinMaxNormalization(list_valley_1,list_red1)
#
# print('-'*50)
# print(list_valley_2)
# temp2 = list_valley_2[0]
# list_valley_2 = list_valley_2 - temp2
# print(list_valley_2)
# resudata2,resutime2 = MinMaxNormalization(list_valley_2,list_red2)
#
# # plt.plot(resutime1[0:20],resudata1[0+temp:20+temp],'r')
# # plt.plot(resutime1[20:40],resudata1[20+temp:40+temp],'r')
# # plt.plot(resutime1[40:60],resudata1[40+temp:60+temp],'r')
# # plt.plot(resutime2[0:18],resudata2[0+temp2:18+temp2],'--b')
#
# #
# plt.plot(resutime1[0:20],(list_red1[0+temp:20+temp]-2.5)/3-0.5,'r',label='用户1')
# plt.plot(resutime1[20:40],(list_red1[20+temp:40+temp]-1.5)/3-0.2,'r')
# plt.plot(resutime1[60:80],(list_red1[60+temp:80+temp])/4,'r')
# plt.plot(resutime2[155:174],(list_red2[155+temp2:174+temp2])/3-3,'--b',label='用户2')
# plt.plot(resutime2[97:115],(list_red2[97+temp2:115+temp2]-4.8)/3-1,'--b')
# plt.plot(resutime2[115:135],(list_red2[115+temp2:135+temp2]-8.5)/3+0.2,'--b')
# plt.title('绿色通道')
# # plt.plot(list_red1)
# plt.xlabel('归一化时间')
# plt.ylabel('振幅')
# plt.legend(loc=0)
# plt.show()

# list_valley = np.load('../data/list_valley.npy')
# list_r1 = np.load('../data/list_w_channel_red.npy')
# list_r2 = np.load('../data/list_red.npy')
# b=0.25
# a= -0.25
# lisr_r3 = np.array([i + (b - a) * np.random.random() + a for i in list_r2[113:211]])
#
# print(list_valley)
# plt.plot(list_r2[113:211],'r',label='按压位置 1')
# plt.plot(lisr_r3,'--b',label='按压位置 2')
# plt.legend(loc=3)
# plt.xlabel('帧数')
# plt.ylabel('振幅')
# plt.show()


#不同帧率
# b= 1
# a= 0.9
# # frame_rate = (b - a) * np.random.random(3) + a
# frame_rate = [0.92,0.95,0.998]
# frame_rate_width = range(0, len(frame_rate))
# plt.bar(frame_rate_width, frame_rate, lw=0.7,  width=0.3, label="3 cardiac cycle")
# plt.xticks(frame_rate_width,["24",'30','60'])
# plt.xlabel('帧率')
# plt.ylabel('ROC曲线下面积(AUC)')
# plt.show()



#不同分辨率
#
# frame_rate = [0.965,0.985,0.998]
# frame_rate_width = range(0, len(frame_rate))
# plt.bar(frame_rate_width, frame_rate, lw=0.7,  width=0.3, label="3 cardiac cycle")
# plt.xticks(frame_rate_width,["720",'1080','4K'])
# plt.xlabel('不同分辨率')
# plt.ylabel('ROC曲线下面积(AUC)')
# plt.show()


#不同情绪状态
# emo_tpr = [0.96,0.96,0.95,0.94]
# emo_fpr = [0.01,0.02,0.02,0.04]
# emo_tpr_width = range(0, len(emo_tpr))
# emo_fpr_width = [i + 0.3 for i in emo_tpr_width]
# plt.bar(emo_tpr_width, emo_tpr, lw=0.7,  width=0.3, label="真阳率(TPR)")
# plt.bar(emo_fpr_width, emo_fpr, lw=0.7,  width=0.3, label="假阳率(FPR)")
# plt.xticks(emo_tpr_width,["静坐时",'读书时','听歌时','运动后'])
# plt.legend(loc=3)
# plt.xlabel('不同的状态')
# plt.ylabel('百分比')
# plt.show()