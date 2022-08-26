import cv2
import numpy as np
import os
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal

from detecta import detect_peaks



# # set minimum peak height = 0 and minimum peak distance = 20
# list_test1_mpd1 = detect_peaks(test1,mpd=1,valley= True, show=False)
# list_test1_mpd35 = detect_peaks(test1,mpd=35,valley= True, show=False)
# list_test2_mpd1 = detect_peaks(test2,mpd=1,valley= True, show=False)
# list_test2_mpd35 = detect_peaks(test2,mpd=35,valley= True, show=False)
# print(list_test1_mpd1,list_test1_mpd1.size)
# print(list_test1_mpd35,list_test1_mpd35.size)
# print(list_test2_mpd1,list_test2_mpd1.size)
# print(list_test2_mpd35,list_test2_mpd35.size)
#
#
# # fig,axes = plt.subplots(4,1,figsize = (30,20) , dpi = 180)
# # # axes[0].plot(test1,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery = list_test1_mpd1)
# # # axes[0].set(xlabel = "date" , ylabel = "red_ava" ,title = "test1-mpd1")
# # # axes[1].plot(test1,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery = list_test1_mpd35)
# # # axes[1].set(xlabel = "date" , ylabel = "red_ava" ,title = "test1-mpd35")
# # axes[2].plot(test2,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery = list_test2_mpd1)
# # axes[2].set(xlabel = "date" , ylabel = "red_ava" ,title = "test2-mpd1")
# # axes[3].plot(test2,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery = list_test2_mpd35)
# # axes[3].set(xlabel = "date" , ylabel = "red_ava" ,title = "test2-mpd35")
# # plt.plot(test2,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery = list_test2_mpd1)
# # plt.show()
#
#
#
#
#
# list_test = test2[0:200]
#
#
# list_test2_mpd1 = detect_peaks(list_test,valley= True, show=False,edge=None)
# plt.figure(figsize=(14, 7), dpi=500)
# plt.plot(list_test,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery = list_test2_mpd1)
#
# plt.show()

# list_valley = [34, 54, 75, 97]
# m = 0
# n = 1
# res = np.zeros(98)
# for j in range(len(list_valley) -1 ):
#     # print(f"j={j}")
#     # print(f"m={m},n={n}")
#     for i in range(list_valley[m] , list_valley[n] ):
#         result = (i-list_valley[m] )/(list_valley[n] - list_valley[m] -1)
#         res[i]=result
#         # print(i,end=" ")
#         # print(result,end=" ")
#     print()
#     if n != len(list_valley):
#         m += 1
#         n += 1
# print(res)
#
# tem = np.zeros(20)
# for j in range(0,10):
#     tem[j]=j
# print(tem)


#
# x = [0,   0.1,                 0.2,                 0.3,                 0.4,                0.5,               0.6,                0.7,               0.8,                0.9,                1,                  0.1,                 0.2,                 0.3,                 0.4,                0.5,               0.6,                0.7,               0.8,                0.9]
# y = [0.0, 0.09983341664682815, 0.19866933079506122, 0.29552020666133955, 0.3894183423086505, 0.479425538604203, 0.5646424733950354, 0.644217687237691, 0.7173560908995228, 0.7833269096274834, 0.8414709848078965, 0.09983341664682815, 0.19866933079506122, 0.29552020666133955, 0.3894183423086505, 0.479425538604203, 0.5646424733950354, 0.644217687237691, 0.7173560908995228, 0.7833269096274834]
# x1= [0.1,                 0.2,                 0.3,                 0.4,                0.5,               0.6,                0.7,               0.8,                0.9]
# y1 = [0.09983341664682815, 0.19866933079506122, 0.29552020666133955, 0.3894183423086505, 0.479425538604203, 0.5646424733950354, 0.644217687237691, 0.7173560908995228, 0.7833269096274834]
# x3 = [1,2,3,2,1]
# y3 = np.random.rand(len(x3))
#
# # y = np.sin(x)
# # print(y.tolist())
# plt.plot(x,y,color= 'r')
# plt.plot(x1,y1,color = 'b')
# plt.plot(x3,y3,color= 'g')
# plt.show()
#
#
# def fun1():
#     a=b=c=d = 10
#     b =20
#     print(a,b,c,d)
#
# fun1()
# fig, axs = plt.subplots(2,1)
# axs[0].plot



# def MinMaxNormalization(list_valley,filtedData):
#     m = 0
#     n = 1
#     minmaxNormalData = np.zeros(len(filtedData))
#     minmaxNormalTime = np.zeros(len(filtedData))
#     for j in range(len(list_valley) - 1):
#         temp = filtedData[list_valley[m]-list_valley[0]:list_valley[n] -list_valley[0] + 1]
#         min = np.min(temp)
#         max_min = np.ptp(temp)
#         result = np.array([(i-min)/max_min for i in temp])
#         minmaxNormalData[list_valley[m]-list_valley[0]:list_valley[n] -list_valley[0] + 1] = result
#         print(minmaxNormalData)
#         for i in range(list_valley[m] - list_valley[0], list_valley[n] - list_valley[0] ):
#             print(f"i={i}")
#             result = (i-list_valley[m] + list_valley[0])/(list_valley[n] - list_valley[m] -1)
#             # print(i,end=" ")
#             # print(result,end=" ")
#             minmaxNormalTime[i] = result
#         input()
#         if n != len(list_valley):
#             m += 1
#             n += 1
#     return minmaxNormalData,minmaxNormalTime
#
# nonzero_w_red = np.load("nonzero_w_red.npy")
# list_w_channel_red = np.load("list_w_channel_red.npy")
# nonzero_w_green = np.load("nonzero_w_green.npy")
# list_w_channel_green = np.load("list_w_channel_green.npy")
# nonzero_w_blue = np.load("nonzero_w_blue.npy")
# list_w_channel_blue = np.load("list_w_channel_blue.npy")
# list_valley = np.load("list_valley.npy")
# filtedData = np.load("filtedData_green.npy")
# # print(list_w_channel_green[354:379].tolist())
# # print(list_w_channel_blue[354:379].tolist())
# # print(nonzero_w_green.tolist())
# # print(nonzero_w_blue.tolist())
# # print(np.nonzero(list_w_channel_green))
# # print(np.nonzero(list_w_channel_blue))
# print(list_w_channel_red)
# print(list_w_channel_green)
# print(list_w_channel_blue)
# print(len(list_w_channel_red))
# print(len(list_w_channel_green))
# print(len(list_w_channel_blue))

# print(list_w_channel_green.tolist())

# r1,r2 = MinMaxNormalization(list_valley,filtedData)

def genMaskandW(list_red, list_green, list_blue, list_valley, r,rg,rb):  # 计算dif Mask矩阵 和 W
    cap = cv2.VideoCapture("test-5.mp4")
    m = 0
    n = 1
    list_w_channel_red = np.zeros(len(list_red))
    list_w_channel_green = np.zeros(len(list_red))
    list_w_channel_blue = np.zeros(len(list_red))
    # print("max_red      maxnum      min_red     minnum")
    for j in range(len(list_valley) - 1):  # 遍历每一个心脏周期
        max_red = max_blue = max_green = 0
        index_max_red = index_max_blue = index_max_green = 0
        min_red = min_green = min_blue = 10000
        index_min_red = index_min_green = index_min_blue = 0

        # 处理红色通道
        for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
            if list_red[i] > max_red:
                max_red = list_red[i]
                index_max_red = i
            if list_red[i] < min_red:
                min_red = list_red[i]
                index_min_red = i
        # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_red)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error! -1")
            exit(-1)
        mat_max = np.mat(frame[:, :, 2])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_red)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error! -2")
            exit(-1)
        mat_min = np.mat(frame[:, :, 2])
        dif = mat_max - mat_min  # 计算dif
        # print(dif1)
        mat_mask_red = (mat_max - mat_min > r) + 0  # 计算Mask
        # print(Mask)
        # input()

        # 处理蓝色通道
        # for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
        #     if list_blue[i] > max_blue:
        #         max_blue = list_blue[i]
        #         index_max_blue = i
        #     if list_blue[i] < min_blue:
        #         min_blue = list_blue[i]
        #         index_min_blue = i
        # # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
        # cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_blue)
        # ret, frame = cap.read()
        # if (ret == False):
        #     print("read frame Error! -1")
        #     exit(-1)
        # mat_max = np.mat(frame[:, :, 0])
        # cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_blue)
        # ret, frame = cap.read()
        # if (ret == False):
        #     print("read frame Error! -2")
        #     exit(-1)
        # mat_min = np.mat(frame[:, :, 0])
        # dif = mat_max - mat_min  # 计算dif
        # # print(dif1)
        # mat_mask_blue = (mat_max - mat_min > rb) + 0  # 计算Mask
        # # print(mat_mask_blue)
        # # input()

        # # 处理绿色通道
        # for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
        #     if list_green[i] > max_green:
        #         max_green = list_green[i]
        #         index_max_green = i
        #     if list_green[i] < min_green:
        #         min_green = list_green[i]
        #         index_min_green = i
        # # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
        # cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_green)
        # ret, frame = cap.read()
        # if (ret == False):
        #     print("read frame Error! -1")
        #     exit(-1)
        # mat_max = np.mat(frame[:, :, 1])
        # cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_green)
        # ret, frame = cap.read()
        # if (ret == False):
        #     print("read frame Error! -2")
        #     exit(-1)
        # mat_min = np.mat(frame[:, :, 1])
        # dif = mat_max - mat_min  # 计算dif
        # # print(dif1)
        # mat_mask_green = (mat_max - mat_min > rg) + 0  # 计算Mask
        # # print(Mask)
        # # input()

        for i in range(list_valley[m], list_valley[n] + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if (ret == False):
                print("read frame Error! -3")
                exit(-1)
            mat_c_channel_red = np.mat(frame[:, :, 2])
            list_w_channel_red[i] = np.sum(np.multiply(mat_c_channel_red, mat_mask_red)) / np.sum(mat_mask_red)  # 计算W序列的值
            # mat_c_channel_green = np.mat(frame[:, :, 1])
            # list_w_channel_green[i] = np.sum(np.multiply(mat_c_channel_green, mat_mask_green)) / np.sum(mat_mask_green)
            # mat_c_channel_blue = np.mat(frame[:, :, 0])
            # list_w_channel_blue[i] = np.sum(np.multiply(mat_c_channel_blue, mat_mask_blue)) / np.sum(mat_mask_blue)
        if n != len(list_valley):
            m += 1
            n += 1
    plt.figure(figsize=(14, 7))
    plt.plot(list_w_channel_red,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery =list_valley)
    plt.title(f'r={r}')
    plt.show()
    print(n)
    return list_w_channel_red, list_w_channel_green, list_w_channel_blue


list_red = np.load("../data/list_red.npy")
list_green = np.load("../data/list_green.npy")
list_blue = np.load("../data/list_blue.npy")
list_valley = np.load("../data/list_valley.npy")
list_w_channel_red, list_w_channel_green, list_w_channel_blue = genMaskandW(list_red,
                                                                            list_green,
                                                                            list_blue,
                                                                            list_valley,
                                                                            r=15,rg=6,rb=0.3)
# list_valley12 = [33,53,73,93,113,134,152,172,192,211,231,251,271,291,311,331,352,373,392,412,432,451,470,489,509,530,550,570,590,610]
# plt.plot(list_w_channel_red[0:33])
# plt.show()

# def getFeature(list_valley1,Data,minmaxNormalTime):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
#     m = 0
#     n = 1
#     tmp = 0
#     count = 0
#     feature = np.zeros((3,3,10, 5), dtype=np.float64)
#     # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
#     # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
#     list_valley = np.array(list_valley1) - list_valley1[0]
#     for j in range(len(list_valley) - 1):
#         print(f'j={j}')
#         a = list_valley[m]
#         b = list_valley[n]
#         plt.plot(minmaxNormalTime[a:b], Data[a:b])
#         plt.show()
#         input()
#         SP_index = Data[a:b].tolist().index(np.max(Data[a:b])) + a
#         print(f"SP_index = {SP_index}")
#         DN_list = list(signal.argrelmin(Data[a:b])[0])
#         print(f"DN_list_1 = {DN_list + a}")
#         if  DN_list:   #如果DN_list非空
#             if (DN_list[0] < 5):
#                 tmp = DN_list[0]
#                 del (DN_list[0])
#             if DN_list:   #如果DN_list非空
#                 print(f"tmp = {tmp}")
#                 print(f"DN_list_2 = {DN_list + a}")
#                 DN_index = Data[a:b].tolist().index(np.min([Data[i+a] for i in DN_list])) + a
#                 print(f"DN_index = {DN_index}")
#                 # if DN_index > SP_index:
#                 #     print("loop")
#                 #     continue
#                 DP_index = Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index])) + a + tmp
#                 print(f'Data[{tmp+a}:{DN_index}]={Data[tmp+a:DN_index]}')
#                 print(f'max = {np.max(Data[tmp+a:DN_index])}')
#                 print(f'index = {Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index]))}')
#                 print(f"DP_index = {DP_index}")
#                 plt.plot(minmaxNormalTime[a:b], Data[a:b], marker=6, mfc='r', mec='r', ms=5,
#                          markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
#                 plt.show()
#                 print(f'cout={count}')
#                 input()
#                 # r通道featu_h
#                 feature[0][0][count][1] = Data[DP_index]
#                 feature[0][0][count][2] = Data[DN_index]
#                 feature[0][0][count][3] = Data[SP_index]
#                 feature[0][0][count][4] = Data[b - 1]
#                 # r通道featu_t
#                 feature[0][1][count][1] = minmaxNormalTime[DP_index]
#                 feature[0][1][count][2] = minmaxNormalTime[DN_index] - feature[0][1][count][1]
#                 feature[0][1][count][3] = minmaxNormalTime[SP_index] - feature[0][1][count][2]
#                 feature[0][1][count][4] = minmaxNormalTime[b - 1] - feature[0][1][count][3]
#                 # r通道featu_s
#                 feature[0][2][count][1] = abs(feature[0][0][count][1] / feature[0][1][count][1])
#                 feature[0][2][count][2] = abs((feature[0][0][count][2] - feature[0][0][count][1])/feature[0][1][count][2])
#                 feature[0][2][count][3] = abs((feature[0][0][count][3] - feature[0][0][count][2])/feature[0][1][count][3])
#                 feature[0][2][count][4] = abs((feature[0][0][count][4] - feature[0][0][count][3])/feature[0][1][count][4])
#
#                 # print(f'-------{count}------')
#                 # print(f'SP_index={SP_index}')
#                 # print(f'DN_index={DN_index}')
#                 # print(f'DP_index={DP_index}')
#                 # print(f'h1={feature_h[count][1]}')
#                 # print(f'h2={feature_h[count][2]}')
#                 # print(f'h3={feature_h[count][3]}')
#                 # print(f'h4={feature_h[count][4]}')
#                 # print(feature_h[count])
#                 count += 1
#
#         # plt.plot(minmaxNormalTime[a:b],Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery=[DP_index-a,DN_index-a,SP_index-a,b-a-1])
#         # plt.show()
#         input()
#         if count >= 10:
#             return feature
#         if n != len(list_valley):
#             m += 1
#             n += 1
#     print("Error -4")
#     return feature
#     # print(list_valley)
# #
# list_valley = np.load("../data/list_valley.npy")
# minmaxNormalData = np.load("../data/minmaxNormalData_red.npy")
# minmaxNormalTime = np.load("../data/minmaxNormalTime_red.npy")
# minmaxNormalData = np.load("minmaxNormalData_blue.npy")
# minmaxNormalTime = np.load("minmaxNormalTime_blue.npy")
# feature = getFeature(list_valley,minmaxNormalData,minmaxNormalTime)


# def ButterFilt(data, n, fs):  # 巴特沃斯带通滤波，n为阶数，fs为采样频率
#     (b, a) = signal.butter(n, [0.3 * 2 / fs, 10 * 2 / fs], 'bandpass')
#     filtedData = signal.filtfilt(b, a, data)
#     return filtedData
# n=10
# nonzero_w_red = np.load("nonzero_w_red.npy")
# nonzero_w_green = np.load("nonzero_w_green.npy")
# nonzero_w_blue = np.load("nonzero_w_blue.npy")
# filtedData_red = ButterFilt(nonzero_w_red, n, 30)
# filtedData_green = ButterFilt(nonzero_w_green, n, 30)
# filtedData_blue = ButterFilt(nonzero_w_blue, n, 30)
# fig,axes = plt.subplots(2,1)
# axes[0].plot(filtedData_red,color = 'r')
# axes[0].plot(filtedData_green,color = 'g')
# axes[0].plot(filtedData_blue,color = 'b')
# axes[0].set(title = f'{n}')
# axes[1].plot(nonzero_w_red,color = 'r')
# axes[1].plot(nonzero_w_green,color = 'g')
# axes[1].plot(nonzero_w_blue,color = 'b')
# print(n)
# plt.show()

# filepath = 'test-5.mp4'
# cap = cv2.VideoCapture(filepath)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 646)
# ret, frame = cap.read()
# print(ret)