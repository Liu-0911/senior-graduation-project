#测试getnonfidFeature函数

import cv2
import numpy as np
import os
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal

from detecta import detect_peaks


def ButterFiltHigh(data,n,ft,fs):  #巴特沃斯高通滤波，n为阶数，fs为采样频率,ft为高通滤波器截止频率
    (b, a) = signal.butter(n, ft*2 / fs, 'highpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def MinMaxNormalization(list_valley,filtedData):
    m = 0
    n = 1
    minmaxNormalData = np.zeros(len(filtedData))
    minmaxNormalTime = np.zeros(len(filtedData))
    for j in range(len(list_valley) - 1):
        temp = filtedData[list_valley[m]-list_valley[0]:list_valley[n] -list_valley[0] + 1]
        min = np.min(temp)
        max_min = np.ptp(temp)
        result = np.array([(i-min)/max_min for i in temp])
        minmaxNormalData[list_valley[m]-list_valley[0]:list_valley[n] -list_valley[0] + 1] = result
        for i in range(list_valley[m] - list_valley[0], list_valley[n] - list_valley[0] ):
            result = (i-list_valley[m] + list_valley[0])/(list_valley[n] - list_valley[m] -1)
            # print(i,end=" ")
            # print(result,end=" ")
            minmaxNormalTime[i] = result

        if n != len(list_valley):
            m += 1
            n += 1
    return minmaxNormalData,minmaxNormalTime

def getnonfidFeature(list_valley1,Data,Time):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    num = 2   #轮空开头的num个周期，减少误差干扰
    count = 0
    feature = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        print(j)
        if j < num:
            m += 1
            n += 1
            continue

        a = list_valley[m]
        b = list_valley[n]
        print(a,b)



        #处理红色通道
        # print(signal.argrelmax(Data[a:b])[0])
        list_max_index = signal.argrelmax(Data[a:b])[0]+a
        # print(type(list_max_index))
        if len(list_max_index) >= 3 :
            list_max = np.array([Data[i] for i in list_max_index])
            print(f"list_max_index={list_max_index}")
            print(f"list_max={list_max}")
            maxtominorder = np.argsort(-list_max)
            print(f"maxtominorder={maxtominorder}")
            # nmax1 = list_max[maxtominorder[0]]
            # nmax2 = Data[list_max_index[maxtominorder[0]]+a]
            # print(nmax1,nmax2)
            max_index = np.sort(list_max_index[maxtominorder[0:3]])
            print("--------------------------------------------")
            list_min_index = signal.argrelmin(Data[max_index[0]:max_index[2]+1])[0] +  max_index[0]
            if len(list_min_index) >= 2:
                list_min = np.array([Data[i] for i in list_min_index])
                # print(f"list_min_index={list_min_index-a}")
                # print(f"list_min={list_min}")
                mintomaxorder = np.argsort(list_min)
                # print(f"mintomaxorder={mintomaxorder}")
                min_index = np.sort(list_min_index[mintomaxorder[0:2]])
                # print(max_index,min_index)
                print(max_index-a, min_index-a)
                x1 = abs(Time[min_index[0]] - Time[max_index[0]])
                x3 = abs(Time[min_index[1]] - Time[max_index[1]])
                x5 = abs(1 - Time[max_index[2]])
                y12 = abs(Data[min_index[0]] - Data[max_index[0]])
                y34 = abs(Data[min_index[1]] - Data[max_index[1]])
                y5 = abs(Data[max_index[2]])
                print(f'x1={x1},x3={x3},x5={x5},y12={y12},y34={y34},y5 = {y5}')
                fig, axes = plt.subplots(2, 1)
                mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                axes[0].plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                axes[0].set_title(f'{j}')
                fig = plt.gcf()
                plt.show()



        # plt.plot(Time[a:b],Data[a:b])
        # plt.title('2')
        # fig = plt.gcf()
        # plt.show()
        input()
        # plt.close(fig)
        # count += 1
        # if count >= 10:
        #     return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature
    # print(list_valley)


list_w_channel_red = np.load("../data/list_w_channel_red.npy")
nonzero_w_red = np.load("../data/nonzero_w_red.npy")
nonzero_w_green = np.load("../data/nonzero_w_green.npy")
nonzero_w_blue = np.load("../data/nonzero_w_blue.npy")
list_valley = np.load("../data/list_valley.npy")

filtedData_red_1 = ButterFiltHigh(nonzero_w_red, 8,8, 30)
filtedData_green_1 = ButterFiltHigh(nonzero_w_green, 8,5, 30)
filtedData_blue_1 = ButterFiltHigh(nonzero_w_blue, 8,5, 30)
minmaxNormalData_red_1,minmaxNormalTime_red_1 = MinMaxNormalization(list_valley,filtedData_red_1)
minmaxNormalData_green_1,minmaxNormalTime_green_1 = MinMaxNormalization(list_valley,filtedData_green_1)
minmaxNormalData_blue_1,minmaxNormalTime_blue_1 = MinMaxNormalization(list_valley,filtedData_blue_1)
getnonfidFeature(list_valley,minmaxNormalData_red_1,minmaxNormalTime_red_1)
# getnonfidFeature(list_valley,minmaxNormalData_green_1,minmaxNormalTime_green_1)
# getnonfidFeature(list_valley,minmaxNormalData_blue_1,minmaxNormalTime_blue_1)