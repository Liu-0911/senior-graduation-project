#测试getnonfidFeature函数

import cv2
import numpy as np
import os
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal

from detecta import detect_peaks



num_of_features = 70


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

def getnonfidFeature_r_h1(list_valley1,Data,Time,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    num = 2   #轮空开头的num个周期，减少误差干扰
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(j)
        if j < num:
            m += 1
            n += 1
            continue

        a = list_valley[m]
        b = list_valley[n]
        # print(a,b)
        # print(signal.argrelmax(Data[a:b])[0])
        list_max_index = signal.argrelmax(Data[a:b])[0]+a
        # print(type(list_max_index))
        if len(list_max_index) >= 3 :
            list_max = np.array([Data[i] for i in list_max_index])
            # print(f"list_max_index={list_max_index}")
            # print(f"list_max={list_max}")
            maxtominorder = np.argsort(-list_max)
            # print(f"maxtominorder={maxtominorder}")
            # nmax1 = list_max[maxtominorder[0]]
            # nmax2 = Data[list_max_index[maxtominorder[0]]+a]
            # print(nmax1,nmax2)
            max_index = np.sort(list_max_index[maxtominorder[0:3]])
            # print("--------------------------------------------")
            list_min_index = signal.argrelmin(Data[max_index[0]:max_index[2]+1])[0] +  max_index[0]
            if len(list_min_index) >= 2:
                list_min = np.array([Data[i] for i in list_min_index])
                # print(f"list_min_index={list_min_index-a}")
                # print(f"list_min={list_min}")
                mintomaxorder = np.argsort(list_min)
                # print(f"mintomaxorder={mintomaxorder}")
                min_index = np.sort(list_min_index[mintomaxorder[0:2]])
                # print(max_index,min_index)
                # print(max_index-a, min_index-a)
                x1 = abs(Time[min_index[0]] - Time[max_index[0]])
                x3 = abs(Time[min_index[1]] - Time[max_index[1]])
                x5 = abs(1 - Time[max_index[2]])
                y12 = abs(Data[min_index[0]] - Data[max_index[0]])
                y34 = abs(Data[min_index[1]] - Data[max_index[1]])
                y5 = abs(Data[max_index[2]])

                feature[count][30] = x1
                feature[count][31] = x3
                feature[count][32] = x5
                feature[count][33] = y12
                feature[count][34] = y34
                feature[count][35] = y5

                # print(f'x1={x1},x3={x3},x5={x5},y12={y12},y34={y34},y5 = {y5}')
                # fig, axes = plt.subplots(2, 1)
                # mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                # plt.plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # # axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # # axes[0].set_title(f'{j}')
                # plt.ylim(-0.1, 1.2)
                # plt.xlabel('归一化心脏周期')
                # plt.ylabel('归一化振幅')
                # plt.show()
                # input()
                count += 1



        # plt.plot(Time[a:b],Data[a:b])
        # plt.title('2')
        # fig = plt.gcf()
        # plt.show()
        # input()
        # plt.close(fig)
        # count += 1
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature
    # print(list_valley)

def getnonfidFeature_g_h1(list_valley1,Data,Time,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    num = 2   #轮空开头的num个周期，减少误差干扰
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(j)
        if j < num:
            m += 1
            n += 1
            continue

        a = list_valley[m]
        b = list_valley[n]
        # print(a,b)
        # print(signal.argrelmax(Data[a:b])[0])
        list_max_index = signal.argrelmax(Data[a:b])[0]+a
        # print(type(list_max_index))
        if len(list_max_index) >= 3 :
            list_max = np.array([Data[i] for i in list_max_index])
            # print(f"list_max_index={list_max_index}")
            # print(f"list_max={list_max}")
            maxtominorder = np.argsort(-list_max)
            # print(f"maxtominorder={maxtominorder}")
            # nmax1 = list_max[maxtominorder[0]]
            # nmax2 = Data[list_max_index[maxtominorder[0]]+a]
            # print(nmax1,nmax2)
            max_index = np.sort(list_max_index[maxtominorder[0:3]])
            # print("--------------------------------------------")
            list_min_index = signal.argrelmin(Data[max_index[0]:max_index[2]+1])[0] +  max_index[0]
            if len(list_min_index) >= 2:
                list_min = np.array([Data[i] for i in list_min_index])
                # print(f"list_min_index={list_min_index-a}")
                # print(f"list_min={list_min}")
                mintomaxorder = np.argsort(list_min)
                # print(f"mintomaxorder={mintomaxorder}")
                min_index = np.sort(list_min_index[mintomaxorder[0:2]])
                # print(max_index,min_index)
                # print(max_index-a, min_index-a)
                x1 = abs(Time[min_index[0]] - Time[max_index[0]])
                x3 = abs(Time[min_index[1]] - Time[max_index[1]])
                x5 = abs(1 - Time[max_index[2]])
                y12 = abs(Data[min_index[0]] - Data[max_index[0]])
                y34 = abs(Data[min_index[1]] - Data[max_index[1]])
                y5 = abs(Data[max_index[2]])

                feature[count][36] = x1
                feature[count][37] = x3
                feature[count][38] = x5
                feature[count][39] = y12
                feature[count][40] = y34
                feature[count][41] = y5

                # print(f'x1={x1},x3={x3},x5={x5},y12={y12},y34={y34},y5 = {y5}')
                # fig, axes = plt.subplots(2, 1)
                # mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                # axes[0].plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[0].set_title(f'{j}')
                # fig = plt.gcf()
                # plt.show()
                count += 1



        # plt.plot(Time[a:b],Data[a:b])
        # plt.title('2')
        # fig = plt.gcf()
        # plt.show()
        # input()
        # plt.close(fig)
        # count += 1
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature

def getnonfidFeature_b_h1(list_valley1,Data,Time,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    num = 2   #轮空开头的num个周期，减少误差干扰
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(j)
        if j < num:
            m += 1
            n += 1
            continue

        a = list_valley[m]
        b = list_valley[n]
        # print(a,b)
        # print(signal.argrelmax(Data[a:b])[0])
        list_max_index = signal.argrelmax(Data[a:b])[0]+a
        # print(type(list_max_index))
        if len(list_max_index) >= 3 :
            list_max = np.array([Data[i] for i in list_max_index])
            # print(f"list_max_index={list_max_index}")
            # print(f"list_max={list_max}")
            maxtominorder = np.argsort(-list_max)
            # print(f"maxtominorder={maxtominorder}")
            # nmax1 = list_max[maxtominorder[0]]
            # nmax2 = Data[list_max_index[maxtominorder[0]]+a]
            # print(nmax1,nmax2)
            max_index = np.sort(list_max_index[maxtominorder[0:3]])
            # print("--------------------------------------------")
            list_min_index = signal.argrelmin(Data[max_index[0]:max_index[2]+1])[0] +  max_index[0]
            if len(list_min_index) >= 2:
                list_min = np.array([Data[i] for i in list_min_index])
                # print(f"list_min_index={list_min_index-a}")
                # print(f"list_min={list_min}")
                mintomaxorder = np.argsort(list_min)
                # print(f"mintomaxorder={mintomaxorder}")
                min_index = np.sort(list_min_index[mintomaxorder[0:2]])
                # print(max_index,min_index)
                # print(max_index-a, min_index-a)
                x1 = abs(Time[min_index[0]] - Time[max_index[0]])
                x3 = abs(Time[min_index[1]] - Time[max_index[1]])
                x5 = abs(1 - Time[max_index[2]])
                y12 = abs(Data[min_index[0]] - Data[max_index[0]])
                y34 = abs(Data[min_index[1]] - Data[max_index[1]])
                y5 = abs(Data[max_index[2]])

                feature[count][42] = x1
                feature[count][43] = x3
                feature[count][44] = x5
                feature[count][45] = y12
                feature[count][46] = y34
                feature[count][47] = y5

                # print(f'x1={x1},x3={x3},x5={x5},y12={y12},y34={y34},y5 = {y5}')
                # fig, axes = plt.subplots(2, 1)
                # mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                # axes[0].plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[0].set_title(f'{j}')
                # fig = plt.gcf()
                # plt.show()
                count += 1



        # plt.plot(Time[a:b],Data[a:b])
        # plt.title('2')
        # fig = plt.gcf()
        # plt.show()
        # input()
        # plt.close(fig)
        # count += 1
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature

def getnonfidFeature_r_h2(list_valley1,Data,Time,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    num = 2   #轮空开头的num个周期，减少误差干扰
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(j)
        if j < num:
            m += 1
            n += 1
            continue

        a = list_valley[m]
        b = list_valley[n]
        # print(a,b)
        # print(signal.argrelmax(Data[a:b])[0])
        list_max_index = signal.argrelmax(Data[a:b])[0]+a
        # print(type(list_max_index))
        if len(list_max_index) >= 3 :
            list_max = np.array([Data[i] for i in list_max_index])
            # print(f"list_max_index={list_max_index}")
            # print(f"list_max={list_max}")
            maxtominorder = np.argsort(-list_max)
            # print(f"maxtominorder={maxtominorder}")
            # nmax1 = list_max[maxtominorder[0]]
            # nmax2 = Data[list_max_index[maxtominorder[0]]+a]
            # print(nmax1,nmax2)
            max_index = np.sort(list_max_index[maxtominorder[0:3]])
            # print("--------------------------------------------")
            list_min_index = signal.argrelmin(Data[max_index[0]:max_index[2]+1])[0] +  max_index[0]
            if len(list_min_index) >= 2:
                list_min = np.array([Data[i] for i in list_min_index])
                # print(f"list_min_index={list_min_index-a}")
                # print(f"list_min={list_min}")
                mintomaxorder = np.argsort(list_min)
                # print(f"mintomaxorder={mintomaxorder}")
                min_index = np.sort(list_min_index[mintomaxorder[0:2]])
                # print(max_index,min_index)
                # print(max_index-a, min_index-a)
                x1 = abs(Time[min_index[0]] - Time[max_index[0]])
                x3 = abs(Time[min_index[1]] - Time[max_index[1]])
                x5 = abs(1 - Time[max_index[2]])
                y12 = abs(Data[min_index[0]] - Data[max_index[0]])
                y34 = abs(Data[min_index[1]] - Data[max_index[1]])
                y5 = abs(Data[max_index[2]])

                feature[count][48] = x1
                feature[count][49] = x3
                feature[count][50] = x5
                feature[count][54] = y12
                feature[count][52] = y34
                feature[count][53] = y5

                # fig, axes = plt.subplots(2, 1)
                # mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                # plt.plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # # # axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # # # axes[0].set_title(f'{j}')
                # plt.ylim(-0.1, 1.2)
                # plt.xlabel('归一化心脏周期')
                # plt.ylabel('归一化振幅')
                # plt.show()
                # input()
                count += 1



        # plt.plot(Time[a:b],Data[a:b])
        # plt.title('2')
        # fig = plt.gcf()
        # plt.show()
        # input()
        # plt.close(fig)
        # count += 1
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature


def getnonfidFeature_g_h2(list_valley1,Data,Time,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    num = 2   #轮空开头的num个周期，减少误差干扰
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(j)
        if j < num:
            m += 1
            n += 1
            continue

        a = list_valley[m]
        b = list_valley[n]
        # print(a,b)
        # print(signal.argrelmax(Data[a:b])[0])
        list_max_index = signal.argrelmax(Data[a:b])[0]+a
        # print(type(list_max_index))
        if len(list_max_index) >= 3 :
            list_max = np.array([Data[i] for i in list_max_index])
            # print(f"list_max_index={list_max_index}")
            # print(f"list_max={list_max}")
            maxtominorder = np.argsort(-list_max)
            # print(f"maxtominorder={maxtominorder}")
            # nmax1 = list_max[maxtominorder[0]]
            # nmax2 = Data[list_max_index[maxtominorder[0]]+a]
            # print(nmax1,nmax2)
            max_index = np.sort(list_max_index[maxtominorder[0:3]])
            # print("--------------------------------------------")
            list_min_index = signal.argrelmin(Data[max_index[0]:max_index[2]+1])[0] +  max_index[0]
            if len(list_min_index) >= 2:
                list_min = np.array([Data[i] for i in list_min_index])
                # print(f"list_min_index={list_min_index-a}")
                # print(f"list_min={list_min}")
                mintomaxorder = np.argsort(list_min)
                # print(f"mintomaxorder={mintomaxorder}")
                min_index = np.sort(list_min_index[mintomaxorder[0:2]])
                # print(max_index,min_index)
                # print(max_index-a, min_index-a)
                x1 = abs(Time[min_index[0]] - Time[max_index[0]])
                x3 = abs(Time[min_index[1]] - Time[max_index[1]])
                x5 = abs(1 - Time[max_index[2]])
                y12 = abs(Data[min_index[0]] - Data[max_index[0]])
                y34 = abs(Data[min_index[1]] - Data[max_index[1]])
                y5 = abs(Data[max_index[2]])

                feature[count][54] = x1
                feature[count][55] = x3
                feature[count][56] = x5
                feature[count][57] = y12
                feature[count][58] = y34
                feature[count][59] = y5

                # print(f'x1={x1},x3={x3},x5={x5},y12={y12},y34={y34},y5 = {y5}')
                # fig, axes = plt.subplots(2, 1)
                # mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                # axes[0].plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[0].set_title(f'{j}')
                # fig = plt.gcf()
                # plt.show()
                count += 1


        # plt.plot(Time[a:b],Data[a:b])
        # plt.title('2')
        # fig = plt.gcf()
        # plt.show()
        # input()
        # plt.close(fig)
        # count += 1
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature

def getnonfidFeature_b_h2(list_valley1,Data,Time,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    num = 2   #轮空开头的num个周期，减少误差干扰
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(j)
        if j < num:
            m += 1
            n += 1
            continue

        a = list_valley[m]
        b = list_valley[n]
        # print(a,b)
        # print(signal.argrelmax(Data[a:b])[0])
        list_max_index = signal.argrelmax(Data[a:b])[0]+a
        # print(type(list_max_index))
        if len(list_max_index) >= 3 :
            list_max = np.array([Data[i] for i in list_max_index])
            # print(f"list_max_index={list_max_index}")
            # print(f"list_max={list_max}")
            maxtominorder = np.argsort(-list_max)
            # print(f"maxtominorder={maxtominorder}")
            # nmax1 = list_max[maxtominorder[0]]
            # nmax2 = Data[list_max_index[maxtominorder[0]]+a]
            # print(nmax1,nmax2)
            max_index = np.sort(list_max_index[maxtominorder[0:3]])
            # print("--------------------------------------------")
            list_min_index = signal.argrelmin(Data[max_index[0]:max_index[2]+1])[0] +  max_index[0]
            if len(list_min_index) >= 2:
                list_min = np.array([Data[i] for i in list_min_index])
                # print(f"list_min_index={list_min_index-a}")
                # print(f"list_min={list_min}")
                mintomaxorder = np.argsort(list_min)
                # print(f"mintomaxorder={mintomaxorder}")
                min_index = np.sort(list_min_index[mintomaxorder[0:2]])
                # print(max_index,min_index)
                # print(max_index-a, min_index-a)
                x1 = abs(Time[min_index[0]] - Time[max_index[0]])
                x3 = abs(Time[min_index[1]] - Time[max_index[1]])
                x5 = abs(1 - Time[max_index[2]])
                y12 = abs(Data[min_index[0]] - Data[max_index[0]])
                y34 = abs(Data[min_index[1]] - Data[max_index[1]])
                y5 = abs(Data[max_index[2]])

                feature[count][60] = x1
                feature[count][61] = x3
                feature[count][62] = x5
                feature[count][63] = y12
                feature[count][64] = y34
                feature[count][65] = y5

                # print(f'x1={x1},x3={x3},x5={x5},y12={y12},y34={y34},y5 = {y5}')
                # fig, axes = plt.subplots(2, 1)
                # mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                # axes[0].plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                # axes[0].set_title(f'{j}')
                # fig = plt.gcf()
                # plt.show()
                count += 1


        # plt.plot(Time[a:b],Data[a:b])
        # plt.title('2')
        # fig = plt.gcf()
        # plt.show()
        # input()
        # plt.close(fig)
        # count += 1
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature

list_valley = np.load("../data/list_valley.npy")
feature = np.load('../data/feature_70.npy')
# list_w_channel_red = np.load("../data/list_w_channel_red.npy")
nonzero_w_red = np.load("../data/nonzero_w_red.npy")
nonzero_w_green = np.load("../data/nonzero_w_green.npy")
nonzero_w_blue = np.load("../data/nonzero_w_blue.npy")

filtedData_red_h1 = ButterFiltHigh(nonzero_w_red, 8,4, 30)
filtedData_green_h1 = ButterFiltHigh(nonzero_w_green, 8,4, 30)
filtedData_blue_h1 = ButterFiltHigh(nonzero_w_blue, 8,4, 30)
filtedData_red_h2 = ButterFiltHigh(nonzero_w_red, 8,5, 30)
filtedData_green_h2 = ButterFiltHigh(nonzero_w_green, 8,5, 30)
filtedData_blue_h2 = ButterFiltHigh(nonzero_w_blue, 8,5, 30)


# filtedData_red = np.load("../data/filtedData_red.npy")
# filtedData_green = np.load("../data/filtedData_green.npy")
# filtedData_blue = np.load("../data/filtedData_blue.npy")
# filtedData_red_h1 = ButterFiltHigh(filtedData_red, 8,1, 30)
# filtedData_green_h1 = ButterFiltHigh(filtedData_green, 8,1, 30)
# filtedData_blue_h1 = ButterFiltHigh(filtedData_blue, 8,1, 30)
# filtedData_red_h2 = ButterFiltHigh(filtedData_red, 8,2, 30)
# filtedData_green_h2 = ButterFiltHigh(filtedData_green_h1,8,2, 30)
# filtedData_blue_h2 = ButterFiltHigh(filtedData_blue, 8,2, 30)

# fig,ax = plt.subplots(3,2,figsize=(12.8,6.4))
# ax[0,0].plot(filtedData_red_h1)
# ax[1,0].plot(filtedData_green_h1)
# ax[2,0].plot(filtedData_blue_h1)
# ax[0,1].plot(filtedData_red_h2)
# ax[1,1].plot(filtedData_green_h2)
# ax[2,1].plot(filtedData_blue_h2)
# plt.show()



#old
# minmaxNormalData_red_h1,minmaxNormalTime_red_h1 = MinMaxNormalization(list_valley,filtedData_red_h1)
# minmaxNormalData_green_h1,minmaxNormalTime_green_h1 = MinMaxNormalization(list_valley,filtedData_green_h1)
# minmaxNormalData_blue_h1,minmaxNormalTime_blue_h1 = MinMaxNormalization(list_valley,filtedData_blue_h1)
# minmaxNormalData_red_h2,minmaxNormalTime_red_h2 = MinMaxNormalization(list_valley,filtedData_red_h1)
# minmaxNormalData_green_h2,minmaxNormalTime_green_h2 = MinMaxNormalization(list_valley,filtedData_green_h1)
# minmaxNormalData_blue_h2,minmaxNormalTime_blue_h2 = MinMaxNormalization(list_valley,filtedData_blue_h1)
# print('0')
# feature = getnonfidFeature_r_h1(list_valley,minmaxNormalData_red_h1,minmaxNormalTime_red_h1,feature)
# print('1')
# feature = getnonfidFeature_g_h1(list_valley,minmaxNormalData_green_h1,minmaxNormalTime_green_h1,feature)
# print('2')
# feature = getnonfidFeature_b_h1(list_valley,minmaxNormalData_blue_h1,minmaxNormalTime_blue_h1,feature)
# print('3')
# feature = getnonfidFeature_r_h2(list_valley,minmaxNormalData_red_h2,minmaxNormalTime_red_h2,feature)
# feature = getnonfidFeature_g_h2(list_valley,minmaxNormalData_green_h2,minmaxNormalTime_green_h2,feature)
# feature = getnonfidFeature_b_h2(list_valley,minmaxNormalData_blue_h2,minmaxNormalTime_blue_h2,feature)
# # getnonfidFeature(list_valley,minmaxNormalData_green_1,minmaxNormalTime_green_1,feature)
# # getnonfidFeature(list_valley,minmaxNormalData_blue_1,minmaxNormalTime_blue_1,feature)
# # np.save('../data/feature_70_done.npy',feature)
# print(feature)


minmaxNormalData_red_h1,minmaxNormalTime_red_h1 = MinMaxNormalization(list_valley,filtedData_red_h1)
minmaxNormalData_green_h1,minmaxNormalTime_green_h1 = MinMaxNormalization(list_valley,filtedData_green_h1)
minmaxNormalData_blue_h1,minmaxNormalTime_blue_h1 = MinMaxNormalization(list_valley,filtedData_blue_h1)
minmaxNormalData_red_h2,minmaxNormalTime_red_h2 = MinMaxNormalization(list_valley,filtedData_red_h2)
minmaxNormalData_green_h2,minmaxNormalTime_green_h2 = MinMaxNormalization(list_valley,filtedData_green_h2)
minmaxNormalData_blue_h2,minmaxNormalTime_blue_h2 = MinMaxNormalization(list_valley,filtedData_blue_h2)
print('0')
feature = getnonfidFeature_r_h1(list_valley,minmaxNormalData_red_h1,minmaxNormalTime_red_h1,feature)
print('1')
feature = getnonfidFeature_g_h1(list_valley,minmaxNormalData_green_h1,minmaxNormalTime_green_h1,feature)
print('2')
feature = getnonfidFeature_b_h1(list_valley,minmaxNormalData_blue_h1,minmaxNormalTime_blue_h1,feature)
print('3')
feature = getnonfidFeature_r_h2(list_valley,minmaxNormalData_red_h2,minmaxNormalTime_red_h2,feature)
feature = getnonfidFeature_g_h2(list_valley,minmaxNormalData_green_h2,minmaxNormalTime_green_h2,feature)
feature = getnonfidFeature_b_h2(list_valley,minmaxNormalData_blue_h2,minmaxNormalTime_blue_h2,feature)
# getnonfidFeature(list_valley,minmaxNormalData_green_1,minmaxNormalTime_green_1,feature)
# getnonfidFeature(list_valley,minmaxNormalData_blue_1,minmaxNormalTime_blue_1,feature)
np.save('../data/feature_70_done_new.npy',feature)
print(feature)