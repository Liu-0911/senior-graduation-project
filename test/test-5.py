#测试signal和detect_peaks

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
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

nonzero_w_red = np.load("../data/nonzero_w_red.npy")
list_valley = np.load("../data/list_valley.npy")
filtedData_red_1 = ButterFiltHigh(nonzero_w_red, 8,4, 30)
Data,minmaxNormalTime_red_1 = MinMaxNormalization(list_valley,filtedData_red_1)

list_max_index1 = signal.argrelmax(Data[60:80])[0]
list_max_index2 = detect_peaks(Data[60:80],valley=False, show=False, edge=None)
print(list_max_index1)
print(list_max_index2)

list_min_index1 = signal.argrelmin(Data[60:80])[0]
list_min_index2 = detect_peaks(Data[60:80],valley=True, show=False, edge="rising")
print(list_min_index1)
print(list_min_index2)

print(Data[60:80])