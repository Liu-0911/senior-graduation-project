#用于测试原始数据和高通滤波后的数据

import cv2
import numpy as np
import os
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal

from detecta import detect_peaks

# def ButterFiltHigh(data,n,ft,fs):  #巴特沃斯高通滤波，n为阶数，fs为采样频率,ft为高通滤波器截止频率
#     (b, a) = signal.butter(n, ft*2 / fs, 'highpass')
#     filtedData = signal.filtfilt(b, a, data)
#     return filtedData
#
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
#         for i in range(list_valley[m] - list_valley[0], list_valley[n] - list_valley[0] ):
#             result = (i-list_valley[m] + list_valley[0])/(list_valley[n] - list_valley[m] -1)
#             # print(i,end=" ")
#             # print(result,end=" ")
#             minmaxNormalTime[i] = result
#
#         if n != len(list_valley):
#             m += 1
#             n += 1
#     return minmaxNormalData,minmaxNormalTime
#
# list_w_channel_red = np.load("../data/list_w_channel_red.npy")
# nonzero_w_red = np.load("../data/nonzero_w_red.npy")
# list_valley = np.load("../data/list_valley.npy")
# list_valley1 = list_valley - list_valley[0]
# print(list_valley1)
# filtedData_red_1 = filtedData_red_1 = ButterFiltHigh(nonzero_w_red, 8,1, 30)
# minmaxNormalData_red_1,minmaxNormalTime_red_1 = MinMaxNormalization(list_valley,filtedData_red_1)
# fig,axes = plt.subplots(3,1,dpi=300,figsize=(16,18))
# axes[0].plot(nonzero_w_red,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery =list_valley1)
# axes[1].plot(filtedData_red_1,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery =list_valley1)
# axes[2].plot(minmaxNormalData_red_1,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery =list_valley1)
# plt.show()




# import time
#
# start = time.perf_counter()  # Python3.8不支持clock了，使用timer.perf_counter()
# for i in range(10000):
#     print(i)
# # 这里可以放入运行代码，比较直接
# end = time.perf_counter()
# print(str(end - start))

from moviepy.editor import *

# 剪辑50-60秒的音乐 00:00:50 - 00:00:60
video = CompositeVideoClip([VideoFileClip("../MP4/test-4.mp4").subclip(5, 9)])

# 写入剪辑完成的音乐
video.write_videofile("../data_show/legal.mp4")