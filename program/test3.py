import cv2
import numpy as np
import os
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal

from detecta import detect_peaks

# def genMaskandW(list_red, list_green, list_blue, list_valley, rr,rg,rb):  # 计算dif Mask矩阵 和 W
#     cap = cv2.VideoCapture("test-5.mp4")
#     m = 0
#     n = 1
#     list_w_channel_red = np.zeros(len(list_red))
#     list_w_channel_green = np.zeros(len(list_red))
#     list_w_channel_blue = np.zeros(len(list_red))
#     # print("max_red      maxnum      min_red     minnum")
#     for j in range(len(list_valley) - 1):  # 遍历每一个心脏周期
#         max_red = max_blue = max_green = 0
#         index_max_red = index_max_blue = index_max_green = 0
#         min_red = min_green = min_blue = 10000
#         index_min_red = index_min_green = index_min_blue = 0
#
#         # 处理红色通道
#         for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
#             if list_red[i] > max_red:
#                 max_red = list_red[i]
#                 index_max_red = i
#             if list_red[i] < min_red:
#                 min_red = list_red[i]
#                 index_min_red = i
#         # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_red)
#         ret, frame = cap.read()
#         # print(index_max_red,ret)
#         if (ret == False):
#             print("read frame Error! -1r")
#             exit(-1)
#         mat_max = np.mat(frame[:, :, 2])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_red)
#         ret, frame = cap.read()
#         if (ret == False):
#             print("read frame Error! -2r")
#             exit(-1)
#         mat_min = np.mat(frame[:, :, 2])
#         dif = mat_max - mat_min  # 计算dif
#         # print(dif1)
#         mat_mask_red = (mat_max - mat_min > rr) + 0  # 计算Mask
#         # print(Mask)
#         # input()
#
#         # 处理蓝色通道
#         for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
#             if list_blue[i] > max_blue:
#                 max_blue = list_blue[i]
#                 index_max_blue = i
#             if list_blue[i] < min_blue:
#                 min_blue = list_blue[i]
#                 index_min_blue = i
#         # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_blue)
#         ret, frame = cap.read()
#         if (ret == False):
#             print("read frame Error! -1b")
#             exit(-1)
#         mat_max = np.mat(frame[:, :, 0])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_blue)
#         ret, frame = cap.read()
#         if (ret == False):
#             print("read frame Error! -2b")
#             exit(-1)
#         mat_min = np.mat(frame[:, :, 0])
#         dif = mat_max - mat_min  # 计算dif
#         # print(dif1)
#         mat_mask_blue = (mat_max - mat_min > rb) + 0  # 计算Mask
#         # print(mat_mask_blue)
#         # input()
#
#         # 处理绿色通道
#         for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
#             if list_green[i] > max_green:
#                 max_green = list_green[i]
#                 index_max_green = i
#             if list_green[i] < min_green:
#                 min_green = list_green[i]
#                 index_min_green = i
#         # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_green)
#         ret, frame = cap.read()
#         if (ret == False):
#             print("read frame Error! -1g")
#             exit(-1)
#         mat_max = np.mat(frame[:, :, 1])
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_green)
#         ret, frame = cap.read()
#         if (ret == False):
#             print("read frame Error! -2g")
#             exit(-1)
#         mat_min = np.mat(frame[:, :, 1])
#         dif = mat_max - mat_min  # 计算dif
#         # print(dif1)
#         mat_mask_green = (mat_max - mat_min > rg) + 0  # 计算Mask
#         # print(Mask)
#         # input()
#
#         for i in range(list_valley[m], list_valley[n] + 1):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#             ret, frame = cap.read()
#             if (ret == False):
#                 print("read frame Error! -3")
#                 exit(-1)
#             mat_c_channel_red = np.mat(frame[:, :, 2])
#             list_w_channel_red[i] = np.sum(np.multiply(mat_c_channel_red, mat_mask_red)) / np.sum(mat_mask_red)  # 计算W序列的值
#             mat_c_channel_green = np.mat(frame[:, :, 1])
#             list_w_channel_green[i] = np.sum(np.multiply(mat_c_channel_green, mat_mask_green)) / np.sum(mat_mask_green)
#             mat_c_channel_blue = np.mat(frame[:, :, 0])
#             list_w_channel_blue[i] = np.sum(np.multiply(mat_c_channel_blue, mat_mask_blue)) / np.sum(mat_mask_blue)
#         if n != len(list_valley):
#             m += 1
#             n += 1
#     print(n)
#     return list_w_channel_red, list_w_channel_green, list_w_channel_blue
#
# list_red = np.load('list_red.npy')
# list_green = np.load('list_green.npy')
# list_blue = np.load('list_blue.npy')
# list_valley = np.load('list_valley.npy')
# list_w_channel_red, list_w_channel_green, list_w_channel_blue = genMaskandW(list_red,
#                                                                             list_green,
#                                                                             list_blue,
#                                                                             list_valley,
#                                                                             15,6,0.5)
# print('over')


# DN = []
# if not DN:
#     print("!!!")
# if DN:
#     print("___")
minmaxNormalData_red = np.load("../data/minmaxNormalData_red.npy")
plt.plot(minmaxNormalData_red[0:20])
plt.title("1")
plt.show()