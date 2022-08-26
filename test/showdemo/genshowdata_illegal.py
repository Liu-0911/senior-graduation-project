

import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

import os
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from sklearn.decomposition import PCA

from detecta import detect_peaks

Channel_R = 2
Channel_G = 1
Channel_B = 0

filepath = '../../data_show/illegal.mp4'
foldername = '../data_show'

#特征提取数量
num_of_features = 10

t = 0.55
pr = 0.95
result_list = []
ava_red_list = []
ava_green_list = []
ava_blue_list = []
num = 0
ava_red = 0

# cap = cv2.VideoCapture(filepath)  # 打开视频文件
# if cap.isOpened() is False:  # 确认视频是否成果打开
#     print('Error')
#     exit(1)
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图片帧宽度
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取图像帧高度
# # fps = float(cap.get(cv2.CAP_PROP_FPS))                 # 获取FPS
# frame_channel = 3
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
# # print("一共{0}帧".format(frame_count))


# print(frame_width,frame_height,fps,frame_count,frame_channel)


@jit  # 每一帧中对像素进行处理，包括： 1.计算每一帧的红色平均强度 2.计算这一帧是否符合红色主导的条件
def count(img, num, t, total_red, total_green, total_blue):
    for row in range(frame_height):  # 遍历每一行
        for col in range(frame_width):  # 遍历每一列
            total = img[row][col][0] + img[row][col][1] + img[row][col][2]
            total_red += img[row][col][2]
            total_green += img[row][col][1]
            total_blue += img[row][col][0]
            if (img[row][col][2] > total * t):
                num += 1
    ava_red = total_red / (frame_height * frame_width)
    ava_green = total_green / (frame_height * frame_width)
    ava_blue = total_blue / (frame_height * frame_width)
    return num, ava_red, ava_green, ava_blue


def detFingerTouch():  # 实现指尖触摸检测并输出三个颜色通道平均值
    ret, frame = cap.read()  # 读取一帧图像，当视频帧读取完毕ret标识符为False
    num_of_frame = 1
    while ret:
        num, ava_red, ava_green, ava_blue = count(frame, 0, t, 0, 0, 0)
        ava_red_list.append(ava_red)
        ava_green_list.append(ava_green)
        ava_blue_list.append(ava_blue)
        if (num > frame_width * frame_height * pr):  # 判断是否可以开始推导并输出01串
            result_list.append("1")
        else:
            result_list.append("0")
        num_of_frame += 1
        ret, frame = cap.read()  # 读取下一帧
    result = "".join(result_list)
    #     result_red = "".join(ava_red_list)
    red_list = np.array(ava_red_list)
    # print(result)
    # print(ava_red_list)
    # detect_peaks(red_list, mph=-1.2, mpd=20, valley=True, show=False)
    cv2.destroyAllWindows()
    return result, ava_red_list, ava_green_list, ava_blue_list


def findmin(list, min):  # 波谷检测算法，返回符合条件的波谷index
    list_valley = []
    left = right = 0
    list_down = detect_peaks(list, mpd=1, valley=True, show=False, edge=None)  # 寻找所有的波谷

    # print(list_down)
    #     for index,value in enumerate(list_down):
    for index in list_down:  # 遍历所有波谷，计算每个波谷对应的最小相对距离
        # print("--------------------------------------------------")
        list_left_value_tem = []
        list_left_index_tem = []
        list_right_value_tem = []
        list_right_index_tem = []
        leftfindflag = False
        rightfindflag = False
        value = list[index]
        # print(f"value ={value},index = {index},list[index] = {list[index]}")
        tmp = index
        while (index >= 0):  # 找到以当前波谷做水平线与图像的左交点
            index -= 1
            if (list[index] == value):
                left = index
                leftfindflag = True
                break
            elif (list[index] < value):
                left = index + 1
                leftfindflag = True
                break
        if (leftfindflag == False):
            left = tmp
        # print(f"left = {left}")
        for i in range(left, tmp + 1):  # 处理左边，找出波峰的最大值
            # print(f"i={i}")
            list_left_value_tem.append(list[i])
            list_left_index_tem.append(i)
        # print("list_left_value_tem =", end=" ")
        # print(list_left_value_tem)
        list_left_up_tem = detect_peaks(list_left_value_tem, mpd=1, valley=False, show=False, edge=None)
        # print("list_left_up_tem =", end=" ")
        # print(list_left_up_tem)

        maxleft = 0

        for i in list_left_up_tem:
            if (list[list_left_index_tem[i]] > maxleft):
                # print(f"list_left_value_tem[i]={list_left_value_tem[i]},list[list_left_value_tem[i]]={list[list_left_value_tem[i]]}")
                maxleft = list[list_left_index_tem[i]]

        index = tmp
        # print(f"index before = {index}")
        while (index <= len(list) - 2):  # 找到以当前波谷做水平线与图像的右交点
            index += 1
            if (list[index] == value):
                right = index
                rightfindflag = True
                break
            elif (list[index] < value):
                right = index - 1
                rightfindflag = True
                break
        if (rightfindflag == False):
            right = tmp
        # print(f"right = {right}")
        for i in range(tmp, right + 1):  # 处理右边，找出波峰的最大值
            list_right_value_tem.append(list[i])
            list_right_index_tem.append(i)
        # print("list_right_value_tem =",end=" ")
        # print(list_right_value_tem)
        list_right_up_tem = detect_peaks(list_left_value_tem, mpd=1, valley=False, show=False, edge=None)
        # print("list_right_up_tem =", end=" ")
        # print(list_right_up_tem)
        # print(f"len(list)={len(list)}")
        maxright = 0
        for i in list_right_up_tem:
            if (list[list_left_index_tem[i]] > maxright):
                # print(f"list_left_value_tem[i]={list_left_value_tem[i]},list[list_left_value_tem[i]]={list[list_left_value_tem[i]]}")
                maxright = list[list_left_index_tem[i]]
        # print(f"maxright={maxright},maxright-value={maxright-value}")
        max = maxleft if maxright > maxleft else maxleft  # 取左右最高峰的较小值，计算相对最小距离
        if (max - value >= min):
            list_valley.append(tmp)
    # print(list_result)
    tmp = 0
    for index, value in enumerate(list_valley):
        if index < len(list_valley) - 1:
            if list_valley[index + 1] - value >= 10:
                tmp = index
                break
    # print(list_result)
    list_valley = list_valley[tmp:len(list_valley)]
    # print(list_result)
    # plt.figure(figsize=(14, 7), dpi=500)
    # plt.plot(list, marker=6, mfc='r', mec='r', ms=5, markevery=list_valley)
    # plt.show()
    return list_valley


def genMask(list_red, list_valley, r):
    m = 0
    n = 1
    print("max      maxnum      min     minnum")
    for j in range(len(list_valley) - 1):
        max = 0
        index_max = 0
        min = 10000
        index_min = 0
        for i in range(list_valley[m], list_valley[n] + 1):
            if list_red[i] > max:
                max = list_red[i]
                index_max = i
            if list_red[i] < min:
                min = list_red[i]
                index_min = i
        # print(index_max, list_red[index_max], index_min, list_red[index_min])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_max)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error!")
            exit(-1)
        mat_max = np.mat(frame[:, :, 2])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_min)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error!")
            exit(-1)
        mat_min = np.mat(frame[:, :, 2])
        dif1 = mat_max - mat_min
        print(dif1)
        dif = (mat_max - mat_min > r) + 0
        print(dif)
        # input()
        if n != len(list_valley):
            m += 1
            n += 1


def genMaskandW(list_red, list_green, list_blue, list_valley, rr,rg,rb):  # 计算diff Mask矩阵 和 W
    cap = cv2.VideoCapture(filepath)
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
            print("read frame Error! -1r")
            exit(-1)
        mat_max = np.mat(frame[:, :, 2])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_red)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error! -2r")
            exit(-1)
        mat_min = np.mat(frame[:, :, 2])
        dif = mat_max - mat_min  # 计算dif
        # print(dif1)
        mat_mask_red = (mat_max - mat_min > rr) + 0  # 计算Mask
        # print(Mask)
        # input()

        # 处理蓝色通道
        for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
            if list_blue[i] > max_blue:
                max_blue = list_blue[i]
                index_max_blue = i
            if list_blue[i] < min_blue:
                min_blue = list_blue[i]
                index_min_blue = i
        # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_blue)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error! -1b")
            exit(-1)
        mat_max = np.mat(frame[:, :, 0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_blue)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error! -2b")
            exit(-1)
        mat_min = np.mat(frame[:, :, 0])
        dif = mat_max - mat_min  # 计算dif
        # print(dif1)
        mat_mask_blue = (mat_max - mat_min > rb) + 0  # 计算Mask
        # print(mat_mask_blue)
        # input()

        # 处理绿色通道
        for i in range(list_valley[m], list_valley[n] + 1):  # 找到每个心脏周期的平均红色最大和最小的帧
            if list_green[i] > max_green:
                max_green = list_green[i]
                index_max_green = i
            if list_green[i] < min_green:
                min_green = list_green[i]
                index_min_green = i
        # print(index_max_red, list_red[index_max_red], index_min_red, list_red[index_min_red])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_max_green)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error! -1g")
            exit(-1)
        mat_max = np.mat(frame[:, :, 1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_min_green)
        ret, frame = cap.read()
        if (ret == False):
            print("read frame Error! -2g")
            exit(-1)
        mat_min = np.mat(frame[:, :, 1])
        dif = mat_max - mat_min  # 计算dif
        # print(dif1)
        mat_mask_green = (mat_max - mat_min > rg) + 0  # 计算Mask
        # print(Mask)
        # input()

        for i in range(list_valley[m], list_valley[n] + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if (ret == False):
                print("read frame Error! -3")
                exit(-1)
            mat_c_channel_red = np.mat(frame[:, :, 2])
            list_w_channel_red[i] = np.sum(np.multiply(mat_c_channel_red, mat_mask_red)) / np.sum(mat_mask_red)  # 计算W序列的值
            mat_c_channel_green = np.mat(frame[:, :, 1])
            list_w_channel_green[i] = np.sum(np.multiply(mat_c_channel_green, mat_mask_green)) / np.sum(mat_mask_green)
            mat_c_channel_blue = np.mat(frame[:, :, 0])
            list_w_channel_blue[i] = np.sum(np.multiply(mat_c_channel_blue, mat_mask_blue)) / np.sum(mat_mask_blue)
        if n != len(list_valley):
            m += 1
            n += 1
    # print(n)
    return list_w_channel_red, list_w_channel_green, list_w_channel_blue

def del0(list1,list2,list3):   #删掉列表中的0元素,返回ndarray
    r1 = np.array(list1)
    r2 = np.array(list2)
    r3 = np.array(list3)
    r1 = r1[np.nonzero(r1)]
    r2 = r2[np.nonzero(r2)]
    r3 = r3[np.nonzero(r3)]
    return r1,r2,r3

def ButterFilt(data,n,fs):  #巴特沃斯带通滤波，n为阶数，fs为采样频率
    (b, a) = signal.butter(n, [0.3*2 / fs , 10*2 / fs ], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

filepath = '../../data_show/legal.mp4'
def MinMaxNormalization(list_valley,filtedData): #对每个心脏周期的时间和振幅进行归一化
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


def getFeature_r(list_valley1,Data,minmaxNormalTime,feature):  #收集红色通道基准特征  feature的两个维度分别为 个数 特征(0-9 h1-s4)
    m = 0
    n = 1
    tmp = 0
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(f'j={j}')
        a = list_valley[m]
        b = list_valley[n]
        # plt.plot(minmaxNormalTime[a:b], Data[a:b])
        # plt.show()
        # input()
        SP_index = Data[a:b].tolist().index(np.max(Data[a:b])) + a
        # print(f"SP_index = {SP_index}")
        DN_list = list(signal.argrelmin(Data[a:b])[0])
        # print(f"DN_list_1 = {DN_list + a}")
        if  DN_list:   #如果DN_list非空
            if (DN_list[0] < 5):
                tmp = DN_list[0]
                del (DN_list[0])
            if DN_list:   #如果DN_list非空
                # print(f"tmp = {tmp}")
                # print(f"DN_list_2 = {DN_list + a}")
                DN_index = Data[a:b].tolist().index(np.min([Data[i+a] for i in DN_list])) + a
                # print(f"DN_index = {DN_index}")
                # if DN_index > SP_index:
                #     print("loop")
                #     continue
                DP_index = Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index])) + a + tmp
                # print(f'Data[{tmp+a}:{DN_index}]={Data[tmp+a:DN_index]}')
                # print(f'max = {np.max(Data[tmp+a:DN_index])}')
                # print(f'index = {Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index]))}')
                # print(f"DP_index = {DP_index}")
                # if count == 9:
                #     plt.plot(minmaxNormalTime[a:b], Data[a:b], marker=6, mfc='r', mec='r', ms=5,
                #              markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
                #     plt.ylim(-0.1,1.2)
                #     plt.xlabel('归一化心脏周期')
                #     plt.ylabel('归一化振幅')
                #     plt.title('红色通道基准特征')
                #     plt.show()
                # print(f'cout={count}')
                # input()
                h1 = Data[DP_index]
                h2 = Data[DN_index]
                h3 = Data[SP_index]
                h4 = Data[b - 1]
                t1 = minmaxNormalTime[DP_index]
                t2 = minmaxNormalTime[DN_index] - t1
                t3 = minmaxNormalTime[SP_index] - t2
                t4 = minmaxNormalTime[b - 1] - t3
                s1 = abs(h1 / t1)
                s2 = abs((h2 - h1) / t2)
                s3 = abs((h3 - h2) / t3)
                s4 = abs((h4 - h3) / t4)

                # r通道featu_h
                feature[count][0] = h1  # h1
                feature[count][1] = h2  # h2
                # r通道featu_t
                feature[count][2] = t1  # t1
                feature[count][3] = t2  # t2
                feature[count][4] = t3  # t3
                feature[count][5] = t4  # t4
                # r通道featu_s
                feature[count][6] = s1  # s1
                feature[count][7] = s2  # s2
                feature[count][8] = s3  # s3
                feature[count][9] = s4  # s4

                # print(f'-------{count}------')
                # print(f'SP_index={SP_index}')
                # print(f'DN_index={DN_index}')
                # print(f'DP_index={DP_index}')
                # print(f'h1={feature_h[count][1]}')
                # print(f'h2={feature_h[count][2]}')
                # print(f'h3={feature_h[count][3]}')
                # print(f'h4={feature_h[count][4]}')
                # print(feature_h[count])
                count += 1

        # plt.plot(minmaxNormalTime[a:b],Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery=[DP_index-a,DN_index-a,SP_index-a,b-a-1])
        # plt.show()
        # input()
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print(count)
    print("Error -41")
    return feature
    # print(list_valley)

def getFeature_g(list_valley1,Data,minmaxNormalTime,feature):  #收集绿色通道基准特征  feature的两个维度分别为 个数 特征(10-19 h1-s4)
    m = 0
    n = 1
    tmp = 0
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(f'j={j}')
        a = list_valley[m]
        b = list_valley[n]
        # plt.plot(minmaxNormalTime[a:b], Data[a:b])
        # plt.show()
        # input()
        SP_index = Data[a:b].tolist().index(np.max(Data[a:b])) + a
        # print(f"SP_index = {SP_index}")
        DN_list = list(signal.argrelmin(Data[a:b])[0])
        # print(f"DN_list_1 = {DN_list + a}")
        if  DN_list:   #如果DN_list非空
            if (DN_list[0] < 5):
                tmp = DN_list[0]
                del (DN_list[0])
            if DN_list:   #如果DN_list非空
                # print(f"tmp = {tmp}")
                # print(f"DN_list_2 = {DN_list + a}")
                DN_index = Data[a:b].tolist().index(np.min([Data[i+a] for i in DN_list])) + a
                # print(f"DN_index = {DN_index}")
                # if DN_index > SP_index:
                #     print("loop")
                #     continue
                DP_index = Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index])) + a + tmp
                # print(f'Data[{tmp+a}:{DN_index}]={Data[tmp+a:DN_index]}')
                # print(f'max = {np.max(Data[tmp+a:DN_index])}')
                # print(f'index = {Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index]))}')
                # print(f"DP_index = {DP_index}")
                # plt.plot(minmaxNormalTime[a:b], Data[a:b], marker=6, mfc='r', mec='r', ms=5,
                #          markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
                # plt.show()
                # print(f'cout={count}')
                # input()
                h1 = Data[DP_index]
                h2 = Data[DN_index]
                h3 = Data[SP_index]
                h4 = Data[b - 1]
                t1 = minmaxNormalTime[DP_index]
                t2 = minmaxNormalTime[DN_index] - t1
                t3 = minmaxNormalTime[SP_index] - t2
                t4 = minmaxNormalTime[b - 1] - t3
                s1 = abs(h1 / t1)
                s2 = abs((h2 - h1) / t2)
                s3 = abs((h3 - h2) / t3)
                s4 = abs((h4 - h3) / t4)

                # r通道featu_h
                feature[count][10] = h1  # h1
                feature[count][11] = h2  # h2
                # r通道featu_t
                feature[count][12] = t1  # t1
                feature[count][13] = t2  # t2
                feature[count][14] = t3  # t3
                feature[count][15] = t4  # t4
                # r通道featu_s
                feature[count][16] = s1  # s1
                feature[count][17] = s2  # s2
                feature[count][18] = s3  # s3
                feature[count][19] = s4  # s4

                # print(f'-------{count}------')
                # print(f'SP_index={SP_index}')
                # print(f'DN_index={DN_index}')
                # print(f'DP_index={DP_index}')
                # print(f'h1={feature_h[count][1]}')
                # print(f'h2={feature_h[count][2]}')
                # print(f'h3={feature_h[count][3]}')
                # print(f'h4={feature_h[count][4]}')
                # print(feature_h[count])
                count += 1

        # plt.plot(minmaxNormalTime[a:b],Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery=[DP_index-a,DN_index-a,SP_index-a,b-a-1])
        # plt.show()
        # input()
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print(count)
    print("Error -42")
    return feature
    # print(list_valley)

def getFeature_b(list_valley1,Data,minmaxNormalTime,feature):  #收集蓝色通道基准特征  feature的两个维度分别为 个数 特征(20-29 h1-s4)
    m = 0
    n = 1
    tmp = 0
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(f'j={j}')
        a = list_valley[m]
        b = list_valley[n]
        # plt.plot(minmaxNormalTime[a:b], Data[a:b])
        # plt.show()
        # input()
        DP_index = Data[a:b].tolist().index(np.max(Data[a:b])) + a
        DN_list = list(signal.argrelmin(Data[a:b])[0])
        # print(f"DN_list_1 = {DN_list }")
        if  DN_list:   #如果DN_list非空
            tmp = DN_list[0]
            if (DN_list[0] < 5):
                del (DN_list[0])
            if DN_list:   #如果DN_list非空
                # print(f"DN_list_2 = {DN_list + a}")
                DN_index = Data[a:b].tolist().index(np.min([Data[i+a] for i in DN_list])) + a
                # print(f"DN_index = {DN_index}")
                # if DN_index > SP_index:
                #     print("loop")
                #     continue
                SP_index = Data[DN_index:b].tolist().index(np.max(Data[DN_index:b])) + DN_index
                # print(f'Data[{tmp+a}:{DN_index}]={Data[tmp+a:DN_index]}')
                # print(f'max = {np.max(Data[tmp+a:DN_index])}')
                # print(f'index = {Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index]))}')
                # print(f"DP_index = {DP_index}")
                # plt.plot(minmaxNormalTime[a:b], Data[a:b], marker=6, mfc='r', mec='r', ms=5,
                #          markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
                # plt.plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5,
                #          markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
                # plt.show()
                # print(f'tmp={tmp}')
                # print(f'cout={count}')
                # print(f"DP_index = {DP_index}")
                # print(f"DN_index = {DN_index}")
                # print(f"SP_index = {SP_index}")

                h1 = Data[DP_index]
                h2 = Data[DN_index]
                h3 = Data[SP_index]
                h4 = Data[b - 1]
                t1 = minmaxNormalTime[DP_index]
                t2 = minmaxNormalTime[DN_index] - t1
                t3 = minmaxNormalTime[SP_index] - t2
                t4 = minmaxNormalTime[b - 1] - t3
                s1 = abs(h1 / t1)
                s2 = abs((h2 - h1) / t2)
                s3 = abs((h3 - h2) / t3)
                s4 = abs((h4 - h3) / t4)
                # print(h1, h2, h3, h4)
                # print(t1, t2, t3, t4)
                # print(s1,s2,s3,s4)
                # input()
                # r通道featu_h
                feature[count][20] = h1  # h1
                feature[count][21] = h2  # h2
                # r通道featu_t
                feature[count][22] = t1  # t1
                feature[count][23] = t2  # t2
                feature[count][24] = t3  # t3
                feature[count][25] = t4  # t4
                # r通道featu_s
                feature[count][26] = s1  # s1
                feature[count][27] = s2  # s2
                feature[count][28] = s3  # s3
                feature[count][29] = s4  # s4

                # print(f'-------{count}------')
                # print(f'SP_index={SP_index}')
                # print(f'DN_index={DN_index}')
                # print(f'DP_index={DP_index}')
                # print(f'h1={feature_h[count][1]}')
                # print(f'h2={feature_h[count][2]}')
                # print(f'h3={feature_h[count][3]}')
                # print(f'h4={feature_h[count][4]}')
                # print(feature_h[count])
                count += 1

        # plt.plot(minmaxNormalTime[a:b],Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery=[DP_index-a,DN_index-a,SP_index-a,b-a-1])
        # plt.show()
        # input()
        if count >= num_of_features:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -43")
    return feature
    # print(list_valley)

def ButterFiltHigh(data,n,ft,fs):  #巴特沃斯高通滤波，n为阶数，fs为采样频率,ft为高通滤波器截止频率
    (b, a) = signal.butter(n, ft*2 / fs, 'highpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def getnonfidFeature_r_h1(list_valley1,Data,Time,feature):  #收集第一个红色通道非基准特征  feature的两个维度分别为 个数 特征(30-35 x1-y5)
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


                # if count == 1:
                #     # print(f'x1={x1},x3={x3},x5={x5},y12={y12},y34={y34},y5 = {y5}')
                #     # fig, axes = plt.subplots(2, 1)
                #     mark = [max_index[0]-a,min_index[0]-a,max_index[1]-a,min_index[1]-a,max_index[2]-a]
                #     plt.plot(Time[a:b], Data[a:b],marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                #     # axes[1].plot(Data[a:b], marker=6, mfc='r', mec='r', ms=5, markevery= mark)
                #     # axes[0].set_title(f'{j}')
                #     plt.ylim(-0.1, 1.2)
                #     plt.xlabel('归一化心脏周期')
                #     plt.ylabel('归一化振幅')
                #     plt.title('非基准特征')
                #     plt.show()
                    # print(count)
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
    print("Error -44")
    return feature
    # print(list_valley)

def getnonfidFeature_g_h1(list_valley1,Data,Time,feature):  #收集第一个绿色通道非基准特征  feature的两个维度分别为 个数 特征(36-41 x1-y5)
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
    print("Error -45")
    return feature

def getnonfidFeature_b_h1(list_valley1,Data,Time,feature):  #收集第一个蓝色通道非基准特征  feature的两个维度分别为 个数 特征(42-47 x1-y5)
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
    print("Error -46")
    return feature

def getnonfidFeature_r_h2(list_valley1,Data,Time,feature):  #收集第二个红色通道非基准特征  feature的两个维度分别为 个数 特征(48-53 x1-y5)
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
    print("Error -47")
    return feature

def getnonfidFeature_g_h2(list_valley1,Data,Time,feature):  #收集第二个绿色通道非基准特征  feature的两个维度分别为 个数 特征(54-59 x1-y5)
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
    print("Error -48")
    return feature

def getnonfidFeature_b_h2(list_valley1,Data,Time,feature):  #收集第二个蓝色通道非基准特征  feature的两个维度分别为 个数 特征(60-65 x1-y5)
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
            feature = np.load('../../data_illegal/feature_25_done.npy')[0:10,:]
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -49")
    return feature

def sklearn_pca(legal_feature,legal_feature_70,illegal_feature):        #计算PCA模型
    pca_1 = PCA()
    pca_1.fit(legal_feature)
    # plt.plot(np.arange(1,41), np.cumsum(pca_1.explained_variance_ratio_))
    # plt.xlabel('主成分数量')
    # plt.ylabel('累计可解释方差贡献率')
    # plt.show()
    pca = PCA(0.95)
    pca.fit(legal_feature)
    resu_lf = pca.transform(legal_feature)
    resu_lf_70 = pca.transform(legal_feature_70)
    resu_ill = pca.transform(illegal_feature)
    print(f"保留95%方差的主成分个数为{resu_ill.shape[1]}")
    return resu_lf,resu_lf_70,resu_ill,resu_ill.shape[1]

def CalLegalDist(resu_lf_70,n,num_of_features):         #计算合法用户距离
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

def CalIllegalDist(resu_lf_70,resu_ill,num_of_features):            #计算非法用户距离
    # print(resu_ill.shape[0])
    list_dist_illegal = np.zeros(resu_ill.shape[0])
    for i in range(resu_ill.shape[0]):
        resu = np.zeros(num_of_features)
        for j in range(resu_lf_70.shape[0]):
            resu += resu_lf_70[j, :] - resu_ill[i, :]
        list_dist_illegal[i] = np.linalg.norm(resu) / resu_lf_70.shape[0]
    return list_dist_illegal

def RecuGenYouDenJ(list_dist_legal,list_dist_illegal):              #选择最佳阈值
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
        # print(eit, YouDenJ)
        if YouDenJ > maxYouDenJ:
            maxYouDenJ = YouDenJ
            max_eit = eit
    print(f'最佳距离阈值为{max_eit}, 其最大约登指数为{maxYouDenJ}')
    return max_eit,maxYouDenJ

def verify(list_legalorill,eit):            #用户验证（已经计算过欧氏距离）
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


def verify_ori(unver_feature,legal_feature,eit):            #用户验证（直接使用特征验证）
    pca = PCA(0.95)
    pca.fit(legal_feature[0:40,:])
    unver_resu = pca.transform(unver_feature)
    legal_resu = pca.transform(legal_feature)
    list_dist = np.zeros(unver_resu.shape[0])
    for i in range(unver_resu.shape[0]):
        resu = np.zeros(num_of_features)
        for j in range(legal_resu.shape[0]):
            resu += legal_resu[j, :] - unver_resu[i, :]
        list_dist[i] = np.linalg.norm(resu) / legal_resu.shape[0]
    ver_resu = np.zeros(unver_resu.shape[0])
    print(f'此用户与配置文件的欧氏距离为：{list_dist}')
    index = 0
    for i in list_dist:
        if i <= eit:
            ver_resu[index] = 1
            # print("Yes")
        # else:
        #     # print("No")
        index += 1
    ver_resu_bool = ver_resu > 0
    print(f'此用每个心脏周期的验证结果为：{ver_resu_bool}')
    # print(np.sum(ver_resu))
    # print(ver_resu.shape[0])
    if np.sum(ver_resu) > ver_resu.shape[0]/2:
        print("验证通过")
        return 1
    else:
        print("验证失败")
        return 0

def show1():
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(list_red, color="r")
    axs[0].plot(list_green, color="g")
    axs[0].plot(list_blue, color="b")
    axs[0].set(title='original')
    axs[1].plot(list_w_channel_red, color="r")
    axs[1].plot(list_w_channel_green, color="g")
    axs[1].plot(list_w_channel_blue, color="b")
    axs[1].set(title='w')
    # plt.plot(list_red,color = "k")
    plt.show()



#指尖触摸检测和心脏周期分割

cap = cv2.VideoCapture(filepath)  # 打开视频文件
if cap.isOpened() is False:  # 确认视频是否成果打开
    print('Error')
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图片帧宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取图像帧高度
# fps = float(cap.get(cv2.CAP_PROP_FPS))                 # 获取FPS
frame_channel = 3
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
# print("一共{0}帧".format(frame_count))
list_bool, list_red, list_green, list_blue = detFingerTouch()
list_valley = findmin(list_red, 5)
# print(list_valley)
# np.save("../data/list_valley",list_valley)
# np.save("../data/list_red", list_red)
# np.save("../data/list_green", list_green)
# np.save("../data/list_blue", list_blue)
# print(list_green)
# print(list_blue)
print("-------------------指尖触摸检测和心脏周期分割完成-------------------")

#动态像素选择和心脏波行推导
list_w_channel_red, list_w_channel_green, list_w_channel_blue = genMaskandW(list_red,
                                                                            list_green,
                                                                            list_blue,
                                                                            list_valley,
                                                                            15,6,0.5)
print("-------------------动态像素选择和心脏波行推导完成-------------------")

#数据校准和归一化
nonzero_w_red, nonzero_w_green, nonzero_w_blue = del0(list_w_channel_red, list_w_channel_green, list_w_channel_blue)
filtedData_red = ButterFilt(nonzero_w_red, 8, 30)
filtedData_green = ButterFilt(nonzero_w_green, 8, 30)
filtedData_blue = ButterFilt(nonzero_w_blue, 8, 30)
minmaxNormalData_red,minmaxNormalTime_red = MinMaxNormalization(list_valley,filtedData_red)
minmaxNormalData_green,minmaxNormalTime_green = MinMaxNormalization(list_valley,filtedData_green)
minmaxNormalData_blue,minmaxNormalTime_blue = MinMaxNormalization(list_valley,filtedData_blue)
print("-------------------数据校准和归一化完成-------------------")


# feature  = getFeature(list_valley,minmaxNormalData_red,minmaxNormalData_green,minmaxNormalData_blue,
#                       minmaxNormalTime_red,minmaxNormalTime_green,minmaxNormalTime_blue)

# np.save("../data/minmaxNormalData_red", minmaxNormalData_red)
# np.save("../data/minmaxNormalTime_red", minmaxNormalTime_red)
# np.save("../data/minmaxNormalData_green", minmaxNormalData_green)
# np.save("../data/minmaxNormalTime_green", minmaxNormalTime_green)
# np.save("../data/minmaxNormalData_blue", minmaxNormalData_blue)
# np.save("../data/minmaxNormalTime_blue", minmaxNormalTime_blue)
# np.save("../data/list_red", list_red)
# np.save("../data/list_green", list_green)
# np.save("../data/list_blue", list_blue)
# np.save("../data/list_w_channel_red", list_w_channel_red)
# np.save("../data/list_w_channel_green", list_w_channel_green)
# np.save("../data/list_w_channel_blue", list_w_channel_blue)
# np.save("../data/nonzero_w_red", nonzero_w_red)
# np.save("../data/nonzero_w_green", nonzero_w_green)
# np.save("../data/nonzero_w_blue", nonzero_w_blue)
# np.save("../data/filtedData_red", filtedData_red)
# np.save("../data/filtedData_green", filtedData_green)
# np.save("../data/filtedData_blue", filtedData_blue)



#基准特征提取
feature = np.zeros((num_of_features,66), dtype=np.float64)
feature = getFeature_r(list_valley,minmaxNormalData_red,minmaxNormalTime_red,feature)
# print(feature[:,0:10])
feature = getFeature_g(list_valley,minmaxNormalData_green,minmaxNormalTime_green,feature)
# print(feature[:,10:20])
feature = getFeature_b(list_valley,minmaxNormalData_blue,minmaxNormalTime_blue,feature)
# print(feature[:,20:30])
# np.save(f"../{foldername}/feature_70.npy",feature)
# np.save(f"../{foldername}/feature_{num_of_features}_done.npy",feature)
print("-------------------基准特征提取完成-------------------")

#非基准特征提取
filtedData_red_h1 = ButterFiltHigh(nonzero_w_red, 8,4, 30)
filtedData_green_h1 = ButterFiltHigh(nonzero_w_green, 8,4, 30)
filtedData_blue_h1 = ButterFiltHigh(nonzero_w_blue, 8,4, 30)
filtedData_red_h2 = ButterFiltHigh(nonzero_w_red, 8,5, 30)
filtedData_green_h2 = ButterFiltHigh(nonzero_w_green, 8,5, 30)
filtedData_blue_h2 = ButterFiltHigh(nonzero_w_blue, 8,5, 30)
minmaxNormalData_red_h1,minmaxNormalTime_red_h1 = MinMaxNormalization(list_valley,filtedData_red_h1)
minmaxNormalData_green_h1,minmaxNormalTime_green_h1 = MinMaxNormalization(list_valley,filtedData_green_h1)
minmaxNormalData_blue_h1,minmaxNormalTime_blue_h1 = MinMaxNormalization(list_valley,filtedData_blue_h1)
minmaxNormalData_red_h2,minmaxNormalTime_red_h2 = MinMaxNormalization(list_valley,filtedData_red_h2)
minmaxNormalData_green_h2,minmaxNormalTime_green_h2 = MinMaxNormalization(list_valley,filtedData_green_h2)
minmaxNormalData_blue_h2,minmaxNormalTime_blue_h2 = MinMaxNormalization(list_valley,filtedData_blue_h2)
feature = getnonfidFeature_r_h1(list_valley,minmaxNormalData_red_h1,minmaxNormalTime_red_h1,feature)
feature = getnonfidFeature_g_h1(list_valley,minmaxNormalData_green_h1,minmaxNormalTime_green_h1,feature)
feature = getnonfidFeature_b_h1(list_valley,minmaxNormalData_blue_h1,minmaxNormalTime_blue_h1,feature)
feature = getnonfidFeature_r_h2(list_valley,minmaxNormalData_red_h2,minmaxNormalTime_red_h2,feature)
feature = getnonfidFeature_g_h2(list_valley,minmaxNormalData_green_h2,minmaxNormalTime_green_h2,feature)
feature = getnonfidFeature_b_h2(list_valley,minmaxNormalData_blue_h2,minmaxNormalTime_blue_h2,feature)
# getnonfidFeature(list_valley,minmaxNormalData_green_1,minmaxNormalTime_green_1,feature)
# getnonfidFeature(list_valley,minmaxNormalData_blue_1,minmaxNormalTime_blue_1,feature)
np.save(f'../{foldername}/illegal/feature.npy',feature)
print(feature)
cap.release()
print("-------------------非基准特征提取完成-------------------")
print(f'特征矩阵形状为{feature.shape}')