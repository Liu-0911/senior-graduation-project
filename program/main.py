import cv2
import numpy as np
import os
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal

from detecta import detect_peaks

Channel_R = 2
Channel_G = 1
Channel_B = 0
# filepath = '../MP4/test-5.mp4'
filepath = '../MP4/lbx-tabs7-1.mp4'

t = 0.55
pr = 0.95
cap = cv2.VideoCapture(filepath)  # 打开视频文件
result_list = []
ava_red_list = []
ava_green_list = []
ava_blue_list = []
num = 0
ava_red = 0


if cap.isOpened() is False:  # 确认视频是否成果打开
    print('Error')
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图片帧宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取图像帧高度
# fps = float(cap.get(cv2.CAP_PROP_FPS))                 # 获取FPS
frame_channel = 3
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
print("一共{0}帧".format(frame_count))


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


def fun1():  # 读取视频逐帧处理并输出红色通道平均值
    ret, frame = cap.read()  # 读取一帧图像，当视频帧读取完毕ret标识符为False
    num_of_frame = 1
    while ret:
        num, ava_red, ava_green, ava_blue = count(frame, 0, t, 0, 0, 0)
        ava_red_list.append(ava_red)
        ava_green_list.append(ava_green)
        ava_blue_list.append(ava_blue)
        if (num > frame_width * frame_height * pr):  # 判断是否可以开始推到并输出01串
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
    plt.figure(figsize=(14, 7), dpi=500)
    plt.plot(list, marker=6, mfc='r', mec='r', ms=5, markevery=list_valley)
    plt.show()
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


def genMaskandW(list_red, list_green, list_blue, list_valley, rr,rg,rb):  # 计算dif Mask矩阵 和 W
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
    print(n)
    return list_w_channel_red, list_w_channel_green, list_w_channel_blue


def del0(list1, list2, list3):  # 删掉列表中的0元素,返回ndarray
    r1 = np.array(list1)
    r2 = np.array(list2)
    r3 = np.array(list3)
    r1 = r1[np.nonzero(r1)]
    r2 = r2[np.nonzero(r2)]
    r3 = r3[np.nonzero(r3)]
    return r1, r2, r3


def ButterFilt(data, n, fs):  # 巴特沃斯带通滤波，n为阶数，fs为采样频率
    (b, a) = signal.butter(n, [0.3 * 2 / fs, 10 * 2 / fs], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData


def MinMaxNormalization(list_valley, filtedData):
    m = 0
    n = 1
    minmaxNormalData = np.zeros(len(filtedData))
    minmaxNormalTime = np.zeros(len(filtedData))
    for j in range(len(list_valley) - 1):
        temp = filtedData[list_valley[m] - list_valley[0]:list_valley[n] - list_valley[0] + 1]
        min = np.min(temp)
        max_min = np.ptp(temp)
        result = np.array([(i - min) / max_min for i in temp])
        minmaxNormalData[list_valley[m] - list_valley[0]:list_valley[n] - list_valley[0] + 1] = result
        for i in range(list_valley[m] - list_valley[0], list_valley[n] - list_valley[0]):
            result = (i - list_valley[m] + list_valley[0]) / (list_valley[n] - list_valley[m] - 1)
            # print(i,end=" ")
            # print(result,end=" ")
            minmaxNormalTime[i] = result

        if n != len(list_valley):
            m += 1
            n += 1
    return minmaxNormalData, minmaxNormalTime


def getFeature(list_valley1, Data_red, Data_green, Data_blue, Time_red, Time_green,
               Time_blue):  # feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    count = 0
    feature = np.zeros((3, 3, 10, 5), dtype=np.float64)
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        a = list_valley[m]
        b = list_valley[n]
        # 处理红色通道
        SP_index = Data_red[a:b].tolist().index(np.max(Data_red[a:b])) + a
        DN_list = list(signal.argrelmin(Data_red[a:b])[0])
        if not DN_list:
            continue
        if (DN_list[0] < 5):
            del (DN_list[0])
        DN_index = Data_red[a:b].tolist().index(np.min([Data_red[i + a] for i in DN_list])) + a
        DP_index = Data_red[a:DN_index].tolist().index(np.max(Data_red[a:DN_index])) + a
        # r通道featu_h
        feature[1][0][count][1] = Data_red[DP_index]
        feature[1][0][count][2] = Data_red[DN_index]
        feature[1][0][count][3] = Data_red[SP_index]
        feature[1][0][count][4] = Data_red[b - 1]
        # r通道featu_t
        feature[1][1][count][1] = Time_red[DP_index]
        feature[1][1][count][2] = Time_red[DN_index] - feature[1][1][count][1]
        feature[1][1][count][3] = Time_red[SP_index] - feature[1][1][count][2]
        feature[1][1][count][4] = Time_red[b - 1] - feature[1][1][count][3]
        # r通道featu_s
        feature[1][2][count][1] = abs(feature[1][0][count][1] / feature[1][1][count][1])
        feature[1][2][count][2] = abs((feature[1][0][count][2] - feature[1][0][count][1]) / feature[1][1][count][2])
        feature[1][2][count][3] = abs((feature[1][0][count][3] - feature[1][0][count][2]) / feature[1][1][count][3])
        feature[1][2][count][4] = abs((feature[1][0][count][4] - feature[1][0][count][3]) / feature[1][1][count][4])

        # 处理绿色通道
        SP_index = Data_green[a:b].tolist().index(np.max(Data_green[a:b])) + a
        DN_list = list(signal.argrelmin(Data_green[a:b])[0])
        if not DN_list:
            continue
        if (DN_list[0] < 5):
            del (DN_list[0])
        DN_index = Data_green[a:b].tolist().index(np.min([Data_green[i + a] for i in DN_list])) + a
        DP_index = Data_green[a:DN_index].tolist().index(np.max(Data_green[a:DN_index])) + a
        # g通道featu_h
        feature[0][0][count][1] = Data_green[DP_index]
        feature[0][0][count][2] = Data_green[DN_index]
        feature[0][0][count][3] = Data_green[SP_index]
        feature[0][0][count][4] = Data_green[b - 1]
        # g通道featu_t
        feature[0][1][count][1] = Time_green[DP_index]
        feature[0][1][count][2] = Time_green[DN_index] - feature[0][1][count][1]
        feature[0][1][count][3] = Time_green[SP_index] - feature[0][1][count][2]
        feature[0][1][count][4] = Time_green[b - 1] - feature[0][1][count][3]
        # g通道featu_s
        feature[0][2][count][1] = abs(feature[0][0][count][1] / feature[0][1][count][1])
        feature[0][2][count][2] = abs((feature[0][0][count][2] - feature[0][0][count][1]) / feature[0][1][count][2])
        feature[0][2][count][3] = abs((feature[0][0][count][3] - feature[0][0][count][2]) / feature[0][1][count][3])
        feature[0][2][count][4] = abs((feature[0][0][count][4] - feature[0][0][count][3]) / feature[0][1][count][4])

        # 处理蓝色通道
        SP_index = Data_blue[a:b].tolist().index(np.max(Data_blue[a:b])) + a
        DN_list = list(signal.argrelmin(Data_blue[a:b])[0])
        if not DN_list:
            continue
        if (DN_list[0] < 5):
            del (DN_list[0])
        DN_index = Data_blue[a:b].tolist().index(np.min([Data_blue[i + a] for i in DN_list])) + a
        DP_index = Data_blue[a:DN_index].tolist().index(np.max(Data_blue[a:DN_index])) + a
        # r通道featu_h
        feature[2][0][count][1] = Data_blue[DP_index]
        feature[2][0][count][2] = Data_blue[DN_index]
        feature[2][0][count][3] = Data_blue[SP_index]
        feature[2][0][count][4] = Data_blue[b - 1]
        # r通道featu_t
        feature[2][1][count][1] = Time_blue[DP_index]
        feature[2][1][count][2] = Time_blue[DN_index] - feature[2][1][count][1]
        feature[2][1][count][3] = Time_blue[SP_index] - feature[2][1][count][2]
        feature[2][1][count][4] = Time_blue[b - 1] - feature[2][1][count][3]
        # r通道featu_s
        feature[2][2][count][1] = abs(feature[2][0][count][1] / feature[2][1][count][1])
        feature[2][2][count][2] = abs((feature[2][0][count][2] - feature[2][0][count][1]) / feature[2][1][count][2])
        feature[2][2][count][3] = abs((feature[2][0][count][3] - feature[2][0][count][2]) / feature[2][1][count][3])
        feature[2][2][count][4] = abs((feature[2][0][count][4] - feature[2][0][count][3]) / feature[2][1][count][4])

        # print(f'-------{count}------')
        # print(f'SP_index={SP_index}')
        # print(f'DN_index={DN_index}')
        # print(f'DP_index={DP_index}')
        # print(f'h1={feature_h[count][1]}')
        # print(f'h2={feature_h[count][2]}')
        # print(f'h3={feature_h[count][3]}')
        # print(f'h4={feature_h[count][4]}')
        # print(feature_h[count])

        # plt.plot(minmaxNormalTime[a:b], Data_red[a:b], marker=6, mfc='r', mec='r', ms=5, markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
        # plt.show()
        # input()
        count += 1
        if count >= 10:
            return feature
        if n != len(list_valley):
            m += 1
            n += 1
    print("Error -4")
    return feature
    # print(list_valley)


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


list_bool, list_red, list_green, list_blue = fun1()
list_valley = findmin(list_red, 5)
print(list_valley)
# np.save("../data/list_valley",list_valley)
# np.save("../data/list_red", list_red)
# np.save("../data/list_green", list_green)
# np.save("../data/list_blue", list_blue)
print(list_green)
print(list_blue)
print("over1")

list_w_channel_red, list_w_channel_green, list_w_channel_blue = genMaskandW(list_red,
                                                                            list_green,
                                                                            list_blue,
                                                                            list_valley,
                                                                            15,6,0.5)
nonzero_w_red, nonzero_w_green, nonzero_w_blue = del0(list_w_channel_red, list_w_channel_green, list_w_channel_blue)
filtedData_red = ButterFilt(nonzero_w_red, 8, 30)
filtedData_green = ButterFilt(nonzero_w_green, 8, 30)
filtedData_blue = ButterFilt(nonzero_w_blue, 8, 30)
minmaxNormalData_red,minmaxNormalTime_red = MinMaxNormalization(list_valley,filtedData_red)
minmaxNormalData_green,minmaxNormalTime_green = MinMaxNormalization(list_valley,filtedData_green)
minmaxNormalData_blue,minmaxNormalTime_blue = MinMaxNormalization(list_valley,filtedData_blue)
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
print("Over")
cap.release()
