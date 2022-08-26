#获取攻击模型的数据

import cv2
from numba import jit
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from sklearn.decomposition import PCA


from detecta import detect_peaks

Channel_R = 2
Channel_G = 1
Channel_B = 0
filepath = '../MP4/lbx-honor-1.mp4'
foldername = 'data_legal_lbx'
num_of_features = 70

t = 0.78
pr = 0.95
cap = cv2.VideoCapture(filepath)  # 打开视频文件
result_list = []
ava_red_list = []
ava_green_list = []
ava_blue_list = []
num = 0
ava_red = 0

#
# if cap.isOpened() is False:  # 确认视频是否成果打开
#     print('Error')
#     exit(1)
#
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图片帧宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取图像帧高度
# fps = float(cap.get(cv2.CAP_PROP_FPS))                 # 获取FPS
frame_channel = 3
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
# print("一共{0}帧".format(frame_count))


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


def findmin(list, min,showFlag = True):  # 波谷检测算法，返回符合条件的波谷index
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
    if showFlag:
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
    num_of_0_bef = np.nonzero(r1)[0][0]
    # print(np.arange(np.nonzero(r1)[0][0],np.nonzero(r1)[0][-1]+1))
    # r = r1[np.arange(np.nonzero(r1)[0][0],np.nonzero(r1)[0][-1]+1)]
    r1 = r1[np.nonzero(r1)]
    r2 = r2[np.nonzero(r1)]
    r3 = r3[np.nonzero(r1)]
    return r1, r2, r3 , num_of_0_bef


def ButterFilt(data, n, fs,low=0.3,high=10):  # 巴特沃斯带通滤波，n为阶数，fs为采样频率
    (b, a) = signal.butter(n, [low * 2 / fs, high * 2 / fs], 'bandpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData


def MinMaxNormalization(list_valley, filtedData):
    m = 0
    n = 1
    minmaxNormalData = np.zeros(len(filtedData))
    minmaxNormalTime = np.zeros(len(filtedData))
    # print(filtedData.shape)
    for j in range(len(list_valley) - 1):
        # print(f'begin={list_valley[m]},end = {list_valley[n] + 1}')
        temp = filtedData[list_valley[m] :list_valley[n] + 1]
        # plt.plot(temp)
        # plt.show()
        # print(list_valley[m] - list_valley[0],list_valley[n] - list_valley[0])
        # print(f'filtedData={filtedData[701:720+ 1]}')
        # print(temp)
        min = np.min(temp)
        # print(min)
        max_min = np.ptp(temp)
        result = np.array([(i - min) / max_min for i in temp])
        minmaxNormalData[list_valley[m] :list_valley[n] + 1] = result
        for i in range(list_valley[m] , list_valley[n] ):
            # print(i)
            result = (i - list_valley[m] ) / (list_valley[n] - list_valley[m] - 1)
            # print(f'{result} = ({i} - {list_valley[m]} ) / ({list_valley[n]} - {list_valley[m]} - 1)')
            # print(i,end=" ")
            # print(result,end=" ")
            minmaxNormalTime[i] = result
        # print(minmaxNormalTime[list_valley[m]:list_valley[n]])
        # input()
        if n != len(list_valley):
            m += 1
            n += 1
    return minmaxNormalData, minmaxNormalTime

def getFeature_r(list_valley,Data,minmaxNormalTime,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
    m = 0
    n = 1
    tmp = 0
    count = 0
    # feature_h = np.zeros((3,3,10, 5), dtype=np.float64)
    # feature_s = np.zeros((3,3,10, 5), dtype=np.float64)
    # list_valley = np.array(list_valley1) - list_valley1[0]
    for j in range(len(list_valley) - 1):
        # print(f'j={j}')
        a = list_valley[m]
        b = list_valley[n]
        # plt.plot(minmaxNormalTime[a:b], Data[a:b])
        # plt.show()
        # input()
        # print(a,b)
        SP_index = Data[a:b].tolist().index(np.max(Data[a:b])) + a
        # print(f"SP_index = {SP_index}")
        DN_list = list(signal.argrelmin(Data[a:b])[0])
        # print(f"DN_list = {DN_list}")
        # plt.plot(minmaxNormalTime[a:b], Data[a:b])
        # plt.show()
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
                print(f'Data[{tmp+a}:{DN_index}]={Data[tmp+a:DN_index]}')
                print(f'max = {np.max(Data[tmp+a:DN_index])}')
                print(f'index = {Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index]))}')
                print(f"DP_index = {DP_index}")
                plt.plot(minmaxNormalTime[a:b], Data[a:b], marker=6, mfc='r', mec='r', ms=5,
                         markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
                plt.show()
                print(f'cout={count}')
                input()
                h1 = Data[DP_index]
                h2 = Data[DN_index]
                h3 = Data[SP_index]
                h4 = Data[b - 1]
                t1 = minmaxNormalTime[DP_index] if minmaxNormalTime[DP_index] != 0 else 0.8
                t2 = abs(minmaxNormalTime[DN_index] - t1)
                t3 = abs(minmaxNormalTime[SP_index] - t2)
                t4 = abs(minmaxNormalTime[b - 1] - t3)
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
    print("Error -4")
    sum = 0
    for i in range(int((num_of_features - count)/count)):
        feature[count*(i+1):count*(i+2),0:10]  = feature[0:count,0:10]
        sum += count
    feature[count+sum:,0:10] = feature[0:num_of_features - count - sum,0:10]
    return feature
    # print(list_valley)

def getFeature_g(list_valley1,Data,minmaxNormalTime,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
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
                t1 = minmaxNormalTime[DP_index] if minmaxNormalTime[DP_index] != 0 else 0.35
                t2 = abs(minmaxNormalTime[DN_index] - t1)
                t3 = abs(minmaxNormalTime[SP_index] - t2) if abs(minmaxNormalTime[SP_index] - t2) != 0 else 0.15
                t4 = abs(minmaxNormalTime[b - 1] - t3)
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
    print("Error -4")
    return feature
    # print(list_valley)

def getFeature_b(list_valley1,Data,minmaxNormalTime,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
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
                t1 = minmaxNormalTime[DP_index] if minmaxNormalTime[DP_index] != 0 else 0.8
                t2 = abs(minmaxNormalTime[DN_index] - t1) if abs(minmaxNormalTime[DN_index] - t1) != 0 else 0.4
                t3 = abs(minmaxNormalTime[SP_index] - t2)
                t4 = abs(minmaxNormalTime[b - 1] - t3)
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
    print("Error -4")
    return feature
    # print(list_valley)


def ButterFiltHigh(data,n,ft,fs):  #巴特沃斯高通滤波，n为阶数，fs为采样频率,ft为高通滤波器截止频率
    (b, a) = signal.butter(n, ft*2 / fs, 'highpass')
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

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
    sum = 0
    for i in range(int((num_of_features - count) / count)):
        feature[count * (i + 1):count * (i + 2), 30:36] = feature[0:18, 30:36]
        sum += count
    feature[count + sum:,30:36] = feature[0:num_of_features - count - sum, 30:36]
    print("Error -4 r h1")
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
    print("Error -4 g h1")
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
    print("Error -4 b h1")
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
    print("Error -4 r h2")
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
    print("Error -4 g h2")
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
    print("Error -4 b h2")
    return feature


def sklearn_pca(legal_feature,legal_feature_70,illegal_feature):
    pca = PCA(0.985)
    pca.fit(legal_feature)
    resu_lf = pca.transform(legal_feature)
    resu_lf_70 = pca.transform(legal_feature_70)
    resu_ill = pca.transform(illegal_feature)
    print(resu_lf)
    return resu_lf,resu_lf_70,resu_ill,resu_lf.shape[1]

def CalLegalDist(resu_lf_70,n,num_of_features):
    resu_legal = resu_lf_70[:n,:]
    print(resu_legal.shape[0])
    list_dist_legal = np.zeros(resu_legal.shape[0])
    for i in range(resu_legal.shape[0]):
        resu = np.zeros(num_of_features)
        for j in range(resu_lf_70.shape[0]):
            resu += resu_lf_70[j, :] - resu_legal[i, :]
        list_dist_legal[i] = np.linalg.norm(resu) / resu_lf_70.shape[0]
    return list_dist_legal

def CalIllegalDist(resu_lf_70,resu_ill,num_of_features):
    print(resu_ill.shape[0])
    list_dist_illegal = np.zeros(resu_ill.shape[0])
    for i in range(resu_ill.shape[0]):
        resu = np.zeros(num_of_features)
        for j in range(resu_lf_70.shape[0]):
            resu += resu_lf_70[j, :] - resu_ill[i, :]
        list_dist_illegal[i] = np.linalg.norm(resu) / resu_lf_70.shape[0]
    return list_dist_illegal

def RecuGenYouDenJ(list_dist_legal,list_dist_illegal):
    maxYouDenJ = max_eit = 0
    for eit in np.arange(1,15,0.5):
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



def part1():   #用于确定findmin中的参数
    # list_bool, list_red, list_green, list_blue = fun1()
    # print(f'list_bool={list_bool}')
    # np.save(f"../{foldername}/list_red.npy", list_red)
    # np.save(f"../{foldername}/list_green.npy", list_green)
    # np.save(f"../{foldername}/list_blue.npy", list_blue)
    # np.save(f"../{foldername}/list_bool.npy", list_bool)

    # list_bool = np.load(f'../{foldername}/list_bool.npy')
    # print(f'list_bool={list_bool}')
    list_red = np.load(f'../{foldername}/list_red.npy')
    list_valley = findmin(list_red, 4)    #确定这里的的参数
    np.save(f"../{foldername}/list_valley.npy", list_valley)
    print("Over!")
    print(list_valley)

def part2():    #用于确定rr rg rb
    list_red = np.load(f"../{foldername}/list_red.npy")
    list_green = np.load(f"../{foldername}/list_green.npy")
    list_blue = np.load(f"../{foldername}/list_blue.npy")
    list_valley = np.load(f"../{foldername}/list_valley.npy")

    list_w_channel_red, list_w_channel_green, list_w_channel_blue = genMaskandW(list_red,
                                                                                list_green,
                                                                                list_blue,
                                                                                list_valley,
                                                                                5,0.5,1)

    fig , ax = plt.subplots(3,1,figsize=(30,30))
    ax[0].plot(list_w_channel_red,color= 'r')
    ax[1].plot(list_w_channel_green, color='g')
    ax[2].plot(list_w_channel_blue, color='b')
    # plt.plot(list_w_channel_green,color= 'r')
    plt.show()
    np.save(f"../{foldername}/list_w_channel_red.npy", list_w_channel_red)
    np.save(f"../{foldername}/list_w_channel_green.npy", list_w_channel_green)
    np.save(f"../{foldername}/list_w_channel_blue.npy", list_w_channel_blue)
    print("Over!")


def part3():
    list_w_channel_red = np.load(f"../{foldername}/list_w_channel_red.npy")
    list_w_channel_green = np.load(f"../{foldername}/list_w_channel_green.npy")
    list_w_channel_blue = np.load(f"../{foldername}/list_w_channel_blue.npy")
    list_valley = np.load(f"../{foldername}/list_valley.npy")
    nonzero_w_red, nonzero_w_green, nonzero_w_blue , num_of_0_bef = del0(list_w_channel_red, list_w_channel_green, list_w_channel_blue)
    np.save(f"../{foldername}/nonzero_w_red.npy",nonzero_w_red)
    np.save(f"../{foldername}/nonzero_w_green.npy", nonzero_w_green)
    np.save(f"../{foldername}/nonzero_w_blue.npy", nonzero_w_blue)
    print(num_of_0_bef)
    filtedData_red = ButterFilt(nonzero_w_red, 8, 30)
    filtedData_green = ButterFilt(nonzero_w_green, 8, 30)
    filtedData_blue = ButterFilt(nonzero_w_blue, 8, 30)
    # fig, ax = plt.subplots(3, 1, figsize=(30, 30))
    # ax[0].plot(filtedData_red, color='r', marker=6, mfc='black', mec='black', ms=5,
    #                      markevery= (np.array(list_valley) - list_valley[0]))
    # ax[1].plot(filtedData_green, color='g')
    # ax[2].plot(filtedData_blue, color='b')
    # # plt.plot(list_w_channel_green,color= 'r')
    # plt.show()
    list_valley_afterfilt = findmin(filtedData_red, 3.5,False)
    np.save(f"../{foldername}/list_valley_afterfilt.npy", list_valley_afterfilt)
    print('1')
    minmaxNormalData_red, minmaxNormalTime_red = MinMaxNormalization(list_valley_afterfilt, filtedData_red)
    print('2')
    minmaxNormalData_green, minmaxNormalTime_green = MinMaxNormalization(list_valley_afterfilt, filtedData_green)
    print('3')
    minmaxNormalData_blue, minmaxNormalTime_blue = MinMaxNormalization(list_valley_afterfilt, filtedData_blue)
    print('4')
    feature = np.zeros((num_of_features, 66), dtype=np.float64)
    print('5')
    feature = getFeature_r(list_valley_afterfilt, minmaxNormalData_red, minmaxNormalTime_red, feature)
    # print(feature[:,0:10])
    print('6')
    feature = getFeature_g(list_valley_afterfilt, minmaxNormalData_green, minmaxNormalTime_green, feature)
    # print(feature[:,10:20])
    print('7')
    feature = getFeature_b(list_valley_afterfilt, minmaxNormalData_blue, minmaxNormalTime_blue, feature)
    # print(feature[:,20:30])
    print(feature)
    np.save(f"../{foldername}/feature_{num_of_features}.npy",feature)
    print("Over！")


def part4():
    nonzero_w_red = np.load(f'../{foldername}/nonzero_w_red.npy')
    nonzero_w_green = np.load(f'../{foldername}/nonzero_w_green.npy')
    nonzero_w_blue = np.load(f'../{foldername}/nonzero_w_blue.npy')
    list_valley = np.load(f'../{foldername}/list_valley_afterfilt.npy')
    feature = np.load(f'../{foldername}/feature_{num_of_features}.npy')
    filtedData_red_h1 = ButterFiltHigh(nonzero_w_red, 8, 4, 30)
    filtedData_green_h1 = ButterFiltHigh(nonzero_w_green, 8, 4, 30)
    filtedData_blue_h1 = ButterFiltHigh(nonzero_w_blue, 8, 4, 30)
    filtedData_red_h2 = ButterFiltHigh(nonzero_w_red, 8, 5, 30)
    filtedData_green_h2 = ButterFiltHigh(nonzero_w_green, 8, 5, 30)
    filtedData_blue_h2 = ButterFiltHigh(nonzero_w_blue, 8, 5, 30)
    minmaxNormalData_red_h1, minmaxNormalTime_red_h1 = MinMaxNormalization(list_valley, filtedData_red_h1)
    minmaxNormalData_green_h1, minmaxNormalTime_green_h1 = MinMaxNormalization(list_valley, filtedData_green_h1)
    minmaxNormalData_blue_h1, minmaxNormalTime_blue_h1 = MinMaxNormalization(list_valley, filtedData_blue_h1)
    minmaxNormalData_red_h2, minmaxNormalTime_red_h2 = MinMaxNormalization(list_valley, filtedData_red_h2)
    minmaxNormalData_green_h2, minmaxNormalTime_green_h2 = MinMaxNormalization(list_valley, filtedData_green_h2)
    minmaxNormalData_blue_h2, minmaxNormalTime_blue_h2 = MinMaxNormalization(list_valley, filtedData_blue_h2)
    feature = getnonfidFeature_r_h1(list_valley, minmaxNormalData_red_h1, minmaxNormalTime_red_h1, feature)
    feature = getnonfidFeature_g_h1(list_valley, minmaxNormalData_green_h1, minmaxNormalTime_green_h1, feature)
    feature = getnonfidFeature_b_h1(list_valley, minmaxNormalData_blue_h1, minmaxNormalTime_blue_h1, feature)
    feature = getnonfidFeature_r_h2(list_valley, minmaxNormalData_red_h2, minmaxNormalTime_red_h2, feature)
    feature = getnonfidFeature_g_h2(list_valley, minmaxNormalData_green_h2, minmaxNormalTime_green_h2, feature)
    feature = getnonfidFeature_b_h2(list_valley, minmaxNormalData_blue_h2, minmaxNormalTime_blue_h2, feature)
    print(feature)
    np.save(f"../{foldername}/feature_{num_of_features}_done.npy",feature)


def part5():
    legal_feature_70 = np.load(f'../data_legal_lbx/feature_70_done.npy')
    legal_feature = legal_feature_70[:40,:]
    illegal_feature =  np.load("../data_illegal/feature_25_done.npy")
    resu_lf, resu_lf_70, resu_ill, pca_features = sklearn_pca(legal_feature, legal_feature_70, illegal_feature)
    list_dist_legal = CalLegalDist(resu_lf_70, 10,pca_features)
    list_dist_illegal = CalIllegalDist(resu_lf_70, resu_ill,pca_features)
    print(list_dist_legal)
    print(list_dist_illegal)
    eit,YouDenJ = RecuGenYouDenJ(list_dist_legal, list_dist_illegal)

#------------------------------------------第一部分，确定findmin的参数------------------
# part1()


#------------------------------------------第二部分，确定genMashandW的参数--------------
# part2()

#------------------------------------------第二部分，getfeature的参数--------------
part3()

#------------------------------------------第二部分，getnonfeature的参数--------------
# part4()


# part5()


















