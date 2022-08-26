import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
from detecta import detect_peaks

# feature结构   feature[count][66]  0-1 h1h2 2-5 t1-t4 6-9 s1-s4


num_of_features = 70

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


def getFeature_r(list_valley1,Data,minmaxNormalTime,feature):  #feature前三维度分别为 通道（r-0 g-1 b-2） 特征（h-0 t-1 s-2） 个数
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
                print(f'Data[{tmp+a}:{DN_index}]={Data[tmp+a:DN_index]}')
                print(f'max = {np.max(Data[tmp+a:DN_index])}')
                print(f'index = {Data[tmp+a:DN_index].tolist().index(np.max(Data[tmp+a:DN_index]))}')
                print(f"DP_index = {DP_index}")
                plt.plot(minmaxNormalTime[a:b], Data[a:b], marker=6, mfc='r', mec='r', ms=5,
                         markevery=[DP_index - a, DN_index - a, SP_index - a, b - a - 1])
                plt.ylim(-0.1,1.2)
                plt.xlabel('归一化心脏周期')
                plt.ylabel('归一化振幅')
                plt.show()
                print(f'cout={count}')
                input()
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
    print("Error -4")
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
    print("Error -4")
    return feature
    # print(list_valley)

list_valley = np.load('../data/list_valley.npy')
minmaxNormalData_red = np.load("../data/minmaxNormalData_red.npy")
minmaxNormalTime_red = np.load("../data/minmaxNormalTime_red.npy")
minmaxNormalData_blue = np.load("../data/minmaxNormalData_blue.npy")
minmaxNormalTime_blue = np.load("../data/minmaxNormalTime_blue.npy")
minmaxNormalData_green = np.load("../data/minmaxNormalData_green.npy")
minmaxNormalTime_green = np.load("../data/minmaxNormalTime_green.npy")
feature = np.zeros((num_of_features,66), dtype=np.float64)
feature = getFeature_r(list_valley,minmaxNormalData_red,minmaxNormalTime_red,feature)
# print(feature[:,0:10])
feature = getFeature_g(list_valley,minmaxNormalData_green,minmaxNormalTime_green,feature)
# print(feature[:,10:20])
feature = getFeature_b(list_valley,minmaxNormalData_blue,minmaxNormalTime_blue,feature)
# print(feature[:,20:30])
print("1111")
np.save('../data/feature_70.npy',feature)
print(feature)
