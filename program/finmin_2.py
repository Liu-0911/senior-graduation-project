from detecta import detect_peaks
import matplotlib.pyplot as plt
import numpy as np
from tem import test4,test2
from scipy.signal import argrelextrema


def findmin(list,min):
    list_result = []
    left = right = 0
    list_down = detect_peaks(list,mpd=1,valley= True, show=False,edge=None)

    # print(list_down)
#     for index,value in enumerate(list_down):
    for index in list_down:
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
        while(index >= 0):
            index -= 1
            if(list[index]==value):
                left = index
                leftfindflag = True
                break
            elif(list[index]<value):
                left = index + 1
                leftfindflag = True
                break
        # print(f"left = {left}")
        for i in range(left,tmp+1):
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
        while(index <= len(list)-2):
            index += 1
            if(list[index]==value ):
                right = index
                rightfindflag = True
                break
            elif(list[index]<value):
                right = index - 1
                rightfindflag = True
                break
        if(rightfindflag == False):
            right = tmp
        # print(f"right = {right}")
        for i in range(tmp,right+1):
            list_right_value_tem.append(list[i])
            list_right_index_tem.append(i)
        # print("list_right_value_tem =",end=" ")
        # print(list_right_value_tem)
        list_right_up_tem = detect_peaks(list_left_value_tem,mpd=1,valley= False, show=False,edge=None)
        # print("list_right_up_tem =", end=" ")
        # print(list_right_up_tem)
        # print(f"len(list)={len(list)}")
        maxright = 0
        for i in list_right_up_tem:
            if(list[list_left_index_tem[i]]>maxright):
                # print(f"list_left_value_tem[i]={list_left_value_tem[i]},list[list_left_value_tem[i]]={list[list_left_value_tem[i]]}")
                maxright = list[list_left_index_tem[i]]
        # print(f"maxright={maxright},maxright-value={maxright-value}")
        max = maxleft if maxright > maxleft else maxleft
        if(max-value>=min):
            list_result.append(tmp)
    tmp = 0
    for index,value in enumerate(list_result):
        if index < len(list_result) - 1:
            if list_result[index+1] - value >= 10:
                tmp = index
                break
    # print(list_result)
    list_result = list_result[tmp:len(list_result)]
    # print(list_result)
    plt.figure(figsize=(14, 7), dpi=500)
    plt.plot(list,marker = 6,mfc  = 'r',mec ='r',ms= 5,markevery = list_result)
    plt.show()
    return list_result



list_red = np.load('../data/list_red.npy')
list_valley = findmin(list_red,5)
# print(list_valley)
# print(type(list_valley))



# list_down2 = detect_peaks(test2,mpd=1,valley= True, show=True,edge=None)
# print(list_down2)