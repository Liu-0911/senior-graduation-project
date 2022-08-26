import cv2
import numpy as np
import os
from numba import jit
# path = r'此电脑\马sir小迷弟 的 Tab S7\Tablet\zhixin\resu.txt'
# path1 = r'D:\resu.txt'
# resu_bool = '0'
#
# with open(path,'w') as f:
#    f.write(resu_bool)
t=0.3
img = cv2.imread("../pic/2.jpg")
frame_width = img.shape[1]  # 获取图片帧宽度
frame_height = img.shape[0]

@jit  # 每一帧中对像素进行处理，包括： 1.计算每一帧的红色平均强度 2.计算这一帧是否符合红色主导的条件
def count(img):
    pr = 0
    for row in range(frame_height):  # 遍历每一行
        for col in range(frame_width):  # 遍历每一列
            total = img[row][col][0] + img[row][col][1] + img[row][col][2]
            # total_red += img[row][col][2]
            # total_green += img[row][col][1]
            # total_blue += img[row][col][0]
            pr = pr + (img[row][col][2]/total if total != 0 else 0)

            # if (img[row][col][2] > total * t):
            #     num += 1
    ava_pr = pr / (frame_height * frame_width)
    # ava_red = total_red / (frame_height * frame_width)
    # ava_green = total_green / (frame_height * frame_width)
    # ava_blue = total_blue / (frame_height * frame_width)
    return ava_pr


num = count(img)
print(num)