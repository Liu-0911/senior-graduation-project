import cv2

filepath = '../../MP4/test-4.mp4'

cap = cv2.VideoCapture(filepath)  # 打开视频文件
if cap.isOpened() is False:  # 确认视频是否成果打开
    print('Error')
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图片帧宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取图像帧高度
# fps = float(cap.get(cv2.CAP_PROP_FPS))                 # 获取FPS
frame_channel = 3
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
print("一共{0}帧".format(frame_count))
