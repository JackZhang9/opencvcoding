# _*_ coding:utf-8 _*_
# 捕获保存视频

import cv2

# 创建一个窗口
cv2.namedWindow('vdo',0)
cv2.resizeWindow('vdo',640,480)

# 实例化视频捕获
video_cap=cv2.VideoCapture(0)  # 获取第一个摄像头

# 实例化写入
file_ourcc=cv2.VideoWriter_fourcc(*'XVID')
video_out=cv2.VideoWriter('video_output.avi',file_ourcc,20.0,(640,480))

while(video_cap.isOpened()):
    ret,frame=video_cap.read()  # 读取每一帧图像
    if ret==True:
        frame=cv2.flip(frame,180)

        video_out.write(frame)  # 写入每一帧图像

        cv2.imshow('vdo',frame)

        if cv2.waitKey(20)==ord('q'):
            break
    else:
        break

# 释放
video_cap.release()
video_out.release()
cv2.destroyAllWindows()
