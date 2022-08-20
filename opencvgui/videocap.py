# _*_ coding:utf-8 _*_
# 获取视频
import cv2
# 实例化一个窗口
cv2.namedWindow('vdo',cv2.WINDOW_NORMAL)
cv2.resizeWindow('vdo',640,480)

#实例化验摄像头
vdocap = cv2.VideoCapture(0)  #打开第一个摄像头

while(True):
    ret,frame=vdocap.read()  # 摄像头读取
    if not ret:
        break
    g_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #rgb转换成灰度图

    cv2.imshow('vdo',g_frame)
    if cv2.waitKey(20) == ord('q'):   # 按q退出
        break

vdocap.release()
cv2.destroyAllWindows()