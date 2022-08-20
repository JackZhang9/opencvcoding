# _*_ coding:utf-8 _*_
# 程序性能检测

import cv2
import numpy as np

cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img1',680,480)
print(cv2.useOptimized())  # 查看优化是否开启
# 1.opencv检测程序效率
img1=cv2.imread('musk.jpg')  # 导入图像
e1=cv2.getTickCount()

# 中值滤波
for i in range(3,9,2):
    img1=cv2.medianBlur(img1,i)

e2=cv2.getTickCount()
cv2.imwrite('mblmusk.png',img1)
time=(e2-e1)/cv2.getTickFrequency()
print(time)
cv2.imshow('img1',img1)
cv2.waitKey(0)


