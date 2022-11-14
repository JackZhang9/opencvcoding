# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/11/14 17:08
'''


import cv2

# 图片读取并转换为灰度图
img=cv2.imread("img0.png",cv2.COLOR_BGR2GRAY)
img1=cv2.imread("img1.png",cv2.COLOR_BGR2GRAY)

# 图片缩小5倍
x,y=img.shape
x1,y1=img1.shape
img=cv2.resize(img,(int(y/5),int(x/5)))
img1=cv2.resize(img1,(int(y1/2),int(x1/2)))

# 图像平滑，高斯滤波
img1=cv2.GaussianBlur(img1,(5,5),1)

# 图片均值滤波，
img1_mean=cv2.blur(img1,(15,15))
# img1_mean=cv2.medianBlur(img1,55)

# 提取划痕灰度图
img11=img1_mean-img1

# 阈值处理
_,img11=cv2.threshold(img11,25,255,cv2.THRESH_BINARY)

# 提取边缘
contours, hierarchy = cv2.findContours(img11, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_img=cv2.drawContours(img1,contours,-1,(255,0,0),2)
cv2.imwrite("qq.jpg",contour_img)
print(type(img11))
# cv2.imshow("de",img1)
# cv2.waitKey(0)

























