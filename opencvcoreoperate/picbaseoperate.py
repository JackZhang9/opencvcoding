# _*_ coding:utf-8 _*_
# 基础操作:获取像素值并修改，获取图像属性，图像的ROI，图像通道的拆分及合并
import cv2
import numpy as np

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',680,480)

# 1.获取并修改像素值
# 读入图像
img=cv2.imread('../opencvchap01/missing.png')

# 获取像素值
print(img.item(10,10,2))

img.itemset((10,10,2),100)  # 修改像素值
print(img.item(10,10,2))

# 2.获取图像属性
print(img.shape,img.size,img.dtype)

# 3.拆分合并图像通道
b,g,r=cv2.split(img)  # 拆分图像
# img=cv2.merge(b,g,r)  # 合并图像

# 4.为图像填充，扩边
replac = cv2.copyMakeBorder(img,100,0,0,0,cv2.BORDER_CONSTANT,value=0)


# cv2.imshow('img',img)
cv2.imshow('img',replac)
cv2.waitKey(0)











