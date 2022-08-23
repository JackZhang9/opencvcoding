# _*_ coding:utf-8 _*_
# @Time: 2022/8/23 10:45
# @Author: JackZhang9

import cv2
import numpy as np
import numpy as py
from matplotlib import pyplot as plt



# 导入图像
img=cv2.imread('trump.jpeg',0)

lowerResolution1=cv2.pyrDown(img)
lowerResolution2=cv2.pyrDown(lowerResolution1)
lowerResolution3=cv2.pyrDown(lowerResolution2)



cv2.imshow('img',img)
cv2.imshow('img1',lowerResolution1)
cv2.imshow('img2',lowerResolution2)
cv2.imshow('img3',lowerResolution3)


cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.subplot(2,2,1)
# plt.title('origin')
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# #
# plt.subplot(2,2,2)
# plt.title('pyramid pic1')
# plt.imshow(lowerResolution1,cmap='gray')
# plt.xticks([])
# plt.yticks([])
#
# plt.subplot(2,2,3)
# plt.title('pyramid pic')
# plt.imshow(lowerResolution2,cmap='gray')
# plt.xticks([])
# plt.yticks([])
#
# plt.subplot(2,2,4)
# plt.title('pyramid pic')
# plt.imshow(lowerResolution3,cmap='gray')
# plt.xticks([])
# plt.yticks([])
#
# plt.show()








