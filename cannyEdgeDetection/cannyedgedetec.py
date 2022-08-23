# _*_ coding:utf-8 _*_
# @Time: 2022/8/23 9:15
# @Author: JackZhang9

# canny边缘检测

import cv2
from matplotlib import pyplot as plt

# 导入图像
img=cv2.imread('dog.jpg',0)

cannypic=cv2.Canny(img,100,200,5,L2gradient=True)



plt.subplot(2,1,1)
plt.title('origin')
plt.imshow(img,cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(2,1,2)
plt.title('canny pic')
plt.imshow(cannypic,cmap='gray')
plt.xticks([])
plt.yticks([])

plt.show()












