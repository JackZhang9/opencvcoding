# _*_ coding:utf-8 _*_
# @Time: 2022/8/23 9:15
# @Author: JackZhang9

# 图像梯度，高通滤波实现,图像显示,比较不同高通滤波之间差别
import cv2
from matplotlib import pyplot as plt

# 导入图像
img=cv2.imread('CNGRIDS.jpg',0)

# Sobelx
Sobelx=cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
# Sobely
Sobely=cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
# Laplacian
Laplacian2=cv2.Laplacian(img,ddepth=cv2.CV_8U,ksize=3)

plt.subplot(2,2,1)
plt.imshow(img)
plt.title('origin pic')
plt.xticks()
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(Sobelx)
plt.title('Sobelx pic')
plt.xticks()
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(Sobely)
plt.title('Sobely pic')
plt.xticks()
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(Laplacian2)
plt.title('Laplacian pic')
plt.xticks()
plt.yticks([])

plt.show()
















