# _*_ coding:utf-8 _*_

import cv2
import numpy as np
# 绘制不同几何图形，直线line,矩形rectangle,圆形circle,椭圆ellipse,添加文字putText
# 设置，img,color,thickness,linetype,

# 创建窗口
cv2.namedWindow('drawpic')
cv2.resizeWindow('drawpic',1920,1080)

img=np.zeros((700,850,3),np.uint8)

cv2.line(img,(0,0),(811,811),(255,0,0),5)  # 起点和终点，画直线

cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)  # 左上角和右下角，画矩形

cv2.circle(img,(447,63),63,(0,0,255),-1)  # 圆心和半径，画圆

cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)  # 椭圆

cv2.putText(img,'draw opencv',(10,500),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),2)  # 添加图片

# 展现图像
cv2.imshow('drawpic',img)
if cv2.waitKey(0)==ord('q'):
    cv2.destroyAllWindows()  # 关闭窗口













