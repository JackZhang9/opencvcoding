# _*_ coding:utf-8 _*_
import cv2

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',640,480)

# 读取图像
img1 = cv2.imread(r"C:\Users\84394\Pictures\v2.jpg",1)

cv2.imwrite('missing1.png',img1)

cv2.imshow('img',img1)

  # 注意K是大写
if cv2.waitKey(0)==ord('q'):  # 按q退出
    cv2.destroyAllWindows()






