# _*_ coding:utf-8 _*_
# 加法，混合，按位运算

import cv2
import numpy as np

cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.resizeWindow('dst',680,480)

# 1.加法,饱和运算，最多255
x=np.uint8([250])
y=np.uint8([10])
print(cv2.add(x,y))

# 2.混合，权重不一样，透明感
img1=cv2.imread('../opencvchap01/missing.png')
img2=cv2.imread('../opencvchap01/missing1.png')

dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imwrite('dst1.png',dst)
cv2.imshow('dst',dst)
if cv2.waitKey(0)==ord('Q'):
    cv2.destroyAllWindows()




