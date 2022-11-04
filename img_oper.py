# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/11/2 18:19
'''

'''
图像(视频)的一系列基本操作：
如，读取，查看，保存
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


def img_read(path):
    # 读取一张图形,是一个ndarray的数据结构
    img = cv2.imread(path,flags=1)  # opencv默认读取BGR，和matlab相反
    return img


def video_read(path,name):
    # 读取一个视频
    video=cv2.VideoCapture(path)
    # 检查是否打开
    if(video.isOpened()):
        # 读取状态和帧
        open,frame=video.read()
    else:
        open=False

    # 当视频处于打开状态时
    while open:
        # 读取状态和帧
        ret, frame = video.read()
        if (frame is None):
            # 如果帧为空，结束读取
            break
        if (ret is True):
            # 如果ret为真，读取帧,转化为灰度图
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow(name,gray)
            cv2.waitKey(100)
    cv2.destroyAllWindows()

def img_write(name,img):
    cv2.imwrite(name,img)



def cv_show(name,img):
    # 定义一个图像显示函数
    cv2.imshow(name,img)  # 创建一个窗口
    cv2.waitKey(0)  # 按任意键终止
    cv2.destroyWindow(name)


def ROI_img(name,path):
    # ROI区域
    img=img_read(path)
    img=img[0:600,0:400]
    cv_show(name,img)


def c_split(img):
    '''

    :param img:
    :return:b,g,r各通道的像素矩阵
    '''
    # 颜色通道提取
    b,g,r=cv2.split(img)
    return b,g,r


def blue_img(img):
    '''

    :param img:一张图片
    :return: 蓝色图片
    '''
    cur_img=img.copy()
    # G,R通道不要
    cur_img[:,:,1]=0
    cur_img[:,:,2]=0
    return cur_img


def red_img(img):
    '''

    :param img: 一张图片
    :return:红色图片
    '''
    cur_img=img.copy()
    # G,R通道不要
    cur_img[:,:,0]=0
    cur_img[:,:,1]=0
    return cur_img


def green_img(img):
    '''

    :param img:一张图片
    :return: 绿色通道的图片
    '''
    cur_img=img.copy()
    # G,R通道不要
    cur_img[:,:,0]=0
    cur_img[:,:,2]=0
    return cur_img


def c_merge(b,g,r):
    '''

    :param b:蓝色通道像素
    :param g: 绿色通道像素
    :param r: 红色通道像素
    :return: 蓝，绿，红3个通道的像素组合
    '''
    # 各颜色通道组合
    img=cv2.merge(b,g,r)
    return img



def edge_pad(img,top_size,bottom_size,left_size,right_size,border_type):
    '''

    :param img: a image
    :param top_size: int
    :param bottom_size: int
    :param left_size: int
    :param right_size: int
    :param border_type: cv2.BORDER_
    :return:
    '''
    # 边界填充
    # 上下左右分别填充的值的大小
    top_size,bottom_size,left_size,right_size=top_size,bottom_size,left_size,right_size
    img=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,border_type)
    return img



def num_caculate(img1,img2):
    # 数值计算
    img1_1=img1+10
    img2_1=img2+50
    res=img1_1+img2_1
    return res


def img_add(img1,img2):
    '''

    :param img1:
    :param img2:
    :return:不越界，取最大值255
    '''
    # 图像add
    res=cv2.add(img1,img2)
    return res


def img_merge(img1,img2):
    '''
    图像融合,需要将两图像的size变成一样
    :return:
    '''
    # 将img2变成img1的size
    img2_resized=cv2.resize(img2,(img1.shape[1],img1.shape[0]))
    # 图像融合
    img_merged=cv2.addWeighted(img1,0.3,img2_resized,0.7,0)
    return img_merged


def threshold_process(img,thresh,maxval,type):
    '''

    :param img:
    :param thresh:
    :param maxval:
    :param type:cv2.THRESH_
    :return:
    '''
    # 图像阈值处理，对于大于阈值的值怎么处理，处理方法，怎么判断阈值，以及怎么处理
    ret, imgthresh=cv2.threshold(img,thresh,maxval,type)
    return imgthresh


def plt_show(img):
    # plt
    plt.figure("fig")
    plt.imshow(img)
    plt.show()
    return 0


def img_blur_mean(img):
    # 均值滤波,把3*3的滤波器和对应像素相乘相加
    blur_img=cv2.blur(img,(3,3))
    return blur_img


def img_box_blur(img):
    # 方框滤波
    box_blur_img=cv2.boxFilter(img,-1,(3,3),normalize=True)
    return box_blur_img


def img_gauss_blur(img):
    # 权重矩阵，离越近，越权重大,卷积核数值满足高斯分布，更重视中间的像素值
    gauss_blur_img=cv2.GaussianBlur(img,(3,3),1)
    return gauss_blur_img


def img_median_blur(img):
    '''

    :param img:'ksize' is required to be an integer
    :return:
    '''
    # 中值滤波，即找到中位数的像素值,去掉椒盐，抑制噪声点
    median_blur_img=cv2.medianBlur(img,3)
    return median_blur_img


def show_imgs(img1,img2,img3):
    # 把所有图像展示在一起
    imgs=np.hstack((img1,img2,img3))
    return imgs


def erode_oper(img):
    # 形态学操作，腐蚀操作,往里缩一些,削弱，去毛刺，变小
    # 生成一个卷积核
    kernel=np.ones((3,3),np.uint8)
    erode_img=cv2.erode(img,kernel,iterations=5)
    return erode_img


def dilate(img):
    # 膨胀
    kernel=np.ones((3,3),np.uint8)
    dilate_img=cv2.dilate(img,kernel,iterations=5)
    return dilate_img


def morphology_open(img):
    # 开运算：先腐蚀后膨胀
    kernel=np.ones((3,3),np.uint8)
    morphology_open_img=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    return morphology_open_img


def morphology_close(img):
    # 闭运算：先膨胀后腐蚀
    kernel=np.ones((3,3),np.uint8)
    morphology_close_img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    return morphology_close_img


def gradient_img(img):
    # 图像梯度，提取边缘轮廓，先膨胀后腐蚀，膨胀-腐蚀，
    kernel = np.ones((3, 3), np.uint8)
    morphology_grad_img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return morphology_grad_img


def tophat_img(img):
    # 图像礼帽，提取毛刺:原图像-开运算图像
    kernel = np.ones((3, 3), np.uint8)
    morphology_tophat_img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    return morphology_tophat_img


def blackhat_img(img):
    # 图像礼帽，提取毛刺:原图像-开运算图像,提取图像轮廓
    kernel = np.ones((3, 3), np.uint8)
    morphology_blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return morphology_blackhat_img
























