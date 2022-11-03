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

























