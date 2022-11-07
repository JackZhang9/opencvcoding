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


def sobel_gradient(img):
    # sobel算子计算梯度,水平和竖直，高斯的理念，左右点像素值的差异，dx方向还是dy方向，只有边界的地方才有梯度
    # 先求绝对值
    gradient_sobel_res_x=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    gradient_sobel_res_x=cv2.convertScaleAbs(gradient_sobel_res_x)
    gradient_sobel_res_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_sobel_res_y=cv2.convertScaleAbs(gradient_sobel_res_y)
    gradient_sobel_res_xy=cv2.addWeighted(gradient_sobel_res_x,0.5,gradient_sobel_res_y,0.5,0)
    return gradient_sobel_res_x,gradient_sobel_res_y,gradient_sobel_res_xy


def scharr_gradient(img):
    # scharr算子计算梯度，水平和竖直，算子的值更大，更敏感
    gradient_scharr_res_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    gradient_scharr_res_x = cv2.convertScaleAbs(gradient_scharr_res_x)
    gradient_scharr_res_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    gradient_scharr_res_y = cv2.convertScaleAbs(gradient_scharr_res_y)
    gradient_sobel_res_xy = cv2.addWeighted(gradient_scharr_res_x, 0.5, gradient_scharr_res_y, 0.5, 0)
    return gradient_scharr_res_x,gradient_scharr_res_y,gradient_sobel_res_xy


def laplace_gradient(img):
    # 缺点：对噪音点敏感，跟其他方法一起使用,求二阶导，
    # 中间点和边缘的计算，没有x和y，也不需要把x和y求出来合并
    gradient_laplace_res=cv2.Laplacian(img,cv2.CV_64F)
    gradient_laplace_res=cv2.convertScaleAbs(gradient_laplace_res)
    return gradient_laplace_res


def canny_detect(img):
    # 1.去噪：使用滤波器，图像平滑，过滤掉噪声，高斯滤波器,中间点大
    # 2.求梯度：计算图像中每个像素点的梯度强度和方向，sobel算子，
    # 3.非极大值抑制NMS：把最明显的体现出来，消除边缘检测的杂散响应，把一些概率小的框去掉
    # 4.双阈值：只保留最真实的。梯度>maxval，处理为边界，minval<梯度<maxval，连有边界为边界，
    # 5.抑制孤立的边缘完成边缘检测
    canny_img=cv2.Canny(img,80,160)
    return canny_img


def contour_detect(img):
    # 轮廓检测，连在一块，是一个整体
    # 1.转换成灰度图
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 2.图像阈值，二值处理，0/1
    _,img_thresh=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy=cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def contour_draw(img,contour):
    img_copy=img.copy()  # 复制img
    # 输入图像，轮廓，轮廓索引-1，颜色模式blue，线条厚度
    contour_img=cv2.drawContours(img_copy,contour,-1,(255,0,0),2)
    return contour_img


def template_match(img,template):
    # 模板匹配，使用模板在原图像上从原点开始滑动，
    # 比较像素点差异，对应位置的，如平方差，进行减法，比较差异有多少；相关系数，越等于1越相近
    # 返回每个窗口的匹配结果
    res=cv2.matchTemplate(img,template,cv2.TM_SQDIFF_NORMED)
    return res


def template_value(res):
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
    return min_val,max_val,min_loc,max_loc


def draw_rectangle(img,top_left,bottom_right):
    rec_tangle=cv2.rectangle(img,top_left,bottom_right,255,2)
    return rec_tangle


def gauss_pyrDown(img):
    # 向下采样，缩小,与高斯内核相乘卷积，偶数行，列去掉
    pyr_down=cv2.pyrDown(img)
    return pyr_down


def gauss_pyrUp(img):
    # 向上采样，放大
    pyr_up=cv2.pyrUp(img)
    return pyr_up


def laplace_pyr(img):
    # laplace金字塔，先down后up,原始图像减去up_down
    down=gauss_pyrDown(img)
    up=gauss_pyrUp(down)
    res=img-up
    return res



























