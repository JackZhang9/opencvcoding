# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/11/3 15:16
'''
import numpy as np

import img_oper
import cv2

'''
主函数，实现各种处理方法
'''

if __name__ == '__main__':
    path="bus.jpg"
    path1="zidane.jpg"
    path2="person.png"
    img=img_oper.img_read(path)
    img2=img_oper.img_read(path1)
    img8=img_oper.img_read("F:\OD\opencvpy\erodeimg.png")
    # # 查看像素矩阵，矩阵的shape
    # print(img,img.shape)
    # # (1080, 810, 3)BGR的彩色图，h,w,c，高1080，宽810，通道数3
    # cv_show("bus",img)

    # 读取一段视频，转化为灰度图
    # video_path="city.mp4"
    # img_oper.video_read(video_path,"city")

    # ROI查看
    # ROI_img=img_oper.ROI_img("ROI_city",path)

    # 蓝色通道查看
    blue_img=img_oper.blue_img(img)
    # img_oper.cv_show("blue_ img",blue_img)

    # 边界填充
    pad_img=img_oper.edge_pad(img,70,70,70,70,cv2.BORDER_CONSTANT)
    # img_oper.cv_show("pad_img",pad_img)

    # 数值计算
    res=img_oper.num_caculate(img,blue_img)
    # print(res)
    # img_oper.cv_show("res",res)

    # 图像add
    img_add=img_oper.img_add(img,blue_img)
    # img_oper.cv_show("img_add",img_add)

    # 图像融合
    img_merge=img_oper.img_merge(img,img2)
    # img_oper.cv_show("merge",img_merge)

    # 阈值处理
    img_thresh=img_oper.threshold_process(img,127,255,cv2.THRESH_TOZERO_INV)
    # ret,img_thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # img_oper.cv_show("thresh",img_thresh)
    # img_oper.plt_show(img_thresh)
    # cv2.imshow("thresh",img_thresh)
    # cv2.waitKey(0)

    # 图像平滑处理
    # 均值滤波
    blur_img=img_oper.img_blur_mean(img)
    # img_oper.cv_show("mean",blur_img)
    # 方框滤波
    box_blur_img=img_oper.img_box_blur(img)
    # img_oper.cv_show("box_blur_img",box_blur_img)
    # 高斯滤波
    gauss_blur_img=img_oper.img_gauss_blur(img)
    # img_oper.cv_show("gauss_img",gauss_blur_img)
    # 中值滤波
    median_blur_img=img_oper.img_median_blur(img)
    # img_oper.cv_show("median_img",median_blur_img)
    # 集体展示
    imgs=img_oper.show_imgs(box_blur_img,gauss_blur_img,median_blur_img)
    # img_oper.cv_show("imgs",imgs)

    # 形态学操作
    # 腐蚀/削弱
    erode_img=img_oper.erode_oper(img)
    # img_oper.cv_show("erode_img",erode_img)
    # 膨胀
    dilate_img=img_oper.dilate(img8)
    dilate_img1 = img_oper.dilate(dilate_img)
    dilate_img2 = img_oper.dilate(dilate_img1)
    dilate_img3 = img_oper.dilate(dilate_img2)
    d_imgs=np.hstack((dilate_img,dilate_img1,dilate_img2,dilate_img3))
    img_oper.img_write("dilate.png",d_imgs)
    img_oper.cv_show("dilate",d_imgs)
    # 形态学变化：开运算和闭运算
    # 开运算
    morphology_open_img=img_oper.morphology_open(img)
    # img_oper.cv_show("morphology_open",morphology_open_img)
    # 闭运算
    morphology_close_img=img_oper.morphology_close(img)
    # img_oper.cv_show("morphology_close",morphology_close_img)
    # 图像梯度，提取图像轮廓，边缘位置才能产生梯度
    gradient_img=img_oper.gradient_img(img)
    # img_oper.cv_show("grad",gradient_img)
    # 礼帽，提取毛刺
    tophat_img=img_oper.tophat_img(img)
    # 黑帽，
    blackhat_img=img_oper.blackhat_img(img)
    hats_img=np.hstack((tophat_img,blackhat_img))
    # img_oper.cv_show("hats",hats_img)

    # 图像处理，梯度
    # 求图像梯度，提取边缘,sobelx求水平边缘，sobely求竖直边缘
    sobelx,sobely,sobelxy=img_oper.sobel_gradient(img)
    # img_oper.cv_show("sobel_img",sobelxy)
    # scharr算子
    _, _, scharr=img_oper.scharr_gradient(img)
    # img_oper.cv_show("scharr",scharr)
    # laplace算子
    laplace=img_oper.laplace_gradient(img)
    img_gradient=np.hstack((img,sobelxy,scharr,laplace))
    # img_oper.cv_show("gradient",img_gradient)

    # canny边缘检测
    canny_img=img_oper.canny_detect(img)
    # print(type(canny_img))
    # img_oper.cv_show("canny",canny_img)

    # 图像轮廓检测
    contours, hierarchy=img_oper.contour_detect(img)
    contour_img=img_oper.contour_draw(img,contours)
    # 计算轮廓面积，轮廓周长
    print(contours[0],cv2.contourArea(contours[0]),cv2.arcLength(contours[0],True))
    # img_oper.cv_show("contour",contour_img)

    # 模板匹配
    img=cv2.imread(path,0)
    person_img=cv2.imread(path2,flags=0)
    h,w=person_img.shape
    print(h,w)  # (454, 129, 3)
    print(img.shape)  # (1080, 810, 3)
    res=img_oper.template_match(img,person_img)
    print(res.shape)
    min_val, max_val, min_loc, max_loc=img_oper.template_value(res)
    print(min_val,max_val,min_loc,max_loc)
    # 矩形显示
    img=img_oper.img_read(path)
    img2=img.copy()
    top_left=min_loc
    bottom_right=(top_left[0]+w,top_left[1]+h)
    rec_tangle=img_oper.draw_rectangle(img2,top_left,bottom_right)
    # img_oper.img_write("rectangle_img.jpg",rec_tangle)
    # img_oper.cv_show("rectangle",rec_tangle)



    # 图像金字塔
    # 高斯金字塔
    # 上采样，放大
    pyrup_img=img_oper.gauss_pyrUp(img)
    # img_oper.cv_show("up",pyrup_img)
    # 下采样，缩小
    pyrdonw_img=img_oper.gauss_pyrDown(img)
    pyrdonw_img = img_oper.gauss_pyrDown(pyrdonw_img)
    pyrdonw_img = img_oper.gauss_pyrDown(pyrdonw_img)
    # img_oper.cv_show("down",pyrdonw_img)
    # 拉普拉斯金字塔
    laplace_pyr_img=img_oper.laplace_pyr(img)
    laplace_pyr_img = img_oper.laplace_pyr(laplace_pyr_img)
    # img_oper.cv_show("laplace",laplace_pyr_img)


    # 直方图



























