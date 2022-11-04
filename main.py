# -*- coding: utf-8 -*-
'''
@Author  : JackZhang9
@Time    : 2022/11/3 15:16
'''
import img_oper
import cv2

'''
主函数，实现各种处理方法
'''

if __name__ == '__main__':
    path="bus.jpg"
    path1="zidane.jpg"
    img=img_oper.img_read(path)
    img2=img_oper.img_read(path1)
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
    dilate_img=img_oper.dilate(img)
    # img_oper.cv_show("dilate",dilate_img)
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
    # img_oper.cv_show("tophat",tophat_img)
    # 黑帽，
    blackhat_img=img_oper.blackhat_img(img)
    img_oper.cv_show("blackhat",blackhat_img)















