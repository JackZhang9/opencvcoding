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
    img_oper.cv_show("merge",img_merge)
