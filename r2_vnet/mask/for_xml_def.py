

from xml.dom.minidom import parse
import xml
import SimpleITK as sitk
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch
import scipy
from scipy import ndimage
import os
import pandas as pd
from scipy.ndimage.interpolation import zoom
import csv




def read_xml(xml_path,child="0",child_child="0",child_child1="0"):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    a = root.getElementsByTagName(child)
    child_node = a[0].getElementsByTagName(child_child)
    if child_node==[]:
        child_node = a[0].getElementsByTagName(child_child1)
    child_value = child_node[0].childNodes[0].nodeValue
    # print(child_value)
    return child_value

def name(xml_path):
    child = "ResponseHeader"
    child_child = "SeriesInstanceUid"
    child_child1 = "CTSeriesInstanceUid"#
    name = read_xml(xml_path,child,child_child,child_child1)
    return name


def read_data(mhd_file):
    ### 读取图像数据
    with open(mhd_file) as f:
        mhd_data = f.readlines()
        # print(mhd_data)
        ### 判断是否反转，其中 TransformMatrix = 1 0 0 0 1 0 0 0 1\n  代表反转为正True
        for i in mhd_data:
            if i.startswith('TransformMatrix'):
                fanzhuan = i.split(' = ')[1]
                if fanzhuan == '1 0 0 0 1 0 0 0 1\n':
                    ### 100代表x，010代表y，001代表z
                    fanzhuan = True

    itkimage = sitk.ReadImage(mhd_file)    ###读取mhd
    numpyImage = sitk.GetArrayFromImage(itkimage)   ###从mhd读取到raw
    print("读取数据，读取的图片大小（zyx）：",numpyImage.shape)
    ###   深 depth  *  宽 width  *  高 height

    origin = itkimage.GetOrigin()
    print("读取数据，读取的坐标原点（xyz）：",origin)
    ###   坐标原点   x,y,z

    spacing = itkimage.GetSpacing()
    print("读取数据，读取的像素间隔（xyz）：",spacing)
    ###   像素间隔   x,y,z
    return numpyImage,origin,spacing,fanzhuan

def plot_3d(image, threshold=-600):   # threshold是阈值
    # 基于轴的转置,原本是(0,1,2)   也就是x,y,z
    p = image.transpose(2, 1, 0)   #  变成了z，y，x

    verts, faces = measure.marching_cubes_classic(p, threshold)  #做二值分类    查找3D体积数据中的曲面

    fig = plt.figure(figsize=(10, 10))    #要一个10*10大小的画布
    ax = fig.add_subplot(111, projection='3d')   # 画一张图，3d的

    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)       #设置区域背景颜色
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def plot_2d(image,z = 132):
    # z,y,x#查看第100张图像
    plt.figure()
    plt.imshow(image[z, :, :])
    plt.show()

def get_raw_label(img_name,img,annos,origin,isflip):    ####   把结节坐标 从世界坐标 准换到 图像坐标 （[1,1,1]像距）
    annos_these_list = []    ####   用来装所有当前名字的 结节数据
    for i in range(len(annos)):   ###  遍历所有数据
        if annos[i][0] == img_name:   ###  如果名字相符
            annos_these_list.append(list(annos[i]))   ###  装进去
    print(annos_these_list)
    return_list = []
    for i in annos_these_list:
        one_annos_list = i
        print("one_annos_list:",one_annos_list)    ####    打印出[["图片名字" 结节位置（x,y,z） 结节直径]]
        ###  annotations.csv中给出的结节中心（x，y，z）坐标是     世界坐标    ！！！！！！！！！
        w_center = [one_annos_list[1],one_annos_list[2],one_annos_list[3]]
        print("世界坐标的   结节中心（x，y，z） ",w_center)
        v_center = list(abs(w_center - np.array(origin)))# /np.array(spacing)  由于我已经把图像转换至像素间隔为[1,1,1],因此不用再除 像素间隔
        print("图像坐标的   结节中心（x，y，z） ",v_center)
        if isflip is False:    ####    如果是反的，由于图反过来了，结节坐标也要反过来
            v_center = [img.shape[2] - 1 - v_center[0],img.shape[1] - 1 - v_center[1],v_center[2]]
        diam = one_annos_list[4]
        print("结节直径",diam)
        one_annos = []
        one_annos.append(v_center[0])
        one_annos.append(v_center[1])
        one_annos.append(v_center[2])
        one_annos.append(diam/2)
        return_list.append(one_annos)
        print("one_annos:",one_annos,"[坐标(x,y,z)]")
    return return_list


def use_plot_2d(image,output,z = 132,batch_index=0,i=0,true_label=False):
    # z,y,x#查看第100张图像
    plt.figure()
    p = image ## 96*96     这是归一化后的
    p = torch.unsqueeze(p,dim=2)
    q = output  ##96*96
    q = (q * 0.2).float()
    q = torch.unsqueeze(q,dim=2)
    q = p + q
    q[q >1] = 1
    r = p
    cat_pic = torch.cat([r,q,p],dim=2)  #  红色为空，my_label为绿色，原图为蓝色
    plt.imshow(cat_pic)
    plt.show()

def read_xml_node(xml_path,child="0",child_child="0"):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    a = root.getElementsByTagName(child)
    child_node = a[0].getElementsByTagName(child_child)
    child_value = child_node[0].childNodes[0].childNodes[0].childNodes[0].nodeValue
    print(child_value)
    return child_value




xml_path = r'D:\datasets\LIDC-IDRI\LIDC-XML-only\tcia-lidc-xml\185\296.xml'
def point(xml_path,origin2):
    a = []
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    nodeid = root.getElementsByTagName("readingSession")
    for u in nodeid:  ###  所有readingSession
        child = u.getElementsByTagName("unblindedReadNodule")
        for i in child:   ###  所有nodule
            id = i.getElementsByTagName("noduleID")
            id1 = id[0].childNodes[0].nodeValue
            # if id1.find("Nodule") != -1 :
            if id1 :
                one_all_iou = i.getElementsByTagName("roi")
                # if len(one_all_iou) > 1:
                for r in one_all_iou:
                    z = r.getElementsByTagName("imageZposition")
                    z1 = float(z[0].childNodes[0].nodeValue)-origin2
                    ioux = r.getElementsByTagName("xCoord")
                    iouy = r.getElementsByTagName("yCoord")

                    ioux1 = np.array([int(k.childNodes[0].nodeValue) for k in ioux])
                    iouy1 = np.array([int(l.childNodes[0].nodeValue) for l in iouy])
                    iou = np.array([ioux1,iouy1])
                    point1 = np.transpose(iou)
                    a.append([z1,point1])
    return a   ### [ [ z层 ， 点 ] ]


def find_xml_path(name1):
    list1 = []
    file_path = r'D:\datasets\LIDC-IDRI\LIDC-XML-only\tcia-lidc-xml'
    for file_list in os.listdir(file_path):
        print(file_list)
        for ii in os.listdir(file_path + "\\" + file_list):
            # print(file_path +"\\" +file_list+"\\"+ii)
            aim_path = file_path + "\\" + file_list + "\\" + ii
            with open(aim_path) as f:
                if name(f) == name1:
                    path = file_path + "\\" + file_list + "\\" + ii
                    list1.append(path)
                    print(path)
        if list1 !=[]:
            return list1

def find_mhd_path(name1):
    luna_path = r"D:\datasets\LUNA16"
    for file_list in os.listdir(luna_path):
        if file_list.find("subset") != -1:
            for ii in os.listdir(luna_path + "\\" + file_list):
                if len(ii.split(".m")) >1:
                    if ii.split((".m"))[0] == name1:
                        path = luna_path + "\\" + file_list + "\\" + ii
                        print(path)
                        return path

def for_one_(name,wrong):
    # anno_path = r'D:\datasets\LUNA16\CSVFILES\annotations.csv'
    # for_node_name_path = r"D:\datasets\sk_output\bbox_annos\bbox_annos.xls"

    xml_path_list = find_xml_path(name)
    ct_image_path = find_mhd_path(name)

    # annos = np.array(pd.read_csv(anno_path))   #用来坐标变换时 得到结节中心的，吐出新的结节中心（用来截取小块时用）
    # an = np.array(pd.read_excel(for_node_name_path))
    # for_node_name_list = []    ###  有结节的案例名称
    # for i in an:
    #     if i[2] != '[]':
    #         for_node_name_list.append(i[1])


    ct_image,origin,spacing,fanzhuan = read_data(ct_image_path)
    # y = get_raw_label(name,ct_image,annos,origin,fanzhuan)   #（截小块用）
    # print(y)
    # print(spacing[2])
    # plot_2d(ct_image,z=int(y[0][2]/spacing[2]))
    s = ct_image.shape
    # mm  = np.zeros((s[0],s[1],s[2]), dtype=np.int32)   ###  全0的mask
    mm  = np.zeros((s[0],s[1],s[2]), dtype=np.int8)   ###  全0的mask
    #取截面  描点
    for i in xml_path_list:
        list1 = point(i,origin[2])   #[ [ z层 ， 点 ] ]
        print(len(list1))
        for ii in list1:
            ceng = ii[0]
            print("ceng",ceng)
            pts = ii[1]
            color = 1  # (0, 255, 0)
            # isClosed = True
            # thickness = 1.
            mm[int(ceng/spacing[2]-1),:,:] = cv.drawContours(mm[int(ceng/spacing[2]-1),:,:], [pts], -1, color=color, thickness=-1)  ###填充
            mm[int(ceng/spacing[2]-1),:,:] = scipy.ndimage.binary_fill_holes(mm[int(ceng/spacing[2]-1),:,:], structure=None, output=None, origin=0)
            mm[int(ceng/spacing[2]-1),:,:] = mm[int(ceng/spacing[2]-1),:,:] + 0.
    if (mm==np.zeros((s[0],s[1],s[2]), dtype=np.int32)).all():
        wrong.append(name)
    return mm,ct_image_path,wrong

def annos():  ###仅取名
    annos = r"D:\datasets\sk_output\bbox_annos\bbox_annos.xls"
    annos = pd.read_excel(annos)#, index_col=0
    # print(annos)
    annos = np.array(annos)
    annos = annos.tolist()
    a = []
    for k in annos:  ###  去掉没有结节的元素【“名字”，“中心+半径”】
        if k[2] != "[]":  ###   不同版本读出来竟然不一样，无奈用了len(k)-1  而不是1
            a.append(k[1])
    return a


def now_annos():  #取结节点小块
    annos = r"D:\datasets\LUNA16\CSVFILES\annotations.csv"
    annos =csv.reader(open(annos))


def resample(imgs, spacing, new_spacing=[1,1,1]):   ###   img是（zyx），spacing是（xyz）
    ###   重采样,坐标原点位置为0
    if len(imgs.shape)==3:   #如果是3维的话
        new_shape = []
        for i in range(3):
            print("（zyx）像素间隔",i,":",spacing[-i-1])   ###   spacing（zyx）
            new_zyx = np.round(imgs.shape[i]*spacing[-i-1]/new_spacing[-i-1])
            new_shape.append(new_zyx)
        print("（zyx）新图大小：",new_shape)
        #new_shape = np.round(imgs.shape * spacing / new_spacing)   #新shape = 图片shape * 像素距离 / 新像素距离   （四舍五入）
        true_spacing = []
        for i in range(3):
            spacing_zyx = np.round(spacing[-i-1]*imgs.shape[i]/new_shape[-i-1])
            true_spacing.append(spacing_zyx)
        print("（zyx）真实间隔:",true_spacing)
            # true_spacing = spacing * imgs.shape / new_shape   #真实像素距离 = 图片shape * 像素距离 / 新形状
        resize_factor = []
        for i in range(3):
            resize_zyx = new_shape[i]/imgs.shape[i]
            resize_factor.append(resize_zyx)
            #resize_factor = new_shape / imgs.shape   用来服务zoom函数的
        imgs = zoom(imgs, resize_factor, mode = 'nearest')   #参数可加：缩放数组，order=2即采样多2倍
        return imgs, true_spacing
    else:
        raise ValueError('wrong shape')