from for_xml_def import annos,for_one_,read_data,resample
import os
import numpy as np
import pandas

#用于生成  预处理后的image  和  带有轮廓信息的mask


output_path = r"D:\datasets\sk_output"
mask_path = r'D:\datasets\LUNA16\seg-lungs-LUNA16'
#
anno_name_list = annos()  ###仅取名
wrony = []
for i in anno_name_list:
    name = i
    mask,ct_image_path,wrony = for_one_(name,wrony)
    print(ct_image_path) # D:\datasets\LUNA16\subset0\1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd
    path = ct_image_path.split("LUNA16")[1].split(".m")[0]
    ct_image, origin, spacing, isflip = read_data(ct_image_path)

    ct_image1, origin1, spacing1, isflip1 = read_data(mask_path + r"\\" + name+".mhd")  ###   读取肺部掩膜数据
    ct_image1[ct_image1>1]=1
    ct_image = ct_image * ct_image1
    print(ct_image.shape)
    print(mask.shape)
    image, newspacing = resample(ct_image, spacing)
    msk, newspacing1 = resample(mask,spacing)
    print(image.shape)
    print(msk.shape)
       # LUNA16竞赛中常用来做归一化处理的阈值集是-1000和400
    max_num = 400
    min_num = -1000
    image = (image - min_num) / (max_num - min_num)
    image[image > 1] = 1.
    image[image < 0] = 0.
    ##   LUNA16竞赛中的均值大约是0.25
    img = image - 0.25
    np.save(output_path+"\\bbox_image"+path,img)
    np.save(output_path+"\\bbox_mask"+path,msk)
    print(wrony)

bbox_annos_path = r"D:\datasets\wrong_annos.xls"
bbox_annos = pandas.DataFrame(wrony)###    工具，用来查看预处理后的图片的肺结节中心和半径，用来显示和排错，可以不要
bbox_annos.to_excel(bbox_annos_path)
print("wrony",wrony)