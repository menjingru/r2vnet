import pandas
import numpy as np
from for_xml_def import find_mhd_path,read_data,get_raw_label
import pandas as pd
import os

# 用于产生bbox_annos.xls——预处理后的肺结节中心和直径

data_path = r'D:\datasets\LUNA16'
annos_csv = r'D:\datasets\LUNA16\CSVFILES\annotations.csv'
#
c = np.array(pd.read_csv(annos_csv))
d = []
for i in range(10):  # 10
    file_list = os.listdir(data_path + r"\subset%d" % i)
    for ii in file_list:
        if len(ii.split(".m")) == 2:  ####    如果是mhd文件的话
            name = ii.split(".m")[0]
            ct_image_path = find_mhd_path(name)
            numpyImage, origin, spacing, fanzhuan = read_data(ct_image_path)
            one_annos = get_raw_label(name, numpyImage, c, origin, fanzhuan)
            # print("11111111111111111",one_annos)

            d.append([name,one_annos])
            # print("dddddddddddddddddddd",d)
new_bbox_annos_path = r"D:\datasets\sk_output\bbox_annos\bbox_annos.xls"
bbox_annos = pandas.DataFrame(d)
bbox_annos.to_excel(new_bbox_annos_path)
print(d)

