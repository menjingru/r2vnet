
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

class dice_loss(nn.Module):      #####   我的思路是每个batch做一次反向传播。我的 batch = 2，但有的subset不够用了，最后一个batch就是1个图了
    def __init__(self,c_num = 2):
        super(dice_loss, self).__init__()
    def forward(self,data,label):
        n = data.size(0)  ### data.size(0)指  batchsize  的值
        dice_list = []
        all_dice = 0.

        for i in range(n):

            my_label11 = label[i]  #【1】
            my_label1 = torch.abs(1 - my_label11)   #【0】


            my_data1 = data[i][0]       #【0】
            my_data11 = data[i][1]      #【1】

            #----------------------------------------------



            m1 = my_data1.view(-1)       #【0】         ### 第一个 batch size （c,96,96,96）  的  第一个通道
            m2 = my_label1.view(-1)      #【0】

            m11 = my_data11.view(-1)     #【1】  ### 两通道嘛  就 不写for了
            m22 = my_label11.view(-1)    #【1】

            dice = 0
            dice += (1-(( 2. * (m1 * m2).sum() +1 ) / (m1.sum() + m2.sum() +1)))
            dice += (1-(( 2. * (m11 * m22).sum() + 1) / ( m11.sum()+m22.sum()+ 1)))
            dice_list.append(dice)

        for i in range(n):
            all_dice += dice_list[i]
        dice_loss = all_dice/n

        return dice_loss
Loss = dice_loss().cuda()
def train_model(model, device, train_loader, optimizer, epoch):
    # 模型训练-----调取方法
    model.train()
    loss_need = []
    tqdr = tqdm(enumerate(train_loader))
    for batch_index, (data, target) in tqdr:   #  m(enumerate(train_loader)):
        data, target = data.cuda(), target.cuda()    ###  输入去batch了
        output = model(data)#.to(device)
        loss = Loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        loss_need.append(train_loss)

        tqdr.set_description("Train Epoch : {} \t train Loss : {:.6f} ".format(epoch, loss.item()))
    train_loss = np.mean(loss_need)
    print("train_loss",train_loss)
    return train_loss,loss_need

def test_model(model, device, test_loader, epoch,test):    # 加了个test  1是想打印时好看（区分valid和test）  2是test要打印图，需要特别设计
    # 模型训练-----调取方法
    model.eval()
    test_loss = 0.0
    PA = IOU = DICE = P =R =F1 = 0
    tqrr = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_index,(data, target) in tqrr:    #dm(enumerate(test_loader)):

            if test:
                data_cpu = data.clone().cpu()    #4*1*96*96*96
                my_label_cpu = target.clone().cpu()
                for i in range(len(data_cpu)):
                    true_img_tensor = data_cpu[i][0]  # 96 * 96 * 96
                    true_label_tensor = my_label_cpu[i] # 96 * 96 * 96
                    use_plot_2d(true_img_tensor,true_label_tensor,z=16,batch_index=batch_index,i=i,true_label=True)
                    # one_minute_plot_2d(true_img_tensor,true_label_tensor,z=16,batch_index=batch_index,i=i,true_label=True)

            data, target = data.cuda(), target.cuda()
            torch.cuda.empty_cache()
            output = model(data)    #(output.shape) torch.Size([4, 2, 96, 96, 96])
            loss = Loss(output, target)
            test_loss += loss.item()

            PA0, IOU0, DICE0, P0, R0, F10,tn, fp, fn, tp = zhibiao(output, target)
            PA += PA0
            IOU += IOU0
            DICE += DICE0
            P += P0
            R += R0
            F1 += F10
            if test:
                name = 'Test'
            else:
                name = 'Valid'
            tqrr.set_description("{} Epoch : {} \t {} Loss : {:.6f} \t tn, fp, fn, tp:  {:.0f}  {:.0f}  {:.0f}  {:.0f} ".format(name,epoch,name, loss.item(),tn, fp, fn, tp))
            if test:
                data_cpu = data.clone().cpu()
                my_output_cpu = output.clone().cpu()
                for i in range(len(data_cpu)):
                    img_tensor = data_cpu[i][0]  # 96 * 96 * 96
                    label_tensor = torch.gt(my_output_cpu[i][1], my_output_cpu[i][0])  # 96 * 96 * 96
                    use_plot_2d(img_tensor,label_tensor,z=16,batch_index=batch_index,i=i)
                    # one_minute_plot_2d(img_tensor,label_tensor,z=16,batch_index=batch_index,i=i)

        test_loss /= len(test_loader)
        PA /= len(test_loader)
        IOU /= len(test_loader)
        DICE /= len(test_loader)
        P /= len(test_loader)
        R /= len(test_loader)
        F1 /= len(test_loader)

        print(" Epoch : {} \t {} Loss : {:.6f} \t DICE :{:.6f} PA :{:.6f}".format(epoch, name,test_loss,DICE,PA))

        return test_loss, [PA, IOU, DICE, P, R, F1]




def annos():
    annos = r"/home/zhangfuchun/menjingru/dataset/sk_output/bbox_annos/bbox_annos2.xls"
    annos = pd.read_excel(annos)#, index_col=0
    annos = np.array(annos)
    annos = annos.tolist()
    a = []
    for k in annos:  ###  去掉没有结节的元素【“名字”，“中心+半径”】
        if k[1] != "[]":  ###   不同版本读出来竟然不一样，无奈用了len(k)-1  而不是1
            kk = str_to_int(k[1])
            a.append([k[0],kk]) ###        【“名字”，【中心+半径】】
    return a


class myDataset(Dataset):

    def __init__(self, data_path, label_path,annos):   ###  transform 我没写
        self.data = self.get_img_label(data_path)   ## 图的位置列表
        self.label = self.get_img_label(label_path)   ## 标签的位置列表

        self.annos_img = self.get_annos_label(self.data,annos)  # 图的位置列表 输入进去  吐出  结节附近的图的【【图片位置，结节中心，半径】列表】
        self.annos_label = self.get_annos_label(self.label,annos)    #112


    def __getitem__(self, index):
        img_all = self.annos_img[index]
        # print(img_all)
        label_all = self.annos_label[index]
        # print(label_all)
        img = np.load(img_all[0])    # 载入的是图片地址
        # print(img)
        label = np.load(label_all[0])    # 载入的是label地址
        cut_list = []      ##  切割需要用的数

        for i in range(len(img.shape)):   ###  0,1,2   →  z,y,x
            if i == 0:
                # print(img_all[1][-i-1])
                a = img_all[1][-i-1] - 16  ### z
                b = img_all[1][-i-1] + 16
            else:
                a = img_all[1][-i-1]-32   ### z
                b = img_all[1][-i-1]+32   ###
            if a<0:
                if i == 0:
                    a = 0
                    b = 32
                else:
                    a = 0
                    b = 64
            elif b>img.shape[i]: #   z
                if i == 0 :
                    a = img.shape[i] - 32
                    b = img.shape[i]
                else:
                    a = img.shape[i]-64
                    b = img.shape[i]
            else:
                pass


            cut_list.append(a)
            cut_list.append(b)


        img = img[cut_list[0]:cut_list[1],cut_list[2]:cut_list[3],cut_list[4]:cut_list[5]]   ###  z,y,x
        label = label[cut_list[0]:cut_list[1],cut_list[2]:cut_list[3],cut_list[4]:cut_list[5]]   ###  z,y,x
        for i in label:
            if label.any() !=0 and label.any() != 1:
                print(i)

        img = np.expand_dims(img,0)  ##(1, 96, 96, 96)
        img = torch.tensor(img)
        img = img.type(torch.FloatTensor)
        label = torch.Tensor(label).long()  ##(96, 96, 96) label不用升通道维度
        torch.cuda.empty_cache()
        return img,label    ### 从这里出去还是96*96*96


    def __len__(self):
        return len(self.annos_img)


    @staticmethod
    def get_img_label(data_path):   ###  list 地址下所有图片的绝对地址

        img_path = []
        for t in data_path:  ###  打开subset0，打开subset1
            data_img_list = os.listdir(t)  ## 列出图
            img_path += [os.path.join(t, j) for j in data_img_list]  ##'/public/home/menjingru/dataset/sk_output/bbox_image/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.104562737760173137525888934217.npy'
        img_path.sort()
        return img_path  ##返回的也就是图像路径 或 标签路径

    @staticmethod
    def get_annos_label(img_path,annos):
        annos_path = []  # 这里边要装图的地址，结节的中心，结节的半径    要小于96/4 # ###半径最大才12

        ### ok   ,   anoos 是处理好的列表了，我只需要把他们对比一下是否在列表里，然后根据列表里的坐标输出一个列表  就可以了   在__getitem__里边把它切下来就行

        for u in img_path:  # 图的路径
            name = u.split("/")[-1].split(".np")[0]  # 拿到图的名字
            for one in annos:  # 遍历有结节的图
                if one[0] == name:  # 如果有结节的图的名字 == 输入的图的名字
                    for l in range(len(one[1])):  # 数一数有几个结节
                        annos_path.append(
                            [u, [one[1][l][0], one[1][l][1], one[1][l][2]]])  # 图的地址，结节的中心

        return annos_path  # ###半径最大才12






def zhibiao(data,label):   #   data  n,2,96,96,96  label  n,96,96,96

    ###        这里需要把data变换成label形式，方法是取大为1

    n = data.size(0)
    PA, IOU, DICE, P, R, F1 ,TN, FP, FN, TP= 0,0,0,0,0,0,0,0,0,0


    for i in range(n):

        empty_data = torch.gt(data[i][1], data[i][0])
        empty_data = empty_data.long()  #pred label

        my_data = empty_data  ##  得到处理好的 pred label（96*96*96）
        my_label = label[i]   ##      标准答案     label


        my_data = my_data.cpu().numpy()
        my_data = numpy_list(my_data)
        # print(my_data)

        my_label = my_label.cpu().numpy()
        my_label = numpy_list(my_label)


        confuse = confusion_matrix(my_label,my_data,labels=[0,1])  ### 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(my_label,my_data, labels=[0,1]).ravel()
        all = tn + fp + fn + tp
        # print("tn, fp, fn, tp",tn, fp, fn, tp)
        diag = torch.diag(torch.from_numpy(confuse))
        b = 0
        for ii in diag:
            b += ii
        diag = b

        PA += float(torch.true_divide(diag , all ))  ##  混淆矩阵  对角线/总数

        IOU += float(torch.true_divide(tp,tp+fp+fn))    ##  交并比
        DICE += float(torch.true_divide(2*tp,fp+fn+2*tp))
        if tp + fp ==0:
            P += tp/(tp + fp+1)    ## 精准率  （注意不是精度）
        else:
            P += tp/(tp + fp)    ## 精准率  （注意不是精度）

        if tp + fn == 0:
            R += tp/(tp + fn+1)    ## 召回率
        else:
            R += tp/(tp + fn)    ## 召回率

        TN += tn
        FP += fp
        FN += fn
        TP += tp
    TN /= n
    FP /= n
    FN /= n
    TP /= n

    PA = PA/n
    IOU = IOU/n
    DICE = DICE/n
    P = P/n
    R = R/n
    if P + R == 0:
        F1 += 2 * P * R / (P + R + 1)
    else:
        F1 += 2 * P * R / (P + R)
    return PA,IOU,DICE,P,R,F1,TN, FP, FN, TP



def numpy_list(numpy):
    x = []
    numpy_to_list(x,numpy)
    return x


def numpy_to_list(x,numpy):
    for i in range(len(numpy)):
        if type(numpy[i]) is np.ndarray:
            numpy_to_list(x,numpy[i])
        else:
            x.append(numpy[i])

def str_to_int(aaa):    ## 这个是处理str格式的结节数据 → int格式
    if aaa == "[]":
        b = []
    else:
        aaa = aaa.lstrip("'[[")
        aaa = aaa.rstrip("]]'")
        b = aaa.split("], [")
        for i in range(len(b)):
            b[i] = b[i].split(",")
            b[i][0] = int(float(b[i][0]))
            b[i][1] = int(float(b[i][1]))
            b[i][2] = int(float(b[i][2]))

    return b






def show_loss(loss_list,STR,path):  ###  损失列表，损失名称，保存位置
    EPOCH = len(loss_list)  ##  训练集中是  总epoch   验证集中是  总epoch/每多少epoch进行验证集的epoch数   测试集中就一个数不用画
    x1 = range(0, EPOCH)
    y1 = loss_list

    plt.plot(x1, y1, "-" ,label=STR)
    plt.legend()

    plt.savefig(path +'/%s.jpg'%STR)
    plt.close()


def use_plot_2d(image,output,z = 132,batch_index=0,i=0,true_label=False):
    # z,y,x#查看第100张图像
    plt.figure()
    p = image[z, :, :] +0.25 ## 96*96     这是归一化后的
    # plt.show(p)
    p = torch.unsqueeze(p,dim=2)
    q = output[z, :, :]  ##96*96
    q = (q * 0.2).float()
    q = torch.unsqueeze(q,dim=2)
    q = p + q
    q[q >1] = 1
    r = p
    cat_pic = torch.cat([r,q,p],dim=2)  #  红色为空，my_label为绿色，原图为蓝色
    plt.imshow(cat_pic)

    path = '/home/zhangfuchun/menjingru/dataset/sk_output/sk_zhibiao2'       #  我真的懒得引入参数了，这个path 就是 zhibiao_path
    if true_label:
        plt.savefig(path +'/true_pic/%d_%d.jpg'%(batch_index,i))
    else:
        plt.savefig(path +'/pic/%d_%d.jpg'%(batch_index,i))
    plt.close()


def one_minute_plot_2d(image,output,z = 132,batch_index=0,i=0,true_label=False):
    # z,y,x#查看第100张图像
    plt.figure()
    p = image[z, :, :] +0.25 ## 96*96     这是归一化后的

    p = torch.unsqueeze(p,dim=2)
    q = output[z, :, :]  ##96*96
    n = q

    cat_pic = torch.cat([p,p,p],dim=2)  #  红色为空，my_label为绿色，原图为蓝色
    plt.imshow(cat_pic)

    one_path = '/home/zhangfuchun/menjingru/dataset/sk_output/sk_zhibiao2'       #  我真的懒得引入参数了，这个path 就是 zhibiao_path
    if true_label:
        plt.imshow(cat_pic)
        plt.savefig(one_path +'/one_true_pic/%d_%d.jpg'%(batch_index,i))

        plt.imshow(n)
        plt.savefig(one_path +'/one_pic/%d_%d.jpg'%(batch_index,i))
    plt.close()