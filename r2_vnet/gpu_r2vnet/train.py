import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch.utils.data
import torch.optim as optim
from r2train_def import *
from r2vnet import VNet
import pandas as pd
import time


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    ###   不知道为啥!!!4核都没报错！！！
os.environ["CUDA_VISIBLE_DEVICES"] = "2"#

BATCH_SIZE = 4#8
EPOCH = 100#30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
model = VNet(2)
# model = torch.load(r'/home/zhangfuchun/menjingru/dataset/sk_output/sk_model2/best_model.pth')


print("Let's use", torch.cuda.device_count(), "GPUs!")

# model = nn.DataParallel(model)    ###   这几句并行   4核满核不报错，3核报错，1核不报错   报的错是模型参数不在同一个gpu上  未解决

model = model.to(DEVICE)

zhibiao_path ='/home/zhangfuchun/menjingru/dataset/sk_output/sk_zhibiao2'

torch.cuda.empty_cache()


annos = annos()
start = time.clock()

######       数据准备

data_path = []
label_path = []
for i in range(0, 8):  ### 0,1,2,3,4,5,6,7   训练集
    data_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_image/subset%d' % i)
    label_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_mask/subset%d' % i)
dataset_train = myDataset(data_path, label_path,annos)  ## 送入dataset
train_loader = torch.utils.data.DataLoader(dataset_train,  ##  生成dataloader
                                               batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=16)  # 24
print("train_dataloader_ok")


data_valid_path = []
label_valid_path = []
for j in range(8, 9):  ### 8   验证集
    data_valid_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_image/subset%d' % j)
    label_valid_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_mask/subset%d' % j)
dataset_valid = myDataset(data_valid_path, label_valid_path, annos)
valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=BATCH_SIZE, shuffle=False,
                                               num_workers=16)
print("valid_dataloader_ok")

data_test_path = []  ### 测试用
label_test_path = []
for ii in range(9, 10):  ### 9   测试集
    data_test_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_image/subset%d' % ii)
    label_test_path.append('/home/zhangfuchun/menjingru/dataset/sk_output/bbox_mask/subset%d' % ii)
dataset_test = myDataset(data_test_path, label_test_path, annos)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=16)
print("Test_dataloader_ok")


######       数据准备完成，开始训练

train_loss_list = []
valid_loss_list = []

minnum = 0
lr_rant = 0
# wd = 0
mome = 0.99
loss_i_need = []
dice123 = []
train_loss = 0.0
train_loss1 = 0.0
for epoch in range(1, EPOCH + 1):  # 每一个epoch  训练一轮   检测一轮
    if epoch == 80:
        mome=0.9



    optimizer = optim.SGD(model.parameters(), lr=0.001 / (10 ** lr_rant), momentum=mome, weight_decay=1e-8)#5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1, last_epoch=-1)


    train_loss,loss_need = train_model(model, DEVICE, train_loader, optimizer, epoch)  ##  开始训练
    train_loss1 = train_loss
    loss_i_need += loss_need
    train_loss_list.append(train_loss)  ###   记录每个epoch的train_loss   在  train_loss_list里   （训练的如何）
    train_loss_pd = pd.DataFrame(train_loss_list)
    train_loss_pd.to_excel(
        zhibiao_path + "/第%d个epoch的训练损失.xls" % (epoch))  ##这里的训练损失是一个epoch结束的时候的训练损失

    torch.save(model, r'/home/zhangfuchun/menjingru/dataset/sk_output/sk_model2/train_model.pth')
    torch.cuda.empty_cache()


    if epoch%5 == 0:   #  每五轮验证一次

        valid_loss, valid_zhibiao = test_model(model, DEVICE, valid_loader,epoch,test=False)   ###   记录每个epoch的valid_loss   在  valid_loss_list里  （泛化能力）
        print("valid_loss",valid_loss)
        dice1 = valid_zhibiao[2]
        dice123.append(dice1)
        valid_loss_list.append(valid_loss)         ####  验证集  损失列表
        valid_loss_pd = pd.DataFrame(valid_loss_list)
        valid_loss_pd.to_excel(zhibiao_path + "/第%d个epoch的验证损失.xls" % (epoch))  ##这里的训练损失是一个epoch结束的时候的训练损失

        if epoch == 5:     ###  一个epoch==5  刚开始，令min为该loss
            torch.save(model, r'/home/zhangfuchun/menjingru/dataset/sk_output/sk_model2/best_model.pth')
            minnum = valid_loss
            zhibiao = valid_zhibiao
            print("minnum", minnum)
        elif valid_loss < minnum:     ##  这个是经过验证   认为最合适的模型
            print("valid_loss < minnum", valid_loss, "<", minnum)
            minnum = valid_loss
            torch.save(model, r'/home/zhangfuchun/menjingru/dataset/sk_output/sk_model2/best_model.pth')
            zhibiao = valid_zhibiao
            zhibiao_pd = pd.DataFrame(zhibiao)      ###  保存这个最合适的指标参数
            zhibiao_pd.to_excel(zhibiao_path + "/目前为止最合适的model指标：第%d个epoch的验证指标[PA, IOU, DICE, P, R, F1].xls" % epoch)
        else:
            pass
        torch.cuda.empty_cache()




end = time.clock()
train_time =end-start
print('Running time: %s Seconds'%train_time)
time_list = []
time_list.append(train_time)
train_time_pd = pd.DataFrame(time_list)
train_time_pd.to_excel(zhibiao_path + "/总epoch的训练时间（不包含测试）.xls")


show_loss(loss_i_need,"train_loss",zhibiao_path)
show_loss(dice123,"valid_dice",zhibiao_path)
# show_loss(train_loss_list,"train_loss_list",zhibiao_path)
# show_loss(valid_loss_list,"valid_loss_list",zhibiao_path)


###  训练 和 验证 结束，保存的模型在r'/public/home/menjingru/dataset/sk_output/model/best_model.pth'

test_start = time.clock()
torch.cuda.empty_cache()
test_loss_list = []
test_zhibiao_list = []


model = torch.load(r'/home/zhangfuchun/menjingru/dataset/sk_output/sk_model2/best_model.pth')
model = model.to(DEVICE)

test_loss, test_zhibiao = test_model(model, DEVICE, test_loader,EPOCH,test=True)
test_loss_list.append(test_loss)     ###  得到测试的损失loss
test_zhibiao_list.append(test_zhibiao)

test_loss_pd = pd.DataFrame(test_loss_list)
test_loss_pd.to_excel(zhibiao_path + "/测试损失.xls")
test_zhibiao_pd = pd.DataFrame(test_zhibiao_list)
test_zhibiao_pd.to_excel(zhibiao_path + "/测试验证指标[PA, IOU, DICE, P, R, F1].xls")


test_end = time.clock()
test_time =test_end-test_start
print('Running time: %s Seconds'%test_time)
test_time_list = []
test_time_list.append(test_time)
test_time_pd = pd.DataFrame(test_time_list)
test_time_pd.to_excel(zhibiao_path + "/测试时间.xls")

