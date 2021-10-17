
import torch.nn as nn
import torch
import torch.nn.functional as f





class r2_block(nn.Module):
    def __init__(self,in_channel,out_channel,s=4):
        super(r2_block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prelu = nn.PReLU()
        if in_channel==1 and out_channel==16:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=16, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(16)
            self.slicon = nn.Conv3d(in_channels=4,out_channels=4,kernel_size=3,stride=1,padding=1)
            self.bn2 = nn.BatchNorm3d(4)
            self.point2 = nn.Conv3d(in_channels=16, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel==16 and out_channel==32:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(32)
            self.slicon = nn.Conv3d(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1)
            self.bn2 = nn.BatchNorm3d(8)
            self.point2 = nn.Conv3d(in_channels=32, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel == 32 and out_channel==64:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(64)
            self.slicon = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(16)
            self.point2 = nn.Conv3d(in_channels=64, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel==64 and out_channel==128:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(128)
            self.slicon = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
            self.bn2 = nn.BatchNorm3d(32)
            self.point2 = nn.Conv3d(in_channels=128, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel==128 and out_channel==256:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=256, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(256)
            self.slicon = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
            self.bn2 = nn.BatchNorm3d(64)
            self.point2 = nn.Conv3d(in_channels=256, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel==256 and out_channel==256:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=256, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(256)
            self.slicon = nn.Conv3d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1)
            self.bn2 = nn.BatchNorm3d(64)
            self.point2 = nn.Conv3d(in_channels=256, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel==128+256 and out_channel==128:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=128, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(128)
            self.slicon = nn.Conv3d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1)
            self.bn2 = nn.BatchNorm3d(32)
            self.point2 = nn.Conv3d(in_channels=128, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel==64+128 and out_channel==64:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=64, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(64)
            self.slicon = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(16)
            self.point2 = nn.Conv3d(in_channels=64, out_channels=out_channel, kernel_size=1, stride=1)
        elif in_channel==32+64 and out_channel==32:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=32, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(32)
            self.slicon = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(8)
            self.point2 = nn.Conv3d(in_channels=32, out_channels=out_channel, kernel_size=1, stride=1)
        else:
            self.point1 = nn.Conv3d(in_channels=in_channel, out_channels=16, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm3d(16)
            self.slicon = nn.Conv3d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(4)
            self.point2 = nn.Conv3d(in_channels=16, out_channels=out_channel, kernel_size=1, stride=1)
    def forward(self,x):
        # print(x.shape)   #torch.Size([4, 1, 32, 64, 64])
        x = self.point1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        # print(x.shape)   #torch.Size([4, 16, 32, 64, 64])
        w = x.shape[1]//4
        # print(w)     #4
        x = torch.split(x,w,dim=1)
        # for i in x:
        #     print(i.size())
        # print(len(x))
        x0 = x[0]
        x1 = x[1]
        x1 = self.slicon(x1)
        x1 = self.bn2(x1)
        x1 = self.prelu(x1)
        x2 = x[1]+x[2]
        x2 = self.slicon(x2)
        x2 = self.bn2(x2)
        x2 = self.prelu(x2)
        x3 = x[2]+x[3]
        x3 = self.slicon(x3)
        x3 = self.bn2(x3)
        x3 = self.prelu(x3)
        x = torch.cat((x0,x1,x2,x3),1)
        # print(x.shape)
        x = self.point2(x)
        return x



class res_block(nn.Module):  ##nn.Module
    def __init__(self, i_channel, o_channel,lei):
        super(res_block, self).__init__()
        self.in_c = i_channel
        self.out_c = o_channel

        # self.conv11 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=1,padding=1)   ###  从 输入channel 到 输出channel
        # self.conv2 = nn.Conv3d(in_channels=o_channel, out_channels=o_channel, kernel_size=5, stride=1,padding=2)   ###  从 输出channel 到 输出channel  （叠加层）
        # self.conv1 = r2_block(in_channel=i_channel,out_channel=o_channel).cuda()
        # self.conv2 = r2_block(in_channel=o_channel,out_channel=o_channel).cuda()
        if self.in_c == 1:
            self.conv1 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()
        elif self.in_c ==80:
            self.conv1 = r2_block(in_channel=i_channel, out_channel=o_channel).cuda()
        else:
            self.conv1 = r2_block(in_channel=i_channel,out_channel=i_channel).cuda()
        self.conv2 = r2_block(in_channel=i_channel,out_channel=o_channel).cuda()

        self.conv3 = nn.Conv3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()  ###  卷积下采样

        self.conv4 = nn.ConvTranspose3d(in_channels=o_channel, out_channels=o_channel, kernel_size=2, stride=2).cuda()   ###  反卷积上采样

        self.conv5 = nn.Conv3d(in_channels=i_channel, out_channels=o_channel, kernel_size=1, stride=1).cuda()   ###  点卷积

        self.bn = nn.BatchNorm3d(i_channel).cuda()
        self.bn1 = nn.BatchNorm3d(o_channel).cuda()
        self.prelu = nn.ELU().cuda()
        self.lei = lei
        self.drop = nn.Dropout3d()

    def forward(self,x):
        if self.lei == "forward1":
            out = self.forward1(x)
        elif self.lei == "forward2":
            out = self.forward2(x)
        elif self.lei == "forward3":
            out = self.forward3(x)
        elif self.lei == "deconv":
            out = self.deconv(x)
        elif self.lei == "upconv":
            out = self.upconv(x)
        else:
            out = self.pointconv(x)
        return out




    def forward1(self, x):
        x = x.to(torch.float32)
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c,self.out_c,"pointconv")
        res = res1(res)
        # print(x.shape)           ####记下   torch.Size([1, 1, 192, 160, 160])
        out = self.conv1(x)
        # print(out.shape)         ####记下   torch.Size([1, 16, 192, 160, 160])
        out = self.bn1(out)
        out = res.add(out)
        out = self.drop(out)
        out = self.prelu(out)
        return out

    def forward2(self,x ):
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")
        res = res1(res)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn1(out)

        out = res.add(out)
        out = self.drop(out)
        out = self.prelu(out)

        return out

    def forward3(self, x):
        res = x   ###   记录下输入时的 x
        res1 = res_block(self.in_c, self.out_c, "pointconv")
        res = res1(res)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv1(out)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn1(out)

        out = res.add(out)
        out = self.drop(out)
        out = self.prelu(out)

        return out

    def deconv(self,x):
        out = self.conv3(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def upconv(self,x):
        out = self.conv4(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out

    def pointconv(self,x):
        out = self.conv5(x)
        out = self.bn1(out)
        out = self.prelu(out)
        return out
