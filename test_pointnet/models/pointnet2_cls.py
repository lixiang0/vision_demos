import os,sys,torch

src_dir=os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith('test_pointnet'):
    src_dir=os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)
print(src_dir)
import torch.nn.functional as F
from models.pointnet2 import SA,SAMsg
from models.pointnet import Linear
class PointNet2Cls(torch.nn.Module):
    def __init__(self,norm_channel=False,num_classes=40):
        super(PointNet2Cls,self).__init__()
        inchannel=3 if not norm_channel else 6
        self.sa1=SA(npoint=512,radius=.2,nsample=32,inchannel=inchannel,mlp=[64,64,128],is_group_all=False)
        self.sa2=SA(npoint=256,radius=.4,nsample=64,inchannel=128+3,mlp=[128,128,256],is_group_all=False)
        self.sa3=SA(npoint=None,radius=None,nsample=None,inchannel=256+3,mlp=[256,512,1024],is_group_all=True)
        self.linear1=Linear(1024,512,is_norm=True,is_dropout=True,is_relu=True)
        self.linear2=Linear(512,256,is_norm=True,is_dropout=True,is_relu=True)
        self.linear3=Linear(256,num_classes,is_norm=False,is_dropout=False,is_relu=False)
    def forward(self,x):
        B,_,_=x.shape
        assert x.shape[-1]==3 
        l1_xyz,l1_point=self.sa1(x)
        l2_xyz,l2_point=self.sa2(l1_xyz,l1_point)
        l3_xyz,l3_point=self.sa3(l2_xyz,l2_point)
        x=l3_point.view(B,1024)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.linear3(x)
        x=F.log_softmax(x,-1)
        # print(l1_xyz.shape,l2_xyz.shape,l3_xyz.shape,)
        return x,l3_point

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
    def forward(self,pred_y,target):
        # print(pred_y.size(),target.size())
        loss=F.nll_loss(pred_y,target)
        return loss

class PointNet2MSGCls(torch.nn.Module):
    def __init__(self,norm_channel=False,num_classes=40):
        super(PointNet2MSGCls,self).__init__()
        inchannel=6 if norm_channel else 3
        self.sm1=SAMsg(npoint=512,radius=[.1,.2,.4],nsamples=[16,32,128],inchannel=inchannel,mlp_list=[[32,32,64],[64,64,128],[64,96,128]])
        self.sm2=SAMsg(npoint=128,radius=[.2,.4,.8],nsamples=[32,64,128],inchannel=320+3,mlp_list=[[64,64,128],[128,128,256],[128,128,256]])
        self.sa1=SA(npoint=None,radius=None,nsample=None,inchannel=640+3,mlp=[256,512,1024],is_group_all=True)
        self.linear1=Linear(1024,512,is_norm=True,is_dropout=True,is_relu=True)
        self.linear2=Linear(512,256,is_norm=True,is_dropout=True,is_relu=True)
        self.linear3=Linear(256,num_classes,is_norm=False,is_dropout=False,is_relu=False)
    def forward(self,x):
        B,_,_=x.shape
        lx1,lp1=self.sm1(x)
        # print('lx1 size=',lx1.shape)
        # print('lp1 size=',lp1.shape)
        lx2,lp2=self.sm2(lx1,lp1)
        lx3,lp3=self.sa1(lx2,lp2)
        x = lp3.view(B, 1024)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.linear3(x)
        x=F.log_softmax(x,-1)
        # print(l1_xyz.shape,l2_xyz.shape,l3_xyz.shape,)
        return x,lp3
if __name__=='__main__':
    import numpy as np
    pcd_path='/home/ruben/workspace/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/airplane/airplane_0001.txt'
    pcd_np=np.loadtxt(pcd_path,delimiter=',').astype(np.float32).reshape(-1,6)[:,:3]
    model=PointNet2Cls()
    x=torch.from_numpy(pcd_np[None,:,:])
    print(x.size())
    print(model(torch.cat([x,x]))[0].size())