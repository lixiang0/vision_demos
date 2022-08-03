
import re
from turtle import forward
import torch

import numpy as np


class Conv(torch.nn.Module):
    def __init__(self,in_channel,out_channel,is_norm=True,is_relu=True) -> None:
        super(Conv,self).__init__()
        self.conv=torch.nn.Conv1d(in_channel,out_channel,1)
        self.norm=torch.nn.BatchNorm1d(out_channel)
        self.is_norm=is_norm
        self.relu=torch.nn.ReLU()
        self.is_relu=is_relu
    def forward(self,x):
        x=self.conv(x)
        if self.is_norm:
            x=self.norm(x)
        if self.is_relu:
            x=self.relu(x)
        return x
class Linear(torch.nn.Module):
    def __init__(self,in_features,out_features,is_norm=True,is_dropout=True,is_relu=True) -> None:
        super(Linear,self).__init__()
        self.linear=torch.nn.Linear(in_features,out_features)
        self.norm=torch.nn.BatchNorm1d(out_features)
        self.relu=torch.nn.ReLU()
        self.dropout=torch.nn.Dropout(p=.4)
        self.is_norm=is_norm
        self.is_dropout=is_dropout
        self.is_relu=is_relu
    def forward(self,x):
        x=self.linear(x)
        if self.is_dropout:
            x=self.dropout(x)
        if self.is_norm:
            x=self.norm(x)
        if self.is_relu:
            x=self.relu(x)
        return x

class STNnd(torch.nn.Module):
    def __init__(self,k,in_channel) -> None:
        super(STNnd,self).__init__()
        self.k=k
        self.conv1=Conv(in_channel,64)
        self.conv2=Conv(64,128)
        self.conv3=Conv(128,1024)

        self.linear1=Linear(1024,512,is_dropout=False)
        self.linear2=Linear(512,256,is_dropout=False)
        self.linear3=Linear(256,self.k*self.k,is_norm=False,is_dropout=False,is_relu=False)

    def forward(self,x):
        batch_size=x.size()[0]
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=torch.max(x3,2,keepdim=True)[0]
        # print(x4.size())
        x5=x4.view(batch_size,-1)
        
        x6=self.linear1(x5)
        x7=self.linear2(x6)
        x8=self.linear3(x7)
        # print(x8.size())
        # self.k=x8.size(1)
        ibias=torch.eye(self.k).view(1,self.k*self.k).repeat(batch_size,1)
        if x.is_cuda:
            ibias=ibias.cuda()
        x8+=ibias
        x9=x8.view(-1,self.k,self.k)
        return x9

class PointNetEncoder(torch.nn.Module):
    def __init__(self,global_feature=True) -> None:
        super(PointNetEncoder,self).__init__()
        self.stn3d=STNnd(3,3)
        self.stn64d=STNnd(64,64)
        self.conv1=Conv(3,64)
        self.conv2=Conv(64,128)
        self.conv3=Conv(128,1024,is_relu=False)
        self.global_feature=global_feature
    def forward(self,x):
        #point transform
        batch_size,D,N=x.size()
        trans=self.stn3d(x)
        x=x.transpose(2,1) #D,N--->N,D
        x=torch.bmm(x,trans)
        x=x.transpose(2,1)#N,D-->D,N
        x=self.conv1(x)
        #feature transform
        
        feature_trans=self.stn64d(x)
        x=x.transpose(2,1)  #D,N--->N,D
        x=torch.bmm(x,feature_trans)
        x=x.transpose(2,1)  #N,D-->D,N

        point_feature=x
        #global
        x=self.conv2(x)
        x=self.conv3(x)
        x=torch.max(x,2,keepdim=True)[0]
        
        x=x.view(-1,1024)
        if self.global_feature:
            return x,trans,feature_trans
        else:
            x=x.view(-1,1024,1).repeat(1,1,N)
            return torch.cat([x, point_feature], 1), trans, feature_trans


if __name__=='__main__':
    x=torch.randn(1,3,1000)
    conv=Conv(3,16)
    output=conv(x)
    print('conv output size=',output.size())

    x=torch.randn(2,100)
    linear=Linear(100,50)
    output=linear(x)
    print('conv output size=',output.size())

    eyex=torch.eye(3)
    print(eyex)


    x=torch.randn(2,3,1000)
    model=STNnd(3,3)
    output=model(x)
    print('output size=',output.size())


    x=torch.randn(8,3,10240)
    model=PointNetEncoder()
    output=model(x)
    print('output size=',output[0].size())