# export PYTHONPATH="$PYTHONPATH:/home/ruben/workspace/tutorial/test_pointnet/"
from turtle import forward
from models.pointnet import Conv,Linear,PointNetEncoder
import torch
import torch.nn.functional as F
class PointNetCls(torch.nn.Module):
    def __init__(self,K=40) -> None:
        super(PointNetCls,self).__init__()
        self.encoder=PointNetEncoder()
        self.linear1=Linear(1024,512,is_dropout=False)
        self.linear2=Linear(512,256)
        self.linear3=Linear(256,K,is_norm=False,is_dropout=False,is_relu=False)
        self.dropout=torch.nn.Dropout(p=.4)
        self.softmax=torch.nn.Softmax()

    def forward(self,x):
        x,trans,feature_trans=self.encoder(x)
        x1=self.linear1(x)
        x2=self.linear2(x1)
        x3=self.linear3(x2)
        x4=self.softmax(x3)
        return x4,feature_trans

class Loss(torch.nn.Module):
    def __init__(self,scale) -> None:
        super(Loss,self).__init__()
        self.scale=scale
    
    def forward(self,pred,target,trans):
        # print(pred.size(),target.size(),trans.size())
        loss=F.cross_entropy(pred,target)
        D=trans.size()[1]
        I=torch.eye(D)[None,:,:]
        # print('I',I.size())
        if trans.is_cuda:
            I=I.cuda()
        loss+=torch.mean(torch.norm(torch.bmm(trans,trans.transpose(2,1))-I,dim=(1,2)))*self.scale
        return loss

if __name__=='__main__':
    model=PointNetCls(K=10)
    x=torch.randn(2,3,1000)
    y,trans=model(x)
    print(y.size(),y,trans.size())
    pass