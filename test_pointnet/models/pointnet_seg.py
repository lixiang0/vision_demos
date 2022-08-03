from turtle import forward
from models.pointnet import Conv,Linear,PointNetEncoder
import torch
import torch.nn.functional as F
class PointNetSeg(torch.nn.Module):
    def __init__(self,K=50) -> None:
        super(PointNetSeg,self).__init__()
        self.encoder=PointNetEncoder(global_feature=False)
        self.conv1=Conv(1088,512)
        self.conv2=Conv(512,256)
        self.conv3=Conv(256,128)
        self.conv4=Conv(128,K,is_relu=False,is_norm=False)
        self.dropout=torch.nn.Dropout(p=.4)
        self.softmax=torch.nn.Softmax()
    def forward(self,x):
        x,trans,feature_trans=self.encoder(x)
        x1=self.conv1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.softmax(x4)
        return x5.permute(0,2,1),feature_trans

class Loss(torch.nn.Module):
    def __init__(self,scale) -> None:
        super(Loss,self).__init__()
        self.scale=scale
    
    def forward(self,pred,target,trans):
        # print(pred.size(),target.size(),trans.size())
        # print(type(pred),type(target),type(trans))
        loss=F.cross_entropy(pred,target)
        D=trans.size()[1]
        I=torch.eye(D)[None,:,:]
        # print('I',I.size())
        if trans.is_cuda:
            I=I.cuda()
        loss+=torch.mean(torch.norm(torch.bmm(trans,trans.transpose(2,1))-I,dim=(1,2)))*self.scale
        return loss

if __name__=='__main__':
    model=PointNetSeg(K=40)
    x=torch.randn(2,3,2500)
    y,trans=model(x)
    print(y.size(),trans.size())
    pass