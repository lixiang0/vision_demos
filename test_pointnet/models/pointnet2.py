import os,torch,sys
import torch.nn.functional as F
src_dir = os.path.dirname(os.path.realpath(__file__))
if not src_dir.endswith("test_pointnet"):
    src_dir = os.path.dirname(src_dir)
print(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import utils

class SA(torch.nn.Module):
    def __init__(self,inchannel=3,mlp=[64, 64, 128],is_group_all=False,npoint=1024,nsample=32,radius=.2):
        super(SA,self).__init__()
        self.mlp_convs=torch.nn.ModuleList()
        self.mlp_bns=torch.nn.ModuleList()
        self.is_group_all=is_group_all
        self.npoint=npoint
        self.nsample=nsample
        self.radius=radius
        in_channel=3
        last_channel=inchannel
        for out_channel in mlp:
            self.mlp_convs.append(torch.nn.Conv2d(last_channel,out_channel,1))
            self.mlp_bns.append(torch.nn.BatchNorm2d(out_channel))
            last_channel=out_channel
    def forward(self,x,points=None):
        # x        B 3 N
        # x=x.permute(0,2,1)
        B,N,_=x.shape
        if self.is_group_all:
            new_x,grouped_xyz_norm=utils.sample_and_group_all(x,points)
        else:
            new_x,grouped_xyz_norm=utils.sample_and_group(x,points=points,radius=self.radius,npoint=self.npoint,nsample=self.nsample)
        grouped_xyz_norm=grouped_xyz_norm.permute(0,3,2,1)
        for i, conv in enumerate(self.mlp_convs):
            # conv bn relu
            bn=self.mlp_bns[i]
            grouped_xyz_norm=F.relu(bn(conv(grouped_xyz_norm)))  # B C nsample npoint
        grouped_xyz_norm=torch.max(grouped_xyz_norm,2)[0]
        return new_x,grouped_xyz_norm.permute(0,2,1)
class SAMsg(torch.nn.Module):
    def __init__(self,radius=[1.,.2,.4],nsamples=[32,64,64],npoint=512,inchannel=3,mlp_list=[64,64,64]):
        super(SAMsg,self).__init__()
        self.radius=radius
        self.nsamples=nsamples
        self.npoint=npoint
        self.mlp_list=mlp_list
        self.conv_blocks=torch.nn.ModuleList()
        self.bns_blocks=torch.nn.ModuleList()
        for i in range(len(self.mlp_list)):
            convs=torch.nn.ModuleList()
            bns=torch.nn.ModuleList()
            last_channel=inchannel
            for out_channel in self.mlp_list[i]:
                convs.append(torch.nn.Conv2d(last_channel,out_channel,1))
                bns.append(torch.nn.BatchNorm2d(out_channel))
                last_channel=out_channel
            self.conv_blocks.append(convs)
            self.bns_blocks.append(bns)
        
    
    def forward(self,x,points=None):
        # x :B N 3
        B,N,C=x.shape
        new_xyz=utils.index_points(x,utils.FPS(x,self.npoint))
        new_points_list=[]
        # print(f'x size={x.size()}')
        # print(f'new_xyz size={new_xyz.size()}')
        
        for i in range(len(self.radius)):
            radius=self.radius[i]
            nsample=self.nsamples[i]
            # def query_ball_point(radius, nsample, xyz, new_xyz):
            group_idx=utils.query_ball_point(radius,nsample,x,new_xyz)
            group_x=utils.index_points(x,group_idx)
            group_x-=new_xyz.view(B,self.npoint,1,C)
            # print(f'group_x size={group_x.size()}')
            if points is not None:
                group_points=utils.index_points(points,group_idx)
                group_points=torch.cat([group_points,group_x],dim=-1)
            else:
                group_points=group_x
            group_points=group_points.permute(0,3,2,1)
            # print(f'group_points1 size={group_points.size()}')
            for j in range(len(self.conv_blocks[i])):
                conv=self.conv_blocks[i][j]
                bn=self.bns_blocks[i][j]
                group_points=F.relu(bn(conv(group_points)))
                # print(f'group_points2 size={group_points.size()}')
            new_points=torch.max(group_points,2)[0]
            # print(f'new_points size={new_points.size()}')
            new_points_list.append(new_points)
        new_xyz=new_xyz
        new_points_cat=torch.cat(new_points_list,dim=1)
        # print(f'new_points_cat size={new_points_cat.size()}')
        return new_xyz,new_points_cat.permute(0, 2, 1)

if __name__=='__main__':
    import numpy as np
    pcd_path='/home/ruben/workspace/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/airplane/airplane_0001.txt'
    pcd_np=np.loadtxt(pcd_path,delimiter=',').astype(np.float32).reshape(-1,6)[:,:3]

    # sa1=SA(npoint=512,nsample=32)
    # sa2=SA(inchannel=128+3,npoint=256,nsample=64)
    # sa3=SA(inchannel=128+3,is_group_all=True,npoint=None, radius=None, nsample=None)

    # xyz=torch.from_numpy(pcd_np[None,:,:])
    # xyz=torch.cat([xyz,xyz])
    # print('xyz:::',xyz.size())
    # l1,lp1=sa1(xyz)
    # print('--'*20,l1.size(),lp1.size())

    # l2,lp2=sa2(l1,lp1)
    # print('--'*20,l2.size(),l2.size())

    # l3,lp3=sa3(l1,lp1)
    # print('--'*20,l3.size(),lp3.size())

    xyz=torch.from_numpy(pcd_np[None,:,:])
    xyz=torch.cat([xyz,xyz])
    # radius=[],nsamples=[],npoint,inchannel,mlp_list=[]
    sas1 = SAMsg(npoint=512, radius=[0.1, 0.2, 0.4], nsamples=[16, 32, 128], inchannel=3,mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
    l1,lp1=sas1(xyz)
    print('--'*20,l1.size(),lp1.size()) 

        