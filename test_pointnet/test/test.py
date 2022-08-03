import open3d as o3d
import torch
import argparse
import torch.nn.functional as F
arg_parser=argparse.ArgumentParser('test')
arg_parser.add_argument('--show',action='store_true',default=False,help='show pcd or not')
pcd_path='/home/ruben/workspace/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/airplane/airplane_0001.txt'

parser=arg_parser.parse_args()

import numpy as np
pcd_path='/home/ruben/workspace/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/airplane/airplane_0001.txt'
pcd_np=np.loadtxt(pcd_path,delimiter=',').astype(np.float32).reshape(-1,6)[:,:3]

def show_pcd(pcd_np):
    if parser.show:
        pcd_o3d=o3d.geometry.PointCloud()
        pcd_o3d.points=o3d.utility.Vector3dVector(pcd_np)
        line_sets=[pcd_o3d]
        o3d.visualization.draw_geometries(line_sets)
# show_pcd(pcd_np)
# 
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # print(f'farthest:{farthest.size()} distance:{distance.size()} centroids:{centroids.size()}')
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        # print(farthest)
        # print(f'centroids1:{centroids.size()}')
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        # print(f'centroids2:{centroids.size()}')
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# npoint=1024

# idx=farthest_point_sample(torch.from_numpy(pcd_np[None,:,:]),npoint)


def index_points1(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B,N,C = points.shape
    if parser.show:
        print('index_points1: points',points.size())
        print('index_points1: idx',idx.size())
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    if parser.show:
        print('index_points1: batch_indices',batch_indices.size())
    new_points = points[batch_indices, idx, :]
    return new_points
# fps_point=index_points1(torch.from_numpy(pcd_np[None,:,:]),idx)
# print('fps_point size=',fps_point.size())
# show_pcd(fps_point[0].numpy())


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    if parser.show:
        print('square_distance: src',src.size())
        print('square_distance: dst',dst.size())
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))# (B N 3) (B 3 M)
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    # xyz=xyz.permute(0,2,1)
    # new_xyz=new_xyz.permute(0,2,1)
    if parser.show:
        print('xyz, new_xyz',xyz.size(), new_xyz.size())
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)  # B S N
    if parser.show:
        print('sqrdists:',sqrdists.size())
    group_idx[sqrdists > radius ** 2] = N   # B S N 
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]#  b s nsamle
    if parser.show:
        print('group_idx:',group_idx.size())
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample]) #  b S nsamle
    # print('group_first:',group_first)
    if parser.show:
        print('group_first:',group_first.size())
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    if parser.show:
        print(f'group_idx size={group_idx.size()} group_first size={group_first.size()} ')
    return group_idx
# group_idx=query_ball_point(.2,64,torch.from_numpy(pcd_np[None,:,:]),fps_point)
# group_point=index_points1(torch.from_numpy(pcd_np[None,:,:]),group_idx)

# B,N,C=1,1000,3
# S=npoint
# grouped_xyz_norm = group_point - fps_point.view(B, S, 1, C)
# print('fps_point size=',fps_point.shape,'group_point size=',group_point.permute(0, 3, 2, 1).shape)
# show_pcd(group_point.view(1,-1,3)[0,:,:])
# print(group_point[0,0,:,:])


def sample_and_group(x,points=None,nsample=64,npoint=1024,radius=.2):
    if parser.show:
        print('\n\n\n\nsample_and_group:\n')
        print('sample_and_group x:',x.size())  
        if points is not None:
            print('sample_and_group points:',points.size())  
    B,N,C=x.shape
    S=nsample
    
    fps_point=farthest_point_sample(x,npoint)# B S
    if parser.show:
        print('fps_point:',fps_point.size())  
    new_x=index_points1(x,fps_point)    # B S 3
    if parser.show:
        print('sample_and_group  new_x:',new_x.size())  
        print('sample_and_group  x:',x.size())  
    group_idx=query_ball_point(radius,S,x,new_x)
    if parser.show:
        print('group_idx  new_x:',group_idx.size())  
    group_point=index_points1(x,group_idx) # B npoint sample 3
    if parser.show:
        print('group_point:',group_point.size())  
        print('sample_and_group  new_x:',new_x.size())  
    group_point_norm=group_point-new_x.view(B,npoint,1,C)

    if points is not None:
        group_points=index_points1(points,group_idx)
        if parser.show:
            print('group_points:',group_points.size())
            print('group_point_norm:',group_point_norm.size())
        new_points=torch.cat([group_point_norm,group_points],dim=-1)
    else:
        new_points=group_point_norm
    return new_x,new_points
def sample_and_group_all(x,points=None):
    B,N,C=x.shape
    group_x=x.view(B,1,N,C)
    new_x=torch.zeros(B,1,C)
    if points is not None:
        new_points=torch.cat([group_x,points.view(B,1,N,-1)],dim=-1)
    else:
        new_points=group_x
    return new_x,new_points
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
        if parser.show:
            print('SA last_channel:',last_channel)
    def forward(self,x,points=None):
        # x        B 3 N
        # x=x.permute(0,2,1)
        B,N,_=x.shape
        if parser.show:
            print('B,N',B,N)
            print('self.is_group_all:',self.is_group_all)
            if points is not None:
                print('self.points:',points.shape)
            print('self.x:',x.shape)
        if self.is_group_all:
            new_x,grouped_xyz_norm=sample_and_group_all(x,points)
        else:
            new_x,grouped_xyz_norm=sample_and_group(x,points=points,radius=self.radius,npoint=self.npoint,nsample=self.nsample)
        if parser.show:
            print('new_x,grouped_xyz_norm',new_x.size(),grouped_xyz_norm.size())
        grouped_xyz_norm=grouped_xyz_norm.permute(0,3,2,1)
        for i, conv in enumerate(self.mlp_convs):
            # conv bn relu
            bn=self.mlp_bns[i]
            grouped_xyz_norm=F.relu(bn(conv(grouped_xyz_norm)))  # B C nsample npoint
        grouped_xyz_norm=torch.max(grouped_xyz_norm,2)[0]
        if parser.show:
            print('SA   new_xyz, new_points:::',new_x.shape, grouped_xyz_norm.shape)
        return new_x,grouped_xyz_norm.permute(0,2,1)
sa1=SA(npoint=512,nsample=32)
sa2=SA(inchannel=128+3,npoint=256,nsample=64)
sa3=SA(inchannel=128+3,is_group_all=True,npoint=None, radius=None, nsample=None)

xyz=torch.from_numpy(pcd_np[None,:,:])#.permute(0,2,1)
xyz=torch.cat([xyz,xyz])
if parser.show:
    print('xyz:::',xyz.size())
l1,lp1=sa1(xyz)
print('--'*20,l1.size(),lp1.size())

l2,lp2=sa2(l1,lp1)
print('--'*20,l2.size(),l2.size())

l3,lp3=sa3(l1,lp1)
print('--'*20,l3.size(),lp3.size())
