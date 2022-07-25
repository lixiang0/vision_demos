import open3d as o3d
import torch


pcd_path='/home/ruben/workspace/Pointnet_Pointnet2_pytorch/data/modelnet40_normal_resampled/airplane/airplane_0001.txt'


import numpy as np
pcd_np=np.loadtxt(pcd_path,delimiter=',').astype(np.float32).reshape(-1,6)[:,:3]

def show_pcd(pcd_np):
    pcd_o3d=o3d.geometry.PointCloud()
    pcd_o3d.points=o3d.utility.Vector3dVector(pcd_np)
    line_sets=[pcd_o3d]
    o3d.visualization.draw_geometries(line_sets)
show_pcd(pcd_np)

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
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # print(f'centroids2:{centroids.size()}')
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

npoint=1024

idx=farthest_point_sample(torch.from_numpy(pcd_np[None,:,:]),npoint)


def index_points1(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
fps_point=index_points1(torch.from_numpy(pcd_np[None,:,:]),idx)
show_pcd(fps_point[0].numpy())


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
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)  # B S N
    group_idx[sqrdists > radius ** 2] = N   # B S N 
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]#  b s nsamle
    # print('group_idx:',group_idx)
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample]) #  b s nsamle
    # print('group_first:',group_first)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    # print(f'group_idx size={group_idx.size()} group_first size={group_first.size()} ')
    return group_idx
group_idx=query_ball_point(.2,64,torch.from_numpy(pcd_np[None,:,:]),fps_point)
group_point=index_points1(torch.from_numpy(pcd_np[None,:,:]),group_idx)

B,N,C=1,1000,3
S=npoint
grouped_xyz_norm = group_point - fps_point.view(B, S, 1, C)
print(fps_point.shape,group_point.permute(0, 3, 2, 1).shape)
show_pcd(group_point.view(1,-1,3)[0,:,:])
print(group_point[0,0,:,:])