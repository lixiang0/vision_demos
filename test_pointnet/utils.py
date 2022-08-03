import open3d as o3d

import torch
import torch.nn.functional as F

def show_pcd(pcd_np):
    '''
    input: point cloud   type:numpy
    output: 
    '''
    pcd_o3d=o3d.geometry.PointCloud()
    pcd_o3d.points=o3d.utility.Vector3dVector(pcd_np)
    line_sets=[pcd_o3d]
    o3d.visualization.draw_geometries(line_sets)

def FPS(xyz, npoint):
    """
    farthest point sample
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


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B,N,C = points.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


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
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample]) #  b S nsamle
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(x,points=None,nsample=64,npoint=1024,radius=.2):
    B,N,C=x.shape
    S=nsample
    
    fps_point=FPS(x,npoint)# B S
    new_x=index_points(x,fps_point)    # B S 3
    group_idx=query_ball_point(radius,S,x,new_x)
    group_point=index_points(x,group_idx) # B npoint sample 3
    group_point_norm=group_point-new_x.view(B,npoint,1,C)

    if points is not None:
        group_points=index_points(points,group_idx)
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
