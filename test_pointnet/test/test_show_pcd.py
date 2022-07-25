import numpy as np
import open3d as o3d


def show_pcd(pcd_np):
    pcd_o3d=o3d.geometry.PointCloud()
    pcd_o3d.points=o3d.utility.Vector3dVector(pcd_np)
    line_sets=[pcd_o3d]
    o3d.visualization.draw_geometries(line_sets)

PCD_PATH='/home/ruben/workspace/tutorial/test_pointnet/datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1a04e3eab45ca15dd86060f189eb133.txt'
pcd_np=np.loadtxt(PCD_PATH,delimiter=' ').astype(np.float32).reshape(-1,7)[:,:3]
show_pcd(pcd_np)