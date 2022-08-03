import os,json
from turtle import color
import numpy as np
from torch.utils.data import Dataset

class PartDataset(Dataset):
    def __init__(self,npoint=2500,path='./datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',ttype='train'):
        self.npoint=npoint
        self.path=path
        self.type=ttype
        # Motorbike	03790512
        self.cat_file=os.path.join(self.path,'synsetoffset2category.txt')

        self.index2cat={}
        self.cat2index={}
        self.fname2cat={}
        self.cat2fname={}
        with open(self.cat_file,'r') as freader:
            for line in freader:
                arrs=line.strip().split()
                fname=arrs[1]
                cat_name=arrs[0]
                # print(fname,cat_name)
                id=len(self.index2cat)
                self.index2cat[id]=cat_name
                self.cat2index[cat_name]=id
                self.fname2cat[fname]=cat_name
                self.cat2fname[cat_name]=fname
        # if self.type=='train':
        fjson=os.path.join(self.path,'train_test_split',f'shuffled_{self.type}_file_list.json')
        with open(fjson,'r') as f:
            self.cat_fnames=[(item.split('/')[1],item.split('/')[2])  for item in json.load(f)]# cat filename
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
    
    def _sample(self,points_set,npoint):
        N,D=points_set.shape
        # print('npoint111',npoint)
        center_ids=np.zeros(npoint)
        distance=np.ones(N)*1e2
        farthest=np.random.randint(0,N)

        for i in range(npoint):
            center_ids[i]=farthest
            center_point=points_set[farthest,:]
            dist=np.sum((points_set-center_point)**2,-1)
            mask=dist<distance
            distance[mask]=dist[mask]
            farthest=np.argmax(distance,-1)
        return center_ids.astype(np.int32)
    def __getitem__(self, index):
        index%=len(self.cat_fnames)
        cat_name,filename = self.cat_fnames[index]
        fname=os.path.join(self.path,cat_name,filename+'.txt')
        #catgory
        cls_id=self.cat2index[self.fname2cat[cat_name]]
        cls_id = np.array([cls_id]).astype(np.int32)

        #point cloud
        data = np.loadtxt(fname).astype(np.float32)
        point_set = data[:, 0:3]
        #seg
        seg = data[:, -1].astype(np.int32)
        #sample
        sample_ids = self._sample(point_set,self.npoint)

        # mask
        point_set = point_set[sample_ids, :]
        seg = seg[sample_ids]
        
        return point_set, cls_id, seg

    def __len__(self):
        return len(self.cat_fnames)*4
if __name__ == "__main__":
    import torch
    dataset=PartDataset(path='./datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal',npoint=2048)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
    colors=np.array([[int(200/(i%5+1)),int(200/(i%10+1)),int(200/(i%5+1))] for i in range(1,51)])
    for point_set, cls_id, seg in dataloader:
        # colors1=np.concatenate((colors)*1024).reshape((-1,3))
        print(point_set.shape,cls_id.shape,seg.shape,colors.shape)
        # print(dataset.id2cat[label.item()])
        colors1=[]
        for i in range(2048):
            colors1.append(colors[seg[0][i]]/200)
            # colors1.append([255,0,0])
            print(colors[seg[0][i]])
        colors1=np.array(colors1)
        print(colors1.shape)
        if True:
            import open3d as o3d
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(point_set[0,:,:3])
            pcd.colors = o3d.utility.Vector3dVector(colors1.astype(np.float))
            line_sets=[pcd]
            o3d.visualization.draw_geometries(line_sets)
        # break