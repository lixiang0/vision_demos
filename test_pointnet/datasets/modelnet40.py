from turtle import distance
from torch.utils.data import Dataset
import os,pickle
from tqdm import tqdm
import numpy as np
class Datasets(Dataset):

    def __init__(self,root_path,split='train',num_catagory=40,npoint=1024,version='v1') -> None:
        self.root_path=root_path
        self.split=split
        self.num_catagory=num_catagory
        self.cat2id=self._load_catagory()
        self.id2cat=dict()
        self.npoint=npoint
        self.version=version
        if self.version=='v1':
            self.save_path='datasets/modelnet40_v1'
        else:
            self.save_path='datasets/modelnet40'
        
        for key in self.cat2id:
            self.id2cat[self.cat2id[key]]=key
        self.train_list,self.test_list=self._load_id_list()
        
    def _load_catagory(self):
        cat_path=os.path.join(self.root_path,f'modelnet{self.num_catagory}_shape_names.txt')
        cats=[ line.rstrip() for line in open(cat_path,'r').readlines()]
        cat2id=dict(zip(cats,range(len(cats))))
        return cat2id
    def _load_id_list(self):
        if not os.path.exists(self.save_path):
            train_path=os.path.join(self.root_path,f'modelnet{self.num_catagory}_train.txt')
            test_path=os.path.join(self.root_path,f'modelnet{self.num_catagory}_test.txt')
            train_list=[ line.rstrip() for line in open(train_path,'r').readlines()]
            test_list=[ line.rstrip() for line in open(test_path,'r').readlines()]
            _train_list=[]
            _test_list=[]
            for i,flist in enumerate([train_list,test_list]):
                for _,fid in tqdm(enumerate(flist),total=len(flist),smoothing=.9):
                    index_path=os.path.join(self.root_path,'_'.join(fid.split('_')[0:-1]),fid+'.txt')
                    points_set=np.loadtxt(index_path,delimiter=',').astype(np.float32).reshape(-1,6)[:,:3]
                    # if self.preprocess:
                    class_id=self.cat2id['_'.join(fid.split('_')[0:-1])]
                    if i==0:
                        _train_list.append((points_set,class_id))
                    else:
                        _test_list.append((points_set,class_id))
            
            with open(self.save_path, 'wb') as f:
                pickle.dump([_train_list, _test_list], f)
            return _train_list,_test_list
        else:
            with open(self.save_path, 'rb') as f:
                _train_list,_test_list=pickle.load(f)
            return _train_list,_test_list
    
    def __len__(self):
        if self.split is 'train':
            return len(self.train_list)*2
        else:
            return len(self.test_list)*2
    def _sample(self,points_set,npoint):
        N,D=points_set.shape

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
        return points_set[center_ids.astype(np.int32)]
    def _norm(self,point_set):
        center_point=np.mean(point_set,axis=0)
        point_set-=center_point
        max_xyz=np.max(np.sqrt(np.sum(point_set**2,axis=1)))
        point_set/=max_xyz
        return point_set
    def _getitem(self,index):
        if self.split is not 'train':
            index%=len(self.test_list)
            (points_set,class_id)=self.test_list[index]
        else:
            index%=len(self.train_list)
            (points_set,class_id)=self.train_list[index]
        # # print('index:',index,fid)
        # index_path=os.path.join(self.root_path,'_'.join(fid.split('_')[0:-1]),fid+'.txt')
        # points_set=np.loadtxt(index_path,delimiter=',').astype(np.float32).reshape(-1,6)[:,:3]
        # # if self.preprocess:
        # class_id=self.cat2id['_'.join(fid.split('_')[0:-1])]
        # points_set=self._sample(points_set,self.npoint)
        # points_set=self._norm(points_set)
        # points_set=points_set.reshape(-1,6)[:,:3]
        # points_set=self._sample(points_set,self.npoint)
        # points_set=self._norm(points_set)
        # if self.version=='v1':
            # points_set=self._sample(points_set,self.npoint) 
            # points_set=self._norm(points_set)
            # points_set.transpose(1,0)
        return points_set,class_id
    def __getitem__(self,index):
        return self._getitem(index)


if __name__ == "__main__":
    import torch
    dataset=Datasets('./datasets/modelnet40_normal_resampled')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True)
    for point,label in dataloader:
        print(point.shape,point[:10],label)
        # print(dataset.id2cat[label.item()])

        if True:
            import open3d as o3d
            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(point[0,:,:3])
            line_sets=[pcd]
            o3d.visualization.draw_geometries(line_sets)
        break