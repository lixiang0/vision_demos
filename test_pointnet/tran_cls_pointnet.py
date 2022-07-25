from random import shuffle
import torch,glob,os
from datasets.modelnet40 import Datasets
from tqdm import tqdm
from models.pointnet_cls import PointNetCls,Loss
import numpy as np
import torch.nn.functional as F
DATA_PATH='datasets/modelnet40_normal_resampled'
train_dataset=Datasets(DATA_PATH,split='train')
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=8,shuffle=True)

test_dataset=Datasets(DATA_PATH,split='test')
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True)


is_cuda=True
use_checkpoint=True
PRE_TRAIN_PATH='checkpoints/'
TRAIN_EPOCH=300
BEST_ACC=0
MODEL_PATH='checkpoints'
model=PointNetCls()
# optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=.9)
optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-2,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
loss_fun=Loss(scale=1e-3)
if is_cuda:
    model=model.cuda()
    loss_fun=loss_fun.cuda()

if use_checkpoint:
    pths=glob.glob(f'{PRE_TRAIN_PATH}/*.pth')
    if pths:
        pth_path=sorted(pths,key=os.path.getctime)[::-1][0]
        checkpoint=torch.load(pth_path)
        start_epoch=checkpoint['current_epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        print('load pretrained model',pth_path)
    else:
        print('not use pretrained model')
        start_epoch=0 
else:
    print('not use pretrained model')
    start_epoch=0


for epoch in range(start_epoch,TRAIN_EPOCH):
    mean_correct=[]
    model.train()
    loss_count=0
    for batch_id,(points,target) in tqdm(enumerate(train_dataloader),total=len(train_dataloader),smoothing=.9):
        optimizer.zero_grad()
        if is_cuda:
            points=points.cuda()
            target=target.cuda()
        pred,trans_feat=model(points)
        loss=loss_fun(pred,target,trans_feat)
        loss.backward()

        pred_index=pred.max(1)[1]
        correct=pred_index.eq(target.long()).cpu().sum()
        mean_correct.append(correct.item()/points.size()[0])
        loss_count+=loss.item()
    print('train accuracy=',np.mean(mean_correct))
    print('train loss=',loss_count/(epoch+1))
    with torch.no_grad():
        model.eval()
        mean_correct=[]
        for batch_id,(points,target) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),smoothing=.9):
            if is_cuda:
                points=points.cuda()
                target=target.cuda()
            pred,_=model(points)
            pred_index=pred.max(1)[1]
            correct=pred_index.eq(target.long()).cpu().sum()
            mean_correct.append(correct/points.size()[0])
        test_acc=np.mean(mean_correct)
        print('test accuracy=',test_acc)
        if test_acc>BEST_ACC:
            BEST_ACC=test_acc
            print('save model')
            state={
                'current_epoch':epoch,
                'model_state_dict':model.state_dict()
            }
            torch.save(state,os.path.join(MODEL_PATH,f'{epoch}_{BEST_ACC}.pth'))