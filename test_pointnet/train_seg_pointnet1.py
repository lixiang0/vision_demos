from random import shuffle
import torch,glob,os
from datasets.shapenet import PartDataset
from tqdm import tqdm
from models.pointnet_seg import PointNetSeg,Loss
import numpy as np
import torch.nn.functional as F
DATA_PATH='datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal'
train_dataset=PartDataset(path=DATA_PATH,ttype='train')
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)

test_dataset=PartDataset(path=DATA_PATH,ttype='test')
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=4,shuffle=True)


is_cuda=True
use_checkpoint=True
PRE_TRAIN_PATH='checkpoints/pointnet1/seg/'
TRAIN_EPOCH=300
BEST_ACC=0
MODEL_PATH=PRE_TRAIN_PATH
model=PointNetSeg()
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
    for batch_id,(points,cls_id,target) in tqdm(enumerate(train_dataloader),total=len(train_dataloader),smoothing=.9):
        optimizer.zero_grad()
        if is_cuda:
            points=points.cuda()
            target=target.cuda()
        points = points.transpose(2, 1)
        pred,trans_feat=model(points)
        pred = pred.contiguous().view(-1, 50)
        target = target.contiguous().view(-1).long()
        loss=loss_fun(pred,target,trans_feat)
        loss.backward()
        optimizer.step()
        pred_index=pred.max(1)[1]
        correct=pred_index.long().eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/pred_index.size()[0])
        loss_count+=loss.item()
    print('train accuracy=',np.mean(mean_correct))
    print('train loss=',loss_count/(epoch+1))
    with torch.no_grad():
        model.eval()
        mean_correct=[]
        cls_correct=np.zeros((50,3))
        for batch_id,(points,_,target) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),smoothing=.9):
            if is_cuda:
                points=points.cuda()
                target=target.cuda()
            points = points.transpose(2, 1)
            pred,_=model(points)
            pred = pred.contiguous().view(-1, 50)
            pred_index=pred.max(1)[1]
            # pred_index = pred_index.contiguous().view(-1, 50)
            target = target.contiguous().view(-1).long()
            # correct=pred_index.eq(target.long()).cpu().sum()
            # mean_correct.append(correct/points.size()[0])
            # print(np.unique(target.cpu()))
            for id in np.unique(target.cpu()):
                count_cls=target[target==id].size()[0]
                correct_cls=pred_index[target==id].long().eq(target[target==id].long().data).cpu().sum()
                # classacc = pred_choice[target == cat].long().eq(target[target == cat].long().data).cpu().sum()
                # print(count_cls,correct_cls)
                cls_correct[id,0]+=correct_cls.item()
                cls_correct[id,1]+=count_cls
        cls_correct[:,2]=cls_correct[:,0]/cls_correct[:,1]
        test_acc=np.mean(cls_correct[:,2])
        print('test accuracy=',test_acc)
        print('class accuracy=',cls_correct)
        if test_acc>BEST_ACC:
            BEST_ACC=test_acc
            print('save model')
            state={
                'current_epoch':epoch,
                'model_state_dict':model.state_dict()
            }
            torch.save(state,os.path.join(MODEL_PATH,f'{epoch}_{BEST_ACC}.pth'))