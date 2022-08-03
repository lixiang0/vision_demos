from random import shuffle
import torch,glob,os
from datasets.modelnet40 import Datasets
from tqdm import tqdm
from models.pointnet2_cls import PointNet2Cls,Loss
import numpy as np
import torch.nn.functional as F

DATA_PATH='datasets/modelnet40_normal_resampled'
train_dataset=Datasets(DATA_PATH,split='train',version='v2')
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=4,shuffle=True)

test_dataset=Datasets(DATA_PATH,split='test',version='v2')
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=8,shuffle=True)


is_cuda=True
use_checkpoint=True
PRE_TRAIN_PATH='checkpoints/pointnet2_ssg/cls'
TRAIN_EPOCH=300
BEST_ACC=0
MODEL_PATH='checkpoints/pointnet2_ssg/cls'
model=PointNet2Cls()

#################################################
# Torch Pruning (Begin)
#################################################
import torch_pruning as tp
model.eval()
num_params_before_pruning = tp.utils.count_params( model )
# 1. build dependency graph
strategy = tp.strategy.L1Strategy()
DG = tp.DependencyGraph()
out = model(torch.randn([1,3, 1024]))
DG.build_dependency(model, example_inputs=torch.randn([1,3,1024]))
# modules=[model.conv1,model.layer1,model.layer2,model.layer3,model.layer4,model.conv_up_level1,model.conv_up_level2,model.conv_up_level3,model.bn1]
modules=[model.encoder.stn3d.conv1.conv,model.encoder.stn3d.conv2.conv,model.encoder.stn3d.conv3.conv,\
        model.encoder.stn3d.linear1.linear,model.encoder.stn3d.linear2.linear,model.encoder.stn3d.linear3.linear,\
            model.encoder.stn64d.conv1.conv,model.encoder.stn64d.conv2.conv,model.encoder.stn64d.conv3.conv,\
        model.encoder.stn64d.linear1.linear,model.encoder.stn64d.linear2.linear,model.encoder.stn64d.linear3.linear,\
            # model.encoder.conv1,model.encoder.conv2,model.encoder.conv3,
    ]
for i in range(len(modules)):
    for m in modules[i].modules():
        # print('m::::',m)
        if isinstance(m, torch.nn.Conv1d):
            print('m::::',m,':::::::::::::i',i)
            # pruning_plan = DG.get_pruning_plan( m, tp.prune_conv_in_channel, idxs=strategy(m.weight, amount=0.4) )
            pruning_plan = DG.get_pruning_plan( m, tp.prune_conv_out_channel, idxs=strategy(m.weight, amount=0.4) )
            # execute the plan (prune the model)
            pruning_plan.exec()
        if isinstance(m, torch.nn.Linear):
            # print('m::::',m)
            pruning_plan = DG.get_pruning_plan( m, tp.prune_linear_in_channel, idxs=strategy(m.weight, amount=0.4) )
            # pruning_plan = DG.get_pruning_plan( m, tp.prune_linear_out_channel, idxs=strategy(m.weight, amount=0.4) )
            # execute the plan (prune the model)
            pruning_plan.exec()
num_params_after_pruning = tp.utils.count_params( model )
print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
#################################################
# Torch Pruning (End)
#################################################

# optimizer=torch.optim.SGD(model.parameters(),lr=1e-3,momentum=.9)
optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4
        )
loss_fun=Loss()
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
        pred,_=model(points)
        loss=loss_fun(pred,target)
        loss.backward()
        optimizer.step()
        pred_index=pred.max(1)[1]
        correct=pred_index.eq(target.long()).cpu().sum()
        mean_correct.append(correct.item()/points.size()[0])
        loss_count+=loss.item()
    print('train accuracy=',np.mean(mean_correct))
    print('train loss=',loss_count/(epoch+1))
    with torch.no_grad():
        model.eval()
        mean_correct=[]
        cls_correct=np.zeros((40,3))
        for batch_id,(points,target) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),smoothing=.9):
            if is_cuda:
                points=points.cuda()
                target=target.cuda()
            pred,_=model(points)
            pred_index=pred.max(1)[1]
            # correct=pred_index.eq(target.long()).cpu().sum()
            # mean_correct.append(correct/points.size()[0])
        # test_acc=np.mean(mean_correct)

            # print(np.unique(target.cpu()))
            for id in np.unique(target.cpu()):
                count_cls=target[target==id].size()[0]
                correct_cls=pred_index[target==id].long().eq(target[target==id].long().data).cpu().sum()
                # classacc = pred_choice[target == cat].long().eq(target[target == cat].long().data).cpu().sum()
                # print('correct_cls=',correct_cls.item())
                # print('count_cls=',count_cls.item())
                cls_correct[id,0]+=correct_cls.item()
                cls_correct[id,1]+=count_cls
        # print(cls_correct)
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