import torch
from torch import optim
import torch.nn as nn
#from Loss_function import CombinedLoss
from LossFunction import CombinedLoss
from LossFunction import AutomaticWeightedLoss,BCEandDiceLoss
from mutitask_nnn import MultiTaskNetwork
from MTL_model import CompositeModel
from LossFunction import  adaptative_CombinedLoss
from Multitaskk_no_die import MultiNetwork
from tool.plt_utils import plt_xOy
import os
import numpy as np
from torch.utils.data import DataLoader
from tool.log_utils import get_logger
from tool.MTL_Dataset import train_dataset,test_dataset
from Trainandval import train_one_epoch,evaluate_one_epoch

'--------------------------------------------基本设置-----------------------------------------------------'

epochs = 100    #150
batch_size = 4
in_channel = 1
n_cls = 1
n_seg = 1

outcome_txt = r'E:\ljn\Seg_NEW\Kfold_projecct\outcome.txt'
weight_save_path = r'E:\ljn\Seg_NEW\outcome\seg_output\best_weight0.pth'
plt_save_path = r'E:\ljn\Seg_NEW\Kfold_projecct\fold_0'   #还没开outcome文件夹啊!
if not os.path.exists(plt_save_path):
    os.makedirs(plt_save_path)
seg_nii_output = r'E:\ljn\Seg_NEW\Kfold_projecct\seg_output'
if not os.path.exists(seg_nii_output):
    os.makedirs(seg_nii_output)

affine = np.array([[-1.4888, 0.0000, 0.0000, 99.0314],
                   [0.0000, -1.4888, 0.0000, 58.6667],
                   [0.0000, 0.0000, 2.0641, -544.5540],
                   [0.0000, 0.0000, 0.0000, 1.0000]])


# device = torch.device('cuda:0')  #0的比较小
logger = get_logger(outcome_txt)
logger.info('我开始啦！')
dropout_prob = 0.5
lr = 0.0001  #太小过拟合了   1.0.0001
weight_decay = 0.0001
gamma = 0.7
Momentum = 0.9
L2_weights = 0.0001   #0.0001


logger.info(f"当前使用初始学习率为为{lr}")
logger.info(f"当前 NCE 是否点乘Mask ：boxes")
logger.info(f"当前图像裁剪区间为 [0, 200]")
logger.info(f"dropout概率为{dropout_prob}")
#logger.info(f"模型是否加载权重{load_model}")
logger.info(f"当前L2正则化是{L2_weights}")


cls_weights = torch.tensor([1.],dtype=torch.float)        #1:1
seg_weights = torch.tensor([1.],dtype=torch.float)

'-----------------------------------------optim|loss|model----------------------------------------------'
model = MultiNetwork(n_cls, n_seg)  #seg,cls,rec
#model = CompositeModel(num_class_cls=n_cls,num_class_seg=n_seg,in_channel=in_channel,prob=dropout_prob)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'
model = model.to(device)
if torch.cuda.device_count() > 1:
    print('use', torch.cuda.device_count(), 'gpus')
    model = nn.DataParallel(model)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'------------------uncertainty automaticWeightedloss----------------------------------------------------'
loss_function = adaptative_CombinedLoss(cls_weights=cls_weights,seg_weights=seg_weights,n_cls=1,reduction='mean')
#loss_function = CombinedLoss(cls_weights=cls_weights, seg_weights=seg_weights, reduction='mean')

#optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=L2_weights, momentum=Momentum) #优化器
optimizer = optim.Adam(model.parameters(),  lr=lr, weight_decay=L2_weights)
Lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.7)

'可视化参数'
epoch_plt = np.array([])
train_loss_plt = np.array([])
train_seg_loss_plt = np.array([])
train_cls_loss_plt = np.array([])
val_loss_plt = np.array([])
val_seg_loss_plt = np.array([])
val_cls_loss_plt = np.array([])

acc_plt = np.array([])
sens_plt = np.array([])
spec_plt = np.array([])
F1_plt = np.array([])
auc_plt = np.array([])

dice_plt = np.array([])  # 存储Dice系数
ji_plt = np.array([])    # 存储Jaccard指数
miou_plt = np.array([])  # 存储Mean IoU
fwiou_plt = np.array([]) # 存储Frequency Weighted IoU

'Dataloader'
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

early_stop_patience = 15
no_improvement = 0
best_val_loss = 100.0
best_performance = 10.0
best_acc, best_sens, best_spec, best_F1, best_auc_score = 0., 0., 0., 0., 0.
best_dice, best_ji, best_miou, best_fwiou = 0., 0., 0., 0.
best_acc_epoch = []

'训练验证过程'
for epoch in range(epochs):
    train_loss, train_seg_loss, train_cls_loss= train_one_epoch(dataloader=train_loader, model=model, device=device, optimizer=optimizer,
                                                                loss_function=loss_function, epoch=epoch, epochs=epochs, logger=logger)
    val_loss, dice, ji, miou, fwiou, acc, sens, spec, F1, auc_score, val_seg_loss, val_cls_loss = evaluate_one_epoch(dataloader=val_loader, model=model,
                                                                              device=device, loss_function=loss_function,
                                                                              epoch=epoch, epochs=epochs, logger=logger)
    if auc_score >= best_auc_score:
        if auc_score >best_auc_score:
            best_acc, best_sens, best_spec, best_F1, best_auc_score = acc, sens, spec, F1, auc_score
            best_dice, best_ji, best_miou, best_fwiou = dice, ji, miou, fwiou
        elif auc_score == best_auc_score:
            if sens > best_sens:
                best_acc, best_sens, best_spec, best_F1, best_auc_score = acc, sens, spec, F1, auc_score
                best_dice, best_ji, best_miou, best_fwiou = dice, ji, miou, fwiou

    Lr_scheduler.step()   #'以loss作为标准’
    #if dice > best_performance:
        #best_performance = dice
        #best_model_weights = model.state_dict()
        #torch.save(best_model_weights, weight_save_path)
    if train_loss < best_performance:
        best_performance = val_loss
        best_model_weights = model.state_dict()
        torch.save(best_model_weights, weight_save_path)
     # 早停
    if no_improvement < early_stop_patience:
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement = 0
        else:
                no_improvement += 1
    else:
        print(f"Early stopping at epoch {epoch}")
        break

    # 结果可视化
    epoch_plt = np.append(epoch_plt, epoch)
    train_loss_plt = np.append(train_loss_plt, train_loss)
    train_seg_loss_plt = np.append(train_seg_loss_plt, train_seg_loss)
    train_cls_loss_plt = np.append(train_cls_loss_plt, train_cls_loss)
    #train_rec_loss_plt = np.append(train_rec_loss_plt,train_rec_loss)
    val_loss_plt = np.append(val_loss_plt, val_loss)
    val_seg_loss_plt = np.append(val_seg_loss_plt, val_seg_loss)
    val_cls_loss_plt = np.append(val_cls_loss_plt, val_cls_loss)
    #val_rec_loss_plt = np.append(val_rec_loss_plt, val_rec_loss)

    acc_plt = np.append(acc_plt, acc)
    sens_plt = np.append(sens_plt, sens)
    spec_plt = np.append(spec_plt, spec)
    F1_plt = np.append(F1_plt, F1)
    auc_plt = np.append(auc_plt, auc_score)

    dice_plt = np.append(dice_plt, dice)  # val_epoch已经完成选择类别1的Dice值，并取item()
    ji_plt = np.append(ji_plt, ji)  # 选择类别1的Jaccard指数
    miou_plt = np.append(miou_plt, miou)
    fwiou_plt = np.append(fwiou_plt, fwiou)

    plt_xOy(x_ndarray=epoch_plt, y_ndarray=[train_loss_plt, val_loss_plt], png_name="Loss_foldo.png",
            legend=["Train", "Val"], dir=plt_save_path, figure_num=1)

    #plt_xOy(x_ndarray=epoch_plt, y_ndarray=[train_rec_loss_plt, val_rec_loss_plt], png_name="Loss_rec_fold0.png",
            #legend=["Train", "Val"], dir=plt_save_path, figure_num=1)

    plt_xOy(x_ndarray=epoch_plt, y_ndarray=[train_cls_loss_plt, val_cls_loss_plt], png_name="cls_loss_fold0.png",
            legend=["Train_cls", "Val_cls"], dir=plt_save_path, figure_num=1)

    plt_xOy(x_ndarray=epoch_plt, y_ndarray=[train_seg_loss_plt, val_seg_loss_plt], png_name="seg_Loss_fold0.png",
            legend=["Train_seg", "Val_seg"], dir=plt_save_path, figure_num=1)

    plt_xOy(x_ndarray=epoch_plt, y_ndarray=[dice_plt, ji_plt, miou_plt, fwiou_plt],
            legend=["DSC", "JI", "MIOU", "FWIOU"],
            png_name="Metrics_seg_fold0.png",dir=plt_save_path, figure_num=1)

    plt_xOy(x_ndarray=epoch_plt, y_ndarray=[acc_plt, sens_plt, spec_plt, F1_plt], legend=["acc", "sens", "spec", "F1"],
            png_name="Metrics_cls_fold0.png",dir=plt_save_path, figure_num=1)

    plt_xOy(x_ndarray=epoch_plt, y_ndarray=[auc_plt],
            legend=["AUC"], png_name="AUC_fold0.png",dir=plt_save_path, figure_num=1)

logger.info(f"\n best_output \n"
            f"acc: {best_acc}---------sens: {best_sens}-------spec: {best_spec}---------F1: {best_F1}-------AUC :{best_auc_score}\n"
            f"DICE:{best_dice}--------JI:{best_ji}--------MIOU:{best_miou}--------FWIOU:{best_fwiou}\n")

