import torch
import numpy as np
from torch import nn
from tool.Metrices import Confusion_Matrix
from tool.save_nii import save_as_nii
import datetime
from sklearn.metrics import roc_curve, auc
import os
from torch import optim
import torch.nn as nn
from LossFunction import  adaptative_CombinedLoss
from Multitaskk_no_die import MultiNetwork
from tool.plt_utils import plt_xOy
from torch.utils.data import DataLoader
from tool.log_utils import get_logger
from tool.MTL_Dataset import train_dataset,test_dataset
import matplotlib.pyplot as plt
'！！！loss/cls_loss/seg_loss'

seg_nii_output = r'E:\ljn\MTL_AD\Kfold_test_result\seg_output'
if not os.path.exists(seg_nii_output):
    os.makedirs(seg_nii_output)


'--------------------------------------------基本设置-----------------------------------------------------'

batch_size = 4
in_channel = 1
n_cls = 1
n_seg = 1

outcome_txt = r'E:\ljn\MTL_AD\Kfold_test_result\outcome.txt'
weight_save_path = r'fold4_weights_save.pth'

logger = get_logger(outcome_txt)
logger.info('我开始啦！')
'-----------------------------------------optim|loss|model----------------------------------------------'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_one_epoch(dataloader,model,logger):
    model.eval()

    confmat_seg_batch = Confusion_Matrix(num_classes=2)
    confmat_seg_batch.reset()
    val_cls_comfmat = Confusion_Matrix(num_classes=2)
    val_cls_comfmat.reset()
    val_seg_comfmat = Confusion_Matrix(num_classes=2)
    val_seg_comfmat.reset()

    val_pre_array = np.array([])
    val_label_array = np.array([])

    model.eval()   #确定已经设置成eval

    with torch.no_grad():  # 梯度关闭

        print(model.state_dict())



        for step,data in enumerate(dataloader):
            ORI, label, Mask, patient_name = data

            ORI = torch.unsqueeze(input=ORI, dim=1)  # 增加一个维度
            ORI = ORI.to(device).float()
            label = label.to(device).float()
            Mask = Mask.to(device).float()

            seg_output, cls_output = model(ORI)

            seg_output = seg_output.squeeze(1)

            cls_predict = (torch.sigmoid(cls_output) > 0.5).long()
            seg_predict = (torch.sigmoid(seg_output) > 0.5).long()

            val_cls_comfmat.update(label.flatten(), cls_predict.flatten())
            val_seg_comfmat.update(Mask.flatten(), seg_predict.flatten())
            confmat_seg_batch.update(Mask.flatten(), seg_predict.flatten())

            val_pre_array = np.append(val_pre_array, cls_output.cpu().numpy())
            val_label_array = np.append(val_label_array, label.cpu().numpy())

            seg_output = torch.sigmoid(seg_output)
            output_img = seg_output.cpu()
            output_img = (output_img>0.5).float() #二值化结果保存
            print(output_img)
            save_as_nii(output_img, img_names=patient_name, save_dir=seg_nii_output)
            confmat_seg_batch.reset()


     #计算指标 logger
    dice = val_seg_comfmat.compute_Dice()
    ji = 2 * dice / (dice + 1)  # 计算Jaccard指数
    miou = val_seg_comfmat.compute_IoU().mean()  # 计算Mean IoU
    freq_weights = val_seg_comfmat.mat.sum(1) / val_seg_comfmat.mat.sum()  # 计算每个类别的频率权重
    fwiou = (freq_weights * val_seg_comfmat.compute_IoU()).sum()  # 计算Frequency Weighted IoU
    # 分类指标
    acc = val_cls_comfmat.compute_globle_acc()
    spec, sens = val_cls_comfmat.compute_Recall_or_specificity_with_sensitivity()
    F1 = val_cls_comfmat.compute_F1_score()
    # 计算FPR, TPR和阈值
    fpr, tpr, thresholds = roc_curve(val_label_array, val_pre_array)
    # 计算AUC
    auc_score = auc(fpr, tpr)
    #画ROC曲线
    plt.plot(fpr, tpr, label=f'trace (AUC = {auc_score:.2f})')


    logger.info(
        f"evaluating"
        f"---DSC:{dice[1].item()}---JI:{ji[1].item()}---MIOU:{miou.item()}---FWIOU:{fwiou.item()}"
        f"\n---acc:{acc.item()}---sens:{sens.item()}---spec:{spec.item()}---F1:{F1[1].item()}\n"
        f"---AUC:{auc_score}"
    )

    return {'Dice':dice[1].item(),
            'Jacc':ji[1].item(),
            'MIOU':miou.item(),
            'FWIOU':fwiou.item(),
            'ACC':acc.item(),
            'SENS':sens.item(),
            'SPEC':spec.item(),
            'F1':F1[1].item(),
            'AUC':auc_score}

model = MultiNetwork(n_cls, n_seg)  #seg,cls,rec

# 将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 重新初始化 DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
model.load_state_dict(torch.load(weight_save_path))

test = evaluate_one_epoch(test_loader,model,logger)
print(test)