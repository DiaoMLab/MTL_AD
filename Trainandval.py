import torch
import numpy as np
from torch import nn
from tool.Metrices import Confusion_Matrix
from tool.save_nii import save_as_nii
import datetime
from sklearn.metrics import roc_curve, auc
import os
'！！！loss/cls_loss/seg_loss'

seg_nii_output = r'E:\ljn\MTL_AD\Kfold_test_result\seg_output'
if not os.path.exists(seg_nii_output):
    os.makedirs(seg_nii_output)
weight_save_path = r'fold0_weights_save.pth'

def train_one_epoch(dataloader,model,optimizer,loss_function,epoch,epochs,logger,device):
    model.train()
    train_cls_comfmat = Confusion_Matrix(num_classes=2)
    train_cls_comfmat.reset()
    train_seg_comfmat = Confusion_Matrix(num_classes=2)
    train_seg_comfmat.reset()


    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_seg_loss = 0.0

    train_pre_array = np.array([])
    train_label_array = np.array([])


    for step,data in enumerate(dataloader):  #遍历dataloader中的数据
        ORI, label, Mask, _ = data

        ORI = torch.unsqueeze(input=ORI,dim=1)   #增加一个维度
        ORI = ORI.to(device).float()
        label = label.to(device).float()
        Mask = Mask.to(device).float()


        seg_output, cls_output = model(ORI)   #这里要改一下model 把输入改成ori

        seg_output = seg_output.squeeze(1)
        batch_loss,seg_loss,cls_loss = loss_function(cls_output,seg_output,label,Mask)
        epoch_loss += batch_loss.item()
        epoch_cls_loss += cls_loss.item()
        epoch_seg_loss += seg_loss.item()


        cls_predict = (torch.sigmoid(cls_output)>0.5).long()
        seg_predict = (torch.sigmoid(seg_output)>0.5).long()



        train_cls_comfmat.update(label.flatten(),cls_predict.flatten())
        train_seg_comfmat.update(Mask.flatten(),seg_predict.flatten())

        train_pre_array = np.append(train_pre_array, cls_output.cpu().detach().numpy())
        train_label_array = np.append(train_label_array, label.cpu().detach().numpy())


        batch_loss.backward()  #反向传播
        optimizer.step()
        #optimizer.step()  #更新模型参数
        optimizer.zero_grad()  #梯度归零
        print(f"training---epoch:[{epoch}/{epochs}]---[{step}/{len(dataloader)}]---batch_loss:{batch_loss.item()}---"
              f"cls_loss:{cls_loss.item()}---seg_loss:{seg_loss.item()}"
              f"\n")
    '计算分割分类的各种指标并且print出来'
    # 分割指标，计算Dice、Jaccard、Mean IoU和Frequency Weighted IoU
    dice = train_seg_comfmat.compute_Dice()
    ji = 2 * dice / (dice + 1)  # 计算Jaccard指数
    miou = train_seg_comfmat.compute_IoU().mean()  # 计算Mean IoU
    freq_weights = train_seg_comfmat.mat.sum(1) / train_seg_comfmat.mat.sum()  # 计算每个类别的频率权重
    fwiou = (freq_weights * train_seg_comfmat.compute_IoU()).sum()  # 计算Frequency Weighted IoU
    # 分类指标
    acc = train_cls_comfmat.compute_globle_acc()
    spec, sens = train_cls_comfmat.compute_Recall_or_specificity_with_sensitivity()
    F1 = train_cls_comfmat.compute_F1_score()
    # 计算FPR, TPR和阈值
    fpr, tpr, thresholds = roc_curve(train_label_array, train_pre_array)
    # 计算AUC
    auc_score = auc(fpr, tpr)
    logger.info(
        f"evaluating---epoch:[{epoch}/{epochs}]---epoch_loss:[{epoch_loss / len(dataloader)}]\n"
        f"---DSC:{dice[1].item()}---JI:{ji[1].item()}---MIOU:{miou.item()}---FWIOU:{fwiou.item()}"
        f"\n---acc:{acc.item()}---sens:{sens.item()}---spec:{spec.item()}---F1:{F1[1].item()}\n"
        f"---AUC:{auc_score}"
    )
    logger.info(
        f"training---epoch:[{epoch}/{epochs}]---epoch_loss:[{epoch_loss / len(dataloader)}]--lr:[{optimizer.param_groups[0]['lr']}]")

    return epoch_loss / len(dataloader), dice[1].item(), ji[
        1].item(), miou.item(), fwiou.item(), acc.item(), sens.item(), \
        spec.item(), F1[1].item(), auc_score, epoch_seg_loss / len(dataloader), epoch_cls_loss / len(dataloader)


    #return epoch_loss / len(dataloader) , epoch_seg_loss / len(dataloader), epoch_cls_loss / len(dataloader)


def evaluate_one_epoch(dataloader,model,loss_function,epoch,epochs,logger,device):


    #model = torch.load(weight)


    model.eval()
    confmat_seg_batch = Confusion_Matrix(num_classes=2)
    confmat_seg_batch.reset()
    val_cls_comfmat = Confusion_Matrix(num_classes=2)
    val_cls_comfmat.reset()
    val_seg_comfmat = Confusion_Matrix(num_classes=2)
    val_seg_comfmat.reset()


    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_seg_loss = 0.0

    val_pre_array = np.array([])
    val_label_array = np.array([])
    with torch.no_grad():  # 梯度关闭
        for step,data in enumerate(dataloader):
            ORI, label, Mask, patient_name = data

            ORI = torch.unsqueeze(input=ORI, dim=1)  # 增加一个维度
            ORI = ORI.to(device).float()
            label = label.to(device).float()
            Mask = Mask.to(device).float()

            seg_output, cls_output = model(ORI)

            seg_output = seg_output.squeeze(1)

            batch_loss, seg_loss, cls_loss = loss_function(cls_output, seg_output,label, Mask)
            epoch_loss += batch_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_seg_loss += seg_loss.item()

            cls_predict = (torch.sigmoid(cls_output) > 0.5).long()
            seg_predict = (torch.sigmoid(seg_output) > 0.5).long()

            val_cls_comfmat.update(label.flatten(), cls_predict.flatten())
            val_seg_comfmat.update(Mask.flatten(), seg_predict.flatten())
            confmat_seg_batch.update(Mask.flatten(), seg_predict.flatten())

            val_pre_array = np.append(val_pre_array, cls_output.cpu().numpy())
            val_label_array = np.append(val_label_array, label.cpu().numpy())

            seg_output = torch.sigmoid(seg_output)
            dice_batch = confmat_seg_batch.compute_Dice()[1].item()
            #考虑换成dice_epoch
            if dice_batch > 0.85:
                output_img = seg_output.cpu()
                output_img = (output_img>0.5).float() #二值化结果保存
            #output = 255output_img
                save_as_nii(output_img, img_names=patient_name, save_dir=seg_nii_output)
            #if dice_batch > 0.85:
                #save_as_nii(output_img, img_names=patient_name)
            confmat_seg_batch.reset()
            print(
                f"valuating---epoch:[{epoch}/{epochs}]---[{step}/{len(dataloader)}]---batch_loss:{batch_loss.item()}---"
                f"cls_loss:{cls_loss.item()}---seg_loss:{seg_loss.item()}"
                f"\n")

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

    logger.info(
        f"evaluating---epoch:[{epoch}/{epochs}]---epoch_loss:[{epoch_loss / len(dataloader)}]\n"
        f"---DSC:{dice[1].item()}---JI:{ji[1].item()}---MIOU:{miou.item()}---FWIOU:{fwiou.item()}"
        f"\n---acc:{acc.item()}---sens:{sens.item()}---spec:{spec.item()}---F1:{F1[1].item()}\n"
        f"---AUC:{auc_score}"
    )

    return epoch_loss / len(dataloader), dice[1].item(), ji[1].item(), miou.item(), fwiou.item(), acc.item(), sens.item(), \
           spec.item(), F1[1].item(), auc_score, epoch_seg_loss / len(dataloader), epoch_cls_loss / len(dataloader)











