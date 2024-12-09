import torch
from torch import nn


device = torch.device('cpu')

class BCEandDiceLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor = None, reduction: str = 'mean'):
        super(BCEandDiceLoss, self).__init__()
        if pos_weight is None:
            pos_weight = torch.Tensor([1.], dtype=torch.float)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.cpu().flatten(1)
        predict = torch.sigmoid(logits).cpu()
        target = target.cpu().flatten(1).float()

        loss_BCE = self.bce_loss(logits, target).mean(1)
        loss_dice = self.dice_loss(predict, target)
        loss = loss_BCE + loss_dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss

    @staticmethod
    def dice_loss(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculating the dice loss
        Args:
            predict = 经激活的预测, (bs, z*x*y)
            target = Targeted image, (bs, z*x*y)
        Output:
            dice_loss
        """

        smooth = 1.0
        predict = predict.flatten(1)
        target = target.flatten(1)

        intersection = torch.sum(predict * target, dim=1)
        union = predict.sum(1) + target.sum(1)

        return 1. - ((2. * intersection + smooth) / (union + smooth))


class WeightsMSELoss(nn.Module):
    def __init__(self):
        super(WeightsMSELoss, self).__init__()

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculating the mean squared error loss for each sample in the batch
        Args:
            predict = Predicted tensor, (bs, z*x*y)
            target = Target tensor, (bs, z*x*y)
        Returns:
            mse_loss_per_sample: A tensor of shape (bs,) containing the MSE for each sample
        """
        # 确保predict和target具有相同的形状
        assert predict.shape == target.shape, "predict and target must have the same shape"

        # 计算每个样本的MSE
        mse_per_sample = ((predict - target) ** 2).mean(dim=1)  # 对每个样本计算MSE

        return mse_per_sample

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum += 1.0 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class CombinedLoss(nn.Module):
    def __init__(self, cls_weights: torch.Tensor = None, seg_weights: torch.Tensor = None, rec_weights: torch.Tensor = None,
                 weights: torch.Tensor = torch.ones(2), n_cls: int = 2, reduction: str = 'mean') -> None:
        super(CombinedLoss, self).__init__()
        if cls_weights is None:
            cls_weights = torch.ones(n_cls, dtype=torch.float)
        if seg_weights is None:
            seg_weights = torch.tensor([1.], dtype=torch.float)

        self.cls_criterion = nn.BCEWithLogitsLoss(weight=cls_weights, reduction='none')
        self.seg_criterion = BCEandDiceLoss(pos_weight=seg_weights, reduction='none')


        self.weight = weights.view(-1, 1)

        self.reduction = reduction

    def forward(self, cls_logits: torch.Tensor, seg_logits: torch.Tensor, label: torch.Tensor,
                target: torch.Tensor) -> [torch.Tensor]:
        cls_logits = cls_logits.squeeze(1)
        seg_logits = seg_logits.squeeze(1)
        #rec_logits = rec_logits.squeeze(1)
        cls_logits = cls_logits.cpu()
        seg_logits = seg_logits.cpu()
        #rec_logits = rec_logits.cpu()
        label = label.cpu()
        target = target.cpu()

        cls_loss = self.cls_criterion(cls_logits, label)

        seg_loss = self.seg_criterion(seg_logits, target)

        loss = (self.weight * torch.stack([cls_loss, seg_loss], dim=0)).sum(0)

        return loss.mean(), seg_loss.mean(), cls_loss.mean()


class adaptative_CombinedLoss(nn.Module):
    def __init__(self, cls_weights: torch.Tensor = None, seg_weights: torch.Tensor = None, n_cls: int = 2,
                 reduction: str = 'mean') -> None:   #reduction定义了计算损失的方式
        super(adaptative_CombinedLoss, self).__init__()
        if cls_weights is None:
            cls_weights = torch.ones(n_cls, dtype=torch.float)
        if seg_weights is None:
            seg_weights = torch.tensor([1.], dtype=torch.float)

        self.cls_criterion = nn.BCEWithLogitsLoss(weight=cls_weights, reduction='none')
        self.seg_criterion = BCEandDiceLoss(pos_weight=seg_weights, reduction='none')
        self.awl = AutomaticWeightedLoss(num=2)
        #self.weights = nn.Parameter(torch.ones((2, 1), device=device, dtype=torch.float), requires_grad=True)
        params = torch.ones(2)
        self.params = torch.nn.Parameter(params,requires_grad=True)
        self.reduction = reduction

    def forward(self, cls_logits: torch.Tensor, seg_logits: torch.Tensor, label: torch.Tensor,
                mask: torch.Tensor) -> [torch.Tensor]:
        cls_logits = cls_logits.squeeze(1)
        seg_logits = seg_logits.squeeze(1)
        #rec_logits = rec_logits.squeeze(1)
        cls_logits = cls_logits.cpu()
        seg_logits = seg_logits.cpu()
        #rec_logits = rec_logits.cpu()
        label = label.cpu()
        mask = mask.cpu()

        cls_loss = self.cls_criterion(cls_logits, label)
        seg_loss = self.seg_criterion(seg_logits, mask)
        #rec_loss = self.rec_criterion(rec_logits,mask)

        #loss = (torch.pow(self.weights, -2) * torch.stack([cls_loss, seg_loss], dim=0)).sum(0)      #总loss还没改,还没加山rec了
        loss = self.awl(cls_loss,seg_loss)
        return loss.mean(), seg_loss.mean(), cls_loss.mean()
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # elif self.reduction == 'none':
        #     return loss




if __name__ == '__main__':
    cls_logits = torch.randn((2, 1))     #两个样本
    seg_logits = torch.randn((2, 1, 128, 128, 256))
    #rec_logits = torch.randn((2, 1, 128, 128, 256))
    labels = torch.randint(0, 2, (2,)).to(torch.float)  #0 2之间01
    target = torch.randint(0, 2, (2, 128, 128, 256)).to(torch.float)   #（128X128x256)

    criterion = adaptative_CombinedLoss()

    loss, seg_loss, cls_loss = criterion(cls_logits, seg_logits, labels, target)

    print(loss, seg_loss, cls_loss)
