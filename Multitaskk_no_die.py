import torch
import torch.nn as nn
import torchvision.models as models
from model.resnet import resnet10
from model.resnet import resnet18
from model.resnet import resnet50
from model.model_zzl import resnet50_threeD
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = None, out_channels: int = None, residual: bool = False):
        super(BasicBlock, self).__init__()
        self.residual = residual

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels * 2

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feature = self.conv(x)

        if self.residual:
            feature = x + feature

        return feature
# 定义分割分类多任务模型
class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str = 'resnet50', in_channels: int = 1) -> None:
        super(ResNetEncoder, self).__init__()
        #self.cfg = [64, 64, 128, 256, 512]
        backbone = 'resnet50'
        encoder = resnet50_threeD(in_channels, include_top=False)
        self.cfg = [64, 256, 512, 1024, 2048]


        self.conv = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu
        )
        self.layer1 = nn.Sequential(
            encoder.maxpool,
            encoder.layer1
        )
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    def forward(self, x):
        x1 = self.conv(x)
        # 四次下采样
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5

class OutConv(nn.Sequential):
    '''输出模块，包含了一个卷积层，用于输出预测结果'''
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv3d(in_channels, num_classes, kernel_size=1)
        )

class MultiNetwork(nn.Module):
    def __init__(self, num_classes_seg, num_classes_cls):
        super(MultiNetwork, self).__init__()
        #cfg = [64, 128, 256, 512]
        cfg = [64, 256, 512, 1024, 2048]
        # Encoder
        self.encoder = ResNetEncoder(in_channels=1)  # 将使用自定义的ResNet编码器
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = nn.MaxPool3d(2, 2)
        self.conv_ = nn.Sequential(
            nn.Conv3d(cfg[-1], 1024, 3, 1, 1),
            nn.Conv3d(1024, cfg[-1], 3, 1, 1)
        )
        #self.conv6 = nn.Sequential(
            #nn.Conv3d(64, 1, kernel_size=1),  # 减少通道数到1
            #nn.Upsample(scale_factor=2, mode='trilinear'),  # 上采样深度、高度、宽度
        #)
        self.conv6 = nn.Conv3d(cfg[-1], cfg[-2], 3, 1, 1)

        # Decoder
        self.up_conv1 = up_conv(last_channels=cfg[-2], skip_channels=cfg[-2])
        self.up_conv2 = up_conv(cfg[-2])
        self.up_conv3 = up_conv(cfg[-3])
        self.up_conv4 = up_conv(last_channels=cfg[-4], skip_channels=cfg[-5])#(2,64,64,64,128)
        #self.up_last = up_conv(last_channels=cfg[-5], skip_channels=32)
        self.up_conv5 = nn.ConvTranspose3d(cfg[0], cfg[0], 2, 2, 0)
        self.conv7 = nn.Conv3d(cfg[0], 1, 3, 1, 1)


        self.dropout = nn.Dropout(0.5)

        # Seg
        #self.seg_conv = nn.Conv3d(cfg[0], num_classes_seg, 3, 1, 1)
        self.uplast = nn.ConvTranspose3d(in_channels=64, out_channels=1, kernel_size=2, stride=2)

        self.output = nn.Sigmoid()

        # Classification head
        self.Avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classification_head = nn.Linear(2048, num_classes_cls)  # 注意修改这里的输入维度

    def forward(self, x):
        # Encoder
        #print('x')
        #print(x.shape)
        x1, x2, x3, x4, x5 = self.encoder(x)  # 调用自定义的ResNet编码器部分
        #print('x1')
        #print(x1.shape)
        x6 = self.conv6(x5)  #torch.Size([2, 1024, 4, 4, 8])

        # Decoder
        x7 = self.dropout(self.up_conv1(x4, x6))
        x8 = self.dropout(self.up_conv2(x3, x7))
        x9 = self.dropout(self.up_conv3(x2, x8))
        x10 = self.dropout(self.up_conv4(x1, x9)) #(2,64,64,64,128)
        #print('x10')
        #print(x10.shape)
        # Seg head
        #seg_output = self.seg_conv(x10) #torch.Size([2, 1, 64, 64, 128])
        #x10 = self.up_conv5(x10)
        #seg_output = self.conv7(x10)
        #print(seg_output)
        #seg_output = self.output(seg_output) #激活了
        #print(seg_output)
        seg_output = self.uplast(x10)




        # Classification head
        x_cls = self.Avgpool(x5)
        x_cls = x_cls.view(x_cls.size(0), -1)   # 【batch, channel】
        cls_output = self.classification_head(self.dropout(x_cls))

        return seg_output, cls_output


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """两个不改变尺寸的卷积"""

        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

class up_conv(nn.Module):
    def __init__(self, last_channels: int, skip_channels: int = None, out_channels: int = None, deconv: bool = True)\
            -> None:
        """
        上采样及skip_connection

        :param last_channels:
            上一层卷积层的输出维度
        :param skip_channels:
            跳跃连接的特征通道数
        :param out_channels:
            上采样及skip_connection后卷积输出维度
        :param deconv:
            是否使用反卷积
        """
        super(up_conv, self).__init__()
        if skip_channels is None:
            skip_channels = last_channels // 2
        if out_channels is None:
            out_channels = skip_channels

        if deconv:
            self.up_sample = nn.ConvTranspose3d(
                in_channels=last_channels, out_channels=skip_channels, kernel_size=2, stride=2
            )
        else:
            self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear')

        self.conv = BasicBlock(in_channels=2 * skip_channels, out_channels=out_channels)

    def forward(self, skip, last):
        last = self.up_sample(last)

        if skip.size() != last.size():
            _, _, z1, x1, y1 = skip.size()
            _, _, z2, x2, y2 = last.size()
            # F.pad的pad参数为(前，后，左，右，上，下)
            last = F.pad(last, [0, y1 - y2, 0, x1 - x2, 0, z1 - z2])

        x = torch.concat((skip, last), dim=1)
        x = self.conv(x)

        return x


def count_params(MultiNetwork):
    return sum(p.numel() for p in MultiNetwork.parameters())

#if __name__ == "__main__":
model = MultiNetwork(num_classes_cls=1, num_classes_seg=1)

print(count_params((model)))
print(model)
x = torch.ones((2, 1, 128, 128, 256))  # [batch, channel, H,W,D]
#x = x.to(device)
y, z  = model(x)
print(y.shape, z.shape)


# # 打印模型架构
# print(model)