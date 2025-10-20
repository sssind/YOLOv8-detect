import torch
import torch.nn as nn


class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向全局平均池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向全局平均池化

        mid_channels = max(8, in_channels // reduction)  # 计算中间层通道数

        # 共享的1x1卷积，用于降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU()

        # 两个独立的1x1卷积，分别用于水平方向和垂直方向的注意力权重
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x  # 保留原始输入，用于后续的shortcut连接
        n, c, h, w = x.size()

        # 水平方向全局平均池化, 形状从 [B, C, H, W] 变为 [B, C, H, 1]
        x_h = self.pool_h(x)
        # 垂直方向全局平均池化, 形状从 [B, C, H, W] 变为 [B, C, 1, W]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 最后进行转置以适应拼接

        # 将水平和垂直方向的特征拼接在一起
        y = torch.cat([x_h, x_w], dim=2)  # 形状变为 [B, C, H+W, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        # 将拼接后的特征分割为水平方向和垂直方向两部分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 对垂直方向特征进行转置，恢复形状

        # 分别计算水平方向和垂直方向的注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # 形状 [B, out_channels, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # 形状 [B, out_channels, 1, W]

        # 将注意力权重应用到原始特征上
        out = identity * a_w * a_h  # 结合了水平和垂直注意力权重的输出
        return out