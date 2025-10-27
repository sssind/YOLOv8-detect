import torch
import torch.nn as nn

from ultralytics.nn.modules import CA, Bottleneck, Conv


# num1
class C2f_CA(nn.Module):
    """将CA注意力机制融合到C2f模块中."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, ca_ratio=0.25, ca_position="output"):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck数量
            shortcut: 是否使用shortcut连接
            g: 分组卷积的组数
            e: 扩展比率
            ca_ratio: CA模块的通道缩减比率
            ca_position: CA添加位置 - 'output', 'bottleneck', 'both'.
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数
        self.ca_position = ca_position

        # 第一个卷积层
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # 第二个卷积层
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 根据位置策略初始化CA模块
        if ca_position in ["bottleneck", "both"]:
            # 在Bottleneck中添加CA
            self.bottleneck_ca = CA(self.c, self.c, reduction=int(1 / ca_ratio))

        if ca_position in ["output", "both"]:
            # 在最终输出添加CA
            self.output_ca = CA(c2, c2, reduction=int(1 / ca_ratio))

        # Bottleneck模块
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3))) for _ in range(n))

    def forward(self, x):
        # 第一部分：通过cv1并分割
        y = list(self.cv1(x).chunk(2, 1))

        # 第二部分：通过多个Bottleneck
        for i, m in enumerate(self.m):
            # 通过Bottleneck
            bottleneck_output = m(y[-1])

            # 在Bottleneck后应用CA
            if hasattr(self, "bottleneck_ca"):
                bottleneck_output = self.bottleneck_ca(bottleneck_output)

            y.append(bottleneck_output)

        # 第三部分：连接所有特征并通过cv2
        x = self.cv2(torch.cat(y, 1))

        # 在最终输出应用CA
        if hasattr(self, "output_ca"):
            x = self.output_ca(x)

        return x


# num2
class C2f_Asparagus_CA(nn.Module):
    """针对成熟期芦笋优化的C2f_CA模块."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, vertical_enhance=True):
        super().__init__()
        self.c = int(c2 * e)
        self.vertical_enhance = vertical_enhance

        # 卷积层
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 针对芦笋的CA模块 - 在关键位置添加
        # 1. 在中间Bottleneck后添加（捕捉中层特征）
        self.mid_ca = AsparagusCA(self.c, vertical_enhance=vertical_enhance)

        # 2. 在最后Bottleneck后添加（捕捉深层特征）
        self.final_bottleneck_ca = AsparagusCA(self.c, vertical_enhance=vertical_enhance)

        # 3. 在最终输出前添加（全局优化）
        self.output_ca = AsparagusCA(c2, vertical_enhance=vertical_enhance)

        # Bottleneck模块
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3))) for _ in range(n))

        # 记录中间层位置
        self.mid_index = n // 2

    def forward(self, x):
        # 通过cv1并分割
        y = list(self.cv1(x).chunk(2, 1))

        # 逐个通过Bottleneck并选择性添加CA
        for i, m in enumerate(self.m):
            bottleneck_output = m(y[-1])

            # 在中间Bottleneck后添加CA
            if i == self.mid_index:
                bottleneck_output = self.mid_ca(bottleneck_output)

            # 在最后一个Bottleneck后添加CA
            if i == len(self.m) - 1:
                bottleneck_output = self.final_bottleneck_ca(bottleneck_output)

            y.append(bottleneck_output)

        # 连接并通过cv2
        x = self.cv2(torch.cat(y, 1))

        # 最终输出CA
        x = self.output_ca(x)

        return x


class AsparagusCA(nn.Module):
    """芦笋专用的CA注意力，增强垂直方向感知."""

    def __init__(self, channels, reduction=16, vertical_enhance=True):
        super().__init__()
        self.channels = channels
        self.vertical_enhance = vertical_enhance

        # 坐标注意力机制
        self.avg_pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平全局池化
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直全局池化

        mid_channels = max(channels // reduction, 8)

        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, 1, bias=True)

        # 芦笋垂直增强
        if vertical_enhance:
            self.vertical_enhancer = nn.Sequential(
                nn.Conv2d(channels, channels // 4, 3, padding=1),
                nn.BatchNorm2d(channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        _batch_size, _c, h, w = x.size()

        # 坐标注意力
        # 水平方向特征 [b, c, h, 1]
        x_h = self.avg_pool_h(x)
        # 垂直方向特征 [b, c, 1, w]
        x_w = self.avg_pool_w(x)

        # 拼接并通过共享MLP
        x_cat = torch.cat([x_h, x_w], dim=2)  # [b, c, h+w, 1]
        y = self.conv1(x_cat)
        y = self.relu(y)

        # 分离特征
        x_h, x_w = torch.split(y, [h, w], dim=2)

        # 生成注意力权重
        att_h = self.sigmoid(self.conv_h(x_h))
        att_w = self.sigmoid(self.conv_w(x_w))

        # 基础CA输出
        out = identity * att_h * att_w

        # 芦笋垂直增强
        if self.vertical_enhance:
            vertical_att = self.vertical_enhancer(out)
            out = out + out * vertical_att  # 残差式增强

        return out
