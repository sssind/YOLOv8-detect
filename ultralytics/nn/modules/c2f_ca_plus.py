import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, Bottleneck, CA

class C2f_CA(nn.Module):
    """基础C2f与CA融合模块"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, ca_type='coordatt'):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 选择CA类型
        if ca_type == 'coordatt':
            self.ca = CA(c2, c2)
        elif ca_type == 'asparagus':
            self.ca = CA(c2, vertical_enhance=True)
        else:
            self.ca = CA(c2, c2)

        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)))
            for _ in range(n)
        )

    def forward(self, x):
        # 标准C2f前向传播
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x = self.cv2(torch.cat(y, 1))

        # 在输出应用CA注意力
        x = self.ca(x)

        return x


class C2f_Asparagus_CA(nn.Module):
    """芦笋专用的C2f-CA融合模块"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 多层次CA融合
        # 1. 在中间Bottleneck后添加
        self.mid_ca = AsparagusCA(self.c, vertical_enhance=True)
        # 2. 在最终输出添加
        self.output_ca = AsparagusCA(c2, vertical_enhance=True)

        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)))
            for _ in range(n)
        )

        self.mid_index = n // 2  # 中间位置

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))

        # 逐个通过Bottleneck
        for i, m in enumerate(self.m):
            bottleneck_output = m(y[-1])

            # 在中间Bottleneck后应用CA
            if i == self.mid_index:
                bottleneck_output = self.mid_ca(bottleneck_output)

            y.append(bottleneck_output)

        # 最终输出
        x = self.cv2(torch.cat(y, 1))
        # 在输出应用CA
        x = self.output_ca(x)

        return x