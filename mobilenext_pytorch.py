import torch
import torch.nn as nn

BN_MOMENTUM = 0.001
BN_EPSILON = 1e-3

def _make_divisible(channels, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_channels = max(min_value, int(channels+divisor/2)//divisor*divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels

class SandGlassBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction):
        super(SandGlassBlock, self).__init__()

        self.conv = nn.Sequential(
            # depthwise
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, eps=BN_EPSILON, momentum=BN_MOMENTUM),
            nn.ReLU6(),
            # pointwise reduction
            nn.Conv2d(in_channels, in_channels//reduction, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels//reduction, eps=BN_EPSILON, momentum=BN_MOMENTUM),
            # pointwise expansion
            nn.Conv2d(in_channels//reduction, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels, eps=BN_EPSILON, momentum=BN_MOMENTUM),
            nn.ReLU6(),
            # depthwise
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels, eps=BN_EPSILON, momentum=BN_MOMENTUM)
        )

        self.residual = (in_channels == out_channels and stride == 1)

    def forward(self, x):
        if self.residual:
            return self.conv(x) + x
        else:
            return self.conv(x)

class MobileNeXt(nn.Module):
    config = [
        # channels, stride, reduction, blocks
        [96,   2, 2, 1],
        [144,  1, 6, 1],
        [192,  2, 6, 3],
        [288,  2, 6, 3],
        [384,  1, 6, 4],
        [576,  2, 6, 4],
        [960,  1, 6, 2],
        [1280, 1, 6, 1]
    ]

    def __init__(self, input_size=224, num_classes=1000, width_mult=1.):
        super(MobileNeXt, self).__init__()

        stem_channels = 32
        stem_channels = _make_divisible(int(stem_channels*width_mult), 8)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels, eps=BN_EPSILON, momentum=BN_MOMENTUM),
            nn.ReLU6()
        )

        blocks = []
        in_channels = stem_channels
        for c, s, r, b in self.config:
            out_channels = _make_divisible(int(c*width_mult), 8)
            for i in range(b):
                stride = s if i == 0 else 1
                blocks.append(SandGlassBlock(in_channels, out_channels, stride, r))
                in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = x.view(-1, x.size(1))
        y = self.classifier(x)

        return y

if __name__ == '__main__':
    model = MobileNeXt()
    x_image = nn.Parameter(torch.randn(1, 3, 224, 224), requires_grad=False)
    y = model(x_image)
