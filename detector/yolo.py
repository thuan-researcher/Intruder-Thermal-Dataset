import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, f):
        # beta = 1
        # f_soft = (1 / beta) * torch.log(1 + torch.exp(beta * f))
        return f * torch.tanh(F.softplus(f))


class ConvBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, stride=1, padding=0, activation='mish'):
        super(ConvBlock, self).__init__()

        if padding == 0:
            padding = 0 if stride == 2 else (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding, padding_mode='zeros', bias=False)
        self.bn = nn.BatchNorm2d(out_filters)
        self.act = Mish() if activation == 'mish' else nn.LeakyReLU(0.01)

    def forward(self, f):
        f = self.act(self.bn(self.conv(f)))
        return f


class BasicBlock(nn.Module):
    def __init__(self, in_filters, out_filters, block_size, activation='mish', narrow=False):
        super(BasicBlock, self).__init__()
        mid_filters = out_filters // 2 if narrow else out_filters

        # self.zero_pad = nn.ConstantPad2d((1, 0, 1, 0), 0)
        self.conv_block1 = ConvBlock(in_filters, out_filters, kernel_size=3, stride=2, padding=1, activation=activation)
        self.conv_block2 = ConvBlock(out_filters, mid_filters, kernel_size=1, activation=activation)
        self.conv_block3 = ConvBlock(out_filters, mid_filters, kernel_size=1, activation=activation)

        self.block_size = block_size
        self.block_layers = nn.ModuleList([])
        for _ in range(self.block_size):
            layers = []
            layers.append(ConvBlock(mid_filters, out_filters // 2, kernel_size=1, activation=activation))
            layers.append(ConvBlock(out_filters // 2, mid_filters, kernel_size=3, activation=activation))
            self.block_layers.append(nn.Sequential(*layers))

        self.conv_block4 = ConvBlock(mid_filters, mid_filters, kernel_size=1, activation=activation)
        self.conv_block5 = ConvBlock(mid_filters * 2, out_filters, kernel_size=1, activation=activation)

    def forward(self, f):
        # f = self.zero_pad(f)
        f = self.conv_block1(f)
        f_short = self.conv_block2(f)
        f = self.conv_block3(f)

        for idx in range(self.block_size):
            f_out = self.block_layers[idx](f)
            f += f_out

        f = self.conv_block4(f)
        f = torch.cat([f, f_short], dim=1)
        f = self.conv_block5(f)

        return f


class Darknet(nn.Module):
    def __init__(self, in_filters=3, out_filters=1024):
        super(Darknet, self).__init__()
        self.conv_block = ConvBlock(in_filters, 32, kernel_size=3, activation='mish')
        self.dark_layer1 = BasicBlock(32, 64, block_size=1, activation='mish', narrow=False)
        self.dark_layer2 = BasicBlock(64, 128, block_size=2, activation='mish', narrow=True)
        self.dark_layer3 = BasicBlock(128, 256, block_size=8, activation='mish', narrow=True)
        self.dark_layer4 = BasicBlock(256, 512, block_size=8, activation='mish', narrow=True)
        self.dark_layer5 = BasicBlock(512, out_filters, block_size=4, activation='mish', narrow=True)

    def forward(self, f):
        f = self.conv_block(f)
        f = self.dark_layer1(f)
        f = self.dark_layer2(f)
        f1 = self.dark_layer3(f)
        f2 = self.dark_layer4(f1)
        f3 = self.dark_layer5(f2)
        return [f1, f2, f3]


class Upsample2d(nn.Module):
    def __init__(self, stride):
        super(Upsample2d, self).__init__()
        if isinstance(stride, int):
            stride = [stride, stride]
        self.stride = stride

    def forward(self, x):
        sh = list(x.shape)
        rsize = (int(sh[2] * self.stride[0]), int(sh[3] * self.stride[1]))
        return F.interpolate(x, size=rsize, mode='nearest')


class YoloV4(nn.Module):
    def __init__(self,
                 num_anchors=3,
                 num_classes=80):
        super(YoloV4, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.darknet = Darknet(in_filters=3, out_filters=1024)
        self.pre_head_32, self.mid_head_32, self.post_head_32, self.out_head_32 = self._create_head_32(
            [1024, 2048, 1024])
        self.pre_head_16, self.mid_head_16, self.post_head_16, self.out_head_16 = self._create_head_16([512, 512, 512])
        self.pre_head_8, self.mid_head_8, self.out_head_8 = self._create_head_8([256, 256, 128])
        self.max_pool_1 = nn.MaxPool2d(13, stride=1, padding=6)
        self.max_pool_2 = nn.MaxPool2d(9, stride=1, padding=4)
        self.max_pool_3 = nn.MaxPool2d(5, stride=1, padding=2)
        self.upsample_32 = nn.Sequential(*[
            ConvBlock(512, 256, kernel_size=1, activation='leaky'),
            Upsample2d(stride=2)
        ])
        self.upsample_16 = nn.Sequential(*[
            ConvBlock(256, 128, kernel_size=1, activation='leaky'),
            Upsample2d(stride=2)
        ])
        self.downsample_8 = nn.Sequential(*[
            # nn.ConstantPad2d((1, 0, 1, 0), 0),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1, activation='leaky')
        ])
        self.downsample_16 = nn.Sequential(*[
            # nn.ConstantPad2d((1, 0, 1, 0), 0),
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1, activation='leaky')
        ])

    def _create_head_32(self, in_filters_list):
        pre_layers = [
            ConvBlock(in_filters_list[0], 512, kernel_size=1, activation='leaky'),
            ConvBlock(512, 1024, kernel_size=3, activation='leaky'),
            ConvBlock(1024, 512, kernel_size=1, activation='leaky')
        ]
        mid_layers = [
            ConvBlock(in_filters_list[1], 512, kernel_size=1, activation='leaky'),
            ConvBlock(512, 1024, kernel_size=3, activation='leaky'),
            ConvBlock(1024, 512, kernel_size=1, activation='leaky')
        ]
        post_layers = [
            ConvBlock(in_filters_list[2], 512, kernel_size=1, activation='leaky'),
            ConvBlock(512, 1024, kernel_size=3, activation='leaky'),
            ConvBlock(1024, 512, kernel_size=1, activation='leaky'),
            ConvBlock(512, 1024, kernel_size=3, activation='leaky'),
            ConvBlock(1024, 512, kernel_size=1, activation='leaky'),
        ]
        out_layers = [
            ConvBlock(512, 1024, kernel_size=3, activation='leaky'),
            nn.Conv2d(1024, self.num_anchors * (self.num_classes + 5), kernel_size=1)
        ]

        return nn.Sequential(*pre_layers), nn.Sequential(*mid_layers), nn.Sequential(*post_layers), nn.Sequential(
            *out_layers)

    def _create_head_16(self, in_filters_list):
        pre_layers = [
            ConvBlock(in_filters_list[0], 256, kernel_size=1, activation='leaky')
        ]
        mid_layers = [
            ConvBlock(in_filters_list[1], 256, kernel_size=1, activation='leaky'),
            ConvBlock(256, 512, kernel_size=3, activation='leaky'),
            ConvBlock(512, 256, kernel_size=1, activation='leaky'),
            ConvBlock(256, 512, kernel_size=3, activation='leaky'),
            ConvBlock(512, 256, kernel_size=1, activation='leaky')
        ]
        post_layers = [
            ConvBlock(in_filters_list[2], 256, kernel_size=1, activation='leaky'),
            ConvBlock(256, 512, kernel_size=3, activation='leaky'),
            ConvBlock(512, 256, kernel_size=1, activation='leaky'),
            ConvBlock(256, 512, kernel_size=3, activation='leaky'),
            ConvBlock(512, 256, kernel_size=1, activation='leaky')
        ]
        out_layers = [
            ConvBlock(256, 512, kernel_size=3, activation='leaky'),
            nn.Conv2d(512, self.num_anchors * (self.num_classes + 5), kernel_size=1)
        ]

        return nn.Sequential(*pre_layers), nn.Sequential(*mid_layers), nn.Sequential(*post_layers), nn.Sequential(
            *out_layers)

    def _create_head_8(self, in_filters_list):
        pre_layers = [
            ConvBlock(in_filters_list[0], 128, kernel_size=1, activation='leaky'),
        ]
        mid_layers = [
            ConvBlock(in_filters_list[1], 128, kernel_size=1, activation='leaky'),
            ConvBlock(128, 256, kernel_size=3, activation='leaky'),
            ConvBlock(256, 128, kernel_size=1, activation='leaky'),
            ConvBlock(128, 256, kernel_size=3, activation='leaky'),
            ConvBlock(256, 128, kernel_size=1, activation='leaky')
        ]
        out_layers = [
            ConvBlock(in_filters_list[2], 256, kernel_size=3, activation='leaky'),
            nn.Conv2d(256, self.num_anchors * (self.num_classes + 5), kernel_size=1)
        ]

        return nn.Sequential(*pre_layers), nn.Sequential(*mid_layers), nn.Sequential(*out_layers)

    def forward(self, f):
        f1, f2, f3 = self.darknet(f)

        f3 = self.pre_head_32(f3)
        mp1 = self.max_pool_1(f3)
        mp2 = self.max_pool_2(f3)
        mp3 = self.max_pool_3(f3)
        f3 = torch.cat([mp1, mp2, mp3, f3], dim=1)
        f3 = self.mid_head_32(f3)

        f3_up = self.upsample_32(f3)

        f2 = self.pre_head_16(f2)
        f2 = torch.cat([f2, f3_up], dim=1)
        f2 = self.mid_head_16(f2)

        f2_up = self.upsample_16(f2)

        f1 = self.pre_head_8(f1)
        f1 = torch.cat([f1, f2_up], dim=1)
        f1 = self.mid_head_8(f1)

        f1_out = self.out_head_8(f1)

        f1_down = self.downsample_8(f1)
        f2 = torch.cat([f1_down, f2], dim=1)
        f2 = self.post_head_16(f2)

        f2_out = self.out_head_16(f2)

        f2_down = self.downsample_16(f2)
        f3 = torch.cat([f2_down, f3], dim=1)
        f3 = self.post_head_32(f3)

        f3_out = self.out_head_32(f3)

        return [f1_out, f2_out, f3_out]


if __name__ == '__main__':
    from torchsummary import summary as ts
    import numpy as np

    device = torch.device(device='cpu' if not torch.cuda.is_available() else 'cuda:0')
    model = YoloV4()
    model.to(device)
    ts(model, (3, 608, 608))
    weights_path = 'converted.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    # x = torch.randn(1, 3, 608, 608)
    # predicts = model(x)




