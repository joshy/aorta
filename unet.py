import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv(x))
        x = self.activation(self.conv2(x))
        return x


class UpBlock(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 kernel_size=3,
                 activation=F.relu,
                 space_dropout=False):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)
        out = self.activation(self.conv(out))
        out = self.activation(self.conv2(out))
        return out


class UNet(nn.Module):
    """
    Original U-net.
    """

    def __init__(self):
        super(UNet, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv_block1_64 = ConvBlock(1, 64)
        self.conv_block64_128 = ConvBlock(64, 128)
        self.conv_block128_256 = ConvBlock(128, 256)
        self.conv_block256_512 = ConvBlock(256, 512)
        self.conv_block512_1024 = ConvBlock(512, 1024)

        self.up_block1024_512 = UpBlock(1024, 512)
        self.up_block512_256 = UpBlock(512, 256)
        self.up_block256_128 = UpBlock(256, 128)
        self.up_block128_64 = UpBlock(128, 64)

        self.last = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)
        up2 = self.up_block512_256(up1, block3)
        up3 = self.up_block256_128(up2, block2)
        up4 = self.up_block128_64(up3, block1)

        return self.last(up4)


net = UNet()