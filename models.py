import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

vgg11 = models.vgg11()


def conv3x3(input, output):
    return nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1)


def conv_relu(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        # nn.BatchNorm2d(out_f),
        nn.ReLU(inplace=True)
    )


def conv_block(f_list, *args, **kwargs):
    return nn.Sequential(
        *[conv_relu(in_f, out_f, *args, **kwargs) for in_f, out_f in zip(f_list, f_list[1:])],
        nn.MaxPool2d(kernel_size=2)
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # VGG-11 encoder
        # self.encoder = nn.Sequential(
        #     conv_block([3, 64], kernel_size=3, padding=1),
        #     conv_block([64, 128], kernel_size=3, padding=1),
        #     conv_block([128, 256, 256], kernel_size=3, padding=1),
        #     conv_block([256, 512, 512], kernel_size=3, padding=1),
        #     conv_block([512, 512, 512], kernel_size=3, padding=1)
        # )

        self.encoder = vgg11.features

        # self.decoder =

    def forward(self, x):
        x = self.encoder.forward(x)
        return x





# def conv3x3(in_, out):
#     return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32):
        """
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Convolutions are from VGG11
        self.encoder = models.vgg11().features

        # "relu" layer is taken from VGG probably for generality, but it's not clear
        self.relu = self.encoder[1]

        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1, )

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        # Deconvolutions with copies of VGG11 layers of corresponding size
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return F.sigmoid(self.final(dec1))


def unet11(**kwargs):
    model = UNet11(**kwargs)

    return model


# def get_model():
#     model = unet11()
#     model.train()
#     return model.to(device)