from models.resnet.resnet_blocks import *


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.d1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        )
        self.d2 = nn.Sequential(
            ResNetEncoderBlock(64, 64),
            ResNetEncoderBlock(64, 64),
        )
        self.d3 = nn.Sequential(
            ResNetEncoderBlock(64, 128, downsample=True, stride=True),
            ResNetEncoderBlock(128, 128),
        )
        self.d4 = nn.Sequential(
            ResNetEncoderBlock(128, 256, downsample=True, stride=True),
            ResNetEncoderBlock(256, 256),
        )
        self.d5 = nn.Sequential(
            ResNetEncoderBlock(256, 512, downsample=True, stride=True),
            ResNetEncoderBlock(512, 512),
        )

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)

        return x


class ResNet18Decoder(nn.Module):
    def __init__(self, classes):
        super(ResNet18Decoder, self).__init__()
        self.u1 = nn.Sequential(
            ResNetDecoderBlock(512, 512),
            ResNetDecoderBlock(512, 256, upsample=True, stride=True),
        )
        self.u2 = nn.Sequential(
            ResNetDecoderBlock(256, 256),
            ResNetDecoderBlock(256, 128, upsample=True, stride=True),
        )
        self.u3 = nn.Sequential(
            ResNetDecoderBlock(128, 128),
            ResNetDecoderBlock(128, 64, upsample=True, stride=True),
        )
        self.u4 = nn.Sequential(
            ResNetDecoderBlock(64, 64),
            ResNetDecoderBlock(64, 64),
        )
        self.u5 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.ConvTranspose2d(64, classes, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)

    def forward(self, x):
        x = self.u1(x)
        x = self.u2(x)
        x = self.u3(x)
        x = self.u4(x)
        x = self.u5(x)
        x = self.conv(x)

        return x


class ResNet18SemanticSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18SemanticSegmentation, self).__init__()
        self.encoder = ResNet18Encoder()
        self.decoder = ResNet18Decoder(num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
