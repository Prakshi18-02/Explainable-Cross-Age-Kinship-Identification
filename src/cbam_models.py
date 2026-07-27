import torch
import torch.nn as nn

from torchvision import models


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        b, c, _, _ = x.size()

        avg = self.fc(
            self.avg_pool(x).view(b, c)
        )

        mx = self.fc(
            self.max_pool(x).view(b, c)
        )

        out = avg + mx

        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        mx, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        x = torch.cat([avg, mx], dim=1)

        x = self.conv(x)

        return self.sigmoid(x)


class CBAM(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.ca = ChannelAttention(channels)

        self.sa = SpatialAttention()

    def forward(self, x):

        x = x * self.ca(x)

        x = x * self.sa(x)

        return x


class SiameseCBAMResNet18(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.cbam = CBAM(512)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def extract(self, x):

        x = self.features(x)

        x = self.cbam(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x

    def forward(self, img1, img2):

        f1 = self.extract(img1)

        f2 = self.extract(img2)

        x = torch.cat([f1, f2], dim=1)

        return self.classifier(x)