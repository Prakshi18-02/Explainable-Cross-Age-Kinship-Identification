import torch
import torch.nn as nn

from torchvision import models


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class SiameseSEResNet18(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.se = SEBlock(512)

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

        x = self.se(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x

    def forward(self, img1, img2):

        f1 = self.extract(img1)

        f2 = self.extract(img2)

        x = torch.cat([f1, f2], dim=1)

        return self.classifier(x)