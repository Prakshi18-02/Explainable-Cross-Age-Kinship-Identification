import torch
import torch.nn as nn
from torchvision import models


class HereditaryAttentionBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):

        ca = self.channel_attention(x)

        x = x * ca

        avg_map = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_map, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        spatial_input = torch.cat(
            [avg_map, max_map],
            dim=1
        )

        sa = self.spatial_attention(
            spatial_input
        )

        x = x * sa

        return x


class HereditaryResNet18(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.hereditary_attention = HereditaryAttentionBlock(
            channels=512
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def extract(self, x):

        x = self.features(x)

        x = self.hereditary_attention(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        return x

    def forward(self, img1, img2):

        f1 = self.extract(img1)

        f2 = self.extract(img2)

        combined = torch.cat(
            [f1, f2],
            dim=1
        )

        output = self.classifier(
            combined
        )

        return output