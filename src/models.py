import torch
import torch.nn as nn
from torchvision import models


class SiameseResNet18(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.feature_extractor = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, img1, img2):

        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)

        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)

        combined = torch.cat([feat1, feat2], dim=1)

        output = self.classifier(combined)

        return output