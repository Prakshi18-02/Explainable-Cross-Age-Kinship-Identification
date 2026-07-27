import torch
from torchvision import models


class ResNetFeatureExtractor:

    def __init__(self):

        self.device = torch.device("cpu")

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.extractor = torch.nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.extractor = self.extractor.to(self.device)

        self.extractor.eval()

    def extract(self, image_tensor):

        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():

            feature = self.extractor(image_tensor)

        feature = feature.view(feature.size(0), -1)

        return feature