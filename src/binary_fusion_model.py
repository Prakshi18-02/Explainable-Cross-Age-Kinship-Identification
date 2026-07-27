import torch
import torch.nn as nn


class BinaryFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.face_branch = nn.Sequential(
            nn.Linear(512 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.region_branch = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def pair_features(self, a, b):

        return torch.cat(
            [
                a,
                b,
                torch.abs(a - b)
            ],
            dim=1
        )

    def forward(
        self,
        face1,
        face2,
        eyes1,
        nose1,
        lips1,
        eyes2,
        nose2,
        lips2
    ):

        face_pair = self.pair_features(
            face1,
            face2
        )

        eyes_pair = self.pair_features(
            eyes1,
            eyes2
        )

        nose_pair = self.pair_features(
            nose1,
            nose2
        )

        lips_pair = self.pair_features(
            lips1,
            lips2
        )

        region_pair = torch.cat(
            [
                eyes_pair,
                nose_pair,
                lips_pair
            ],
            dim=1
        )

        face_feat = self.face_branch(face_pair)

        region_feat = self.region_branch(region_pair)

        fusion_input = torch.cat(
            [
                face_feat,
                region_feat
            ],
            dim=1
        )

        weights = self.attention(fusion_input)

        face_w = weights[:, 0:1]
        region_w = weights[:, 1:2]

        fused = torch.cat(
            [
                face_feat * face_w,
                region_feat * region_w
            ],
            dim=1
        )

        output = self.classifier(fused)

        return output, weights