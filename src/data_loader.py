import os
import random
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CrossAgeDataset(Dataset):

    def __init__(self, csv_file, root_dir):

        self.df = pd.read_csv(csv_file)

        self.root_dir = root_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def get_random_image(self, person_path):

        full_path = os.path.join(
            self.root_dir,
            person_path.replace("/", os.sep)
        )

        images = [
            f for f in os.listdir(full_path)
            if f.endswith(".jpg")
        ]

        image_name = random.choice(images)

        return os.path.join(full_path, image_name)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img1_path = self.get_random_image(row["p1"])
        img2_path = self.get_random_image(row["p2"])

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label = row["label"]

        return img1, img2, label