import pickle
import pandas as pd
import torch

from torch.utils.data import Dataset


class FusionEmbeddingDataset(Dataset):

    def __init__(
        self,
        csv_file,
        face_embedding_file,
        region_embedding_file
    ):

        self.df = pd.read_csv(csv_file)

        with open(face_embedding_file, "rb") as f:
            self.face_embeddings = pickle.load(f)

        with open(region_embedding_file, "rb") as f:
            self.region_embeddings = pickle.load(f)

        self.df = self.df[

            self.df["image1"].isin(self.face_embeddings)
            &
            self.df["image2"].isin(self.face_embeddings)
            &
            self.df["image1"].isin(self.region_embeddings)
            &
            self.df["image2"].isin(self.region_embeddings)

        ].reset_index(drop=True)

        print("Valid Pairs:", len(self.df))

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img1 = row["image1"]
        img2 = row["image2"]

        face1 = self.face_embeddings[img1]
        face2 = self.face_embeddings[img2]

        reg1 = self.region_embeddings[img1]
        reg2 = self.region_embeddings[img2]

        return {

            "face1": torch.tensor(face1, dtype=torch.float32),
            "face2": torch.tensor(face2, dtype=torch.float32),

            "eyes1": torch.tensor(reg1["eyes"], dtype=torch.float32),
            "nose1": torch.tensor(reg1["nose"], dtype=torch.float32),
            "lips1": torch.tensor(reg1["lips"], dtype=torch.float32),

            "eyes2": torch.tensor(reg2["eyes"], dtype=torch.float32),
            "nose2": torch.tensor(reg2["nose"], dtype=torch.float32),
            "lips2": torch.tensor(reg2["lips"], dtype=torch.float32),

            "label": torch.tensor(
                row["label"],
                dtype=torch.float32
            )
        }