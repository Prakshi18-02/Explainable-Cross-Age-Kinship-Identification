import pickle
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms


PAIR_FILES = [
    "data/train_pairs.csv",
    "data/val_pairs.csv",
    "data/test_pairs.csv"
]

OUTPUT_FILE = "data/face_embeddings.pkl"

DEVICE = torch.device("cpu")


print("=" * 50)
print("CREATE FACE EMBEDDINGS")
print("=" * 50)


all_images = set()

for csv_file in PAIR_FILES:

    df = pd.read_csv(csv_file)

    all_images.update(df["image1"].tolist())
    all_images.update(df["image2"].tolist())


all_images = sorted(list(all_images))

print("Unique Images:", len(all_images))


resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


embeddings = {}


for image_path in tqdm(all_images):

    try:

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            feature = feature_extractor(image)

        feature = feature.view(-1).cpu().numpy()

        embeddings[image_path] = feature

    except Exception as e:

        print("Skipped:", image_path)
        print("Reason:", e)


print()
print("Saving embeddings...")

with open(
    OUTPUT_FILE,
    "wb"
) as f:

    pickle.dump(
        embeddings,
        f
    )


print("Saved:", len(embeddings), "embeddings")
print("Output:", OUTPUT_FILE)