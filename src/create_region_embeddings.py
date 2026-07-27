import cv2
import pickle
import torch
import numpy as np

from tqdm import tqdm
from torchvision import models, transforms


LANDMARK_CACHE = "data/landmark_cache.pkl"
OUTPUT_FILE = "data/region_embeddings.pkl"


with open(LANDMARK_CACHE, "rb") as f:
    landmark_cache = pickle.load(f)

print("Images Found:", len(landmark_cache))


resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

feature_extractor = torch.nn.Sequential(
    *list(resnet.children())[:-1]
)

feature_extractor.eval()


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def crop_region(
    image,
    landmarks,
    start_idx,
    end_idx,
    padding=10
):

    points = landmarks[start_idx:end_idx]

    x1 = max(
        0,
        int(points[:, 0].min()) - padding
    )

    y1 = max(
        0,
        int(points[:, 1].min()) - padding
    )

    x2 = min(
        image.shape[1],
        int(points[:, 0].max()) + padding
    )

    y2 = min(
        image.shape[0],
        int(points[:, 1].max()) + padding
    )

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


embeddings = {}

for image_path, landmarks in tqdm(
    landmark_cache.items()
):

    img = cv2.imread(image_path)

    if img is None:
        continue

    eyes = crop_region(
        img,
        landmarks,
        33,
        53
    )

    nose = crop_region(
        img,
        landmarks,
        72,
        87
    )

    lips = crop_region(
        img,
        landmarks,
        87,
        106
    )

    if (
        eyes is None
        or nose is None
        or lips is None
    ):
        continue

    region_features = {}

    for name, region in {

        "eyes": eyes,
        "nose": nose,
        "lips": lips

    }.items():

        x = transform(region)

        x = x.unsqueeze(0)

        with torch.no_grad():

            feat = feature_extractor(x)

        feat = feat.squeeze()

        feat = feat.numpy()

        region_features[name] = feat

    embeddings[image_path] = region_features


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

print(
    "Saved:",
    len(embeddings),
    "embeddings"
)