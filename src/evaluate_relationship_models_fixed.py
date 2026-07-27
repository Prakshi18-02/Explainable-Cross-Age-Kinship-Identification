import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from models import SiameseResNet18
from attention_models import SiameseSEResNet18
from cbam_models import SiameseCBAMResNet18
from hereditary_resnet18 import HereditaryResNet18


DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "Father-Son",
    "Father-Daughter",
    "Mother-Son",
    "Mother-Daughter"
]


class FixedCrossAgeDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def get_fixed_image(self, person_path):
        full_path = os.path.join(self.root_dir, person_path.replace("/", os.sep))
        images = sorted([
            f for f in os.listdir(full_path)
            if f.lower().endswith(".jpg")
        ])
        if not images:
            raise ValueError(f"No images found in {full_path}")
        return os.path.join(full_path, images[0])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img1 = Image.open(self.get_fixed_image(row["p1"])).convert("RGB")
        img2 = Image.open(self.get_fixed_image(row["p2"])).convert("RGB")

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        label = int(row["label"])

        return img1, img2, label


def evaluate_model(model, model_name, checkpoint_path):
    print("=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    dataset = FixedCrossAgeDataset(
        csv_file="data/test.csv",
        root_dir="data/FIW/FIDs"
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            outputs = model(img1, img2)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100

    print(f"Accuracy : {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall   : {rec:.2f}%")
    print(f"F1-score : {f1:.2f}%")

    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    os.makedirs("results/metrics", exist_ok=True)

    with open(f"results/metrics/{model_name}_fixed_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy : {acc:.2f}%\n")
        f.write(f"Precision: {prec:.2f}%\n")
        f.write(f"Recall   : {rec:.2f}%\n")
        f.write(f"F1-score : {f1:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "Model": model_name,
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1-score": round(f1, 2)
    }


if __name__ == "__main__":
    results = []

    results.append(
        evaluate_model(
            SiameseResNet18(num_classes=4),
            "Siamese_ResNet18",
            "checkpoints/best_resnet18.pth"
        )
    )

    results.append(
        evaluate_model(
            SiameseSEResNet18(num_classes=4),
            "Siamese_SE_ResNet18",
            "checkpoints/best_se_resnet18.pth"
        )
    )

    results.append(
        evaluate_model(
            SiameseCBAMResNet18(num_classes=4),
            "Siamese_CBAM_ResNet18",
            "checkpoints/best_cbam_resnet18.pth"
        )
    )

    results.append(
        evaluate_model(
            HereditaryResNet18(num_classes=4),
            "Hereditary_ResNet18",
            "checkpoints/best_hereditary_resnet18.pth"
        )
    )

    df = pd.DataFrame(results)
    df.to_csv("results/metrics/relationship_models_fixed_summary.csv", index=False)

    print("\nRelationship Model Summary:")
    print(df)