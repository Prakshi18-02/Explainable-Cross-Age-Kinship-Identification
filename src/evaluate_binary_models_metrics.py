import os
import pickle
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from fusion_embedding_dataset import FusionEmbeddingDataset
from binary_fusion_model import BinaryFusionModel
from triplet_model import TripletEmbeddingNet
from fusion_triplet_model import FusionTripletNet


os.makedirs("results/metrics", exist_ok=True)


def save_report(model_name, labels, preds):
    acc = accuracy_score(labels, preds) * 100
    prec = precision_score(labels, preds, zero_division=0) * 100
    rec = recall_score(labels, preds, zero_division=0) * 100
    f1 = f1_score(labels, preds, zero_division=0) * 100

    report = classification_report(
        labels,
        preds,
        target_names=["Not Kin", "Kin"],
        zero_division=0
    )

    print("=" * 60)
    print(model_name)
    print("=" * 60)
    print(f"Accuracy : {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall   : {rec:.2f}%")
    print(f"F1-score : {f1:.2f}%")
    print(report)

    with open(f"results/metrics/{model_name}_binary_report.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy : {acc:.2f}%\n")
        f.write(f"Precision: {prec:.2f}%\n")
        f.write(f"Recall   : {rec:.2f}%\n")
        f.write(f"F1-score : {f1:.2f}%\n\n")
        f.write(report)

    return {
        "Model": model_name,
        "Accuracy": round(acc, 2),
        "Precision": round(prec, 2),
        "Recall": round(rec, 2),
        "F1-score": round(f1, 2)
    }


def evaluate_binary_fusion():
    val_dataset = FusionEmbeddingDataset(
        csv_file="data/val_pairs.csv",
        face_embedding_file="data/face_embeddings.pkl",
        region_embedding_file="data/region_embeddings.pkl"
    )

    test_dataset = FusionEmbeddingDataset(
        csv_file="data/test_pairs.csv",
        face_embedding_file="data/face_embeddings.pkl",
        region_embedding_file="data/region_embeddings.pkl"
    )

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = BinaryFusionModel()
    model.load_state_dict(torch.load("checkpoints/best_binary_fusion.pth", map_location="cpu"))
    model.eval()

    def collect_probs(loader):
        probs_all = []
        labels_all = []

        with torch.no_grad():
            for batch in loader:
                output, _ = model(
                    batch["face1"], batch["face2"],
                    batch["eyes1"], batch["nose1"], batch["lips1"],
                    batch["eyes2"], batch["nose2"], batch["lips2"]
                )

                probs = torch.sigmoid(output).squeeze(1)

                probs_all.extend(probs.numpy().tolist())
                labels_all.extend(batch["label"].numpy().tolist())

        return np.array(probs_all), np.array(labels_all)

    val_probs, val_labels = collect_probs(val_loader)

    best_threshold = 0.5
    best_val_acc = 0.0

    for threshold in np.arange(0.30, 0.71, 0.01):
        preds = (val_probs >= threshold).astype(int)
        acc = accuracy_score(val_labels, preds) * 100

        if acc > best_val_acc:
            best_val_acc = acc
            best_threshold = threshold

    test_probs, test_labels = collect_probs(test_loader)
    test_preds = (test_probs >= best_threshold).astype(int)

    print(f"\nBinary Fusion Best Threshold: {best_threshold:.2f}")
    print(f"Binary Fusion Best Val Accuracy: {best_val_acc:.2f}%")

    return save_report("Binary_Fusion", test_labels, test_preds)


def evaluate_triplet():
    with open("data/face_embeddings.pkl", "rb") as f:
        face_embeddings = pickle.load(f)

    model = TripletEmbeddingNet()
    model.load_state_dict(torch.load("checkpoints/best_triplet_model.pth", map_location="cpu"))
    model.eval()

    def get_projected_embedding(image_path):
        x = torch.tensor(face_embeddings[image_path], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
        return out.squeeze(0).numpy()

    def compute_distances(csv_file):
        df = pd.read_csv(csv_file)
        distances = []
        labels = []

        for _, row in df.iterrows():
            img1 = row["image1"]
            img2 = row["image2"]

            if img1 not in face_embeddings or img2 not in face_embeddings:
                continue

            emb1 = get_projected_embedding(img1)
            emb2 = get_projected_embedding(img2)

            distances.append(np.linalg.norm(emb1 - emb2))
            labels.append(row["label"])

        return np.array(distances), np.array(labels)

    val_dist, val_labels = compute_distances("data/val_pairs.csv")

    best_threshold = 0.0
    best_val_acc = 0.0

    for threshold in np.arange(val_dist.min(), val_dist.max(), 0.01):
        preds = (val_dist < threshold).astype(int)
        acc = accuracy_score(val_labels, preds) * 100

        if acc > best_val_acc:
            best_val_acc = acc
            best_threshold = threshold

    test_dist, test_labels = compute_distances("data/test_pairs.csv")
    test_preds = (test_dist < best_threshold).astype(int)

    print(f"\nTriplet Best Threshold: {best_threshold:.4f}")
    print(f"Triplet Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Average Kin Distance: {test_dist[test_labels == 1].mean():.4f}")
    print(f"Average Not-Kin Distance: {test_dist[test_labels == 0].mean():.4f}")

    return save_report("Triplet_Model", test_labels, test_preds)


def evaluate_fusion_triplet():
    with open("data/face_embeddings.pkl", "rb") as f:
        face_embeddings = pickle.load(f)

    with open("data/region_embeddings.pkl", "rb") as f:
        region_embeddings = pickle.load(f)

    model = FusionTripletNet()
    model.load_state_dict(torch.load("checkpoints/best_fusion_triplet_model.pth", map_location="cpu"))
    model.eval()

    def make_feature(image_path):
        face = torch.tensor(face_embeddings[image_path], dtype=torch.float32)
        regions = region_embeddings[image_path]

        eyes = torch.tensor(regions["eyes"], dtype=torch.float32)
        nose = torch.tensor(regions["nose"], dtype=torch.float32)
        lips = torch.tensor(regions["lips"], dtype=torch.float32)

        return torch.cat([face, eyes, nose, lips], dim=0).unsqueeze(0)

    def get_projected_embedding(image_path):
        x = make_feature(image_path)
        with torch.no_grad():
            out = model(x)
        return out.squeeze(0).numpy()

    def compute_distances(csv_file):
        df = pd.read_csv(csv_file)
        distances = []
        labels = []

        for _, row in df.iterrows():
            img1 = row["image1"]
            img2 = row["image2"]

            if (
                img1 not in face_embeddings or img2 not in face_embeddings
                or img1 not in region_embeddings or img2 not in region_embeddings
            ):
                continue

            emb1 = get_projected_embedding(img1)
            emb2 = get_projected_embedding(img2)

            distances.append(np.linalg.norm(emb1 - emb2))
            labels.append(row["label"])

        return np.array(distances), np.array(labels)

    val_dist, val_labels = compute_distances("data/val_pairs.csv")

    best_threshold = 0.0
    best_val_acc = 0.0

    for threshold in np.arange(val_dist.min(), val_dist.max(), 0.01):
        preds = (val_dist < threshold).astype(int)
        acc = accuracy_score(val_labels, preds) * 100

        if acc > best_val_acc:
            best_val_acc = acc
            best_threshold = threshold

    test_dist, test_labels = compute_distances("data/test_pairs.csv")
    test_preds = (test_dist < best_threshold).astype(int)

    print(f"\nFusion Triplet Best Threshold: {best_threshold:.4f}")
    print(f"Fusion Triplet Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Average Kin Distance: {test_dist[test_labels == 1].mean():.4f}")
    print(f"Average Not-Kin Distance: {test_dist[test_labels == 0].mean():.4f}")

    return save_report("Fusion_Triplet_Model", test_labels, test_preds)


if __name__ == "__main__":
    results = []

    results.append(evaluate_binary_fusion())
    results.append(evaluate_triplet())
    results.append(evaluate_fusion_triplet())

    df = pd.DataFrame(results)
    df.to_csv("results/metrics/binary_models_summary.csv", index=False)

    print("\nBinary Verification Model Summary:")
    print(df)