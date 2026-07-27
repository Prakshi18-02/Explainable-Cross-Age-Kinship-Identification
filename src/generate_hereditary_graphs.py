import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hereditary_resnet18 import HereditaryResNet18


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = "checkpoints/best_hereditary_resnet18.pth"
IMAGE_ROOT = "data/FIW/FIDs"
OUTPUT_DIR = "results/metrics/hereditary_graphs"

CLASS_NAMES = [
    "Father-Son",
    "Father-Daughter",
    "Mother-Son",
    "Mother-Daughter",
]

NUM_CLASSES = len(CLASS_NAMES)


class FixedCrossAgeDataset(Dataset):
    """
    Uses the same fixed-image selection policy as
    evaluate_relationship_models_fixed.py.

    For every person folder, the alphabetically first JPG file is selected.
    """

    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def get_fixed_image(self, person_path):
        full_path = os.path.join(
            self.root_dir,
            person_path.replace("/", os.sep),
        )

        if not os.path.isdir(full_path):
            raise FileNotFoundError(
                f"Person folder not found: {full_path}"
            )

        images = sorted(
            file_name
            for file_name in os.listdir(full_path)
            if file_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        )

        if not images:
            raise ValueError(
                f"No face images found in: {full_path}"
            )

        return os.path.join(full_path, images[0])

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image1_path = self.get_fixed_image(row["p1"])
        image2_path = self.get_fixed_image(row["p2"])

        image1 = Image.open(image1_path).convert("RGB")
        image2 = Image.open(image2_path).convert("RGB")

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        label = int(row["label"])

        return image1, image2, label


def load_frozen_model():
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = HereditaryResNet18(num_classes=NUM_CLASSES)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def evaluate_split(model, csv_file, split_name):
    dataset = FixedCrossAgeDataset(
        csv_file=csv_file,
        root_dir=IMAGE_ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for image1, image2, labels in loader:
            image1 = image1.to(DEVICE)
            image2 = image2.to(DEVICE)

            outputs = model(image1, image2)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predictions.cpu().numpy()
            )
            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    labels_array = np.asarray(all_labels)
    predictions_array = np.asarray(all_predictions)
    probabilities_array = np.asarray(all_probabilities)

    accuracy = (
        accuracy_score(
            labels_array,
            predictions_array,
        )
        * 100
    )

    print(
        f"{split_name} samples : {len(dataset)}"
    )
    print(
        f"{split_name} accuracy: {accuracy:.2f}%"
    )

    return {
        "split": split_name,
        "accuracy": accuracy,
        "labels": labels_array,
        "predictions": predictions_array,
        "probabilities": probabilities_array,
    }


def create_train_test_graph(train_result, test_result):
    split_names = [
        train_result["split"],
        test_result["split"],
    ]

    accuracies = [
        train_result["accuracy"],
        test_result["accuracy"],
    ]

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        split_names,
        accuracies,
    )

    plt.ylabel("Accuracy (%)")
    plt.xlabel("Dataset split")
    plt.title(
        "Hereditary ResNet18: "
        "Training and Test Accuracy"
    )
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.3)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "hereditary_train_test_accuracy.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def create_multiclass_roc_curve(test_result):
    labels = test_result["labels"]
    probabilities = test_result["probabilities"]

    binary_labels = label_binarize(
        labels,
        classes=np.arange(NUM_CLASSES),
    )

    plt.figure(figsize=(8, 6))

    roc_records = []

    # Per-class ROC curves
    for class_index, class_name in enumerate(CLASS_NAMES):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_labels[:, class_index],
            probabilities[:, class_index],
        )

        class_auc = auc(
            false_positive_rate,
            true_positive_rate,
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{class_name} (AUC = {class_auc:.3f})",
        )

        roc_records.append(
            {
                "Class": class_name,
                "AUC": class_auc,
            }
        )

    # Micro-average ROC
    micro_fpr, micro_tpr, _ = roc_curve(
        binary_labels.ravel(),
        probabilities.ravel(),
    )

    micro_auc = auc(
        micro_fpr,
        micro_tpr,
    )

    plt.plot(
        micro_fpr,
        micro_tpr,
        linestyle="--",
        linewidth=2,
        label=f"Micro-average (AUC = {micro_auc:.3f})",
    )

    # Random classifier reference line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        label="Random classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Multiclass ROC Curve of Hereditary ResNet18"
    )
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "hereditary_multiclass_roc_curve.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    roc_records.append(
        {
            "Class": "Micro-average",
            "AUC": micro_auc,
        }
    )

    pd.DataFrame(roc_records).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "hereditary_roc_auc_values.csv",
        ),
        index=False,
    )

    print(f"Saved: {output_path}")
    print(f"Micro-average ROC-AUC: {micro_auc:.4f}")


def create_precision_recall_curve(test_result):
    labels = test_result["labels"]
    probabilities = test_result["probabilities"]

    binary_labels = label_binarize(
        labels,
        classes=np.arange(NUM_CLASSES),
    )

    plt.figure(figsize=(8, 6))

    pr_records = []

    # Per-class Precision-Recall curves
    for class_index, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(
            binary_labels[:, class_index],
            probabilities[:, class_index],
        )

        average_precision = average_precision_score(
            binary_labels[:, class_index],
            probabilities[:, class_index],
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{class_name} "
                f"(AP = {average_precision:.3f})"
            ),
        )

        pr_records.append(
            {
                "Class": class_name,
                "Average Precision": average_precision,
            }
        )

    # Micro-average Precision-Recall curve
    micro_precision, micro_recall, _ = (
        precision_recall_curve(
            binary_labels.ravel(),
            probabilities.ravel(),
        )
    )

    micro_average_precision = average_precision_score(
        binary_labels,
        probabilities,
        average="micro",
    )

    plt.plot(
        micro_recall,
        micro_precision,
        linestyle="--",
        linewidth=2,
        label=(
            "Micro-average "
            f"(AP = {micro_average_precision:.3f})"
        ),
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Multiclass Precision-Recall Curve "
        "of Hereditary ResNet18"
    )
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "hereditary_precision_recall_curve.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    pr_records.append(
        {
            "Class": "Micro-average",
            "Average Precision": micro_average_precision,
        }
    )

    pd.DataFrame(pr_records).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "hereditary_average_precision_values.csv",
        ),
        index=False,
    )

    print(f"Saved: {output_path}")
    print(
        "Micro-average Average Precision: "
        f"{micro_average_precision:.4f}"
    )


def save_split_summary(train_result, test_result):
    summary = pd.DataFrame(
        [
            {
                "Split": "Training",
                "Samples": len(train_result["labels"]),
                "Accuracy": round(
                    train_result["accuracy"],
                    2,
                ),
            },
            {
                "Split": "Test",
                "Samples": len(test_result["labels"]),
                "Accuracy": round(
                    test_result["accuracy"],
                    2,
                ),
            },
        ]
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "hereditary_train_test_summary.csv",
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("=" * 65)
    print("OFFICIAL HEREDITARY RESNET18 GRAPH GENERATION")
    print("=" * 65)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("This script does not train or overwrite the model.")
    print()

    model = load_frozen_model()

    print("Evaluating training split...")
    train_result = evaluate_split(
        model=model,
        csv_file="data/train.csv",
        split_name="Training",
    )

    print()
    print("Evaluating independent test split...")
    test_result = evaluate_split(
        model=model,
        csv_file="data/test.csv",
        split_name="Test",
    )

    print()

    # Safety check for the official test result
    if abs(test_result["accuracy"] - 90.70) > 0.05:
        print(
            "WARNING: Test accuracy does not reproduce "
            "the expected 90.70%."
        )
        print(
            "Do not place the generated figures in the paper "
            "until the difference is investigated."
        )
        return

    print(
        "Verified: the frozen checkpoint reproduces "
        "the official 90.70% test accuracy."
    )
    print()

    create_train_test_graph(
        train_result,
        test_result,
    )

    create_multiclass_roc_curve(
        test_result,
    )

    create_precision_recall_curve(
        test_result,
    )

    save_split_summary(
        train_result,
        test_result,
    )

    print()
    print("=" * 65)
    print("ALL OFFICIAL GRAPHS GENERATED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()