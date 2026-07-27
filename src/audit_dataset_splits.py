import pandas as pd

files = {
    "Train": "data/train.csv",
    "Val": "data/val.csv",
    "Test": "data/test.csv"
}

for name, path in files.items():
    df = pd.read_csv(path)
    print("\n" + "=" * 50)
    print(name, len(df))
    print("=" * 50)
    print(df["label"].value_counts().sort_index())

    pairs = set(df["pp"].astype(str))
    print("Unique pair IDs:", len(pairs))

train = pd.read_csv("data/train.csv")
val = pd.read_csv("data/val.csv")
test = pd.read_csv("data/test.csv")

train_pairs = set(train["pp"].astype(str))
val_pairs = set(val["pp"].astype(str))
test_pairs = set(test["pp"].astype(str))

print("\nLeakage Check")
print("Train-Val overlap:", len(train_pairs & val_pairs))
print("Train-Test overlap:", len(train_pairs & test_pairs))
print("Val-Test overlap:", len(val_pairs & test_pairs))