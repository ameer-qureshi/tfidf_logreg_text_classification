# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re

RANDOM_STATE = 42

def clean_text(text: str) -> str:
    """
    Simple, defensible cleaning:
    - lowercase
    - remove non-alphanumeric characters
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def main():
    # Load dataset
    df = pd.read_csv(
        "data/raw/SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "text"]
    )

    # Encode labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean text
    df["text"] = df["text"].apply(clean_text)
    # Remove any rows where text is empty after cleaning
    df = df[df["text"].str.strip() != ""].dropna(subset=["text"])

    # Stratified split: train (70%), val (15%), test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.15,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1765,  # 0.15 / 0.85
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    # Save splits
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(
        "data/processed/train.csv", index=False
    )
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv(
        "data/processed/val.csv", index=False
    )
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(
        "data/processed/test.csv", index=False
    )

    print("Preprocessing complete.")
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

if __name__ == "__main__":
    main()