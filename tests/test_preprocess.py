# tests/test_preprocess.py
import pandas as pd

def test_processed_data():
    """
    Sanity check for processed CSV files:
    - No NaNs in text or label
    - Text column contains only strings
    """
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    for df in [train_df, val_df, test_df]:
        # Check for NaNs
        assert df["text"].notna().all(), "Found NaN in text column"
        assert df["label"].notna().all(), "Found NaN in label column"
        # Check all text entries are strings
        assert all(isinstance(t, str) for t in df["text"]), "Non-string entry in text column"

if __name__ == "__main__":
    test_processed_data()
    print("All preprocessing sanity checks passed!")