# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def main():
    test_df = pd.read_csv("data/processed/test.csv")

    model = pd.read_pickle("results/model.pkl")
    vectorizer = pd.read_pickle("results/vectorizer.pkl")

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    with open("results/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")

    with open("results/confusion_matrix.txt", "w") as f:
        f.write(str(cm))

    print("Test Accuracy:", acc)
    print("Test F1-score:", f1)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()