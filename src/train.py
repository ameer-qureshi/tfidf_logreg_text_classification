# # src/train.py
# import pandas as pd
# import numpy as np
# import logging
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score

# RANDOM_STATE = 42
# C_VALUE = 1.0  # tunable hyperparameter

# logging.basicConfig(
#     filename="logs/train.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(message)s"
# )

# def main():
#     # Load data
#     train_df = pd.read_csv("data/processed/train.csv")
#     val_df = pd.read_csv("data/processed/val.csv")

#     # TF-IDF
#     vectorizer = TfidfVectorizer(
#         max_features=5000,
#         ngram_range=(1, 1)
#     )

#     X_train = vectorizer.fit_transform(train_df["text"])
#     X_val = vectorizer.transform(val_df["text"])

#     y_train = train_df["label"]
#     y_val = val_df["label"]

#     # Logistic Regression
#     model = LogisticRegression(
#         C=C_VALUE,
#         penalty="l2",
#         solver="liblinear",
#         random_state=RANDOM_STATE,
#         max_iter=1000
#     )

#     model.fit(X_train, y_train)

#     # Validation evaluation
#     val_preds = model.predict(X_val)
#     val_f1 = f1_score(y_val, val_preds)

#     logging.info(f"C={C_VALUE}, Validation F1={val_f1:.4f}")
#     print(f"Validation F1-score: {val_f1:.4f}")

#     # Save artifacts
#     pd.to_pickle(model, "results/model.pkl")
#     pd.to_pickle(vectorizer, "results/vectorizer.pkl")

# if __name__ == "__main__":
#     main()



# src/train.py
import pandas as pd
import os
import joblib
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RANDOM_STATE = 42
C_VALUE = 1.0  # tunable hyperparameter

logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def main():
    # Load data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # TF-IDF with bigrams
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)  # now unigrams + bigrams
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])

    y_train = train_df["label"]
    y_val = val_df["label"]

    # Logistic Regression with balanced class weights
    model = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    # Validation evaluation
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds)

    logging.info(f"C={C_VALUE}, Validation F1={val_f1:.4f}, ngram=(1,2), class_weight=balanced")
    print(f"Validation F1-score: {val_f1:.4f}")

    # Save artifacts for deployment
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/logreg.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
if __name__ == "__main__":
    main()