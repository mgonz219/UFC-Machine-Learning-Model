import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score



def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    feature_cols = [
        "RedOdds", "BlueOdds",
        "RedAge", "BlueAge",
        "RedHeightCms", "BlueHeightCms",
        "RedReachCms", "BlueReachCms",
        #"RedWins", "BlueWins",
        #"RedLosses", "BlueLosses",
        "RedAvgSigStrLanded", "BlueAvgSigStrLanded",
        "RedAvgTDLanded", "BlueAvgTDLanded",
        "RedAvgSubAtt", "BlueAvgSubAtt",
        "LoseStreakDif","WinStreakDif",
        "AgeDif",
        "BMatchWCRank","RMatchWCRank",
    ]
  
    winner = df["Winner"].astype(str).str.strip().str.lower()
    y = (winner == "red").astype(int)  # 1 = Red wins, 0 = Blue

   
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Helpful debug: show anything that contains "sub"
        sub_cols = [c for c in df.columns if "sub" in c.lower()]
        raise KeyError(f"Missing feature columns: {missing}\nSub-related cols found: {sub_cols}")
    
    X = df[feature_cols].copy()

    # Clean numeric values
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop rows with any missing feature or missing target
    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("y value counts:", y.value_counts().to_dict())
    print("Any NaNs in X?", X.isna().any().any())
    print("Feature columns used:", list(X.columns))
    return X, y


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))


def main():
    df = load_data("data/raw/ufc-master.csv")

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for C in [0.01, 0.1, 1, 10]:
        model = LogisticRegression(max_iter=3000, C=C)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"C={C} Accuracy:", accuracy_score(y_test, preds))
        evaluate_model(model, X_test, y_test)

    pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=5000))
])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print("CV accuracy mean/std:", scores.mean(), scores.std())

    return model, scaler, X.columns


if __name__ == "__main__":
    main()
