import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURES_FILE = "features/features.csv"
RESULTS_FILE  = "results/retrieval_results.csv"
MODELS_DIR    = "models"
RANDOM_SEED   = 42

# MLP hyperparameters
HIDDEN1      = 256
HIDDEN2      = 64
DROPOUT      = 0.3
LR           = 0.001
EPOCHS       = 200
BATCH_SIZE   = 16

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── MLP Model ──────────────────────────────────────────────────────────────────
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, output_dim, dropout):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# ── Train one fold ─────────────────────────────────────────────────────────────
def train_model(X_train, y_train, input_dim, output_dim):
    model = MLPClassifier(input_dim, HIDDEN1, HIDDEN2, output_dim, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_tensor = torch.LongTensor(y_train).to(DEVICE)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    return model

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(DEVICE)
        logits   = model(X_tensor)
        pred     = torch.argmax(logits, dim=1).cpu().numpy()
    return pred

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load and merge data ────────────────────────────────────────────────────
    print("Loading features and results...")
    features_df = pd.read_csv(FEATURES_FILE)
    results_df  = pd.read_csv(RESULTS_FILE)

    df = pd.merge(features_df, results_df[["filename", "best_config"]], on="filename")
    print(f"Merged dataset: {len(df)} samples")

    # ── Prepare features and labels ────────────────────────────────────────────
    feature_cols = [c for c in features_df.columns if c not in ["filename", "domain"]]
    X = df[feature_cols].values.astype(np.float32)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["best_config"].values)
    
    input_dim  = X.shape[1]
    output_dim = len(label_encoder.classes_)
    
    print(f"Input dimensions : {input_dim}")
    print(f"Output classes   : {output_dim}")
    print(f"Class distribution:")
    for cls, count in zip(*np.unique(df["best_config"], return_counts=True)):
        print(f"  {cls}: {count}")

    # ── Leave-One-Out Cross Validation ─────────────────────────────────────────
    print(f"\nRunning Leave-One-Out Cross Validation ({len(df)} folds)...")
    loo      = LeaveOneOut()
    y_true   = []
    y_pred   = []
    
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        if fold % 10 == 0:
            print(f"  Fold {fold+1}/{len(df)}...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Train and predict
        model  = train_model(X_train, y_train, input_dim, output_dim)
        pred   = predict(model, X_test)

        y_true.append(y_test[0])
        y_pred.append(pred[0])

    # ── LOO Results ────────────────────────────────────────────────────────────
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    loo_accuracy = np.mean(y_true == y_pred)
    majority_class_accuracy = np.max(np.bincount(y)) / len(y)

    print(f"\n{'='*60}")
    print(f"LOO Cross-Validation Results")
    print(f"{'='*60}")
    print(f"MLP Accuracy          : {loo_accuracy:.4f} ({loo_accuracy*100:.2f}%)")
    print(f"Majority Class Baseline: {majority_class_accuracy:.4f} ({majority_class_accuracy*100:.2f}%)")
    print(f"Improvement over baseline: {(loo_accuracy - majority_class_accuracy)*100:.2f}%")
    
    print(f"\nPer-class Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # ── Per-domain accuracy ────────────────────────────────────────────────────
    print(f"Per-domain Accuracy:")
    df_eval = df.copy()
    df_eval["y_true"] = label_encoder.inverse_transform(y_true)
    df_eval["y_pred"] = label_encoder.inverse_transform(y_pred)
    df_eval["correct"] = df_eval["y_true"] == df_eval["y_pred"]
    
    for domain, group in df_eval.groupby("domain"):
        acc = group["correct"].mean()
        print(f"  {domain}: {acc:.4f} ({acc*100:.2f}%)")

    # ── Train final model on ALL data ──────────────────────────────────────────
    print(f"\nTraining final model on all 150 samples...")
    scaler_final  = StandardScaler()
    X_scaled      = scaler_final.fit_transform(X)
    final_model   = train_model(X_scaled, y, input_dim, output_dim)

    # ── Save model and artifacts ───────────────────────────────────────────────
    model_path   = os.path.join(MODELS_DIR, "mlp_model.pt")
    scaler_path  = os.path.join(MODELS_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    meta_path    = os.path.join(MODELS_DIR, "model_meta.json")

    torch.save(final_model.state_dict(), model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_final, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    meta = {
        "input_dim"         : input_dim,
        "output_dim"        : output_dim,
        "hidden1"           : HIDDEN1,
        "hidden2"           : HIDDEN2,
        "dropout"           : DROPOUT,
        "loo_accuracy"      : float(loo_accuracy),
        "majority_baseline" : float(majority_class_accuracy),
        "classes"           : list(label_encoder.classes_)
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training complete!")
    print(f"   Model saved to    : {model_path}")
    print(f"   Scaler saved to   : {scaler_path}")
    print(f"   Encoder saved to  : {encoder_path}")
    print(f"   Meta saved to     : {meta_path}")

if __name__ == "__main__":
    main()
