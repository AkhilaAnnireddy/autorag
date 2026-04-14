import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from itertools import combinations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
FEATURES_FILE         = "features/features.csv"
RESULTS_FILE          = "results/retrieval_results.csv"
MODELS_DIR            = "models"
DIAGNOSTICS_DIR       = "diagnostics"
RANDOM_SEED           = 42
MIN_SAMPLES_PER_CLASS = 5   # classes below this are merged into "other"

# MLP hyperparameters
HIDDEN1    = 32
HIDDEN2    = 16
DROPOUT    = 0.3
LR         = 0.001
EPOCHS     = 200
BATCH_SIZE = 8

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def merge_rare_classes(y_series, min_samples=MIN_SAMPLES_PER_CLASS):
    """Merge classes with fewer than min_samples into 'other'."""
    counts = y_series.value_counts()
    rare   = counts[counts < min_samples].index.tolist()
    if rare:
        print(f"  Merging rare classes into 'other': {rare}")
        y_series = y_series.copy()
        y_series[y_series.isin(rare)] = "other"
    else:
        print("  No rare classes to merge.")
    return y_series


def engineer_features(df, feature_cols):
    """
    Add domain-aware engineered features.
    Extend this function once diagnostics reveal which base features matter most.
    """
    X = df[feature_cols].copy()

    # 1. Ratio features (avoid div-by-zero with small epsilon)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    eps = 1e-8

    # Pairwise ratios for all numeric columns (captures relative magnitudes)
    ratio_names = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            c1, c2 = num_cols[i], num_cols[j]
            col_name = f"ratio_{c1}__{c2}"
            X[col_name] = X[c1] / (X[c2] + eps)
            ratio_names.append(col_name)

    # 2. Polynomial features for top-variance columns
    variances  = X[num_cols].var().sort_values(ascending=False)
    top_cols   = variances.head(min(5, len(num_cols))).index.tolist()
    for col in top_cols:
        X[f"sq_{col}"]   = X[col] ** 2
        X[f"sqrt_{col}"] = np.sqrt(np.abs(X[col]))

    # 3. Z-score outlier flag per feature (samples far from mean)
    for col in num_cols:
        mu, sd = X[col].mean(), X[col].std() + eps
        X[f"outlier_{col}"] = ((X[col] - mu).abs() > 2 * sd).astype(int)

    # 4. Replace any inf/nan introduced by ratios
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)

    new_cols = [c for c in X.columns if c not in feature_cols]
    print(f"  Engineered {len(new_cols)} new features  "
          f"(total: {len(X.columns)})")
    return X.values.astype(np.float32), list(X.columns)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(X, y, le, feature_cols, output_dir=DIAGNOSTICS_DIR):
    """
    Produce six diagnostic artefacts that reveal why accuracy is low
    and which features/class-pairs are the bottleneck.
    """
    os.makedirs(output_dir, exist_ok=True)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    classes  = le.classes_

    # ── 1. Feature-class heatmap ──────────────────────────────────────────────
    df_feat = pd.DataFrame(X_scaled, columns=feature_cols)
    df_feat["class"] = le.inverse_transform(y)
    class_means = df_feat.groupby("class")[feature_cols].mean()

    plt.figure(figsize=(max(14, len(feature_cols) // 2), 6))
    sns.heatmap(class_means.T, cmap="RdBu_r", center=0,
                xticklabels=True, yticklabels=True, annot=False)
    plt.title("Mean feature value per class (z-scored)\n"
              "Columns with high variance across rows = discriminative features")
    plt.tight_layout()
    p = f"{output_dir}/1_feature_class_heatmap.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")

    # ── 2. PCA scatter ────────────────────────────────────────────────────────
    pca    = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca  = pca.fit_transform(X_scaled)
    plt.figure(figsize=(9, 7))
    for i, cls in enumerate(classes):
        mask = y == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls,
                    alpha=0.75, s=65, edgecolors="white", linewidths=0.4)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("PCA — 2D projection of feature space\n"
              "Overlapping clusters → features lack discriminative power")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    p = f"{output_dir}/2_pca_scatter.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")

    # ── 3. LDA scatter ────────────────────────────────────────────────────────
    n_comp = min(len(classes) - 1, X_scaled.shape[1], 2)
    lda    = LinearDiscriminantAnalysis(n_components=n_comp)
    X_lda  = lda.fit_transform(X_scaled, y)
    plt.figure(figsize=(9, 7))
    for i, cls in enumerate(classes):
        mask = y == i
        xv   = X_lda[mask, 0]
        yv   = X_lda[mask, 1] if n_comp >= 2 else np.zeros(mask.sum()) + i * 0.1
        plt.scatter(xv, yv, label=cls, alpha=0.75, s=65,
                    edgecolors="white", linewidths=0.4)
    plt.title("LDA — maximum linear class separation\n"
              "If still overlapping, classes are indistinguishable with these features")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    p = f"{output_dir}/3_lda_scatter.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")

    # ── 4. ANOVA F-score per feature ──────────────────────────────────────────
    f_scores, p_values = f_classif(X_scaled, y)
    feat_df = pd.DataFrame({
        "feature": feature_cols,
        "f_score": f_scores,
        "p_value": p_values
    }).sort_values("f_score", ascending=False)

    top15 = feat_df.head(15)
    plt.figure(figsize=(10, 5))
    plt.barh(top15["feature"][::-1], top15["f_score"][::-1])
    plt.xlabel("ANOVA F-score")
    plt.title("Top 15 most discriminative features")
    plt.tight_layout()
    p = f"{output_dir}/4_feature_importance.png"
    plt.savefig(p, dpi=150); plt.close()
    feat_df.to_csv(f"{output_dir}/4_feature_importance.csv", index=False)

    sig_count = (p_values < 0.05).sum()
    print(f"\n  Top 10 discriminative features (ANOVA F-score):")
    print(feat_df.head(10).to_string(index=False))
    print(f"\n  {sig_count}/{len(feature_cols)} features significant at p<0.05")
    print(f"  Saved: {p}")

    # ── 5. OOB confusion matrix ───────────────────────────────────────────────
    rf_oob = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     oob_score=True, random_state=RANDOM_SEED)
    rf_oob.fit(X_scaled, y)
    oob_pred = np.argmax(rf_oob.oob_decision_function_, axis=1)
    cm = confusion_matrix(y, oob_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"RF OOB Confusion Matrix  (OOB acc: {rf_oob.oob_score_:.3f})\n"
              "Off-diagonal = which configs get confused with each other")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    p = f"{output_dir}/5_oob_confusion_matrix.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"\n  RF OOB accuracy : {rf_oob.oob_score_:.4f}")
    print(f"  Saved: {p}")

    # ── 6. Inter-class centroid distance ─────────────────────────────────────
    centroids = {cls: X_scaled[y == i].mean(axis=0)
                 for i, cls in enumerate(classes)}
    dists = [(c1, c2, float(np.linalg.norm(v1 - v2)))
             for (c1, v1), (c2, v2) in combinations(centroids.items(), 2)]
    dists.sort(key=lambda x: x[2])

    dist_df = pd.DataFrame(dists, columns=["class_a", "class_b", "l2_distance"])
    print("\n  10 closest class pairs (hardest to separate):")
    print(dist_df.head(10).to_string(index=False))
    dist_df.to_csv(f"{output_dir}/6_centroid_distances.csv", index=False)

    # Distance heatmap
    n = len(classes)
    dist_matrix = np.zeros((n, n))
    for c1, c2, d in dists:
        i1 = list(classes).index(c1)
        i2 = list(classes).index(c2)
        dist_matrix[i1, i2] = d
        dist_matrix[i2, i1] = d
    plt.figure(figsize=(9, 7))
    sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="YlOrRd_r",
                xticklabels=classes, yticklabels=classes)
    plt.title("Centroid L2 distance between classes\n"
              "Low = hard to separate, High = easy to separate")
    plt.tight_layout()
    p = f"{output_dir}/6_centroid_distance_heatmap.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")

    return feat_df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MLP model
# ══════════════════════════════════════════════════════════════════════════════

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, h1, h2, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, input_dim, output_dim):
    # Compute balanced class weights
    unique_classes = np.unique(y_train)
    raw_weights    = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    weight_vec     = np.ones(output_dim)
    for i, c in enumerate(unique_classes):
        weight_vec[c] = raw_weights[i]
    class_weights = torch.FloatTensor(weight_vec).to(DEVICE)

    model     = MLPClassifier(input_dim, HIDDEN1, HIDDEN2, output_dim, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    X_t    = torch.FloatTensor(X_train).to(DEVICE)
    y_t    = torch.LongTensor(y_train).to(DEVICE)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def predict_mlp(model, X_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(DEVICE))
        return torch.argmax(logits, dim=1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOO evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_loo_sklearn(X, y, make_model_fn, label="Model"):
    """LOO for any sklearn-compatible estimator."""
    loo            = LeaveOneOut()
    y_true, y_pred = [], []
    for fold, (tr, te) in enumerate(loo.split(X)):
        if fold % 25 == 0:
            print(f"    [{label}] fold {fold+1}/{len(X)} ...")
        scaler  = StandardScaler()
        X_tr    = scaler.fit_transform(X[tr])
        X_te    = scaler.transform(X[te])
        model   = make_model_fn()
        model.fit(X_tr, y[tr])
        y_pred.append(model.predict(X_te)[0])
        y_true.append(y[te][0])
    return np.array(y_true), np.array(y_pred)


def run_loo_mlp(X, y, input_dim, output_dim):
    """LOO for the PyTorch MLP."""
    loo            = LeaveOneOut()
    y_true, y_pred = [], []
    for fold, (tr, te) in enumerate(loo.split(X)):
        if fold % 25 == 0:
            print(f"    [MLP] fold {fold+1}/{len(X)} ...")
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr])
        X_te   = scaler.transform(X[te])
        model  = train_mlp(X_tr, y[tr], input_dim, output_dim)
        y_pred.append(predict_mlp(model, X_te)[0])
        y_true.append(y[te][0])
    return np.array(y_true), np.array(y_pred)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Model factory
# ══════════════════════════════════════════════════════════════════════════════

def get_models(output_dim):
    """
    Returns a dict of {name: make_fn} for all models to evaluate.
    make_fn() returns a fresh unfitted estimator.
    """
    return {
        "RF_balanced": lambda: RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=RANDOM_SEED
        ),
        "RF_with_select": lambda: Pipeline([
            ("select", SelectKBest(f_classif, k=min(20, 10))),
            ("clf",    RandomForestClassifier(
                n_estimators=500, class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=RANDOM_SEED
            ))
        ]),
        "GBM": lambda: GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=RANDOM_SEED
        ),
        "SVM_rbf": lambda: SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "SVM_poly": lambda: SVC(
            kernel="poly", degree=3, C=5,
            class_weight="balanced",
            random_state=RANDOM_SEED
        ),
        "LogReg": lambda: LogisticRegression(
            C=0.1, class_weight="balanced",
            max_iter=2000, solver="lbfgs",
            multi_class="multinomial",
            random_state=RANDOM_SEED
        ),
        "Ensemble": lambda: VotingClassifier(
            estimators=[
                ("rf",  RandomForestClassifier(
                    n_estimators=300, class_weight="balanced_subsample",
                    min_samples_leaf=2, random_state=RANDOM_SEED)),
                ("svm", SVC(kernel="rbf", C=10, class_weight="balanced",
                            probability=True, random_state=RANDOM_SEED)),
                ("lr",  LogisticRegression(
                    C=0.1, class_weight="balanced",
                    max_iter=2000, random_state=RANDOM_SEED)),
            ],
            voting="soft"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Results plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_comparison(results, majority_baseline, output_dir=DIAGNOSTICS_DIR):
    names  = list(results.keys())
    accs   = [np.mean(yt == yp) for yt, yp in results.values()]
    colors = ["#4CAF50" if a > majority_baseline else "#EF5350" for a in accs]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)
    plt.axhline(majority_baseline, color="black", linestyle="--",
                linewidth=1.2, label=f"Majority baseline ({majority_baseline:.2f})")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, min(1.0, max(accs) + 0.12))
    plt.ylabel("LOO Accuracy")
    plt.title("Model comparison — LOO cross-validation accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    p = f"{output_dir}/7_model_comparison.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")


def plot_per_domain(df_eval, output_dir=DIAGNOSTICS_DIR):
    domain_acc = (df_eval.groupby("domain")["correct"]
                  .mean().sort_values(ascending=True))
    plt.figure(figsize=(8, 4))
    domain_acc.plot(kind="barh", color="#42A5F5", edgecolor="white")
    plt.xlabel("Accuracy")
    plt.title("Per-domain accuracy (best model)")
    plt.tight_layout()
    p = f"{output_dir}/8_per_domain_accuracy.png"
    plt.savefig(p, dpi=150); plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading and preparing data...")
    features_df = pd.read_csv(FEATURES_FILE)
    results_df  = pd.read_csv(RESULTS_FILE)
    df = pd.merge(features_df,
                  results_df[["filename", "domain", "best_config"]],
                  on=["filename", "domain"])
    print(f"  Loaded {len(df)} samples across "
          f"{df['domain'].nunique()} domains")

    # Merge rare classes
    df["best_config"] = merge_rare_classes(df["best_config"])

    feature_cols = [c for c in features_df.columns
                    if c not in ["filename", "domain"]]

    # ── Feature engineering ────────────────────────────────────────────────────
    print("\n[2/6] Engineering features...")
    X_eng, eng_cols = engineer_features(df, feature_cols)
    X_base          = df[feature_cols].values.astype(np.float32)

    le  = LabelEncoder()
    y   = le.fit_transform(df["best_config"].values)
    n_classes = len(le.classes_)

    print(f"\n  Classes ({n_classes}):")
    for cls, cnt in zip(*np.unique(df["best_config"], return_counts=True)):
        print(f"    {cls}: {cnt}")

    majority_baseline = np.max(np.bincount(y)) / len(y)
    print(f"\n  Majority class baseline: {majority_baseline:.4f} "
          f"({majority_baseline*100:.2f}%)")

    # ── Diagnostics on BASE features ──────────────────────────────────────────
    print("\n[3/6] Running diagnostics on base features...")
    feat_importance = run_diagnostics(X_base, y, le, feature_cols)

    # Select top-K features by F-score for engineered set
    top_k   = min(30, len(feature_cols))
    top_feats = feat_importance.head(top_k)["feature"].tolist()
    print(f"\n  Using top-{top_k} features by ANOVA F-score for modelling")

    # Use only top features from base (avoids noise from poor features)
    top_idx = [feature_cols.index(f) for f in top_feats if f in feature_cols]
    X_top   = X_base[:, top_idx]

    # ── LOO evaluation ─────────────────────────────────────────────────────────
    print("\n[4/6] Running LOO cross-validation on all models...")
    all_results = {}
    models_to_run = get_models(n_classes)

    for name, make_fn in models_to_run.items():
        print(f"\n  >> {name}")
        yt, yp = run_loo_sklearn(X_top, y, make_fn, label=name)
        all_results[name] = (yt, yp)
        acc = np.mean(yt == yp)
        print(f"     accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # MLP on base features
    print(f"\n  >> MLP")
    yt_mlp, yp_mlp = run_loo_mlp(X_top, y, len(top_idx), n_classes)
    all_results["MLP"] = (yt_mlp, yp_mlp)
    mlp_acc = np.mean(yt_mlp == yp_mlp)
    print(f"     accuracy: {mlp_acc:.4f} ({mlp_acc*100:.2f}%)")

    # ── Print comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Model':<22}  {'LOO Acc':>8}  {'vs Baseline':>12}  {'Best?':>6}")
    print(f"{'='*65}")
    print(f"{'Majority baseline':<22}  {majority_baseline:>8.4f}")
    print(f"{'-'*65}")

    best_name, best_acc = None, 0.0
    for name, (yt, yp) in all_results.items():
        acc  = np.mean(yt == yp)
        diff = acc - majority_baseline
        flag = " <-- BEST" if acc > best_acc else ""
        if acc > best_acc:
            best_acc, best_name = acc, name
        print(f"{name:<22}  {acc:>8.4f}  {diff:>+12.4f}{flag}")
    print(f"{'='*65}")

    # ── Detailed report for best model ────────────────────────────────────────
    print(f"\n[5/6] Detailed results for best model: {best_name}")
    best_yt, best_yp = all_results[best_name]
    print(classification_report(best_yt, best_yp,
                                 target_names=le.classes_,
                                 zero_division=0))

    df_eval = df.copy()
    df_eval["y_true"]  = le.inverse_transform(best_yt)
    df_eval["y_pred"]  = le.inverse_transform(best_yp)
    df_eval["correct"] = df_eval["y_true"] == df_eval["y_pred"]

    print("Per-domain accuracy:")
    for domain, grp in df_eval.groupby("domain"):
        acc = grp["correct"].mean()
        print(f"  {domain:<30}: {acc:.4f} ({acc*100:.2f}%)")

    # ── Save comparison plots ──────────────────────────────────────────────────
    plot_model_comparison(all_results, majority_baseline)
    plot_per_domain(df_eval)

    # ── Train final model on ALL data ─────────────────────────────────────────
    print(f"\n[6/6] Training final {best_name} on all {len(df)} samples...")
    scaler_final = StandardScaler()
    X_final      = scaler_final.fit_transform(X_top)

    if best_name == "MLP":
        final_model = train_mlp(X_final, y, len(top_idx), n_classes)
        torch.save(final_model.state_dict(),
                   os.path.join(MODELS_DIR, "best_model_mlp.pt"))
        final_sklearn = None
    else:
        make_fn = models_to_run[best_name]
        final_model = make_fn()
        final_model.fit(X_final, y)
        final_sklearn = final_model

    # ── Save artefacts ────────────────────────────────────────────────────────
    if final_sklearn is not None:
        with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
            pickle.dump(final_sklearn, f)

    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler_final, f)

    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    with open(os.path.join(MODELS_DIR, "top_feature_indices.pkl"), "wb") as f:
        pickle.dump(top_idx, f)

    meta = {
        "best_model"        : best_name,
        "loo_accuracy"      : float(best_acc),
        "majority_baseline" : float(majority_baseline),
        "improvement"       : float(best_acc - majority_baseline),
        "classes"           : list(le.classes_),
        "n_classes"         : n_classes,
        "n_samples"         : len(df),
        "n_features_base"   : len(feature_cols),
        "n_features_used"   : len(top_idx),
        "top_features"      : top_feats,
        "all_model_accuracies": {
            name: float(np.mean(yt == yp))
            for name, (yt, yp) in all_results.items()
        }
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Best model       : {best_name}")
    print(f"  LOO accuracy     : {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"  Majority baseline: {majority_baseline:.4f} ({majority_baseline*100:.2f}%)")
    print(f"  Improvement      : {(best_acc-majority_baseline)*100:+.2f}%")
    print(f"\n  Saved artefacts:")
    print(f"    models/best_model.pkl")
    print(f"    models/scaler.pkl")
    print(f"    models/label_encoder.pkl")
    print(f"    models/top_feature_indices.pkl")
    print(f"    models/model_meta.json")
    print(f"    diagnostics/  (8 plots + 2 CSVs)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()