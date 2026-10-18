#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os, sys, pickle, time

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.neural_network   import MLPClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm              import SVC
from sklearn.metrics          import (classification_report,
                                       confusion_matrix,
                                       accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.getcwd())
from utils.gesture_config import GESTURE_LABELS, SEQUENCE_LENGTH

DATA_FILE   = "data/gesture_data.npz"
MODEL_DIR   = "models"
RESULT_DIR  = "results"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path):
    d = np.load(path)
    X, y = d["X"], d["y"]
    print(f"  Loaded: X={X.shape}  y={y.shape}")
    return X, y


def flatten_sequence(X):
    """Mean-pool + std-pool over time → (N, 2*feat_dim)."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)


def temporal_flatten(X):
    """Full flatten (N, SEQ_LEN*feat_dim) — keeps temporal order."""
    return X.reshape(len(X), -1)


def save_model(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓ Saved → {path}")


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#1a1a2e")

    for ax, data, fmt, cmap, ttl in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Blues", "Greens"],
        ["Count", "Percentage (%)"]
    ):
        ax.set_facecolor("#16213e")
        sns.heatmap(
            data, ax=ax,
            annot=True, fmt=fmt, cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="#0f3460",
            annot_kws={"size": 9, "color": "white"},
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"{title} — {ttl}", color="white", fontsize=13, pad=10)
        ax.set_xlabel("Predicted", color="#a0aec0", fontsize=10)
        ax.set_ylabel("True", color="#a0aec0", fontsize=10)
        ax.tick_params(colors="#a0aec0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#0f3460")

    plt.tight_layout(pad=2)
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Confusion matrix → {path}")


def plot_accuracy_comparison(results: dict, path: str):
    models  = list(results.keys())
    accs    = [results[m]["test_acc"] * 100 for m in models]
    cv_accs = [results[m]["cv_mean"] * 100  for m in models]
    cv_stds = [results[m]["cv_std"]  * 100  for m in models]

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars1 = ax.bar(x - w/2, accs,    w, label="Test Accuracy",      color="#4ecdc4", alpha=0.9)
    bars2 = ax.bar(x + w/2, cv_accs, w, label="5-Fold CV Accuracy", color="#f7b731",
                   alpha=0.9, yerr=cv_stds, capsize=5,
                   error_kw={"ecolor": "white", "linewidth": 1.5})

    ax.set_ylim(50, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(models, color="white", fontsize=10)
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.set_title("Model Accuracy Comparison\n(inspired by Table V in Gao et al., 2021)",
                 color="white", fontsize=13)
    ax.legend(facecolor="#0f3460", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#4ecdc4")
    ax.spines["left"].set_color("#4ecdc4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#0f3460", linewidth=0.8)
    ax.set_axisbelow(True)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center",
                color="white", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{bar.get_height():.1f}%", ha="center",
                color="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Accuracy comparison → {path}")


def plot_training_history(train_losses, val_losses, path):
    """Plot MLP loss curve (approximated via score history)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#4ecdc4", linewidth=2, label="Training Loss")
    if val_losses:
        ax.plot(epochs, val_losses, color="#f7b731", linewidth=2, label="Validation Loss")
    ax.set_xlabel("Epoch / Iteration", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.set_title("MLP Classifier — Training Loss Curve", color="white", fontsize=13)
    ax.legend(facecolor="#0f3460", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#4ecdc4")
    ax.spines["left"].set_color("#4ecdc4")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#0f3460", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Loss curve → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════")
    print("   GESTURE RECOGNITION — MODEL TRAINING")
    print("══════════════════════════════════════════\n")

    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        print("  → Run collect_data.py first.")
        return

    # Load
    print("[1/6] Loading data...")
    X_seq, y = load_data(DATA_FILE)
    label_names = [GESTURE_LABELS[i] for i in sorted(GESTURE_LABELS)]

    # Feature engineering
    print("\n[2/6] Feature engineering...")
    X_flat  = flatten_sequence(X_seq)     # mean+std pool  → for RF / SVM
    X_full  = temporal_flatten(X_seq)     # full flatten   → for MLP

    scaler_flat = StandardScaler().fit(X_flat)
    scaler_full = StandardScaler().fit(X_full)
    Xf = scaler_flat.transform(X_flat)
    Xm = scaler_full.transform(X_full)

    # Train/test split
    Xf_tr, Xf_te, y_tr, y_te = train_test_split(
        Xf, y, test_size=0.2, random_state=42, stratify=y)
    Xm_tr, Xm_te, _, _       = train_test_split(
        Xm, y, test_size=0.2, random_state=42, stratify=y)

    print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")

    results = {}

    # ── Model A: Random Forest ────────────────────────────────────────────────
    print("\n[3/6] Training Random Forest...")
    t0  = time.time()
    rf  = RandomForestClassifier(n_estimators=200, max_depth=None,
                                  random_state=42, n_jobs=-1)
    rf.fit(Xf_tr, y_tr)
    rf_acc = accuracy_score(y_te, rf.predict(Xf_te))
    rf_cv  = cross_val_score(rf, Xf, y, cv=5, scoring="accuracy")
    print(f"  Test Acc : {rf_acc*100:.1f}%")
    print(f"  CV  Acc  : {rf_cv.mean()*100:.1f}% ± {rf_cv.std()*100:.1f}%")
    print(f"  Time     : {time.time()-t0:.1f}s")
    results["Random Forest"] = {
        "model": rf, "scaler": scaler_flat, "feat": "flat",
        "test_acc": rf_acc, "cv_mean": rf_cv.mean(), "cv_std": rf_cv.std(),
        "y_pred": rf.predict(Xf_te)
    }

    # ── Model B: SVM ──────────────────────────────────────────────────────────
    print("\n[4/6] Training SVM (RBF kernel)...")
    t0  = time.time()
    svm = SVC(C=10, kernel="rbf", gamma="scale",
              probability=True, random_state=42)
    svm.fit(Xf_tr, y_tr)
    svm_acc = accuracy_score(y_te, svm.predict(Xf_te))
    svm_cv  = cross_val_score(svm, Xf, y, cv=5, scoring="accuracy")
    print(f"  Test Acc : {svm_acc*100:.1f}%")
    print(f"  CV  Acc  : {svm_cv.mean()*100:.1f}% ± {svm_cv.std()*100:.1f}%")
    print(f"  Time     : {time.time()-t0:.1f}s")
    results["SVM (RBF)"] = {
        "model": svm, "scaler": scaler_flat, "feat": "flat",
        "test_acc": svm_acc, "cv_mean": svm_cv.mean(), "cv_std": svm_cv.std(),
        "y_pred": svm.predict(Xf_te)
    }

    # ── Model C: MLP (our "LSTM" substitute) ─────────────────────────────────
    print("\n[5/6] Training MLP Classifier (deep network)...")
    t0  = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        verbose=False
    )
    mlp.fit(Xm_tr, y_tr)
    mlp_acc = accuracy_score(y_te, mlp.predict(Xm_te))
    mlp_cv  = cross_val_score(mlp, Xm, y, cv=5, scoring="accuracy")
    print(f"  Test Acc : {mlp_acc*100:.1f}%")
    print(f"  CV  Acc  : {mlp_cv.mean()*100:.1f}% ± {mlp_cv.std()*100:.1f}%")
    print(f"  Time     : {time.time()-t0:.1f}s")
    results["MLP (Deep NN)"] = {
        "model": mlp, "scaler": scaler_full, "feat": "full",
        "test_acc": mlp_acc, "cv_mean": mlp_cv.mean(), "cv_std": mlp_cv.std(),
        "y_pred": mlp.predict(Xm_te)
    }

    # Pick best model
    best_name = max(results, key=lambda k: results[k]["test_acc"])
    print(f"\n🏆 Best model: {best_name}  "
          f"({results[best_name]['test_acc']*100:.1f}%)")

    # ── Save models ───────────────────────────────────────────────────────────
    print("\n[6/6] Saving models and results...")
    save_model({"model": results["MLP (Deep NN)"]["model"],
                "scaler": scaler_full, "feat": "full"},
               os.path.join(MODEL_DIR, "mlp_model.pkl"))
    save_model({"model": results["Random Forest"]["model"],
                "scaler": scaler_flat, "feat": "flat"},
               os.path.join(MODEL_DIR, "rf_model.pkl"))
    save_model({"model": results["SVM (RBF)"]["model"],
                "scaler": scaler_flat, "feat": "flat"},
               os.path.join(MODEL_DIR, "svm_model.pkl"))

    # Save loss curve
    if hasattr(mlp, "loss_curve_"):
        plot_training_history(
            mlp.loss_curve_,
            mlp.validation_scores_ if hasattr(mlp, "validation_scores_") else [],
            os.path.join(RESULT_DIR, "loss_curve.png")
        )

    # Plots
    plot_confusion_matrix(
        y_te, results[best_name]["y_pred"],
        label_names, best_name,
        os.path.join(RESULT_DIR, "confusion_matrix.png")
    )
    plot_accuracy_comparison(results,
        os.path.join(RESULT_DIR, "accuracy_comparison.png"))

    # Text report
    report_path = os.path.join(RESULT_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("GESTURE RECOGNITION — TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset : {DATA_FILE}\n")
        f.write(f"Samples : {len(y)}\n")
        f.write(f"Classes : {len(GESTURE_LABELS)}\n\n")
        f.write("MODEL COMPARISON\n" + "-"*60 + "\n")
        f.write(f"{'Model':<20} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>10}\n")
        f.write("-"*60 + "\n")
        for name, r in results.items():
            f.write(f"{name:<20} {r['test_acc']*100:>9.1f}% "
                    f"{r['cv_mean']*100:>9.1f}% {r['cv_std']*100:>9.1f}%\n")
        f.write("\n\nDETAILED CLASSIFICATION REPORT (Best Model)\n")
        f.write("="*60 + "\n")
        f.write(classification_report(
            y_te, results[best_name]["y_pred"],
            target_names=label_names))
    print(f"  ✓ Report → {report_path}")

    print("\n✅ Training complete!")
    print("   Run  python realtime_demo.py  to start the live demo.")


if __name__ == "__main__":
    main()


# In[ ]:



