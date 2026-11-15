"""Logistic Regression from Scratch with Gradient Descent on Palmer Penguins Dataset.
Zero in-built ML library functions: implements Sigmoid, Softmax, Binary Cross-Entropy,
Categorical Cross-Entropy, analytical gradients, simultaneous parameter updates, and
cost convergence tracking strictly using elementary Python and NumPy.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Data Ingestion & Preprocessing (Palmer Penguins Dataset)
# ---------------------------------------------------------------------------
def load_penguins_classification_data():
    """Load Palmer Penguins dataset and extract morphological features for classification.
    
    Returns:
        df: Cleaned dataframe (N=333)
        features: numpy array (N, 2) [bill_length_mm, flipper_length_mm]
        labels_multi: numpy array (N,) with integers 0 (Adelie), 1 (Chinstrap), 2 (Gentoo)
        species_names: list of species strings
        binary_mask: boolean mask for binary task (Adelie vs Gentoo, N=274)
        labels_binary: numpy array (274,) with 0 (Adelie) and 1 (Gentoo)
    """
    csv_path = Path("data/palmer_penguins.csv")
    if not csv_path.exists():
        import seaborn as sns
        df = sns.load_dataset("penguins").dropna()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path).dropna()

    species_list = ["Adelie", "Chinstrap", "Gentoo"]
    species_to_idx = {name: i for i, name in enumerate(species_list)}

    # Filter to known species
    df = df[df["species"].isin(species_list)].copy()
    
    # 2D features for visual decision boundaries: bill length (mm) & flipper length (mm)
    x1 = df["bill_length_mm"].values.astype(float)
    x2 = df["flipper_length_mm"].values.astype(float)
    X = np.column_stack([x1, x2])
    
    species = df["species"].values
    y_multi = np.array([species_to_idx[s] for s in species], dtype=int)

    # Binary subset: Adelie (0) vs Gentoo (1), excluding Chinstrap
    bin_mask = (species == "Adelie") | (species == "Gentoo")
    X_bin = X[bin_mask]
    species_bin = species[bin_mask]
    y_bin = np.array([1 if s == "Gentoo" else 0 for s in species_bin], dtype=int)

    return df, X, y_multi, species_list, X_bin, y_bin


# ---------------------------------------------------------------------------
# 2. Binary Logistic Regression (From Scratch - No ML Libraries)
# ---------------------------------------------------------------------------
def sigmoid(z):
    """Numerically stable Sigmoid activation function: sigma(z) = 1 / (1 + exp(-z))."""
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def compute_binary_cost(y_true, y_prob):
    """Binary Cross-Entropy (Log Loss):
       J(w, b) = -1/m * sum( y*log(p) + (1-y)*log(1-p) )
    """
    eps = 1e-15
    y_p = np.clip(y_prob, eps, 1.0 - eps)
    cost = -np.mean(y_true * np.log(y_p) + (1.0 - y_true) * np.log(1.0 - y_p))
    return float(cost)


def fit_binary_logistic_regression(X_raw, y_raw, alpha=0.1, iterations=600):
    """Batch Gradient Descent for Binary Logistic Regression from first principles.
    
    Hypothesis:
      z = w1*x1 + w2*x2 + b
      p = sigma(z)
      
    Gradients:
      dJ/dw = 1/m * X^T (p - y)
      dJ/db = 1/m * sum(p - y)
      
    Updates:
      w := w - alpha * dJ/dw
      b := b - alpha * dJ/db
    """
    m, d = X_raw.shape
    
    # Feature standardization: z = (x - mu) / sigma
    mu = np.mean(X_raw, axis=0)
    sigma = np.std(X_raw, axis=0)
    X_norm = (X_raw - mu) / sigma

    # Parameter initialization
    w = np.zeros(d, dtype=float)
    b = 0.0
    cost_history = []

    for _ in range(iterations):
        # Forward pass
        z = np.dot(X_norm, w) + b
        p = sigmoid(z)
        
        # Cost calculation
        cost = compute_binary_cost(y_raw, p)
        cost_history.append(cost)

        # Gradient computation
        error = p - y_raw
        dw = (1.0 / m) * np.dot(X_norm.T, error)
        db = (1.0 / m) * np.sum(error)

        # Simultaneous gradient descent updates
        w -= alpha * dw
        b -= alpha * db

    # Final forward pass & evaluation metrics
    z_final = np.dot(X_norm, w) + b
    p_final = sigmoid(z_final)
    preds = (p_final >= 0.5).astype(int)
    accuracy = float(np.mean(preds == y_raw))

    # Analytical unnormalization of decision hyperplane to raw feature scale:
    # w_raw1 * x1 + w_raw2 * x2 + b_raw = 0
    w_raw = w / sigma
    b_raw = b - np.sum((w * mu) / sigma)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn = int(np.sum((preds == 0) & (y_raw == 0)))
    fp = int(np.sum((preds == 1) & (y_raw == 0)))
    fn = int(np.sum((preds == 0) & (y_raw == 1)))
    tp = int(np.sum((preds == 1) & (y_raw == 1)))

    return {
        "weights_norm": w.tolist(),
        "bias_norm": float(b),
        "weights_raw": w_raw.tolist(),
        "bias_raw": float(b_raw),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "cost_history": cost_history,
        "initial_cost": float(cost_history[0]),
        "final_cost": float(cost_history[-1]),
        "accuracy": accuracy,
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "iterations": iterations,
        "alpha": alpha,
        "sample_count": m
    }


# ---------------------------------------------------------------------------
# 3. Multiclass Logistic Regression (Softmax Regression from Scratch)
# ---------------------------------------------------------------------------
def softmax(Z):
    """Numerically stable Softmax activation across rows:
       P_ik = exp(Z_ik - max(Z_i)) / sum_j exp(Z_ij - max(Z_i))
    """
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def compute_multiclass_cost(Y_onehot, P_prob):
    """Categorical Cross-Entropy Loss:
       J(W, b) = -1/m * sum_i sum_k Y_ik * log(P_ik)
    """
    eps = 1e-15
    P_clipped = np.clip(P_prob, eps, 1.0 - eps)
    cost = -np.mean(np.sum(Y_onehot * np.log(P_clipped), axis=1))
    return float(cost)


def fit_multiclass_logistic_regression(X_raw, y_raw, K=3, alpha=0.1, iterations=600):
    """Batch Gradient Descent for Multiclass Softmax Regression from first principles.
    
    Hypothesis:
      Z = X W + b  (shape: m x K)
      P = softmax(Z)
      
    Gradients:
      dJ/dW = 1/m * X^T (P - Y)
      dJ/db = 1/m * sum_i (P_i - Y_i)
      
    Updates:
      W := W - alpha * dJ/dW
      b := b - alpha * dJ/db
    """
    m, d = X_raw.shape

    # One-hot encoding of target labels
    Y_onehot = np.zeros((m, K), dtype=float)
    for i in range(m):
        Y_onehot[i, y_raw[i]] = 1.0

    # Feature standardization
    mu = np.mean(X_raw, axis=0)
    sigma = np.std(X_raw, axis=0)
    X_norm = (X_raw - mu) / sigma

    # Parameter initialization
    W = np.zeros((d, K), dtype=float)
    b = np.zeros(K, dtype=float)
    cost_history = []

    for _ in range(iterations):
        # Forward pass
        Z = np.dot(X_norm, W) + b
        P = softmax(Z)

        # Cross-Entropy Cost
        cost = compute_multiclass_cost(Y_onehot, P)
        cost_history.append(cost)

        # Gradient computation
        error = P - Y_onehot
        dW = (1.0 / m) * np.dot(X_norm.T, error)
        db = (1.0 / m) * np.sum(error, axis=0)

        # Simultaneous gradient descent updates
        W -= alpha * dW
        b -= alpha * db

    # Final forward pass
    Z_final = np.dot(X_norm, W) + b
    P_final = softmax(Z_final)
    preds = np.argmax(P_final, axis=1)
    accuracy = float(np.mean(preds == y_raw))

    # Multiclass Confusion Matrix (3x3)
    cm = np.zeros((K, K), dtype=int)
    for true_label, pred_label in zip(y_raw, preds):
        cm[true_label, pred_label] += 1

    return {
        "W_norm": W.tolist(),
        "b_norm": b.tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "cost_history": cost_history,
        "initial_cost": float(cost_history[0]),
        "final_cost": float(cost_history[-1]),
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "iterations": iterations,
        "alpha": alpha,
        "sample_count": m
    }


# ---------------------------------------------------------------------------
# 4. Publication Plot Generation (Exact Gradient Descent Curves & Boundaries)
# ---------------------------------------------------------------------------
def generate_logistic_regression_charts(output_path):
    """Generate minimal black-and-white 2-panel figure:
    1. Gradient descent cost curves for Binary and Multiclass models.
    2. Empirical Palmer Penguins scatter with learned decision boundaries.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, X, y_multi, species_list, X_bin, y_bin = load_penguins_classification_data()

    # Fit models
    bin_res = fit_binary_logistic_regression(X_bin, y_bin, alpha=0.1, iterations=500)
    multi_res = fit_multiclass_logistic_regression(X, y_multi, K=3, alpha=0.1, iterations=500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    fig.patch.set_facecolor("#ffffff")

    # -----------------------------------------------------------------------
    # PANEL 1: Exact Gradient Descent Cost Convergence Curves
    # -----------------------------------------------------------------------
    ax1.set_facecolor("#ffffff")
    epochs_bin = np.arange(1, len(bin_res["cost_history"]) + 1)
    epochs_multi = np.arange(1, len(multi_res["cost_history"]) + 1)

    ax1.plot(epochs_bin, bin_res["cost_history"], color="#000000", lw=2.0,
             label=f"Binary Log Loss J(w, b) [Final: {bin_res['final_cost']:.3f}, Acc: {bin_res['accuracy']*100:.1f}%]")
    ax1.plot(epochs_multi, multi_res["cost_history"], color="#555555", lw=1.8, linestyle="--",
             label=f"Softmax Cross-Entropy J(W, b) [Final: {multi_res['final_cost']:.3f}, Acc: {multi_res['accuracy']*100:.1f}%]")

    # Start points
    ax1.scatter([1], [bin_res["initial_cost"]], facecolors="none", edgecolors="#000000", s=50, linewidth=1.5, zorder=5)
    ax1.scatter([1], [multi_res["initial_cost"]], facecolors="#555555", edgecolors="#000000", s=50, linewidth=1.5, zorder=5)
    
    # End points
    ax1.scatter([epochs_bin[-1]], [bin_res["final_cost"]], color="#000000", s=50, zorder=5)
    ax1.scatter([epochs_multi[-1]], [multi_res["final_cost"]], color="#555555", s=50, zorder=5)

    ax1.set_title("A. Gradient Descent Cost Convergence Curves", fontsize=11, fontweight="bold", pad=10, color="#000000")
    ax1.set_xlabel("Iteration / Epoch", fontsize=9.5, fontweight="600", color="#000000")
    ax1.set_ylabel("Cross-Entropy Loss J", fontsize=9.5, fontweight="600", color="#000000")
    ax1.grid(True, linestyle=":", color="#cccccc", alpha=0.8)
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True, edgecolor="#000000")

    for spine in ax1.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    # -----------------------------------------------------------------------
    # PANEL 2: Learned Decision Boundaries & Empirical Data Scatter
    # -----------------------------------------------------------------------
    ax2.set_facecolor("#ffffff")

    # Distinct monochrome markers for the 3 species
    markers = {"Adelie": ("o", "none", "#000000"), "Chinstrap": ("s", "#bbbbbb", "#000000"), "Gentoo": ("^", "#333333", "#000000")}
    for idx, sp in enumerate(species_list):
        mask = (y_multi == idx)
        m, fc, ec = markers[sp]
        ax2.scatter(X[mask, 0], X[mask, 1], facecolors=fc, edgecolors=ec,
                    marker=m, alpha=0.85, s=36, linewidth=0.8,
                    label=f"{sp} (Class {idx}, N={np.sum(mask)})")

    # Binary Decision Boundary Line: w1_raw*x1 + w2_raw*x2 + b_raw = 0
    w1_raw, w2_raw = bin_res["weights_raw"]
    b_raw = bin_res["bias_raw"]
    x1_line = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 200)
    x2_line = -(w1_raw * x1_line + b_raw) / (w2_raw + 1e-12)
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    valid_idx = (x2_line >= y_min) & (x2_line <= y_max)
    ax2.plot(x1_line[valid_idx], x2_line[valid_idx], color="#000000", lw=2.2, linestyle="-",
             label="Gentoo/Adelie Boundary (P=0.5)")

    ax2.set_xlim(X[:, 0].min() - 2, X[:, 0].max() + 2)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("B. Learned Decision Boundary on Palmer Penguins", fontsize=11, fontweight="bold", pad=10, color="#000000")
    ax2.set_xlabel("Bill Length (mm)", fontsize=9.5, fontweight="600", color="#000000")
    ax2.set_ylabel("Flipper Length (mm)", fontsize=9.5, fontweight="600", color="#000000")
    ax2.grid(True, linestyle=":", color="#cccccc", alpha=0.8)
    ax2.legend(loc="lower right", fontsize=8.0, frameon=True, edgecolor="#000000")

    for spine in ax2.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor="#ffffff", edgecolor="none")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Interactive Prediction Engine & Specimen Ledger
# ---------------------------------------------------------------------------
def predict_sample(bill_length_mm, flipper_length_mm, bin_res=None, multi_res=None):
    """Predict binary and multiclass species probabilities for continuous user input."""
    df, X, y_multi, species_list, X_bin, y_bin = load_penguins_classification_data()

    if bin_res is None:
        bin_res = fit_binary_logistic_regression(X_bin, y_bin, alpha=0.1, iterations=500)
    if multi_res is None:
        multi_res = fit_multiclass_logistic_regression(X, y_multi, K=3, alpha=0.1, iterations=500)

    # 1. Binary prediction
    mu_b = np.array(bin_res["mu"])
    sigma_b = np.array(bin_res["sigma"])
    x_input = np.array([bill_length_mm, flipper_length_mm], dtype=float)
    x_norm_b = (x_input - mu_b) / sigma_b
    z_b = np.dot(x_norm_b, np.array(bin_res["weights_norm"])) + bin_res["bias_norm"]
    p_gentoo = float(sigmoid(z_b))
    p_adelie = 1.0 - p_gentoo
    binary_pred = "Gentoo" if p_gentoo >= 0.5 else "Adelie"

    # 2. Multiclass prediction
    mu_m = np.array(multi_res["mu"])
    sigma_m = np.array(multi_res["sigma"])
    x_norm_m = (x_input - mu_m) / sigma_m
    z_m = np.dot(x_norm_m, np.array(multi_res["W_norm"])) + np.array(multi_res["b_norm"])
    p_multi = softmax(z_m.reshape(1, -1))[0]
    multi_winner_idx = int(np.argmax(p_multi))
    multi_winner = species_list[multi_winner_idx]

    return {
        "bill_length_mm": bill_length_mm,
        "flipper_length_mm": flipper_length_mm,
        "binary": {
            "p_gentoo": p_gentoo,
            "p_adelie": p_adelie,
            "prediction": binary_pred,
            "confidence": max(p_gentoo, p_adelie),
            "z_score": float(z_b)
        },
        "multiclass": {
            "probabilities": {species_list[i]: float(p_multi[i]) for i in range(3)},
            "winner": multi_winner,
            "confidence": float(p_multi[multi_winner_idx])
        }
    }


def generate_classification_test_ledger():
    """Select representative test specimens across all classes to benchmark zero-library models."""
    df, X, y_multi, species_list, X_bin, y_bin = load_penguins_classification_data()

    bin_res = fit_binary_logistic_regression(X_bin, y_bin, alpha=0.1, iterations=500)
    multi_res = fit_multiclass_logistic_regression(X, y_multi, K=3, alpha=0.1, iterations=500)

    # Pick 2 specimens from each species
    sample_indices = [
        0, 15,       # Adelie
        160, 190,    # Chinstrap
        230, 275     # Gentoo
    ]
    # Bound to valid range
    sample_indices = [i for i in sample_indices if i < len(df)]

    ledger = []
    for idx in sample_indices:
        row = df.iloc[idx]
        bill = float(row["bill_length_mm"])
        flipper = float(row["flipper_length_mm"])
        true_sp = str(row["species"])

        preds = predict_sample(bill, flipper, bin_res, multi_res)
        ledger.append({
            "id": f"PGN-{idx:03d}",
            "species": true_sp,
            "island": str(row["island"]),
            "bill_length": bill,
            "flipper_length": flipper,
            "bin_pred": preds["binary"]["prediction"] if true_sp in ["Adelie", "Gentoo"] else "N/A (Chinstrap)",
            "bin_conf": preds["binary"]["confidence"] if true_sp in ["Adelie", "Gentoo"] else None,
            "bin_correct": (preds["binary"]["prediction"] == true_sp) if true_sp in ["Adelie", "Gentoo"] else None,
            "multi_pred": preds["multiclass"]["winner"],
            "multi_conf": preds["multiclass"]["confidence"],
            "multi_correct": (preds["multiclass"]["winner"] == true_sp),
            "probabilities": preds["multiclass"]["probabilities"]
        })

    return bin_res, multi_res, ledger


# ---------------------------------------------------------------------------
# 6. From-Scratch Python Source Code Snippet
# ---------------------------------------------------------------------------
PYTHON_LOGISTIC_REGRESSION_CODE = '''# ==============================================================================
# LOGISTIC REGRESSION FROM SCRATCH (NO IN-BUILT ML LIBRARIES)
# Batch Gradient Descent for Binary & Multiclass (Softmax) Classification
# Dataset: Palmer Penguins (Predicting Species from Bill & Flipper Measurements)
# ==============================================================================
import numpy as np

# ------------------------------------------------------------------------------
# 1. BINARY LOGISTIC REGRESSION (Sigmoid + Binary Cross-Entropy)
# ------------------------------------------------------------------------------
def sigmoid(z):
    z_clipped = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def fit_binary_logistic_regression(X, y, alpha=0.1, iterations=500):
    """
    X: shape (m, d) continuous features (e.g. bill_length, flipper_length)
    y: shape (m,) binary labels in {0, 1}
    """
    m, d = X.shape
    # Feature standardization
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    w = np.zeros(d)
    b = 0.0
    cost_history = []

    for _ in range(iterations):
        # Forward pass: hypothesis p = sigma(X * w + b)
        z = np.dot(X_norm, w) + b
        p = sigmoid(z)

        # Binary Cross-Entropy Cost J(w, b)
        eps = 1e-15
        p_clipped = np.clip(p, eps, 1.0 - eps)
        cost = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
        cost_history.append(cost)

        # Analytical Gradient Derivation
        error = p - y
        dw = (1.0 / m) * np.dot(X_norm.T, error)
        db = (1.0 / m) * np.sum(error)

        # Simultaneous Parameter Updates
        w -= alpha * dw
        b -= alpha * db

    return w, b, mu, sigma, cost_history


# ------------------------------------------------------------------------------
# 2. MULTICLASS LOGISTIC REGRESSION (Softmax + Categorical Cross-Entropy)
# ------------------------------------------------------------------------------
def softmax(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def fit_multiclass_logistic_regression(X, y, K=3, alpha=0.1, iterations=500):
    """
    X: shape (m, d) continuous features
    y: shape (m,) discrete labels in {0, 1, ..., K-1}
    """
    m, d = X.shape
    # One-hot encode targets: Y has shape (m, K)
    Y = np.zeros((m, K))
    for i in range(m):
        Y[i, y[i]] = 1.0

    # Feature standardization
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    W = np.zeros((d, K))
    b = np.zeros(K)
    cost_history = []

    for _ in range(iterations):
        # Forward pass: Z = X*W + b, P = softmax(Z)
        Z = np.dot(X_norm, W) + b
        P = softmax(Z)

        # Categorical Cross-Entropy Cost J(W, b)
        eps = 1e-15
        cost = -np.mean(np.sum(Y * np.log(np.clip(P, eps, 1.0 - eps)), axis=1))
        cost_history.append(cost)

        # Analytical Gradient Derivation
        error = P - Y
        dW = (1.0 / m) * np.dot(X_norm.T, error)
        db = (1.0 / m) * np.sum(error, axis=0)

        # Simultaneous Parameter Updates
        W -= alpha * dW
        b -= alpha * db

    return W, b, mu, sigma, cost_history
'''
