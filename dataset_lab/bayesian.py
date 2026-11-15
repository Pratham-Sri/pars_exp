"""Bayesian Multi-Class Pattern Classification on Real Benchmark Datasets.
Supports:
1. Palmer Penguins Morphology Dataset (3 Classes: Adelie, Chinstrap, Gentoo)
2. UCI Wheat Seeds Grain X-Ray Dataset (3 Classes: Kama, Rosa, Canadian)

Features:
- Single-Feature Univariate Gaussian Likelihood & Posterior Modeling
- Multiple-Feature Multivariate Gaussian QDA Decision Boundary Fitting
- Train/Test Splitting (75% Train, 25% Stratified Test)
- Empirical Classification of Test Dataset Samples
- Exact Decision Boundary & Test Point Scatter Visualizations
- Real-time Interactive Inference API
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------------------------
# DATASET REGISTRY & CONFIGURATIONS
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "penguins": {
        "id": "penguins",
        "name": "Palmer Penguins Morphology Dataset",
        "csv_path": "data/palmer_penguins.csv",
        "class_col": "species",
        "classes": ["Adelie", "Chinstrap", "Gentoo"],
        "class_symbols": {"Adelie": "Adelie (ω₁)", "Chinstrap": "Chinstrap (ω₂)", "Gentoo": "Gentoo (ω₃)"},
        "class_colors": {"Adelie": "#000000", "Chinstrap": "#444444", "Gentoo": "#888888"},
        "feat_1d": "flipper_length_mm",
        "feat_1d_label": "Flipper Length (mm)",
        "feat_2d": ["bill_length_mm", "flipper_length_mm"],
        "feat_2d_labels": ["Bill Length (mm)", "Flipper Length (mm)"],
        "slider_1_range": [30.0, 60.0, 0.2, 45.0],
        "slider_2_range": [170.0, 240.0, 1.0, 200.0],
        "description": "Morphological measurements of 333 adult foraging penguins near Palmer Station, Antarctica (collected by Dr. Kristen Gorman)."
    },
    "wheat_seeds": {
        "id": "wheat_seeds",
        "name": "UCI Wheat Seeds Grain X-Ray Dataset",
        "csv_path": "data/wheat_seeds.csv",
        "class_col": "species",
        "classes": ["Kama", "Rosa", "Canadian"],
        "class_symbols": {"Kama": "Kama (ω₁)", "Rosa": "Rosa (ω₂)", "Canadian": "Canadian (ω₃)"},
        "class_colors": {"Kama": "#000000", "Rosa": "#444444", "Canadian": "#888888"},
        "feat_1d": "Area",
        "feat_1d_label": "Kernel Area (A in mm²)",
        "feat_2d": ["Area", "Asymmetry_Coeff"],
        "feat_2d_labels": ["Kernel Area (mm²)", "Asymmetry Coefficient"],
        "slider_1_range": [10.0, 22.0, 0.1, 14.5],
        "slider_2_range": [0.5, 9.0, 0.1, 3.5],
        "description": "Geometric properties of wheat grain kernels from soft X-ray photographs across 3 wheat varieties (Institute of Agrophysics, Polish Academy of Sciences)."
    }
}


def load_dataset_data(ds_id: str):
    """Load, clean, and prepare data for specified dataset."""
    cfg = DATASET_CONFIGS[ds_id]
    csv_file = Path(cfg["csv_path"])
    
    if ds_id == "penguins":
        if not csv_file.exists():
            import seaborn as sns
            df = sns.load_dataset("penguins").dropna()
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_file, index=False)
        else:
            df = pd.read_csv(csv_file)
            
    elif ds_id == "wheat_seeds":
        if not csv_file.exists():
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
            cols = ["Area", "Perimeter", "Compactness", "Kernel_Length", "Kernel_Width", "Asymmetry_Coeff", "Kernel_Groove_Length", "Class"]
            df = pd.read_csv(url, sep=r"\s+", names=cols)
            s_map = {1: "Kama", 2: "Rosa", 3: "Canadian"}
            df["species"] = df["Class"].map(s_map)
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_file, index=False)
        else:
            df = pd.read_csv(csv_file)
            if "species" not in df.columns and "Class" in df.columns:
                s_map = {1: "Kama", 2: "Rosa", 3: "Canadian"}
                df["species"] = df["Class"].map(s_map)
                
    return df


def fit_and_evaluate_dataset(ds_id: str):
    """Train Bayesian Gaussian models on train split and evaluate on test split."""
    cfg = DATASET_CONFIGS[ds_id]
    df = load_dataset_data(ds_id)
    classes = cfg["classes"]
    feat_1d = cfg["feat_1d"]
    feat_2d = cfg["feat_2d"]
    
    # 75% Train, 25% Test Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        df[feat_2d], df[cfg["class_col"]], test_size=0.25, random_state=42, stratify=df[cfg["class_col"]]
    )
    
    # Compute Class Priors P(omega_j) from training set
    N_train = len(X_train)
    priors = {c: float(np.mean(y_train == c)) for c in classes}
    
    # Estimate 1D Parameters (Mean, Variance) on training set
    params_1d = {}
    for c in classes:
        vals = X_train.loc[y_train == c, feat_1d].values
        params_1d[c] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "var": float(np.var(vals, ddof=1))
        }
        
    # Estimate 2D Parameters (Mean Vector, Covariance Matrix) on training set
    params_2d = {}
    for c in classes:
        vals = X_train.loc[y_train == c, feat_2d].values
        params_2d[c] = {
            "mean": np.mean(vals, axis=0),
            "cov": np.cov(vals, rowvar=False)
        }
        
    # Evaluate 1D on Test Set
    test_rows = []
    preds_1d = []
    preds_2d = []
    
    for idx, (index_lbl, row) in enumerate(X_test.iterrows()):
        true_cls = y_test.loc[index_lbl]
        val_1d = float(row[feat_1d])
        val_2d = row[feat_2d].values
        
        # 1D Bayesian Likelihoods & Posteriors
        lh_1d = {c: float(norm.pdf(val_1d, params_1d[c]["mean"], params_1d[c]["std"])) for c in classes}
        joint_1d = {c: lh_1d[c] * priors[c] for c in classes}
        ev_1d = sum(joint_1d.values())
        post_1d = {c: (joint_1d[c] / ev_1d if ev_1d > 0 else 1/3) for c in classes}
        pred_1d = max(post_1d, key=post_1d.get)
        preds_1d.append(pred_1d)
        
        # 2D Bayesian Likelihoods & Posteriors (QDA)
        lh_2d = {c: float(multivariate_normal.pdf(val_2d, params_2d[c]["mean"], params_2d[c]["cov"])) for c in classes}
        joint_2d = {c: lh_2d[c] * priors[c] for c in classes}
        ev_2d = sum(joint_2d.values())
        post_2d = {c: (joint_2d[c] / ev_2d if ev_2d > 0 else 1/3) for c in classes}
        pred_2d = max(post_2d, key=post_2d.get)
        preds_2d.append(pred_2d)
        
        test_rows.append({
            "test_id": f"TS-{idx+1:02d}",
            "original_index": int(index_lbl),
            "x1": float(row[feat_2d[0]]),
            "x2": float(row[feat_2d[1]]),
            "true_class": true_cls,
            "true_symbol": cfg["class_symbols"][true_cls],
            "true_color": cfg["class_colors"][true_cls],
            # 1D
            "post_1d": post_1d,
            "pred_1d": pred_1d,
            "pred_1d_symbol": cfg["class_symbols"][pred_1d],
            "conf_1d": post_1d[pred_1d],
            "correct_1d": (pred_1d == true_cls),
            # 2D
            "post_2d": post_2d,
            "pred_2d": pred_2d,
            "pred_2d_symbol": cfg["class_symbols"][pred_2d],
            "conf_2d": post_2d[pred_2d],
            "correct_2d": (pred_2d == true_cls),
        })
        
    acc_1d = accuracy_score(y_test, preds_1d)
    acc_2d = accuracy_score(y_test, preds_2d)
    cm_2d = confusion_matrix(y_test, preds_2d, labels=classes)
    
    metrics = {
        "dataset_name": cfg["name"],
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "acc_1d": acc_1d,
        "acc_2d": acc_2d,
        "acc_1d_pct": f"{acc_1d * 100:.1f}%",
        "acc_2d_pct": f"{acc_2d * 100:.1f}%",
        "gain_pct": f"+{(acc_2d - acc_1d) * 100:.1f}%",
        "confusion_matrix": cm_2d.tolist(),
    }
    
    return {
        "cfg": cfg,
        "classes": classes,
        "priors": priors,
        "params_1d": params_1d,
        "params_2d": params_2d,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_rows": test_rows,
        "metrics": metrics
    }


# ---------------------------------------------------------------------------
# GENERATE PUBLICATION-GRADE VISUALIZATIONS SHOWING REAL DATASET & TEST POINTS
# ---------------------------------------------------------------------------
def generate_dataset_charts(ds_id: str, output_path: Path):
    """Generate 4-panel figure explicitly displaying actual training and test data points."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res = fit_and_evaluate_dataset(ds_id)
    cfg = res["cfg"]
    classes = res["classes"]
    priors = res["priors"]
    params_1d = res["params_1d"]
    params_2d = res["params_2d"]
    X_train = res["X_train"]
    y_train = res["y_train"]
    X_test = res["X_test"]
    y_test = res["y_test"]
    test_rows = res["test_rows"]
    metrics = res["metrics"]
    
    feat_1d = cfg["feat_1d"]
    feat_2d = cfg["feat_2d"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    fig.patch.set_facecolor("#ffffff")

    # -----------------------------------------------------------------------
    # PANEL 1: Single Feature 1D Likelihoods + Real Test Points
    # -----------------------------------------------------------------------
    ax1.set_facecolor("#ffffff")
    all_vals_1d = np.concatenate([X_train[feat_1d].values, X_test[feat_1d].values])
    min_x, max_x = np.min(all_vals_1d), np.max(all_vals_1d)
    pad_x = (max_x - min_x) * 0.15
    grid_1d = np.linspace(min_x - pad_x, max_x + pad_x, 500)
    
    linestyles = ["-", "--", ":"]
    lh_curves = {}
    for idx, c in enumerate(classes):
        lh = norm.pdf(grid_1d, params_1d[c]["mean"], params_1d[c]["std"])
        lh_curves[c] = lh
        ls = linestyles[idx % len(linestyles)]
        ax1.plot(grid_1d, lh, label=f"p({feat_1d}|{c})", color="#000000", linestyle=ls, linewidth=1.8)

    # Plot Actual Test Data Points along bottom rug/strip
    y_rug = 0.02 * max([np.max(lh) for lh in lh_curves.values()])
    for idx, row in enumerate(test_rows):
        val = row["x1"] if feat_1d == feat_2d[0] else row["x2"]
        marker = "o" if row["correct_1d"] else "x"
        fc = "#000000" if row["correct_1d"] else "#666666"
        ax1.scatter(val, y_rug * (1 + (idx % 3) * 0.5), color=fc, marker=marker,
                    s=32, linewidth=1.0, alpha=0.8, zorder=5)

    ax1.set_title(f"A. 1D Likelihoods & Test Samples ({cfg['feat_1d_label']})",
                  fontsize=11, fontweight="bold", pad=10, color="#000000")
    ax1.set_xlabel(cfg["feat_1d_label"], fontsize=9.5, fontweight="600", color="#000000")
    ax1.set_ylabel("Probability Density", fontsize=9.5, fontweight="600", color="#000000")
    ax1.grid(True, linestyle=":", color="#cccccc", alpha=0.8)
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True, edgecolor="#000000")

    for spine in ax1.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    # -----------------------------------------------------------------------
    # PANEL 2: 2D Features QDA Decision Boundaries + Train & Test Points
    # -----------------------------------------------------------------------
    ax2.set_facecolor("#ffffff")
    x1_min, x1_max = X_train[feat_2d[0]].min() - 1, X_train[feat_2d[0]].max() + 1
    x2_min, x2_max = X_train[feat_2d[1]].min() - 1, X_train[feat_2d[1]].max() + 1
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 250), np.linspace(x2_min, x2_max, 250))
    mesh_coords = np.dstack((xx, yy))
    
    # Evaluate 2D posteriors across mesh
    post_mesh_stack = []
    for c in classes:
        p_c = multivariate_normal.pdf(mesh_coords, params_2d[c]["mean"], params_2d[c]["cov"]) * priors[c]
        post_mesh_stack.append(p_c)
    ev_mesh = sum(post_mesh_stack)
    post_mesh_stack = [p / ev_mesh for p in post_mesh_stack]
    decision_grid = np.argmax(np.stack(post_mesh_stack, axis=-1), axis=-1)
    
    # Clean black decision boundary contours
    ax2.contour(xx, yy, decision_grid, levels=[0.5, 1.5], colors=["#000000", "#555555"], linewidths=1.8, linestyles=["-", "--"])

    # Plot Training Points (light background)
    markers_2d = ["o", "s", "^"]
    for idx, c in enumerate(classes):
        pts = X_train[y_train == c]
        m = markers_2d[idx % len(markers_2d)]
        ax2.scatter(pts[feat_2d[0]], pts[feat_2d[1]], facecolors="#ffffff", edgecolors="#666666",
                    marker=m, s=24, alpha=0.6, label=f"Train {c} (N={len(pts)})")

    # Plot Test Points (prominently on foreground)
    for row in test_rows:
        marker = "o" if row["correct_2d"] else "x"
        c_idx = classes.index(row["true_class"]) if row["true_class"] in classes else 0
        fc = "#000000" if row["correct_2d"] else "#444444"
        ax2.scatter(row["x1"], row["x2"], color=fc, marker=marker,
                    s=45, linewidth=1.2, zorder=6)

    ax2.set_title(f"B. 2D Decision Boundaries & Test Data (Acc: {metrics['acc_2d_pct']})",
                  fontsize=11, fontweight="bold", pad=10, color="#000000")
    ax2.set_xlabel(cfg["feat_2d_labels"][0], fontsize=9.5, fontweight="600", color="#000000")
    ax2.set_ylabel(cfg["feat_2d_labels"][1], fontsize=9.5, fontweight="600", color="#000000")
    ax2.grid(True, linestyle=":", color="#cccccc", alpha=0.8)
    ax2.legend(loc="upper right", fontsize=8.0, frameon=True, edgecolor="#000000")

    for spine in ax2.spines.values():
        spine.set_color("#000000")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor="#ffffff", edgecolor="none")
    plt.close()


# ---------------------------------------------------------------------------
# INTERACTIVE POINT CLASSIFIER API
# ---------------------------------------------------------------------------
def classify_custom_point(ds_id: str, x1: float, x2: float, mode: str = "multi", prior_mode: str = "empirical"):
    """Classify user-provided coordinate using fitted Bayesian model."""
    res = fit_and_evaluate_dataset(ds_id)
    cfg = res["cfg"]
    classes = res["classes"]
    priors = {"Adelie": 1/3, "Chinstrap": 1/3, "Gentoo": 1/3} if prior_mode == "equal" else res["priors"]
    
    if mode == "single":
        likelihoods = {c: float(norm.pdf(x1, res["params_1d"][c]["mean"], res["params_1d"][c]["std"])) for c in classes}
    else:
        x_vec = np.array([x1, x2])
        likelihoods = {c: float(multivariate_normal.pdf(x_vec, res["params_2d"][c]["mean"], res["params_2d"][c]["cov"])) for c in classes}
        
    joints = {c: likelihoods[c] * priors[c] for c in classes}
    evidence = sum(joints.values())
    posteriors = {c: (joints[c] / evidence if evidence > 0 else 1.0 / len(classes)) for c in classes}
    pred_cls = max(posteriors, key=posteriors.get)
    
    return {
        "dataset": ds_id,
        "mode": mode,
        "x1": x1,
        "x2": x2,
        "likelihoods": likelihoods,
        "evidence": evidence,
        "posteriors": posteriors,
        "pred_class": pred_cls,
        "pred_symbol": cfg["class_symbols"][pred_cls],
        "pred_color": cfg["class_colors"][pred_cls],
        "confidence": posteriors[pred_cls]
    }


# ---------------------------------------------------------------------------
# PYTHON CODE SNIPPETS FOR PAGE DISPLAY
# ---------------------------------------------------------------------------
PENGUINS_SINGLE_CODE = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.metrics import accuracy_score

# 1. Load Palmer Penguins Dataset & Stratified Split
df = pd.read_csv("data/palmer_penguins.csv").dropna()
classes = ["Adelie", "Chinstrap", "Gentoo"]
feat_1d = "flipper_length_mm"

X_train, X_test, y_train, y_test = train_test_split(
    df[[feat_1d]], df["species"], test_size=0.25, random_state=42, stratify=df["species"]
)

# 2. Fit Univariate Gaussian MLE Parameters on Train Split
priors = {c: float(np.mean(y_train == c)) for c in classes}
params_1d = {}
for c in classes:
    vals = X_train.loc[y_train == c, feat_1d].values
    params_1d[c] = {"mean": np.mean(vals), "std": np.std(vals, ddof=1)}
    print(f"Class {c}: Mean = {params_1d[c]['mean']:.2f} mm, Std = {params_1d[c]['std']:.2f} mm, Prior = {priors[c]:.3f}")

# 3. Classify Test Samples using Bayes' Rule
test_preds = []
for x in X_test[feat_1d].values:
    # Compute likelihoods p(x|wj) and joint probabilities p(x, wj)
    joints = {c: norm.pdf(x, params_1d[c]["mean"], params_1d[c]["std"]) * priors[c] for c in classes}
    evidence = sum(joints.values())
    posteriors = {c: joints[c] / evidence for c in classes}
    # MAP Decision Rule
    pred = max(posteriors, key=posteriors.get)
    test_preds.append(pred)

acc_1d = accuracy_score(y_test, test_preds)
print(f"\\nSingle-Feature (flipper_length_mm) Test Accuracy: {acc_1d * 100:.2f}%")
'''

PENGUINS_MULTI_CODE = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Palmer Penguins Dataset (Bill Length & Flipper Length)
df = pd.read_csv("data/palmer_penguins.csv").dropna()
classes = ["Adelie", "Chinstrap", "Gentoo"]
feat_2d = ["bill_length_mm", "flipper_length_mm"]

X_train, X_test, y_train, y_test = train_test_split(
    df[feat_2d], df["species"], test_size=0.25, random_state=42, stratify=df["species"]
)

# 2. Fit Multivariate Gaussian QDA Parameters on Train Split
priors = {c: float(np.mean(y_train == c)) for c in classes}
params_2d = {}
for c in classes:
    vals = X_train.loc[y_train == c, feat_2d].values
    params_2d[c] = {
        "mean": np.mean(vals, axis=0),
        "cov": np.cov(vals, rowvar=False)
    }

# 3. Classify Test Samples using Multivariate QDA Decision Rule
test_preds = []
test_posteriors = []
for _, row in X_test.iterrows():
    x_vec = row.values
    # Multivariate Normal Likelihood p(x|wj)
    joints = {c: multivariate_normal.pdf(x_vec, params_2d[c]["mean"], params_2d[c]["cov"]) * priors[c] for c in classes}
    evidence = sum(joints.values())
    posteriors = {c: joints[c] / evidence for c in classes}
    pred = max(posteriors, key=posteriors.get)
    test_preds.append(pred)
    test_posteriors.append(posteriors)

acc_2d = accuracy_score(y_test, test_preds)
print(f"Multi-Feature (2D QDA) Test Accuracy: {acc_2d * 100:.2f}%")
print("\\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_preds, labels=classes))
print("\\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=classes))
'''

SEEDS_MULTI_CODE = '''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load UCI Wheat Seeds Dataset (Kama, Rosa, Canadian)
df = pd.read_csv("data/wheat_seeds.csv")
classes = ["Kama", "Rosa", "Canadian"]
feat_2d = ["Area", "Asymmetry_Coeff"]

X_train, X_test, y_train, y_test = train_test_split(
    df[feat_2d], df["species"], test_size=0.25, random_state=42, stratify=df["species"]
)

priors = {c: float(np.mean(y_train == c)) for c in classes}
params_2d = {}
for c in classes:
    vals = X_train.loc[y_train == c, feat_2d].values
    params_2d[c] = {"mean": np.mean(vals, axis=0), "cov": np.cov(vals, rowvar=False)}

# 2. Classify Test Set
preds = []
for _, row in X_test.iterrows():
    x = row.values
    joints = {c: multivariate_normal.pdf(x, params_2d[c]["mean"], params_2d[c]["cov"]) * priors[c] for c in classes}
    ev = sum(joints.values())
    post = {c: joints[c] / ev for c in classes}
    preds.append(max(post, key=post.get))

print(f"UCI Wheat Seeds 2D QDA Test Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds, labels=classes))
'''

DATASET_CODE_SNIPPETS = {
    "penguins_single": PENGUINS_SINGLE_CODE,
    "penguins_multi": PENGUINS_MULTI_CODE,
    "seeds_multi": SEEDS_MULTI_CODE,
}
