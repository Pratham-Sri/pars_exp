"""Palmer Penguins overfitting and L2 regularization comparison.

Uses an RBF feature map and a softmax classifier implemented with NumPy.
The split is stratified and all preprocessing is fitted on the training set.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .logistic_regression import load_penguins_classification_data, softmax


def _stratified_split(y, test_fraction=0.25, seed=42):
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_fraction)))
        test_parts.append(indices[:test_count])
        train_parts.append(indices[test_count:])
    train = np.concatenate(train_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _rbf_features(X, centers, gamma):
    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.exp(-gamma * distances)


def _fit_softmax(Phi, y, class_count, reg_strength, max_iter=2400):
    sample_count, feature_count = Phi.shape
    one_hot = np.eye(class_count)[y]
    weights = np.zeros((feature_count, class_count), dtype=float)
    bias = np.zeros(class_count, dtype=float)
    spectral = np.linalg.norm(Phi, ord=2) ** 2 / sample_count
    step_size = 1.0 / max(0.5 * spectral + reg_strength, 1e-6)

    for _ in range(max_iter):
        probabilities = softmax(Phi @ weights + bias)
        error = probabilities - one_hot
        grad_weights = (Phi.T @ error) / sample_count + reg_strength * weights
        grad_bias = error.mean(axis=0)
        weights -= step_size * grad_weights
        bias -= step_size * grad_bias
    return weights, bias


def _macro_f1(y_true, y_pred, class_count):
    scores = []
    for label in range(class_count):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def _predict(X, means, scales, centers, gamma, weights, bias, chunk_size=2500):
    X_norm = (X - means) / scales
    predictions, probabilities = [], []
    for start in range(0, len(X_norm), chunk_size):
        phi = _rbf_features(X_norm[start:start + chunk_size], centers, gamma)
        prob = softmax(phi @ weights + bias)
        probabilities.append(prob)
        predictions.append(np.argmax(prob, axis=1))
    return np.concatenate(predictions), np.vstack(probabilities)


def _metrics(y, predictions, probabilities, class_count):
    eps = 1e-12
    log_loss = -np.mean(np.log(np.clip(probabilities[np.arange(len(y)), y], eps, 1.0)))
    return {
        "accuracy": float(np.mean(y == predictions)),
        "macro_f1": _macro_f1(y, predictions, class_count),
        "log_loss": float(log_loss),
    }


def _plot_decision_boundaries(X, y, y_train, train_indices, test_indices, species, models, output_path):
    colors = ["#276749", "#b7791f", "#315a9b"]
    cmap = matplotlib.colors.ListedColormap(["#e9f1ec", "#f7f0df", "#e8edf7"])
    x_min, x_max = X[:, 0].min() - 2.2, X[:, 0].max() + 2.2
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 190), np.linspace(y_min, y_max, 190))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=170, sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        pred, _ = _predict(grid, model["means"], model["scales"], model["centers"], model["gamma"], model["weights"], model["bias"])
        regions = pred.reshape(xx.shape)
        ax.contourf(xx, yy, regions, levels=np.arange(-0.5, len(species) + .5), cmap=cmap, alpha=.9)
        ax.contour(xx, yy, regions, levels=np.arange(.5, len(species) - .5, 1), colors="#46504a", linewidths=.75, alpha=.65)
        for label, name in enumerate(species):
            train_mask = train_indices[y_train == label]
            ax.scatter(X[train_mask, 0], X[train_mask, 1], s=20, marker="o", c=colors[label], edgecolors="white", linewidths=.35, label=f"{name} · train")
            test_mask = test_indices[y[test_indices] == label]
            ax.scatter(X[test_mask, 0], X[test_mask, 1], s=39, marker="x", c=colors[label], linewidths=1.1, label=f"{name} · test")
        ax.set_title(model["label"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Bill length (mm)")
        ax.grid(alpha=.12)
    axes[0].set_ylabel("Flipper length (mm)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7.5, frameon=False)
    fig.suptitle("Palmer Penguins · RBF Softmax Decision Regions", fontsize=14, fontweight="bold")
    fig.text(.5, .01, "Circles are training observations; crosses are held-out test observations. Preprocessing and model fitting use training data only.", ha="center", fontsize=8.5, color="#59645e")
    fig.tight_layout(rect=(0, .045, .84, .94))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_performance(models, output_path, train_percent, test_percent):
    metric_defs = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("log_loss", "Log loss (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=170)
    x = np.arange(2)
    width = .33
    for ax, (metric, title) in zip(axes, metric_defs):
        train = [m["train"][metric] for m in models]
        test = [m["test"][metric] for m in models]
        ax.bar(x - width / 2, train, width, color="#8da993", label="Train")
        ax.bar(x + width / 2, test, width, color="#325f46", label="Test")
        for xi, val in zip(x - width / 2, train):
            ax.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)
        for xi, val in zip(x + width / 2, test):
            ax.text(xi, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, ["Unregularized", "L2 regularized"])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, fontsize=8)
        if metric != "log_loss":
            ax.set_ylim(0, 1.12)
    fig.suptitle("Train/Test Performance Before and After L2 Regularization", fontsize=14, fontweight="bold")
    fig.text(.5, .01, f"Same stratified {train_percent}/{test_percent} split; higher is better except log loss.", ha="center", fontsize=8.5, color="#59645e")
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_regularization_experiment(static_dir: Path):
    """Fit the high-capacity baseline and its L2-regularized counterpart; draw both plots."""
    df, X, y, species, _, _ = load_penguins_classification_data()
    # Keep the training subset deliberately small so the high-capacity baseline
    # can expose a meaningful generalization gap on the larger held-out set.
    train_idx, test_idx = _stratified_split(y, test_fraction=.80, seed=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    means = X_train.mean(axis=0)
    scales = X_train.std(axis=0)
    scales[scales == 0] = 1.0
    X_train_norm = (X_train - means) / scales
    X_test_norm = (X_test - means) / scales

    gamma = 35.0
    train_phi = _rbf_features(X_train_norm, X_train_norm, gamma)
    test_phi = _rbf_features(X_test_norm, X_train_norm, gamma)
    configs = [
        ("Unregularized high-capacity model", 0.0),
        ("L2-regularized model (λ = 0.001)", 0.001),
    ]
    models = []
    for label, strength in configs:
        weights, bias = _fit_softmax(train_phi, y_train, len(species), strength)
        train_prob = softmax(train_phi @ weights + bias)
        test_prob = softmax(test_phi @ weights + bias)
        train_pred = np.argmax(train_prob, axis=1)
        test_pred = np.argmax(test_prob, axis=1)
        models.append({
            "label": label,
            "regularization": strength,
            "means": means,
            "scales": scales,
            "centers": X_train_norm,
            "gamma": gamma,
            "weights": weights,
            "bias": bias,
            "train": _metrics(y_train, train_pred, train_prob, len(species)),
            "test": _metrics(y_test, test_pred, test_prob, len(species)),
            "train_n": len(y_train),
            "test_n": len(y_test),
        })
    for model in models:
        model["accuracy_gap"] = model["train"]["accuracy"] - model["test"]["accuracy"]
        model["macro_f1_gap"] = model["train"]["macro_f1"] - model["test"]["macro_f1"]

    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = static_dir / "regularization_decision_boundaries.png"
    performance_path = static_dir / "regularization_performance.png"
    _plot_decision_boundaries(X, y, y_train, train_idx, test_idx, species, models, boundary_path)
    _plot_performance(models, performance_path, round(100 * len(train_idx) / len(y)), round(100 * len(test_idx) / len(y)))

    return {
        "models": models,
        "species": species,
        "sample_count": len(y),
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "train_percent": round(100 * len(train_idx) / len(y)),
        "test_percent": round(100 * len(test_idx) / len(y)),
        "split_seed": 42,
        "test_accuracy_change_pp": 100 * (models[1]["test"]["accuracy"] - models[0]["test"]["accuracy"]),
        "test_macro_f1_change_pp": 100 * (models[1]["test"]["macro_f1"] - models[0]["test"]["macro_f1"]),
        "accuracy_gap_reduction_pp": 100 * (models[0]["accuracy_gap"] - models[1]["accuracy_gap"]),
        "degree_note": f"RBF model · γ = {gamma:g} · L2 λ = 0.001",
        "boundary_file": boundary_path.name,
        "performance_file": performance_path.name,
        "split_counts": {name: {"train": int(np.sum(y_train == i)), "test": int(np.sum(y_test == i))} for i, name in enumerate(species)},
    }
