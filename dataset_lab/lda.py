"""Linear discriminant analysis on a stratified Palmer Penguins holdout."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .logistic_regression import load_penguins_classification_data


def run_lda_experiment(static_dir: Path):
    df, _, labels, species, _, _ = load_penguins_classification_data()
    feature_columns = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    X = df[feature_columns].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    train_parts, test_parts = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * .2))
        test_parts.append(indices[:n_test])
        train_parts.append(indices[n_test:])
    train_idx, test_idx = np.concatenate(train_parts), np.concatenate(test_parts)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1
    Z_train = (X_train - mean) / scale
    Z_test = (X_test - mean) / scale

    class_count, feature_count = len(species), Z_train.shape[1]
    overall_mean = Z_train.mean(axis=0)
    within = np.zeros((feature_count, feature_count))
    between = np.zeros_like(within)
    class_means, priors = [], []
    for label in range(class_count):
        class_rows = Z_train[y_train == label]
        class_mean = class_rows.mean(axis=0)
        class_means.append(class_mean)
        priors.append(len(class_rows) / len(Z_train))
        centered = class_rows - class_mean
        within += centered.T @ centered
        delta = (class_mean - overall_mean)[:, None]
        between += len(class_rows) * (delta @ delta.T)

    within /= max(1, len(Z_train) - class_count)
    between /= len(Z_train)
    eig_w, vec_w = np.linalg.eigh(within)
    eig_w = np.maximum(eig_w, 1e-8)
    inv_sqrt_w = (vec_w * (1 / np.sqrt(eig_w))) @ vec_w.T
    whitened_between = inv_sqrt_w @ between @ inv_sqrt_w
    eigvals, eigvecs = np.linalg.eigh(whitened_between)
    order = np.argsort(eigvals)[::-1][:class_count - 1]
    components = (inv_sqrt_w @ eigvecs[:, order]).T
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    train_scores = Z_train @ components.T
    test_scores = Z_test @ components.T

    covariance = within + np.eye(feature_count) * 1e-8
    precision = np.linalg.pinv(covariance)
    class_means = np.asarray(class_means)
    discriminants = Z_test @ precision @ class_means.T
    discriminants -= .5 * np.sum((class_means @ precision) * class_means, axis=1)
    discriminants += np.log(np.asarray(priors))[None, :]
    predictions = discriminants.argmax(axis=1)
    accuracy = float(np.mean(predictions == y_test))
    confusion = np.zeros((class_count, class_count), dtype=int)
    for actual, predicted in zip(y_test, predictions):
        confusion[actual, predicted] += 1

    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    chart_path = static_dir / "lda_projection.png"
    colors = ["#276749", "#b7791f", "#315a9b"]
    fig, (ax_plane, ax_line) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=170, gridspec_kw={"width_ratios": [1.25, 1]})
    for label, name in enumerate(species):
        tr = y_train == label
        te = y_test == label
        ax_plane.scatter(train_scores[tr, 0], train_scores[tr, 1], s=24, color=colors[label], alpha=.55, label=f"{name} · train")
        ax_plane.scatter(test_scores[te, 0], test_scores[te, 1], s=48, marker="x", linewidth=1.2, color=colors[label], label=f"{name} · test")
        ax_line.scatter(train_scores[tr, 0], np.full(np.sum(tr), label) + .08, s=22, color=colors[label], alpha=.55)
        ax_line.scatter(test_scores[te, 0], np.full(np.sum(te), label) - .08, s=46, marker="x", color=colors[label])
    ax_plane.axhline(0, color="#64748b", lw=.8, alpha=.55)
    ax_plane.axvline(0, color="#64748b", lw=.8, alpha=.55)
    ax_plane.set_title("LD1–LD2 · projected test records", fontweight="bold")
    ax_plane.set_xlabel("LD1 score")
    ax_plane.set_ylabel("LD2 score")
    ax_plane.legend(fontsize=7, frameon=False, ncol=2)
    ax_plane.grid(alpha=.16)
    ax_line.axvline(0, color="#111827", lw=1.3)
    ax_line.set_yticks(np.arange(class_count), species)
    ax_line.set_xlabel("LD1 score · reduced feature on projection line")
    ax_line.set_title("LD1 projection line", fontweight="bold")
    ax_line.grid(axis="x", alpha=.18)
    fig.suptitle("Palmer Penguins · LDA", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(chart_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "chart_file": chart_path.name,
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "accuracy": accuracy,
        "species": species,
        "confusion": confusion.tolist(),
        "test_rows": [
            {"record": int(test_idx[i]) + 1, "actual": species[y_test[i]], "predicted": species[predictions[i]],
             "ld1": float(test_scores[i, 0]), "ld2": float(test_scores[i, 1])}
            for i in range(len(test_idx))
        ],
    }
