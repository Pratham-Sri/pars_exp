"""Four-component PCA of Palmer Penguins, fitted on the training partition."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .logistic_regression import load_penguins_classification_data


def run_pca_experiment(static_dir: Path):
    df, _, labels, species, _, _ = load_penguins_classification_data()
    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    feature_names = ["Bill length (mm)", "Bill depth (mm)", "Flipper length (mm)", "Body mass (g)"]
    X = df[features].to_numpy(dtype=float)

    rng = np.random.default_rng(42)
    train_parts, test_parts = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * .2))
        test_parts.append(indices[:n_test])
        train_parts.append(indices[n_test:])
    train_idx, test_idx = np.concatenate(train_parts), np.concatenate(test_parts)

    mean = X[train_idx].mean(axis=0)
    scale = X[train_idx].std(axis=0)
    scale[scale == 0] = 1
    Z = (X - mean) / scale
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(Z[train_idx], rowvar=False))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order].T
    for row in range(len(components)):
        if components[row, np.argmax(np.abs(components[row]))] < 0:
            components[row] *= -1
    scores = Z @ components.T
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)

    static_dir = Path(static_dir)
    static_dir.mkdir(parents=True, exist_ok=True)
    chart_path = static_dir / "pca_projection.png"
    colors = ["#276749", "#b7791f", "#315a9b"]
    train_set = set(train_idx.tolist())
    fig, (ax_scores, ax_dot) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=170, gridspec_kw={"width_ratios": [1.5, 1]})
    for label, name in enumerate(species):
        ids = np.flatnonzero(labels == label)
        tr = np.array([i for i in ids if i in train_set], dtype=int)
        te = np.array([i for i in ids if i not in train_set], dtype=int)
        ax_scores.scatter(scores[tr, 0], scores[tr, 1], s=23, color=colors[label], alpha=.55, label=f"{name} · train")
        ax_scores.scatter(scores[te, 0], scores[te, 1], s=46, marker="x", linewidth=1.2, color=colors[label], label=f"{name} · test")

    ax_scores.axhline(0, color="#64748b", lw=.8, alpha=.55)
    ax_scores.axvline(0, color="#64748b", lw=.8, alpha=.55)
    ax_scores.set_title(f"Reduced test features · PC1 / PC2 ({cumulative[1] * 100:.1f}%)", fontweight="bold")
    ax_scores.set_xlabel("PC1 score · projected feature line")
    ax_scores.set_ylabel("PC2 score")
    ax_scores.legend(fontsize=7, frameon=False, ncol=2)
    ax_scores.grid(alpha=.16)

    gram = components @ components.T
    image = ax_dot.imshow(gram, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
    ax_dot.set_xticks(np.arange(4), ["PC1", "PC2", "PC3", "PC4"])
    ax_dot.set_yticks(np.arange(4), ["PC1", "PC2", "PC3", "PC4"])
    ax_dot.set_title("Component dot products", fontweight="bold")
    for r in range(4):
        for c in range(4):
            value = 0.0 if abs(gram[r, c]) < .005 else gram[r, c]
            ax_dot.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if gram[r, c] > .55 else "#17212b")
    fig.colorbar(image, ax=ax_dot, fraction=.046, pad=.04)

    fig.suptitle("Palmer Penguins · PCA", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(chart_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    orthogonality = components @ components.T
    test_reconstruction = scores[test_idx, :2] @ components[:2]
    test_rmse_pc12 = float(np.sqrt(np.mean((Z[test_idx] - test_reconstruction) ** 2)))
    component_rows = [
        {
            "name": f"PC{i + 1}",
            "loadings": components[i].tolist(),
            "eigenvalue": float(eigenvalues[i]),
            "explained": float(explained[i]),
            "cumulative": float(cumulative[i]),
        }
        for i in range(4)
    ]
    return {
        "chart_file": chart_path.name,
        "features": feature_names,
        "components": component_rows,
        "orthogonality": orthogonality.tolist(),
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "test_variance_pc12": float(cumulative[1]),
        "test_reconstruction_rmse_pc12": test_rmse_pc12,
        "sample_projections": [
            {"record": int(df.index[i]) + 1, "species": species[labels[i]], "scores": scores[i].tolist()}
            for i in test_idx
        ],
    }
