"""Linear Regression from Scratch with Gradient Descent on Palmer Penguins Dataset.
Zero in-built ML library functions: implements hypothesis, MSE cost function, gradient derivation,
parameter updates, and convergence tracking strictly using elementary Python and NumPy.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data Ingestion & Preprocessing (Palmer Penguins Dataset)
# ---------------------------------------------------------------------------
def load_penguins_regression_data():
    """Load Palmer Penguins dataset and extract continuous features for regression."""
    csv_path = Path("data/palmer_penguins.csv")
    if not csv_path.exists():
        import seaborn as sns
        df = sns.load_dataset("penguins").dropna()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path).dropna()

    # Feature: Flipper Length (mm), Target: Body Mass (g)
    x = df["flipper_length_mm"].values.astype(float)
    y = df["body_mass_g"].values.astype(float)
    species = df["species"].values if "species" in df.columns else np.array(["Penguin"] * len(x))

    return df, x, y, species


# ---------------------------------------------------------------------------
# From-Scratch Gradient Descent Implementation (No In-Built ML Functions)
# ---------------------------------------------------------------------------
def compute_cost_mse(x, y, theta0, theta1):
    """
    Compute Mean Squared Error Cost Function:
      J(theta0, theta1) = 1 / (2 * m) * sum((theta0 + theta1 * x^(i) - y^(i))^2)
    """
    m = len(y)
    predictions = theta0 + theta1 * x
    errors = predictions - y
    cost = (1.0 / (2.0 * m)) * np.sum(errors ** 2)
    return float(cost)


def gradient_descent_linear_regression(x_raw, y_raw, alpha=0.05, iterations=500, init_theta0=0.0, init_theta1=0.0):
    """
    Implements Batch Gradient Descent for Simple Linear Regression from first principles:
      dJ/dtheta0 = 1/m * sum(h(x) - y)
      dJ/dtheta1 = 1/m * sum((h(x) - y) * x)
      theta0 := theta0 - alpha * dJ/dtheta0
      theta1 := theta1 - alpha * dJ/dtheta1
    """
    m = len(y_raw)
    
    # Feature Standardization for smooth numerical stability
    mu_x = float(np.mean(x_raw))
    sigma_x = float(np.std(x_raw))
    mu_y = float(np.mean(y_raw))
    sigma_y = float(np.std(y_raw))

    x_norm = (x_raw - mu_x) / sigma_x
    y_norm = (y_raw - mu_y) / sigma_y

    theta0 = float(init_theta0)
    theta1 = float(init_theta1)

    cost_history = []
    theta0_history = [theta0]
    theta1_history = [theta1]

    for epoch in range(iterations):
        # 1. Hypothesis prediction: h_theta(x) = theta0 + theta1 * x
        y_pred_norm = theta0 + theta1 * x_norm
        errors = y_pred_norm - y_norm

        # 2. Compute cost J(theta)
        cost = (1.0 / (2.0 * m)) * np.sum(errors ** 2)
        cost_history.append(float(cost))

        # 3. Compute partial gradients
        grad0 = (1.0 / m) * np.sum(errors)
        grad1 = (1.0 / m) * np.sum(errors * x_norm)

        # 4. Simultaneous parameter update rule
        theta0 -= alpha * grad0
        theta1 -= alpha * grad1

        theta0_history.append(theta0)
        theta1_history.append(theta1)

    # 5. Transform normalized parameters back to native physical dimensions
    # native y = slope * native x + intercept
    slope = theta1 * (sigma_y / sigma_x)
    intercept = mu_y - slope * mu_x + theta0 * sigma_y

    # 6. Goodness-of-Fit Evaluation Metrics computed from scratch
    y_pred_raw = slope * x_raw + intercept
    residuals = y_raw - y_pred_raw
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_raw - mu_y) ** 2)

    mse = ss_res / m
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    r2_score = 1.0 - (ss_res / ss_tot)

    return {
        "alpha": alpha,
        "iterations": iterations,
        "theta0_norm": theta0,
        "theta1_norm": theta1,
        "slope": float(slope),
        "intercept": float(intercept),
        "initial_cost": cost_history[0],
        "final_cost": cost_history[-1],
        "cost_history": cost_history,
        "theta0_history": theta0_history,
        "theta1_history": theta1_history,
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2_score),
        "mu_x": mu_x,
        "sigma_x": sigma_x,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
    }


# ---------------------------------------------------------------------------
# High-Resolution Visualization of the Gradient Descent Curve & Model Fit
# ---------------------------------------------------------------------------
def generate_gradient_descent_charts(output_path: Path):
    """Generate the B regression-fit and D learning-rate comparison panels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df, x, y, species = load_penguins_regression_data()
    opt_res = gradient_descent_linear_regression(x, y, alpha=0.05, iterations=400)
    fig, (ax_fit, ax_rates) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=180)
    fig.patch.set_facecolor("#ffffff")

    markers = {"Adelie": "o", "Chinstrap": "s", "Gentoo": "^"}
    for sp in np.unique(species):
        mask = (species == sp)
        m = markers.get(sp, "o")
        ax_fit.scatter(x[mask], y[mask], facecolors="#f0f0f0", edgecolors="#333333",
                    marker=m, alpha=0.85, s=32, linewidth=0.8,
                    label=f"{sp} (N={np.sum(mask)})")

    x_line = np.linspace(np.min(x) - 2, np.max(x) + 2, 200)
    y_line = opt_res["slope"] * x_line + opt_res["intercept"]
    ax_fit.plot(x_line, y_line, color="#000000", linewidth=2.2,
             label=f"Fit: y = {opt_res['slope']:.2f}x − {abs(opt_res['intercept']):.1f}")
    ax_fit.plot(x_line, y_line + opt_res["rmse"], color="#666666", linestyle="--", linewidth=1.0,
             label=f"±1 RMSE (±{opt_res['rmse']:.0f} g)")
    ax_fit.plot(x_line, y_line - opt_res["rmse"], color="#666666", linestyle="--", linewidth=1.0)
    ax_fit.set_title(f"B. Linear Regression Fit (R² = {opt_res['r2_score']:.3f}, RMSE = {opt_res['rmse']:.1f} g)",
                  fontsize=11, fontweight="bold", pad=10, color="#000000")
    ax_fit.set_xlabel("Flipper Length (mm)", fontsize=9.5, fontweight="600", color="#000000")
    ax_fit.set_ylabel("Body Mass (g)", fontsize=9.5, fontweight="600", color="#000000")
    ax_fit.grid(True, linestyle=":", color="#cccccc", alpha=0.8)
    ax_fit.legend(loc="upper left", fontsize=8.0, frameon=True, edgecolor="#000000")

    learning_rates = [0.005, 0.02, 0.05, 0.15]
    rate_colors = ["#94a3b8", "#3b82f6", "#164e3d", "#dc2626"]
    for alpha, color in zip(learning_rates, rate_colors):
        result = gradient_descent_linear_regression(x, y, alpha=alpha, iterations=200)
        ax_rates.plot(np.arange(len(result["cost_history"])), result["cost_history"],
                      color=color, linewidth=2, label=f"α = {alpha:g}")
    ax_rates.set_title("D. Learning Rate Sensitivity", fontsize=11, fontweight="bold", pad=10)
    ax_rates.set_xlabel("Iteration")
    ax_rates.set_ylabel("Cost J(θ)")
    ax_rates.grid(True, linestyle=":", alpha=.7)
    ax_rates.legend(frameon=True, fontsize=8)

    for axis in (ax_fit, ax_rates):
        for spine in axis.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor="#ffffff", edgecolor="none")
    plt.close()


# ---------------------------------------------------------------------------
# Test Prediction Ledger
# ---------------------------------------------------------------------------
def generate_sample_predictions():
    """Evaluate linear regression on actual Palmer Penguins test samples."""
    df, x, y, species = load_penguins_regression_data()
    res = gradient_descent_linear_regression(x, y, alpha=0.05, iterations=400)
    slope = res["slope"]
    intercept = res["intercept"]

    indices = [5, 42, 110, 165, 215, 270, 310, 325]
    sample_rows = []
    for idx in indices:
        if idx < len(df):
            row = df.iloc[idx]
            x_val = float(row["flipper_length_mm"])
            actual_y = float(row["body_mass_g"])
            pred_y = slope * x_val + intercept
            err = pred_y - actual_y
            sample_rows.append({
                "index": int(idx),
                "species": str(row["species"]),
                "flipper_length_mm": x_val,
                "actual_mass_g": actual_y,
                "predicted_mass_g": float(pred_y),
                "residual_error_g": float(err),
                "relative_error_pct": float(abs(err) / actual_y * 100)
            })

    return res, sample_rows


# ---------------------------------------------------------------------------
# Python Source Code Snippet for Page Display
# ---------------------------------------------------------------------------
PYTHON_GRADIENT_DESCENT_CODE = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Palmer Penguins Dataset (No external ML library used)
df = pd.read_csv("data/palmer_penguins.csv").dropna()
x_raw = df["flipper_length_mm"].values.astype(float)
y_raw = df["body_mass_g"].values.astype(float)
m = len(y_raw)

# 2. Feature Standardization (Z-score normalization for numerical stability)
mu_x, sigma_x = np.mean(x_raw), np.std(x_raw)
mu_y, sigma_y = np.mean(y_raw), np.std(y_raw)

x_norm = (x_raw - mu_x) / sigma_x
y_norm = (y_raw - mu_y) / sigma_y

# 3. Batch Gradient Descent Implementation from First Principles
alpha = 0.05        # Learning rate
iterations = 400   # Number of epochs
theta0 = 0.0        # Initial bias
theta1 = 0.0        # Initial weight

cost_history = []

for epoch in range(iterations):
    # Hypothesis: h_theta(x) = theta0 + theta1 * x
    y_pred = theta0 + theta1 * x_norm
    errors = y_pred - y_norm
    
    # Mean Squared Error Cost Function: J(theta) = 1/(2m) * sum(errors^2)
    cost = (1.0 / (2.0 * m)) * np.sum(errors ** 2)
    cost_history.append(cost)
    
    # Compute partial derivatives:
    # dJ/dtheta0 = 1/m * sum(h(x) - y)
    # dJ/dtheta1 = 1/m * sum((h(x) - y) * x)
    grad0 = (1.0 / m) * np.sum(errors)
    grad1 = (1.0 / m) * np.sum(errors * x_norm)
    
    # Simultaneous Parameter Updates
    theta0 -= alpha * grad0
    theta1 -= alpha * grad1

# 4. Transform Parameters back to Physical Units (grams / mm)
slope = theta1 * (sigma_y / sigma_x)
intercept = mu_y - slope * mu_x + theta0 * sigma_y

# 5. Goodness-of-Fit Evaluation Metrics
y_pred_raw = slope * x_raw + intercept
r2 = 1.0 - np.sum((y_raw - y_pred_raw) ** 2) / np.sum((y_raw - mu_y) ** 2)
rmse = np.sqrt(np.mean((y_raw - y_pred_raw) ** 2))

print(f"Optimized Equation: Body Mass (g) = {slope:.2f} * Flipper (mm) + {intercept:.2f}")
print(f"R^2 Coefficient:   {r2:.4f}")
print(f"Root Mean Sq Error: {rmse:.2f} g")
print(f"Cost Convergence:   Initial J = {cost_history[0]:.4f} -> Final J = {cost_history[-1]:.4f}")

# 6. Plot Gradient Descent Curve
plt.figure(figsize=(9, 4.5))
plt.plot(cost_history, color="#164e3d", linewidth=2.5, label=f"Cost J(theta) [alpha={alpha}]")
plt.title("Gradient Descent Cost Convergence Curve J(theta) on Palmer Penguins")
plt.xlabel("Iteration")
plt.ylabel("Cost J(theta)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
'''
