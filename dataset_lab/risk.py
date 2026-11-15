"""Bayesian Risk Minimization Module.
Implements Bayesian decision theory for multi-class, multi-action problems under arbitrary loss matrices:
- State of nature / classes: Omega = {omega_1, omega_2, omega_3}
- Actions: Alpha = {alpha_1, alpha_2, alpha_3}
- Loss matrix: Lambda = [lambda(alpha_i | omega_j)]
- Conditional risk: R(alpha_i | x) = sum_j lambda(alpha_i | omega_j) * P(omega_j | x)
- Bayes decision rule: alpha* = argmin_i R(alpha_i | x)
- Action trigger reflection in operational system state.
"""

# ---------------------------------------------------------------------------
# Default Configuration & Class / Action Definitions
# ---------------------------------------------------------------------------
CLASSES = [
    {
        "id": "omega_1",
        "symbol": "ω₁",
        "name": "Normal / Benign State",
        "description": "Nominal operational status, no critical defect present.",
        "color": "#164e3d"
    },
    {
        "id": "omega_2",
        "symbol": "ω₂",
        "name": "Critical Fault / High-Risk Pathology",
        "description": "Severe failure state requiring immediate intervention.",
        "color": "#b91c1c"
    },
    {
        "id": "omega_3",
        "symbol": "ω₃",
        "name": "Moderate Degradation / Suspicious Anomaly",
        "description": "Borderline condition warranting targeted audit.",
        "color": "#b45309"
    }
]

ACTIONS = [
    {
        "id": "alpha_1",
        "symbol": "α₁",
        "name": "Standard Release (Routine Operation)",
        "trigger_title": "ROUTINE DISPATCH & CLEARANCE TRIGGERED",
        "trigger_desc": "Automated pipeline approved. Workcell continues high-speed manufacturing without interruption.",
        "trigger_badge": "DISPATCH_NORMAL",
        "color": "#15803d",
        "light_color": "#dcfce7",
        "border_color": "#86efac"
    },
    {
        "id": "alpha_2",
        "symbol": "α₂",
        "name": "Emergency Halt & Critical Intervention",
        "trigger_title": "CRITICAL EMERGENCY SHUTDOWN TRIGGERED",
        "trigger_desc": "Immediate safety interlocks engaged. Process halted; engineering supervisor alert dispatched.",
        "trigger_badge": "EMERGENCY_HALT",
        "color": "#b91c1c",
        "light_color": "#fee2e2",
        "border_color": "#fca5a5"
    },
    {
        "id": "alpha_3",
        "symbol": "α₃",
        "name": "Escalate to Specialist Review / Deferral",
        "trigger_title": "SECONDARY DIAGNOSTIC AUDIT QUEUED",
        "trigger_desc": "Decision deferred. High-resolution telemetry routed to Tier-2 specialist review queue.",
        "trigger_badge": "SPECIALIST_AUDIT",
        "color": "#1d4ed8",
        "light_color": "#dbeafe",
        "border_color": "#93c5fd"
    }
]

# Standard Loss Matrix (Rows: Actions alpha_i, Columns: Classes omega_j)
# lambda_ij = Cost incurred taking action alpha_i when true class is omega_j
DEFAULT_LOSS_MATRIX = [
    # True omega_1 (Normal), True omega_2 (Critical), True omega_3 (Moderate)
    [0.0,   500.0,   60.0],  # alpha_1: Standard Release
    [100.0,   0.0,   40.0],  # alpha_2: Emergency Halt
    [20.0,   50.0,   10.0]   # alpha_3: Specialist Review
]

LOSS_PRESETS = {
    "asymmetric_critical": {
        "name": "Asymmetric Critical Safety (Heavy Miss Penalty)",
        "matrix": [
            [0.0, 500.0, 60.0],
            [100.0, 0.0, 40.0],
            [20.0, 50.0, 10.0]
        ],
        "description": "Severely penalizes releasing a critical defect (cost 500), encouraging cautious audit or halt."
    },
    "zero_one_loss": {
        "name": "Symmetric 0-1 Loss (Equivalent to Naive MAP)",
        "matrix": [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ],
        "description": "Equal penalty for all errors. Bayesian risk minimization reduces exactly to MAP decision rule."
    },
    "costly_downtime": {
        "name": "High-Cost Downtime (Conservative Intervention)",
        "matrix": [
            [0.0, 150.0, 30.0],
            [350.0, 0.0, 120.0],
            [25.0, 40.0, 15.0]
        ],
        "description": "High false-alarm shutdown penalty (cost 350); prioritizes targeted secondary review."
    }
}

# ---------------------------------------------------------------------------
# Test Scenarios / Samples
# ---------------------------------------------------------------------------
TEST_SCENARIOS = [
    {
        "id": "CASE-01",
        "name": "Clear Nominal Operating Telemetry",
        "posteriors": [0.85, 0.05, 0.10],
        "true_state": "omega_1"
    },
    {
        "id": "CASE-02",
        "name": "Subtle High-Risk Anomaly (Borderline Probabilities)",
        "posteriors": [0.65, 0.18, 0.17],
        "true_state": "omega_2"
    },
    {
        "id": "CASE-03",
        "name": "Acute Sensor Flare / Definite Critical Fault",
        "posteriors": [0.08, 0.82, 0.10],
        "true_state": "omega_2"
    },
    {
        "id": "CASE-04",
        "name": "Moderate Wear with High Uncertainty",
        "posteriors": [0.35, 0.15, 0.50],
        "true_state": "omega_3"
    },
    {
        "id": "CASE-05",
        "name": "Low-Confidence Diagnostic Signal",
        "posteriors": [0.42, 0.22, 0.36],
        "true_state": "omega_3"
    },
    {
        "id": "CASE-06",
        "name": "False-Positive Suspect Transient",
        "posteriors": [0.72, 0.12, 0.16],
        "true_state": "omega_1"
    }
]

# ---------------------------------------------------------------------------
# Computational Engine
# ---------------------------------------------------------------------------
def compute_conditional_risks(posteriors: list[float], loss_matrix=None):
    """
    Compute conditional risk R(alpha_i | x) = sum_j lambda(alpha_i | omega_j) * P(omega_j | x)
    for each candidate action.
    """
    if loss_matrix is None:
        loss_matrix = DEFAULT_LOSS_MATRIX

    # Normalize posteriors to guarantee sum == 1.0
    p_sum = sum(posteriors)
    if p_sum > 0:
        norm_post = [p / p_sum for p in posteriors]
    else:
        norm_post = [1.0 / len(posteriors)] * len(posteriors)

    risks = []
    for i, action in enumerate(ACTIONS):
        # Expected loss for action alpha_i
        r_i = sum(loss_matrix[i][j] * norm_post[j] for j in range(len(norm_post)))
        risks.append({
            "action_id": action["id"],
            "action_symbol": action["symbol"],
            "action_name": action["name"],
            "risk_value": float(r_i),
            "loss_row": loss_matrix[i],
            "color": action["color"]
        })

    # Find optimal action minimizing conditional risk: alpha* = argmin R(alpha_i | x)
    min_risk_idx = min(range(len(risks)), key=lambda idx: risks[idx]["risk_value"])
    optimal_action = ACTIONS[min_risk_idx]
    optimal_risk = risks[min_risk_idx]["risk_value"]

    # Also compute what a naive MAP rule would do for comparison
    map_idx = max(range(len(norm_post)), key=lambda idx: norm_post[idx])
    map_class = CLASSES[map_idx]

    return {
        "norm_posteriors": norm_post,
        "risks": risks,
        "optimal_action": optimal_action,
        "optimal_risk": optimal_risk,
        "map_class": map_class,
        "loss_matrix": loss_matrix,
        "triggered_state": {
            "action_id": optimal_action["id"],
            "title": optimal_action["trigger_title"],
            "description": optimal_action["trigger_desc"],
            "badge": optimal_action["trigger_badge"],
            "color": optimal_action["color"],
            "light_color": optimal_action["light_color"],
            "border_color": optimal_action["border_color"]
        }
    }


def evaluate_test_scenarios(loss_matrix=None):
    """Evaluate pre-configured test scenarios against current loss matrix."""
    if loss_matrix is None:
        loss_matrix = DEFAULT_LOSS_MATRIX

    results = []
    for sc in TEST_SCENARIOS:
        calc = compute_conditional_risks(sc["posteriors"], loss_matrix)
        results.append({
            "id": sc["id"],
            "name": sc["name"],
            "posteriors": sc["posteriors"],
            "true_state": sc["true_state"],
            "true_symbol": next(c["symbol"] for c in CLASSES if c["id"] == sc["true_state"]),
            "true_name": next(c["name"] for c in CLASSES if c["id"] == sc["true_state"]),
            "risks": [r["risk_value"] for r in calc["risks"]],
            "optimal_action": calc["optimal_action"]["symbol"],
            "optimal_action_name": calc["optimal_action"]["name"],
            "optimal_risk": calc["optimal_risk"],
            "triggered_badge": calc["triggered_state"]["badge"],
            "map_symbol": calc["map_class"]["symbol"],
            "map_name": calc["map_class"]["name"],
            "differs_from_map": (
                (calc["optimal_action"]["id"] == "alpha_1" and calc["map_class"]["id"] != "omega_1") or
                (calc["optimal_action"]["id"] == "alpha_2" and calc["map_class"]["id"] != "omega_2") or
                (calc["optimal_action"]["id"] == "alpha_3" and calc["map_class"]["id"] != "omega_3")
            )
        })
    return results


# ---------------------------------------------------------------------------
# Python Source Code for Page Display
# ---------------------------------------------------------------------------
PYTHON_RISK_MINIMIZATION_CODE = '''import numpy as np

# 1. Define States of Nature (Classes) & Actions
CLASSES = ["w1: Normal", "w2: Critical Fault", "w3: Moderate Wear"]
ACTIONS = [
    "a1: Routine Dispatch",
    "a2: Emergency Shutdown",
    "a3: Specialist Review"
]

# 2. Asymmetric Loss Matrix Lambda[i, j] = Loss of Action a_i given State w_j
# Row 0: a1 (Routine Dispatch)   -> Cost 0 if w1, Cost 500 if w2 (catastrophic!), Cost 60 if w3
# Row 1: a2 (Emergency Shutdown) -> Cost 100 if w1, Cost 0 if w2, Cost 40 if w3
# Row 2: a3 (Specialist Review)  -> Cost 20 if w1, Cost 50 if w2, Cost 10 if w3
LOSS_MATRIX = np.array([
    [0.0,   500.0,   60.0],
    [100.0,   0.0,   40.0],
    [20.0,   50.0,   10.0]
])

def bayesian_risk_minimization(posteriors, loss_matrix=LOSS_MATRIX):
    """
    Computes Bayesian Conditional Risk:
      R(a_i | x) = sum_j lambda(a_i | w_j) * P(w_j | x)
    Returns:
      optimal_action = argmin_i R(a_i | x)
    """
    P = np.array(posteriors) / np.sum(posteriors) # Normalize
    
    # Vectorized conditional risk computation: R = Lambda @ P
    risks = np.dot(loss_matrix, P)
    
    best_idx = np.argmin(risks)
    optimal_action = ACTIONS[best_idx]
    
    return optimal_action, risks[best_idx], risks

# --- Demonstration on Borderline Test Sample ---
# Posteriors: P(w1)=0.65 (Normal), P(w2)=0.18 (Critical Fault), P(w3)=0.17 (Moderate)
# Notice: Naive MAP would choose w1 (Normal, 65% probability).
# But Bayesian Risk considers the 500 cost of a missed critical fault!
posteriors = [0.65, 0.18, 0.17]
opt_act, min_risk, all_risks = bayesian_risk_minimization(posteriors)

print(f"Sample Posteriors: P(w1)={posteriors[0]:.2f}, P(w2)={posteriors[1]:.2f}, P(w3)={posteriors[2]:.2f}")
print(f"Naive MAP Choice:  w1 (Normal) with 65.0% probability\\n")

print("Computed Conditional Risks:")
for act, r in zip(ACTIONS, all_risks):
    print(f"  {act}: Expected Risk = {r:.2f}")

print(f"\\n--> Optimal Action Triggered: {opt_act} (Minimum Risk: {min_risk:.2f})")
if opt_act == "a3: Specialist Review":
    print("    [TRIGGER RESPONSE] System halts routine flow and routes sample to Human Tier-2 review.")
'''
