import urllib.request
import json

def check(url, method="GET", data=None):
    req = urllib.request.Request(url, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
            print(f"[{resp.status}] {method} {url} -> Length {len(body)}")
            return resp.status, body
    except Exception as e:
        print(f"[FAIL] {method} {url} -> {e}")
        return None, str(e)

print("--- Testing Bayesian Risk Minimization ---")
s, b = check("http://127.0.0.1:5000/experiments/bayesian-risk-minimization")
if b and "Bayesian Risk" in b:
    print(" Experiment 05 page rendered successfully!")

s2, b2 = check(
    "http://127.0.0.1:5000/api/risk/calculate",
    method="POST",
    data={
        "p1": 0.65, "p2": 0.18, "p3": 0.17,
        "loss_matrix": [
            [0.0, 500.0, 60.0],
            [100.0, 0.0, 40.0],
            [20.0, 50.0, 10.0]
        ]
    }
)
if b2:
    parsed = json.loads(b2)
    print(" API Output Success:", parsed.get("status"))
    print("  Optimal Action:", parsed["calc"]["optimal_action"]["name"])
    print("  Triggered Badge:", parsed["calc"]["triggered_state"]["badge"])
    print("  Minimum Risk:", parsed["calc"]["optimal_risk"])

print("--- Checking All Endpoints ---")
for exp in [
    "dataset-explorer",
    "multimodal-feature-extraction",
    "bayesian-classification",
    "text-nlp-bayesian",
    "bayesian-risk-minimization"
]:
    check(f"http://127.0.0.1:5000/experiments/{exp}")

check("http://127.0.0.1:5000/")
