import json
from pathlib import Path

RESULTS_ROOT = Path("outputs/results")
REPORT = Path("report.md")


def find_latest_results():
    """Return the most recently modified results.json file."""
    results = sorted(
        RESULTS_ROOT.rglob("results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return results[0] if results else None


results_file = find_latest_results()

if results_file is None:
    REPORT.write_text(
        "# ML Report\n\nNo experiment results found.\n",
        encoding="utf-8",
    )
    print("No results.json found.")
    raise SystemExit(0)

with open(results_file, "r", encoding="utf-8") as f:
    data = json.load(f)

summary = data.get("summary", {})

bacc = summary.get("bacc_mean", "N/A")
std = summary.get("bacc_std", "N/A")
gate = summary.get("gate_passed", False)

report = f"""# Machine Learning Report

**Experiment**

`{results_file.parent.relative_to(RESULTS_ROOT)}`

## Results

| Metric | Value |
|--------|------:|
| Balanced Accuracy | {bacc if isinstance(bacc, str) else f"{bacc:.3f}"} |
| Std | {std if isinstance(std, str) else f"{std:.3f}"} |
| Gate | {"✅ PASS" if gate else "❌ FAIL"} |

---
Generated automatically by CML.
"""

REPORT.write_text(report, encoding="utf-8")

print(f"Report generated from: {results_file}")