"""
Generate a comprehensive scientific report from the latest results.json.

Usage:
  py scripts/generate_report.py                          # uses latest results.json
  py scripts/generate_report.py --path path/to/results.json
  py scripts/generate_report.py --gate 0.65              # custom gate threshold

Output: report.md
"""
import json
import sys
import argparse
from pathlib import Path

RESULTS_ROOT = Path("outputs/results")
REPORT = Path("report.md")


def find_latest_results():
    results = sorted(
        RESULTS_ROOT.rglob("results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return results[0] if results else None


def _fmt(v, decimals=3):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _fmt_pair(m, s, decimals=3):
    if m is None or s is None:
        return "N/A"
    return f"{m:.{decimals}f} +- {s:.{decimals}f}"


def _maybe_bool(v):
    if v is None or v is False:
        return "No"
    return "Yes"


def _subj_list_str(cods, max_display=12):
    if not cods:
        return "N/A"
    s = ", ".join(cods[:max_display])
    if len(cods) > max_display:
        s += f", ... ({len(cods)} total)"
    return s


def generate_report(results_path, gate_threshold=0.65):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    exp = data.get("experiment", {})
    dset = data.get("data", {})
    cfg = data.get("config", {})
    val = data.get("validation", {})
    tst = data.get("test", {})
    folds = data.get("folds", [])
    final = data.get("final_model")

    config_name = data.get("config_name", "?")
    git_commit = exp.get("git_commit", "?")
    has_new_format = bool(cfg.get("fusion"))

    val_bacc = val.get("bacc_mean")
    val_bacc_s = val.get("bacc_std")
    gate_passed = val_bacc is not None and val_bacc >= gate_threshold

    lines = []
    lines.append(f"# Crossmodal MDD Report: {config_name}")
    lines.append("")

    # ── Experiment ──
    lines.append("## Experiment")
    lines.append("")
    if has_new_format:
        lines.append(f"- Model: {exp.get('name', '?')}")
        lines.append(f"- Fusion: {cfg.get('fusion', '?')}")
        lines.append(f"- Hidden dimension: {cfg.get('hidden', '?')}")
        lines.append(f"- Attention heads: {cfg.get('n_heads', '?')}")
        lines.append(f"- Dropout: {cfg.get('dropout', '?')}")
        lines.append(f"- Max windows per subject: {cfg.get('max_windows', '?')}")
        lines.append(f"- Self-attention layers: {cfg.get('n_self_attn_layers', 0)}")
        lines.append(f"- Pooling: {cfg.get('pooling', 'mean')}")
        if cfg.get('bottleneck_dim'):
            lines.append(f"- Bottleneck dim: {cfg.get('bottleneck_dim')}")
        lines.append(f"- Backbones: {cfg.get('backbone_eeg', '?')} (EEG) / {cfg.get('backbone_aud', '?')} (Audio)")
        lines.append(f"- Augmentation: {_maybe_bool(cfg.get('augment', False))}")
        lines.append(f"- Backbone lr/wd: {cfg.get('backbone_lr', '?')} / {cfg.get('backbone_wd', '?')}")
        lines.append(f"- Fusion lr/wd: {cfg.get('fusion_lr', '?')} / {cfg.get('fusion_wd', '?')}")
        lines.append(f"- Subjects: {dset.get('n_eeg', '?')} EEG, {dset.get('n_audio', '?')} Audio, "
                     f"{dset.get('n_paired', '?')} paired "
                     f"({dset.get('n_mdd_paired', '?')} MDD / {dset.get('n_hc_paired', '?')} HC)")
        lines.append(f"- Folds: {dset.get('n_folds', '?')}")
    else:
        lines.append(f"- Config: {config_name}")
        lines.append(f"- Git commit: `{git_commit}`")
        lines.append(f"- Timestamp: {exp.get('timestamp', '?')}")
    lines.append(f"- Seed: {exp.get('seed', '?')}")
    lines.append(f"- Git commit: `{git_commit}`")
    lines.append(f"- Timestamp: {exp.get('timestamp', '?')}")
    lines.append("")

    # ── Subject Splits per Fold ──
    has_subject_splits = any(
        'eeg_backbone_train_cods' in fr for fr in folds
    )
    if folds and has_subject_splits:
        lines.append("## Subject Splits per Fold")
        lines.append("")
        lines.append("Each fold has three nested training processes: ")
        lines.append("(1) EEG backbone, (2) Audio backbone, (3) Fusion head. ")
        lines.append("All three have their own inner train/val split. The test set is held out until final evaluation.")
        lines.append("")
        for fr in folds:
            fi = fr.get("fold", "?")
            lines.append(f"### Fold {fi}")
            lines.append("")
            lines.append("| Process | Split | n | Subjects |")
            lines.append("|---------|-------|---|----------|")
            eeg_tr = fr.get('eeg_backbone_train_cods', [])
            eeg_vl = fr.get('eeg_backbone_val_cods', [])
            aud_tr = fr.get('aud_backbone_train_cods', [])
            aud_vl = fr.get('aud_backbone_val_cods', [])
            fuse_tr = fr.get('fusion_train_subjects', [])
            fuse_vl = fr.get('fusion_val_subjects', [])
            test_s = fr.get('test_subjects', [])

            lines.append(f"| EEG Backbone | Train | {len(eeg_tr)} | {_subj_list_str(eeg_tr)} |")
            lines.append(f"| EEG Backbone | Val | {len(eeg_vl)} | {_subj_list_str(eeg_vl)} |")
            lines.append(f"| Audio Backbone | Train | {len(aud_tr)} | {_subj_list_str(aud_tr)} |")
            lines.append(f"| Audio Backbone | Val | {len(aud_vl)} | {_subj_list_str(aud_vl)} |")
            lines.append(f"| Fusion Head | Train | {len(fuse_tr)} | {_subj_list_str(fuse_tr)} |")
            lines.append(f"| Fusion Head | Val | {len(fuse_vl)} | {_subj_list_str(fuse_vl)} |")
            lines.append(f"| **Test** (held-out) | - | {len(test_s)} | **{_subj_list_str(test_s)}** |")
            lines.append("")

    # ── Validation (Nested CV) ──
    if val and val.get("bacc_mean") is not None:
        lines.append("## Validation (Nested CV)")
        lines.append("")
        lines.append("| Metric | Mean +- Std |")
        lines.append("|--------|-----------:|")
        lines.append(f"| Balanced Accuracy | {_fmt_pair(val.get('bacc_mean'), val.get('bacc_std'))} |")
        lines.append(f"| Accuracy | {_fmt_pair(val.get('acc_mean'), val.get('acc_std'))} |")
        lines.append(f"| F1-score | {_fmt_pair(val.get('f1_mean'), val.get('f1_std'))} |")
        lines.append(f"| Sensitivity | {_fmt_pair(val.get('sens_mean'), val.get('sens_std'))} |")
        lines.append(f"| Specificity | {_fmt_pair(val.get('spec_mean'), val.get('spec_std'))} |")
        if val.get("auc_mean") is not None:
            lines.append(f"| ROC AUC | {_fmt_pair(val.get('auc_mean'), val.get('auc_std'))} |")
        lines.append("")

    # ── Test ──
    if tst and tst.get("bacc_mean") is not None:
        lines.append("## Test (held-out per fold)")
        lines.append("")
        lines.append("| Metric | Mean +- Std |")
        lines.append("|--------|:----------:|")
        lines.append(f"| Balanced Accuracy | {_fmt_pair(tst.get('bacc_mean'), tst.get('bacc_std'))} |")
        lines.append(f"| Accuracy | {_fmt_pair(tst.get('acc_mean'), tst.get('acc_std'))} |")
        lines.append(f"| F1-score | {_fmt_pair(tst.get('f1_mean'), tst.get('f1_std'))} |")
        lines.append(f"| Sensitivity | {_fmt_pair(tst.get('sens_mean'), tst.get('sens_std'))} |")
        lines.append(f"| Specificity | {_fmt_pair(tst.get('spec_mean'), tst.get('spec_std'))} |")
        lines.append(f"| ROC AUC | {_fmt_pair(tst.get('auc_mean'), tst.get('auc_std'))} |")
        lines.append("")

    # ── Per-Fold Metrics ──
    if folds:
        lines.append("## Per-Fold Breakdown")
        lines.append("")
        hdr = ("| Fold | Val BACC | Val F1 | Test BACC | Test AUC | "
               "Test ACC | Test F1 | n_train | n_test | Test Subjects |")
        lines.append(hdr)
        lines.append("|:----:|:--------:|:-----:|:---------:|:--------:|"
                      ":--------:|:------:|:-------:|:------:|:-------------|")
        for fr in folds:
            fi = fr.get("fold", "?")
            vb = _fmt(fr.get("best_val_bacc"), 3)
            vf = _fmt(fr.get("best_val_f1"), 3)
            tb = _fmt(fr.get("test_bacc"), 3)
            ta = _fmt(fr.get("test_auc"), 3)
            tc = _fmt(fr.get("test_acc"), 3)
            ts = _fmt(fr.get("test_sens"), 3)
            nt = fr.get("n_train_paired", "?")
            ne = fr.get("n_test", "?")
            te_subjs = ", ".join(fr.get("test_subjects", [])[:5])
            lines.append(f"| {fi} | {vb} | {vf} | {tb} | {ta} | {tc} | {ts} | {nt} | {ne} | {te_subjs} |")
        lines.append("")

        lines.append(f"**Per-Fold Test Summary** &mdash; "
                     f"BACC {_fmt_pair(tst.get('bacc_mean'), tst.get('bacc_std'))}, "
                     f"AUC {_fmt_pair(tst.get('auc_mean'), tst.get('auc_std'))}, "
                     f"F1 {_fmt_pair(tst.get('f1_mean'), tst.get('f1_std'))}")
        lines.append("")

    # ── Final Model ──
    if final is not None:
        lines.append("## Final Model")
        lines.append("")
        lines.append("Trained on 100% of paired data for deployment. Metrics are descriptive (no held-out).")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Balanced Accuracy | {_fmt(final.get('bacc'))} |")
        lines.append(f"| Accuracy | {_fmt(final.get('acc'))} |")
        lines.append(f"| F1-score | {_fmt(final.get('f1'))} |")
        lines.append(f"| Sensitivity | {_fmt(final.get('sens'))} |")
        lines.append(f"| Specificity | {_fmt(final.get('spec'))} |")
        lines.append(f"| ROC AUC | {_fmt(final.get('auc'))} |")
        lines.append("")
        lines.append(f"Model saved at: `{final.get('model_path', 'N/A')}`")
        lines.append("")

    # ── Gate ──
    lines.append("## Gate Check")
    lines.append("")
    gate_str = "PASS" if gate_passed else "FAIL"
    lines.append(f"Balanced Accuracy ({_fmt(val_bacc)} / {gate_threshold}) &mdash; **{gate_str}**")
    lines.append("")

    lines.append("---")
    lines.append("Generated automatically by CML.")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report generated from: {results_path}")
    print(f"Val BACC: {_fmt(val_bacc)} +- {_fmt(val_bacc_s)} -- Gate: {gate_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scientific ML report")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to results.json (default: latest)")
    parser.add_argument("--gate", type=float, default=0.65,
                        help="Gate threshold for BACC (default: 0.65)")
    args = parser.parse_args()

    if args.path:
        results_path = Path(args.path)
        if not results_path.exists():
            print(f"File not found: {results_path}")
            sys.exit(1)
    else:
        results_path = find_latest_results()
        if results_path is None:
            REPORT.write_text(
                "# Crossmodal MDD Report\n\nNo experiment results found.\n",
                encoding="utf-8",
            )
            print("No results.json found.")
            sys.exit(0)

    generate_report(results_path, gate_threshold=args.gate)