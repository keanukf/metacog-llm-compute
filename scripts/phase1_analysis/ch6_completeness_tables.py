#!/usr/bin/env python3
"""Descriptive tables for Chapter 6 completeness (deep-check Batch 1: A3, C5, C6, C7, A12).

No bootstrap, no GPU: stage manipulation-check rates, signal descriptives (VC support and
concentration, TLE by stage, TLE-VC association), per-cell signal-missingness, evaluable N, and
the full H3 GEE coefficient table with 90% Wald CIs. Read-only.

Run from repo root:
  python scripts/phase1_analysis/ch6_completeness_tables.py
"""
from __future__ import annotations

import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.inference import build_h3_frame  # noqa: E402
from src.analysis.phase1_canonical import (  # noqa: E402
    TEXTWORLD_TRUE_HOLDOUT_INSTANCES,
    apply_textworld_holdout_correction,
    load_canonical_dataset_from_manifest,
)

DOMS = ("tower_of_hanoi", "textworld")
STG = ("C0", "C1", "C2")


def _load():
    ds = load_canonical_dataset_from_manifest(
        REPO_ROOT / "data/results/phase1_analysis/stage0/canonical_manifest.json"
    )
    apply_textworld_holdout_correction(ds.steps, TEXTWORLD_TRUE_HOLDOUT_INSTANCES)
    return ds.steps


def c5_manipulation(S):
    print("\n===== C5  Stage manipulation check: step optimal-rate by domain x stage =====")
    for dom in DOMS:
        for s in STG:
            ys = [int(r["y_optimal"]) for r in S
                  if r.get("domain") == dom and str(r.get("compute_stage")) == s
                  and r.get("y_optimal") is not None]
            print(f"  {dom:15s} {s}: n={len(ys):6d}  optimal_rate={sum(ys)/len(ys):.3f}")


def c7_missing(S):
    print("\n===== C7  Signal missingness by domain x stage (truncation proxy) =====")
    for dom in DOMS:
        for s in STG:
            rows = [r for r in S if r.get("domain") == dom and str(r.get("compute_stage")) == s
                    and r.get("y_optimal") is not None]
            present = [r for r in rows if r.get("tle_mean_entropy") is not None and r.get("vc") is not None]
            n = len(rows)
            miss = n - len(present)
            print(f"  {dom:15s} {s}: labeled={n:6d}  signal_present={len(present):6d}  missing_rate={miss/n:.3f}")


def c6_signal(S):
    print("\n===== C6  Signal descriptives =====")
    for dom in DOMS:
        vc = [int(r["vc"]) for r in S if r.get("domain") == dom and r.get("vc") is not None]
        dv = sorted(set(vc))
        c = Counter(vc); tot = sum(c.values())
        modal, modal_n = c.most_common(1)[0]
        print(f"\n  {dom}:")
        print(f"    VC: n={len(vc)}, distinct values={len(dv)}, range=[{min(dv)},{max(dv)}]")
        print(f"        modal value={modal} (share {modal_n/tot:.3f}), VC==100 share={c.get(100,0)/tot:.3f}")
        print(f"    TLE mean/median by stage:")
        for s in STG:
            t = [float(r["tle_mean_entropy"]) for r in S if r.get("domain") == dom
                 and str(r.get("compute_stage")) == s and r.get("tle_mean_entropy") is not None]
            print(f"        {s}: n={len(t):6d}  mean={st.mean(t):.4f}  median={st.median(t):.6f}")
        # TLE-VC association on paired present steps
        pairs = [(float(r["tle_mean_entropy"]), float(r["vc"])) for r in S if r.get("domain") == dom
                 and r.get("tle_mean_entropy") is not None and r.get("vc") is not None]
        xs = [a for a, _ in pairs]; ys = [b for _, b in pairs]
        import numpy as np
        pear = float(np.corrcoef(xs, ys)[0, 1])
        try:
            from scipy.stats import spearmanr
            spear = float(spearmanr(xs, ys).statistic)
        except Exception as e:
            spear = float("nan"); print("    (spearman unavailable:", e, ")")
        print(f"    corr(TLE, VC): Pearson={pear:+.3f}  Spearman={spear:+.3f}  (n={len(pairs)})")


def a12_n(S):
    print("\n===== A12  Evaluable N by domain (for H1a/H1b/H4 table N columns) =====")
    for dom in DOMS:
        rows = [r for r in S if r.get("domain") == dom]
        inst = set(r.get("instance_key") for r in rows)
        evalable = [r for r in rows if r.get("y_optimal") is not None
                    and r.get("tle_mean_entropy") is not None and r.get("vc") is not None]
        nonh = [r for r in evalable if not bool(r.get("holdout"))]
        print(f"  {dom:15s} instances={len(inst)}  evaluable_steps={len(evalable)}  non_holdout_eval={len(nonh)}")


def a3_h3(S):
    print("\n===== A3  Full H3 GEE coefficients with 90% Wald CIs =====")
    import statsmodels.api as sm
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    for dom in DOMS:
        for sig in ("tle", "vc"):
            frame, note = build_h3_frame(S, signal=sig, domain=dom)
            if frame is None:
                print(f"  {dom}/{sig}: NA ({note})")
                continue
            frame["p_c"] = frame["position_norm"] - frame["position_norm"].mean()
            design = sm.add_constant(
                frame[["z_c", "p_c"]].assign(interaction=frame["z_c"] * frame["p_c"])
            )
            res = sm.GEE(frame["y"], design, groups=frame["g"],
                         family=Binomial(), cov_struct=Exchangeable()).fit()
            ci = res.conf_int(alpha=0.10)
            print(f"\n  {dom}/{sig.upper()} (n={len(frame)}, position centred at mean; signal z-std within stage):")
            for name in ("const", "z_c", "p_c", "interaction"):
                b = res.params[name]; lo, hi = ci.loc[name]
                print(f"    {name:11s} = {b:+.4f}  [90% CI {lo:+.4f}, {hi:+.4f}]  p={res.pvalues[name]:.4g}")


def main() -> int:
    S = _load()
    c5_manipulation(S)
    c6_signal(S)
    c7_missing(S)
    a12_n(S)
    a3_h3(S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
