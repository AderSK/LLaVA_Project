"""
compute_concept_importance.py

Compute a concept-importance table that compares the conditional rate of each
trained probe concept across two subsets of images:
   - those LLaVA labelled as containing an arbitrary concept (positive set)
   - those LLaVA labelled as NOT containing it (negative set)

For every probe concept c, we compute the FORWARD direction:
    p_with    = P(probe c == 1  |  arbitrary == 1)        in the positive set
    p_without = P(probe c == 1  |  arbitrary == 0)        in the negative set
    diff      = p_with - p_without

…and the BACKWARD direction:
    p_arb_given_probe    = P(arbitrary == 1  |  probe c == 1)
    p_arb_given_no_probe = P(arbitrary == 1  |  probe c == 0)
    diff_back            = p_arb_given_probe - p_arb_given_no_probe

The two directions answer different questions:
    forward:  "When the arbitrary concept is present, how often does probe c fire?"
    backward: "When probe c fires, how often is the arbitrary concept present?"
The two diffs differ in magnitude in general (different denominators), but
phi and the chi-squared p-value are symmetric and apply to both directions.

Interpretation:
    diff ≈ 0          c is unrelated to the arbitrary concept (no importance).
    diff > 0  large   c tends to co-occur with the arbitrary concept.
    diff < 0  large   c tends to be ABSENT when the arbitrary concept is present.
    (analogous reading for diff_back, with the conditioning variable swapped)

The script also reports:
    - 95% Wilson confidence intervals on each rate
    - φ (phi) coefficient — signed effect size, comparable across probes
    - chi-squared (with Yates correction) p-value — significance of the difference

Inputs:
    --arbitrary_labels   JSON from label_arbitrary_concept.py
    --probe_predictions  JSON from apply_probes.py

Outputs (written next to --output prefix):
    <prefix>.json   full structured results
    <prefix>.csv    flat table for spreadsheets

Usage:
    python compute_concept_importance.py \
        --arbitrary_labels beach_labels.json \
        --probe_predictions probe_predictions.json \
        --output beach_importance
"""

import argparse
import json
import csv
import math
from pathlib import Path


# ─── Statistics ──────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """95% Wilson score confidence interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def chi_squared_2x2(a, b, c, d):
    """
    Yates-corrected chi-squared test on a 2x2 contingency table.
        Layout:
                          probe = 1   probe = 0
            arbitrary=1       a           b
            arbitrary=0       c           d

    Returns (chi2_statistic, p_value, signed_phi).

    The p-value is computed without scipy: for chi-squared with df=1,
        P(X >= x) = erfc(sqrt(x / 2))
    (this follows because chi2_{df=1} is the square of a standard normal).

    phi and chi2 (and the p-value) are direction-symmetric: they describe the
    strength of association in the 2x2 table regardless of which variable we
    condition on.
    """
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0, 0.0

    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if min(row1, row2, col1, col2) == 0:
        return 0.0, 1.0, 0.0

    expected = [
        row1 * col1 / n,
        row1 * col2 / n,
        row2 * col1 / n,
        row2 * col2 / n,
    ]
    observed = [a, b, c, d]

    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            adj = max(0.0, abs(o - e) - 0.5)  # Yates continuity correction
            chi2 += (adj * adj) / e

    p_value = math.erfc(math.sqrt(chi2 / 2)) if chi2 >= 0 else 1.0

    phi = math.sqrt(chi2 / n) if n > 0 else 0.0
    # Sign: positive when (arb=1, probe=1) and (arb=0, probe=0) dominate
    if a * d < b * c:
        phi = -phi

    return chi2, p_value, phi


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def index_by_name(d):
    """Re-key a dict-of-entries by image_name so files from different working
    directories can be joined reliably."""
    out = {}
    for entry in d.values():
        name = entry.get("image_name")
        if name:
            out[name] = entry
    return out


def fmt_pvalue(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def significance_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute concept-importance table from arbitrary-concept "
                    "labels and probe predictions."
    )
    parser.add_argument("--arbitrary_labels", required=True,
                        help="JSON from label_arbitrary_concept.py")
    parser.add_argument("--probe_predictions", required=True,
                        help="JSON from apply_probes.py")
    parser.add_argument("--output", default="importance",
                        help="Output prefix; writes <prefix>.json and <prefix>.csv")
    args = parser.parse_args()

    with open(args.arbitrary_labels) as f:
        arb_data = json.load(f)
    with open(args.probe_predictions) as f:
        probe_data = json.load(f)

    # Detect arbitrary concept name from first valid entry
    arb_concept = None
    for entry in arb_data.values():
        if entry.get("labels"):
            arb_concept = next(iter(entry["labels"].keys()))
            break
    if arb_concept is None:
        print("No valid labeled entries in arbitrary_labels file.")
        return
    print(f"Arbitrary concept: '{arb_concept}'")

    # Join by image_name (more reliable than image_path which can differ
    # between machines / working directories)
    arb_by_name = index_by_name(arb_data)
    probe_by_name = index_by_name(probe_data)
    common = sorted(set(arb_by_name) & set(probe_by_name))
    if not common:
        print("No images appear in BOTH files. Check that image_name fields match.")
        return

    n_arb_only = len(arb_by_name) - len(common)
    n_probe_only = len(probe_by_name) - len(common)
    print(f"Joined on image_name: {len(common)} matching images "
          f"({n_arb_only} only in labels, {n_probe_only} only in predictions)")

    # Filter to entries with valid label and valid predictions
    valid = []
    for name in common:
        ae = arb_by_name[name]
        pe = probe_by_name[name]
        if ae.get("labels") is None or "predictions" not in pe:
            continue
        valid.append((name, ae["labels"][arb_concept], pe["predictions"]))

    n_total = len(valid)
    n_pos = sum(1 for _, l, _ in valid if l == 1)
    n_neg = n_total - n_pos
    print(f"Valid joined images: {n_total}")
    print(f"  {arb_concept} = YES: {n_pos}  ({100*n_pos/max(1,n_total):.1f}%)")
    print(f"  {arb_concept} = NO : {n_neg}  ({100*n_neg/max(1,n_total):.1f}%)")

    if n_pos == 0 or n_neg == 0:
        print(f"\n[error] One subset is empty — cannot compute differences.")
        return

    # Probe concepts present in the data
    probe_concepts = sorted({c for _, _, preds in valid for c in preds.keys()})
    print(f"Probe concepts: {probe_concepts}\n")

    # Compute one row per probe concept
    rows = []
    for c in probe_concepts:
        # 2x2 contingency
        #                  probe=1    probe=0    | row sum
        #     arbitrary=1     a          b       | a+b   (n_with_arbitrary)
        #     arbitrary=0     cc         d       | cc+d  (n_without_arbitrary)
        #     col sum       a+cc       b+d       | total
        #              (n_probe_pos) (n_probe_neg)
        a = sum(1 for _, lbl, p in valid if lbl == 1 and p[c]["label"] == 1)
        b = sum(1 for _, lbl, p in valid if lbl == 1 and p[c]["label"] == 0)
        cc = sum(1 for _, lbl, p in valid if lbl == 0 and p[c]["label"] == 1)
        d = sum(1 for _, lbl, p in valid if lbl == 0 and p[c]["label"] == 0)

        # Forward direction: P(probe | arbitrary)
        p_with = a / (a + b) if (a + b) > 0 else 0.0
        p_without = cc / (cc + d) if (cc + d) > 0 else 0.0
        diff = p_with - p_without

        with_lo, with_hi = wilson_ci(a, a + b)
        without_lo, without_hi = wilson_ci(cc, cc + d)

        # Backward direction: P(arbitrary | probe)
        # Among probe=1: a (arb=1), cc (arb=0). Total = a + cc.
        # Among probe=0: b (arb=1), d  (arb=0). Total = b + d.
        p_arb_given_probe = a / (a + cc) if (a + cc) > 0 else 0.0
        p_arb_given_no_probe = b / (b + d) if (b + d) > 0 else 0.0
        diff_back = p_arb_given_probe - p_arb_given_no_probe

        arb_given_probe_lo, arb_given_probe_hi = wilson_ci(a, a + cc)
        arb_given_no_probe_lo, arb_given_no_probe_hi = wilson_ci(b, b + d)

        # phi, chi2 and p-value are symmetric in the 2x2 table — one set covers
        # both directions.
        chi2, pval, phi = chi_squared_2x2(a, b, cc, d)

        rows.append({
            "probe_concept": c,
            # marginals
            "n_with_arbitrary": a + b,
            "n_without_arbitrary": cc + d,
            "n_probe_pos": a + cc,
            "n_probe_neg": b + d,
            # cell counts
            "n_probe_pos_in_with": a,
            "n_probe_pos_in_without": cc,
            "n_arb_pos_in_probe_pos": a,
            "n_arb_pos_in_probe_neg": b,
            # forward direction: P(probe | arbitrary)
            "p_with": p_with,
            "p_with_ci_low": with_lo,
            "p_with_ci_high": with_hi,
            "p_without": p_without,
            "p_without_ci_low": without_lo,
            "p_without_ci_high": without_hi,
            "diff": diff,
            "abs_diff": abs(diff),
            # backward direction: P(arbitrary | probe)
            "p_arb_given_probe": p_arb_given_probe,
            "p_arb_given_probe_ci_low": arb_given_probe_lo,
            "p_arb_given_probe_ci_high": arb_given_probe_hi,
            "p_arb_given_no_probe": p_arb_given_no_probe,
            "p_arb_given_no_probe_ci_low": arb_given_no_probe_lo,
            "p_arb_given_no_probe_ci_high": arb_given_no_probe_hi,
            "diff_back": diff_back,
            "abs_diff_back": abs(diff_back),
            # symmetric statistics (apply to both directions)
            "phi_coefficient": phi,
            "chi2": chi2,
            "p_value": pval,
        })

    rows.sort(key=lambda r: r["abs_diff"], reverse=True)

    # ─── Forward table: P(probe | arbitrary) ──────────────────────────────
    print(f"FORWARD direction — P(probe concept | '{arb_concept}')")
    print(f"  positive set n = {n_pos}    negative set n = {n_neg}\n")

    header = (f"  {'probe concept':<16} "
              f"{'p(with)':>14}   {'p(without)':>14}   "
              f"{'diff':>8}  {'phi':>7}  {'p-value':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in rows:
        with_str = f"{r['p_with']*100:5.1f}% [{r['p_with_ci_low']*100:.0f}-{r['p_with_ci_high']*100:.0f}]"
        without_str = f"{r['p_without']*100:5.1f}% [{r['p_without_ci_low']*100:.0f}-{r['p_without_ci_high']*100:.0f}]"
        stars = significance_stars(r["p_value"])
        print(f"  {r['probe_concept']:<16} "
              f"{with_str:>14}   {without_str:>14}   "
              f"{r['diff']*100:>+7.1f}%  "
              f"{r['phi_coefficient']:>+7.3f}  "
              f"{fmt_pvalue(r['p_value']):>9} {stars}")

    print(f"\n  CIs are 95% Wilson intervals.  Significance: *** p<0.001  ** p<0.01  * p<0.05")
    print(f"  diff > 0 → probe concept appears MORE often when '{arb_concept}' is present")
    print(f"  diff < 0 → probe concept appears LESS often when '{arb_concept}' is present")
    print(f"  |phi| ≈ 0.1 small,  ≈ 0.3 medium,  ≈ 0.5 large effect (Cohen)")

    # ─── Backward table: P(arbitrary | probe) ─────────────────────────────
    print(f"\n\nBACKWARD direction — P('{arb_concept}' | probe concept)")
    print(f"  probe-pos / probe-neg counts vary per concept\n")

    header2 = (f"  {'probe concept':<16} "
               f"{'n(probe +/-)':>13}   "
               f"{'p(arb|probe)':>16}   {'p(arb|no probe)':>18}   "
               f"{'diff_back':>10}")
    print(header2)
    print("  " + "-" * (len(header2) - 2))

    for r in rows:
        n_str = f"{r['n_probe_pos']}/{r['n_probe_neg']}"
        ap_str = (f"{r['p_arb_given_probe']*100:5.1f}% "
                  f"[{r['p_arb_given_probe_ci_low']*100:.0f}-"
                  f"{r['p_arb_given_probe_ci_high']*100:.0f}]")
        anp_str = (f"{r['p_arb_given_no_probe']*100:5.1f}% "
                   f"[{r['p_arb_given_no_probe_ci_low']*100:.0f}-"
                   f"{r['p_arb_given_no_probe_ci_high']*100:.0f}]")
        stars = significance_stars(r["p_value"])
        print(f"  {r['probe_concept']:<16} "
              f"{n_str:>13}   "
              f"{ap_str:>16}   {anp_str:>18}   "
              f"{r['diff_back']*100:>+9.1f}% {stars}")

    print(f"\n  diff_back > 0 → '{arb_concept}' appears MORE often when probe concept is present")
    print(f"  diff_back < 0 → '{arb_concept}' appears LESS often when probe concept is present")
    print(f"  Note: phi, chi2 and p-value are direction-symmetric (same as the forward table).")

    # Save JSON
    out_prefix = Path(args.output)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")

    with open(json_path, "w") as f:
        json.dump({
            "arbitrary_concept": arb_concept,
            "n_total": n_total,
            "n_with_arbitrary": n_pos,
            "n_without_arbitrary": n_neg,
            "rows": rows,
        }, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "probe_concept",
            # forward
            "p_with", "p_with_ci_low", "p_with_ci_high",
            "p_without", "p_without_ci_low", "p_without_ci_high",
            "diff",
            # backward
            "p_arb_given_probe", "p_arb_given_probe_ci_low", "p_arb_given_probe_ci_high",
            "p_arb_given_no_probe", "p_arb_given_no_probe_ci_low", "p_arb_given_no_probe_ci_high",
            "diff_back",
            # symmetric
            "phi", "chi2", "p_value",
            # counts
            "n_probe_pos_in_with", "n_with_arbitrary",
            "n_probe_pos_in_without", "n_without_arbitrary",
            "n_arb_pos_in_probe_pos", "n_probe_pos",
            "n_arb_pos_in_probe_neg", "n_probe_neg",
        ])
        for r in rows:
            writer.writerow([
                r["probe_concept"],
                r["p_with"], r["p_with_ci_low"], r["p_with_ci_high"],
                r["p_without"], r["p_without_ci_low"], r["p_without_ci_high"],
                r["diff"],
                r["p_arb_given_probe"], r["p_arb_given_probe_ci_low"], r["p_arb_given_probe_ci_high"],
                r["p_arb_given_no_probe"], r["p_arb_given_no_probe_ci_low"], r["p_arb_given_no_probe_ci_high"],
                r["diff_back"],
                r["phi_coefficient"], r["chi2"], r["p_value"],
                r["n_probe_pos_in_with"], r["n_with_arbitrary"],
                r["n_probe_pos_in_without"], r["n_without_arbitrary"],
                r["n_arb_pos_in_probe_pos"], r["n_probe_pos"],
                r["n_arb_pos_in_probe_neg"], r["n_probe_neg"],
            ])
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()
