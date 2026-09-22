"""
Seed replication analysis -> eval/seed_replication_results.md

Read-only on everything it reads. The output is deterministic: the date in the
header is the commit date of the last run, not the current time, so rerunning
reproduces the committed file byte for byte.

    venv/bin/python scripts/analyze_seed_replication.py            write (refuses to overwrite)
    venv/bin/python scripts/analyze_seed_replication.py --verify   regenerate in memory and diff
                                                                   against the existing file
    venv/bin/python scripts/analyze_seed_replication.py --out PATH write somewhere else

Inputs: the six run commits (results + manifests), eval/stage6_1_official_eval.md,
and — local, gitignored — each run's best checkpoint, training.csv and per_step.csv,
plus logs/transformer_6_1_grad_norms.csv.

Hard gate: nothing is computed or written unless all six run commits exist and
verify (4 files, completed, clean-tree launch, final step 10000, both data hashes).

Verdict rule (applied mechanically):
  ROBUST       the two configs' loss ranges do not overlap AND
               (mean baseline - mean 6.1) >= 3 x the larger within-config sample SD
  INCONCLUSIVE otherwise
"""

import argparse
import csv
import difflib
import json
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "eval" / "seed_replication_results.md"
RUNS = ["sr_base20mb_s1337", "sr_data200mb_s1337", "sr_base20mb_s1338",
        "sr_data200mb_s1338", "sr_base20mb_s1339", "sr_data200mb_s1339"]
SHA = {"base": "d4b45f4be5fb1cba325ac7f08fdbeefc007b2426a86d3fe7012b85edfdf5f1c8",
       "data": "5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb"}
VAL_SHA = "9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e"
ROW = re.compile(r"\|\s*\*\*this transformer[^|]*\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|\s*\*\*([\d.]+)\*\*")
ORIGINAL_CLAIM = {"full": -10.4, "clean": -10.5}      # eval/stage6_1_comparison.md
THRESHOLD = 3.0
SPIKE_RATIO = 10.0                                     # same rule as train_transformer.py
EARLY_STEP = 30


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout


def config_of(run):
    return "base" if "base20mb" in run else "data"


def official(path):
    """(loss, ppl, tokens) of the evaluated checkpoint, from a committed eval file."""
    text = git("show", f"HEAD:{path}")
    loss, ppl = ROW.search(text).groups()
    tokens = re.search(r"\*\*([\d,]+) tokens scored\*\*", text).group(1)
    return float(loss), float(ppl), tokens


def run_commits():
    commits = {}
    for line in git("log", "--format=%h|%s", "--grep=^Seed replication:").splitlines():
        h, subject = line.split("|", 1)
        commits[subject.replace("Seed replication: ", "")] = h
    return commits


def verify(commits):
    rows = []
    for run in RUNS:
        if run not in commits:
            rows.append((run, None, False, "not committed"))
            continue
        h = commits[run]
        files = sorted(git("show", "--name-only", "--format=", h).split())
        want = sorted(f"results/seed_replication/{run}{s}" for s in
                      (".md", ".provenance.json", "_official_full.md", "_official_clean.md"))
        m = json.loads(git("show", f"{h}:results/seed_replication/{run}.provenance.json"))
        checks = {
            "exactly 4 files": files == want,
            "completed": m["status"] == "completed",
            "dirty=false": m["git"]["dirty"] is False,
            "final step 10000": m["fingerprints"]["final_step"] == 10000,
            "train hash": m["data"]["train_tokens"]["sha256"] == SHA[config_of(run)],
            "val hash": m["data"]["val_tokens"]["sha256"] == VAL_SHA,
        }
        rows.append((run, h, all(checks.values()),
                     " · ".join(f"{k} {'✓' if v else '✗'}" for k, v in checks.items())
                     + f" · seed {m['seeds']['seed']} · launched from `{m['git']['commit'][:7]}`"))
    return rows


def per_run(run):
    full = official(f"results/seed_replication/{run}_official_full.md")
    clean = official(f"results/seed_replication/{run}_official_clean.md")
    ckpt = torch.load(REPO / "checkpoints" / "seed_replication" / run / f"{run}_best.pt",
                      map_location="cpu", weights_only=False)
    with (REPO / "logs" / "seed_replication" / run / "training.csv").open() as f:
        log = [(int(r["step"]), float(r["train_loss"]), float(r["val_loss"]))
               for r in csv.DictReader(f)]
    with (REPO / "logs" / "seed_replication" / run / "per_step.csv").open() as f:
        norms = [(int(r["step"]), float(r["grad_norm"])) for r in csv.DictReader(f)]
    median = st.median(n for _, n in norms)
    early_max = max((x for x in norms if x[0] <= EARLY_STEP), key=lambda x: x[1])
    best_log_step = min(log, key=lambda r: r[2])[0]     # first minimum, as the best-save rule
    _, final_train, final_val = log[-1]
    return {"run": run, "config": config_of(run), "seed": int(run[-4:]),
            "full_loss": full[0], "full_ppl": full[1], "clean_loss": clean[0],
            "clean_ppl": clean[1], "tokens_full": full[2], "tokens_clean": clean[2],
            "best_step": ckpt["step"], "best_monitoring_val": ckpt["best_val_loss"],
            "best_step_matches_log": ckpt["step"] == best_log_step,
            "final_gap": final_val - final_train,
            "norm_median": median, "early_max": early_max,
            "spikes": [(s, n) for s, n in norms if n > SPIKE_RATIO * median],
            "step17": dict(norms).get(17)}


def historical():
    """Historical seed-1337 official numbers, from the committed Stage 6.1 eval file."""
    text = git("show", "HEAD:eval/stage6_1_official_eval.md")
    out = {}
    for section in text.split("## Official validation set")[1:]:
        name = re.search(r"evaluation of `([^`]+)` \(([^)]+)\)", section)
        loss, ppl = ROW.search(section).groups()
        out[(name.group(1), name.group(2))] = (float(loss), float(ppl))
    return out


def historical_step17():
    with (REPO / "logs" / "transformer_6_1_grad_norms.csv").open() as f:
        return {int(r["step"]): float(r["grad_norm"]) for r in csv.DictReader(f)}[17]


def label(cfg):
    return "20 MB baseline" if cfg == "base" else "200 MB (6.1)"


def build(commits, verification):
    runs = [per_run(r) for r in RUNS]
    hist = historical()
    last = commits[RUNS[-1]]
    as_of = git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M", last).strip()

    stats = {}
    for cfg in ("base", "data"):
        for target in ("full", "clean"):
            v = [r[f"{target}_loss"] for r in runs if r["config"] == cfg]
            stats[(cfg, target)] = {"values": v, "mean": st.mean(v), "sd": st.stdev(v),
                                    "min": min(v), "max": max(v)}
    effect, verdict = {}, {}
    for target in ("full", "clean"):
        b, d = stats[("base", target)], stats[("data", target)]
        eff = b["mean"] - d["mean"]
        max_sd = max(b["sd"], d["sd"])
        no_overlap = d["max"] < b["min"] or b["max"] < d["min"]
        ratio = eff / max_sd if max_sd > 0 else float("inf")
        effect[target] = {"nats": eff, "ppl_change": 100 * (math.exp(d["mean"] - b["mean"]) - 1),
                          "max_sd": max_sd, "ratio": ratio, "no_overlap": no_overlap}
        verdict[target] = "ROBUST" if no_overlap and eff >= THRESHOLD * max_sd else "INCONCLUSIVE"
    min_ratio = min(effect[t]["ratio"] for t in ("full", "clean"))
    both_robust = all(v == "ROBUST" for v in verdict.values())

    hb = {t: hist[("transformer_10k_best.pt", l)][0] for t, l in (("full", "full file"), ("clean", "clean subset"))}
    hd = {t: hist[("transformer_6_1_best.pt", l)][0] for t, l in (("full", "full file"), ("clean", "clean subset"))}
    by_run = {r["run"]: r for r in runs}
    nb = {t: by_run["sr_base20mb_s1337"][f"{t}_loss"] for t in ("full", "clean")}
    nd = {t: by_run["sr_data200mb_s1337"][f"{t}_loss"] for t in ("full", "clean")}

    L = []
    w = L.append
    w("# Seed replication — results")
    w("")
    w(f"_Data as of `{last}` ({as_of}), the last run's commit · read-only analysis of committed "
      "results, generated by `scripts/analyze_seed_replication.py` · every claim labelled "
      "**VERIFIED** (read or computed from a named source) or **INFERRED** (a judgement not "
      "directly measured)._")
    w("")
    w("The question: does Stage 6.1's data intervention (200 MB unique data vs the 20 MB "
      "baseline, fixed 10,000-step compute budget) reduce official validation loss across "
      "seeds, or was the original single-seed result a seed artefact? Three seeds per "
      "config (1337, 1338, 1339), all under the current protocol "
      "(`train_transformer.py` from `2e3f9a1`: fixed eval seed, decoupled generation RNG, "
      "single continuous run).")
    w("")
    w("## Verdict")
    w("")
    for target, name in (("full", "full validation file"), ("clean", "clean subset")):
        e = effect[target]
        w(f"- **{name}: {verdict[target]}.** Effect {e['nats']:.4f} nats "
          f"(perplexity {e['ppl_change']:+.2f}%); ranges overlap: "
          f"{'no' if e['no_overlap'] else 'yes'}; effect = {e['ratio']:.1f}× the larger "
          f"within-config SD ({e['max_sd']:.4f}). **VERIFIED** (computed from the values in §2 "
          "under the rule in §5)")
    w("")
    mean_eff = st.mean(effect[t]["nats"] for t in ("full", "clean"))
    mean_ppl = st.mean(effect[t]["ppl_change"] for t in ("full", "clean"))
    if both_robust:
        w(f"In plain terms: across three seeds per config, the 200 MB data intervention lowered "
          f"official validation loss by about {mean_eff:.2f} nats (perplexity about "
          f"{mean_ppl:.0f}%), with no overlap between the two configs' results. This supports "
          "the original Stage 6.1 perplexity conclusion. **INFERRED** (reading of the verdicts above)")
    else:
        w("In plain terms: under the stated rule the replication does not establish the effect "
          "on at least one target; the original Stage 6.1 perplexity conclusion is not "
          "confirmed there. **INFERRED** (reading of the verdicts above)")
    w("")
    w("**Threshold timing — the user's statement:** the 3× threshold was fixed after runs 1–3's "
      "official losses had been reported and before runs 4–6's were. This is consistent with "
      "the record (**VERIFIED**): the rule reached this analysis in a request made when 5 of the "
      "6 runs were committed; runs 1–3's losses had been reported in status checks, runs 4–5's "
      f"had not, and run 6 had not finished. The verdict is unchanged for any threshold up to "
      f"{math.floor(min_ratio * 10) / 10:.1f}×. **VERIFIED** (arithmetic)")
    w("")
    w("## 1. Verification of the six runs")
    w("")
    w("| run | commit | all checks | detail |")
    w("|---|---|---|---|")
    for run, h, ok, detail in verification:
        w(f"| `{run}` | `{h}` | **{'pass' if ok else 'FAIL'}** | {detail} |")
    w("")
    w("Each commit contains exactly its 4 files; every manifest shows status completed, "
      "dirty=false, final step 10,000, and both data hashes matching the values verified by "
      "regeneration. **VERIFIED** (commits and their `.provenance.json`)")
    w("")
    w("## 2. Per-run results")
    w("")
    w("Official validation loss (nats) and perplexity, full file and clean subset, "
      "`eval_official_valid.py` on each run's best checkpoint. Best checkpoint = lowest "
      "monitoring-split loss within 10,000 steps. Final gap = monitoring val − train loss at "
      "step 10,000 (train = mean of that 250-step interval's batch losses).")
    w("")
    w("| run | config | seed | full loss | full ppl | clean loss | clean ppl | best step | "
      "best monitoring val | final gap |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        w(f"| `{r['run']}` | {'20 MB' if r['config'] == 'base' else '200 MB'} | {r['seed']} | "
          f"{r['full_loss']:.4f} | {r['full_ppl']:.2f} | {r['clean_loss']:.4f} | "
          f"{r['clean_ppl']:.2f} | {r['best_step']:,} | {r['best_monitoring_val']:.4f} | "
          f"{r['final_gap']:+.4f} |")
    w("")
    w(f"Sources: losses and perplexities from the committed `_official_full.md` / "
      f"`_official_clean.md` files (recorded to 4 decimal places; tokens scored "
      f"{runs[0]['tokens_full']} / {runs[0]['tokens_clean']} in every run); best step and "
      f"best monitoring val from each `_best.pt` checkpoint, cross-checked against the first "
      f"minimum in `training.csv` "
      f"({'agrees for all six' if all(r['best_step_matches_log'] for r in runs) else 'DISAGREES for some'}); "
      "final gap from the last row of `training.csv`. **VERIFIED**")
    w("")
    w("## 3. Per-config statistics (n = 3 each)")
    w("")
    w("| config | target | values | mean | sample SD | min – max |")
    w("|---|---|---|---|---|---|")
    for cfg in ("base", "data"):
        for target in ("full", "clean"):
            s = stats[(cfg, target)]
            w(f"| {label(cfg)} | {target} | {', '.join(f'{v:.4f}' for v in s['values'])} | "
              f"{s['mean']:.4f} | {s['sd']:.4f} | {s['min']:.4f} – {s['max']:.4f} |")
    w("")
    w("Sample SD uses n − 1. **VERIFIED** (computed from §2)")
    w("")
    w("## 4. Effect")
    w("")
    w("| target | mean baseline − mean 6.1 (nats) | perplexity change |")
    w("|---|---|---|")
    for target in ("full", "clean"):
        e = effect[target]
        w(f"| {target} | {e['nats']:.4f} | {e['ppl_change']:+.2f}% |")
    w("")
    w("Perplexity change = exp(mean 6.1 loss − mean baseline loss) − 1, computed from the "
      "losses, not from rounded perplexities. **VERIFIED**")
    w("")
    w("## 5. Verdict rule and its evaluation")
    w("")
    w("- **ROBUST**: the two configs' loss ranges do not overlap **and** the effect is at "
      "least 3× the larger within-config sample SD.")
    w("- **INCONCLUSIVE**: otherwise.")
    w("")
    w("| target | ranges overlap? | effect | 3 × larger SD | effect / larger SD | verdict |")
    w("|---|---|---|---|---|---|")
    for target in ("full", "clean"):
        e = effect[target]
        w(f"| {target} | {'no' if e['no_overlap'] else 'yes'} | {e['nats']:.4f} | "
          f"{THRESHOLD * e['max_sd']:.4f} | {e['ratio']:.1f}× | **{verdict[target]}** |")
    w("")
    w("No p-values, t-tests or confidence intervals are reported: with n = 3 per config they "
      "would rest on assumptions three points cannot check. **VERIFIED** (none computed)")
    w("")
    w("The 3× threshold was fixed after runs 1–3's losses were reported and before runs 4–6's "
      "were (the user's statement; see the note under the verdict). The verdict is unchanged "
      f"for any threshold up to {math.floor(min_ratio * 10) / 10:.1f}×. **VERIFIED** (arithmetic)")
    w("")
    w("## 6. Secondary diagnostics (not part of the three-seed statistics)")
    w("")
    w("The historical seed-1337 runs were trained under the earlier protocol and are not "
      "among the six runs above. Historical numbers are from the committed "
      "`eval/stage6_1_official_eval.md`. **VERIFIED**")
    w("")
    w("| comparison | target | historical | new (same seed) | difference (new − historical) | "
      "difference / within-config SD |")
    w("|---|---|---|---|---|---|")
    for name, old, new, cfg in (("baseline 1337: two-process (historical) vs single-pass (new)", hb, nb, "base"),
                                ("6.1 1337: coupled generation RNG (historical) vs decoupled (new)", hd, nd, "data")):
        for t in ("full", "clean"):
            sd = stats[(cfg, t)]["sd"]
            w(f"| {name} | {t} | {old[t]:.4f} | {new[t]:.4f} | {new[t] - old[t]:+.4f} | "
              f"{(new[t] - old[t]) / sd:+.2f} SD |")
    w("")
    w("A difference well under 1 SD means the protocol change at seed 1337 moved the result "
      "by less than the ordinary seed-to-seed spread. **INFERRED** (interpretation of the "
      "ratios above)")
    w("")
    w("**Replicated effect vs the original Stage 6.1 claim:**")
    w("")
    w("| target | original single-seed claim | historical losses → change | three-seed mean → change |")
    w("|---|---|---|---|")
    hist_change = {}
    for t in ("full", "clean"):
        hist_change[t] = 100 * (math.exp(hd[t] - hb[t]) - 1)
        w(f"| {t} | {ORIGINAL_CLAIM[t]:+.1f}% | {hb[t]:.4f} → {hd[t]:.4f} = {hist_change[t]:+.2f}% | "
          f"{effect[t]['ppl_change']:+.2f}% |")
    w("")
    w("**VERIFIED** (original claim from `eval/stage6_1_comparison.md`; changes recomputed "
      "from the recorded losses)")
    w("")
    pp = {t: abs(effect[t]["ppl_change"]) - abs(hist_change[t]) for t in ("full", "clean")}
    gap = {t: effect[t]["nats"] - (hb[t] - hd[t]) for t in ("full", "clean")}
    larger = all(v > 0 for v in pp.values())
    w(f"The replicated effect has the same direction as the original claim and is about "
      f"{st.mean(pp.values()):.1f} percentage points {'larger' if larger else 'different'} "
      f"({effect['full']['ppl_change']:.2f}% vs {hist_change['full']:.2f}% full; "
      f"{effect['clean']['ppl_change']:.2f}% vs {hist_change['clean']:.2f}% clean). In nats, the "
      f"original single-seed effect ({hb['full'] - hd['full']:.4f} full, "
      f"{hb['clean'] - hd['clean']:.4f} clean) sits {gap['full']:.4f} and {gap['clean']:.4f} "
      f"below the three-seed mean effect, "
      f"{'less than half' if max(gap.values()) < 0.5 * stats[('base', 'full')]['sd'] else 'more than half'} "
      "the baseline's within-config SD. **VERIFIED** (arithmetic from the tables above)")
    w("")
    w("## 7. Notes and limitations")
    w("")
    w("- **Run 2's wall-clock time is not a compute measure.** `sr_data200mb_s1337` includes "
      "an ~82.5-minute pause while the Mac slept (between steps ~6,000 and ~6,500), measured "
      "from its `training.csv` elapsed times. Training itself was unaffected: steps continued "
      "from the same state. **VERIFIED** (pause); **INFERRED** (no effect on results — the "
      "process state was preserved and the curve continued smoothly)")
    w("- **Variation between runs is seed + MPS nondeterminism, not seed alone.** Two runs of "
      "identical code and seed already diverge from step 2 on MPS (determinism test). How much "
      "of the spread in §3 comes from each source is not measured. **VERIFIED** "
      "(nondeterminism); **INFERRED** (its share of the spread is unknown)")
    w("- **n = 3 per config.** The SDs themselves are imprecise with three values; the verdict "
      "rule is a pre-stated heuristic, not a statistical test. **VERIFIED** (n) / **INFERRED** (imprecision)")
    w("- **Losses are recorded to 4 decimal places** in the evaluation files, which bounds the "
      "precision of every derived number. **VERIFIED**")
    w("- **Protocol:** all six runs share one protocol (single 10,000-step pass, decoupled "
      "generation RNG, fixed eval seed 1337, best-by-monitoring checkpoint). They are "
      "replicates under that protocol, not reproductions of the historical runs. **VERIFIED** "
      "(manifests and code at `2e3f9a1`)")
    w("- **Scope:** this covers official validation loss only. The entity and causal "
      "evaluation from Stage 6.1 was not repeated across seeds. **VERIFIED**")
    w("")
    w("## 8. Observations")
    w("")
    sd_ratio = {t: stats[("base", t)]["sd"] / stats[("data", t)]["sd"] for t in ("full", "clean")}
    w(f"- **The within-config SDs differ by about {st.mean(sd_ratio.values()):.1f}×**: "
      f"{stats[('base', 'full')]['sd']:.4f} for the 20 MB baseline vs "
      f"{stats[('data', 'full')]['sd']:.4f} for 200 MB on the full file "
      f"({sd_ratio['full']:.1f}×; clean subset {sd_ratio['clean']:.1f}×). **VERIFIED** "
      "(arithmetic). With three values per config this cannot establish that the two "
      "configs genuinely differ in seed-to-seed variability. **INFERRED**. The verdict rule "
      "uses the larger of the two SDs, so the difference does not favour the result. "
      "**VERIFIED** (rule definition)")
    at_end = [r for r in runs if r["best_step"] == 10000]
    others = [r for r in runs if r["best_step"] != 10000]
    w(f"- **{len(at_end)} of 6 runs selected step 10,000 as their best checkpoint**"
      + (f" (the exception: `{others[0]['run']}` at step {others[0]['best_step']:,})" if len(others) == 1 else "")
      + ". **VERIFIED** (checkpoints and `training.csv`). In those runs monitoring loss was "
      "still at its lowest at the final evaluation, so the models had not converged; the "
      "result applies to a fixed 10,000-step budget, not to training to convergence. **INFERRED**")
    with_early = [r for r in runs if r["spikes"] and all(s <= EARLY_STEP for s, _ in r["spikes"])]
    without = [r for r in runs if not r["spikes"]]
    late = [r for r in runs if any(s > EARLY_STEP for s, _ in r["spikes"])]
    seeds_with = sorted({r["seed"] for r in with_early})
    cfgs_with = sorted({label(r["config"]) for r in with_early})
    spike_text = "; ".join(
        f"`{r['run']}` step{'s' if len(r['spikes']) > 1 else ''} "
        f"{', '.join(str(s) for s, _ in r['spikes'])} "
        f"({max(n for _, n in r['spikes']) / r['norm_median']:.0f}× median)" for r in with_early)
    w(f"- **Early gradient spikes (above {SPIKE_RATIO:.0f}× the run's median norm, at step "
      f"≤ {EARLY_STEP}) occurred in {len(with_early)} of 6 runs**: {spike_text}. They occur "
      f"in both configs ({', '.join(cfgs_with)}) but only for seed"
      f"{'s' if len(seeds_with) > 1 else ''} {', '.join(map(str, seeds_with))}; the other "
      f"{len(without)} runs peak at "
      f"{min(r['early_max'][1] / r['norm_median'] for r in without):.1f}–"
      f"{max(r['early_max'][1] / r['norm_median'] for r in without):.1f}× in that window. "
      f"{'No run has a spike after step ' + str(EARLY_STEP) + '.' if not late else 'Some runs also spike later.'} "
      "**VERIFIED** (each run's `per_step.csv`)")
    s17_new, s17_old = by_run["sr_data200mb_s1337"]["step17"], historical_step17()
    w(f"- `sr_data200mb_s1337`'s spike is at the same step as the historical 6.1 run's "
      f"(step 17: {s17_new:.2f} vs {s17_old:.2f}), as expected from identical batches through "
      "step 1,000 at seed 1337. **VERIFIED** (values); **INFERRED** (cause)")
    return "\n".join(L) + "\n", verdict


def main():
    parser = argparse.ArgumentParser(description="Seed replication analysis")
    parser.add_argument("--out", default=None, help="write here instead of the default path")
    parser.add_argument("--verify", action="store_true",
                        help="regenerate in memory and diff against the existing file")
    args = parser.parse_args()
    out = Path(args.out).resolve() if args.out else DEFAULT_OUT

    commits = run_commits()
    verification = verify(commits)
    if not all(ok for _, _, ok, _ in verification):
        for run, h, ok, detail in verification:
            print(f"  {run:<20} {h or '-':<8} ok={ok}  {detail}")
        sys.exit("not all six runs committed and verified: nothing computed, nothing written")

    text, verdict = build(commits, verification)
    if args.verify:
        existing = out.read_text(encoding="utf-8")
        if existing == text:
            print(f"byte-for-byte identical: {out}")
            return
        sys.stdout.writelines(difflib.unified_diff(existing.splitlines(True), text.splitlines(True),
                                                   str(out), "regenerated"))
        sys.exit("DIFFERS")
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    print(f"verdict: full {verdict['full']} | clean {verdict['clean']}")


if __name__ == "__main__":
    main()
