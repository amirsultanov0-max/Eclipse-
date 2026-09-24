"""
Stage 6.3 Amendment 1: equivalence evidence, checks V1-V5 of
eval/stage6_3_preregistration.md section 10.

Run on the training machine, from the repository root, on the commit that carries the
amendment, with a clean tree and the real data/ directory:

    venv/bin/python scripts/equivalence_6_3.py

The pre-change code is train_transformer.py as committed with the preregistration
(sha256 57d4e57e...). It is checked out into a temporary git worktree, so this checkout
is never modified. Evidence goes to results/equivalence_6_3/; the script exits 0 only if
every check passes.

    --smoke   exercise the harness without the real data or a clean tree. Writes to a
              scratch directory and labels everything a smoke test: NOT evidence.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PRE_REF = "b0fd350"   # the preregistration commit; train_transformer.py there is 57d4e57e...
EXPECTED = {
    "pre train_transformer.py": "57d4e57eb7d6e6bff6c8042e2842109b0cd2f72949ef0b3ee21548842fe393a5",
    "model/transformer.py": "63ad52cdf138092233f9752dd7393f7af4790ff5276f4886294d0afbe93ca8f0",
    "tokenizer/bpe_4000.json": "e3876f558896533d13b7818282c6b19b55cbf205390ad4ee9e982d105762d3df",
    "data/stage6_1_train_tokens.npy": "5c25b49be87093ce19f5b7a1d2b67ce2783d062c1a3c3c20c5ce7341ba47d4eb",
    "data/stage3_val_tokens.npy": "9343c0f71f96cc033c4a5ba163af099ea8a33db0d4327af73b3dfae00c52eb9e",
}
TRAIN_TOKENS = "data/stage6_1_train_tokens.npy"
TITLE = "Stage 6.3 equivalence run"
SEED = 1337
COSINE_LINE = ("lr 0.001 peak (linear warm-up over steps 1-1,000, "
               "then cosine decay to 0.0001 at step 10,000)")
D352_LINE = "d_model 352, 4 heads, d_ff 1408, 6 Pre-LN blocks, tied LM head — 10,537,472 parameters"
# Lines of a run summary that legitimately differ between two runs of identical code.
VARIABLE_PREFIXES = ("## ", "- wall clock:", "Logs:")


def section_2_1(step):
    """The preregistered schedule, written out independently of train_transformer.py."""
    if step <= 1000:
        return 1e-3 * step / 1000
    return 1e-4 + 0.5 * (1e-3 - 1e-4) * (1 + math.cos(math.pi * (step - 1000) / 9000))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def read_log(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Report:
    def __init__(self):
        self.rows = []

    def check(self, group, name, ok, detail=""):
        self.rows.append((group, name, bool(ok), detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {group} {name}" + (f"  ({detail})" if detail else ""))
        return ok

    @property
    def passed(self):
        return all(ok for _, _, ok, _ in self.rows)


def run(label, root, work, stamp, device, extra=(), steps=1001):
    """One training run. Returns the paths of everything it wrote."""
    name = f"eq63_{stamp}_{label}"
    results = work / f"{label}.md"
    cmd = [sys.executable, "train_transformer.py", "--steps", str(steps), "--seed", str(SEED),
           "--device", device, "--train-tokens", TRAIN_TOKENS, "--run-name", name,
           "--log-name", f"{name}_training.csv", "--grad-name", f"{name}_grad.csv",
           "--samples-name", f"{name}_samples.txt", "--checkpoint-dir", str(work / f"ckpt_{label}"),
           "--results-file", str(results), "--results-title", TITLE, *extra]
    print(f"\n[{label}] {' '.join(cmd[1:])}\n        in {root}")
    with open(work / f"{label}.stdout.txt", "w") as out:
        subprocess.run(cmd, cwd=root, stdout=out, stderr=subprocess.STDOUT, check=True)
    # Copy the run's logs out of its checkout at once: the pre-change worktree is deleted
    # before the comparisons run.
    logs = root / "logs"
    kept = {}
    for kind, src in (("grad", logs / f"{name}_grad.csv"), ("samples", logs / f"{name}_samples.txt"),
                      ("lr", logs / f"{name}_grad_lr.csv")):
        kept[kind] = work / f"{label}_{kind}{src.suffix}"
        if src.exists():
            shutil.copy2(src, kept[kind])
    return {"label": label, "cmd": cmd[1:], "root": str(root), **kept, "summary": results,
            "manifest": work / f"{label}.provenance.json"}


def fingerprints(r):
    return json.loads(Path(r["manifest"]).read_text())["fingerprints"]


def summary_lines(r):
    return [l for l in Path(r["summary"]).read_text().splitlines()
            if not l.startswith(VARIABLE_PREFIXES)]


def line_starting(r, prefix):
    return next((l for l in Path(r["summary"]).read_text().splitlines() if l.startswith(prefix)), "")


def compare_fp(rep, group, a, b, keys):
    fa, fb = fingerprints(a), fingerprints(b)
    for k in keys:
        rep.check(group, f"{k} bitwise equal", fa.get(k) == fb.get(k) and fa.get(k) is not None,
                  f"{str(fa.get(k))[:16]}…")


def first_difference(rows_a, rows_b, column):
    for ra, rb in zip(rows_a, rows_b):
        if ra[column] != rb[column]:
            return int(ra["step"])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="harness test only; NOT evidence")
    p.add_argument("--steps", type=int, default=1001, help="prefix length (evidence: 1001)")
    p.add_argument("--no-mps", action="store_true", help="skip V3 (only allowed with --smoke)")
    args = p.parse_args()
    if args.no_mps and not args.smoke:
        raise SystemExit("V3 (MPS) is part of the registered verification; --no-mps needs --smoke")
    if not args.smoke and args.steps != 1001:
        raise SystemExit("the registered verification uses --steps 1001")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    work = Path(tempfile.mkdtemp(prefix="eq63_"))
    out = work / "evidence" if args.smoke else REPO / "results" / "equivalence_6_3"
    rep = Report()
    print(f"work directory: {work}\nevidence directory: {out}")

    # --- preconditions ------------------------------------------------------------
    print("\n[preconditions]")
    head = git("rev-parse", "HEAD").strip()
    dirty = git("status", "--porcelain", "--untracked-files=no").strip()
    if not args.smoke:
        if dirty:
            raise SystemExit(f"STOP: the working tree has uncommitted changes:\n{dirty}")
        if out.exists():
            raise SystemExit(f"STOP: {out} already exists; refusing to overwrite evidence")
    pre_source = git("show", f"{PRE_REF}:train_transformer.py")
    hashes = {
        "pre train_transformer.py": hashlib.sha256(pre_source.encode()).hexdigest(),
        "post train_transformer.py": sha256(REPO / "train_transformer.py"),
    }
    for rel in ("model/transformer.py", "tokenizer/bpe_4000.json",
                "data/stage6_1_train_tokens.npy", "data/stage3_val_tokens.npy"):
        hashes[rel] = sha256(REPO / rel)
    for key, want in EXPECTED.items():
        if args.smoke and key.startswith("data/"):
            print(f"  SKIP  {key} hash (smoke test uses synthetic data)")
            continue
        if hashes[key] != want:
            raise SystemExit(f"STOP: {key} is {hashes[key]}, expected {want}")
        print(f"  ok    {key} {want[:16]}…")
    if hashes["post train_transformer.py"] == hashes["pre train_transformer.py"]:
        raise SystemExit("STOP: train_transformer.py is unchanged; nothing to verify")
    import torch
    mps = torch.backends.mps.is_available()
    if not args.no_mps and not mps:
        raise SystemExit("STOP: MPS is not available, and V3 needs it")

    # --- the pre-change code, in its own worktree ------------------------------------
    pre_root = work / "pre"
    git("worktree", "add", "--detach", str(pre_root), PRE_REF)
    # Hard links (or copies), not a symlink: the training script resolves symlinks when it
    # writes the data path into the run summary, which would make V2 differ spuriously.
    (pre_root / "data").mkdir()
    for rel in (TRAIN_TOKENS, "data/stage3_val_tokens.npy"):
        try:
            (pre_root / rel).hardlink_to(REPO / rel)
        except OSError:
            shutil.copy2(REPO / rel, pre_root / rel)
    runs = {}
    try:
        n = args.steps
        runs["pre_cpu"] = run("pre_cpu", pre_root, work, stamp, "cpu", steps=n)
        runs["post_cpu"] = run("post_cpu", REPO, work, stamp, "cpu", steps=n)
        if not args.no_mps:
            runs["pre_mps"] = run("pre_mps", pre_root, work, stamp, "mps", steps=n)
            runs["post_mps"] = run("post_mps", REPO, work, stamp, "mps", steps=n)
        runs["treat_cpu"] = run("treat_cpu", REPO, work, stamp, "cpu",
                                ("--lr-schedule", "warmup_cosine"), steps=n)
        runs["v5_cosine"] = run("v5_cosine", REPO, work, stamp, "cpu",
                                ("--lr-schedule", "warmup_cosine"), steps=20)
        runs["v5_d352"] = run("v5_d352", REPO, work, stamp, "cpu",
                              ("--d-model", "352", "--n-heads", "4", "--d-ff", "1408"), steps=20)
    finally:
        git("worktree", "remove", "--force", str(pre_root))

    pre, post = runs["pre_cpu"], runs["post_cpu"]
    pre_log, post_log = read_log(pre["grad"]), read_log(post["grad"])

    print("\n[V1] default arguments, pre vs post, CPU")
    rep.check("V1", f"per-step log has {n} steps in both", len(pre_log) == len(post_log) == n)
    for column in ("loss", "grad_norm", "batch_starts_sha256"):
        step = first_difference(pre_log, post_log, column)
        rep.check("V1", f"per-step {column} bitwise equal at all steps", step is None,
                  f"first difference at step {step}" if step else "")
    rep.check("V1", "whole per-step log file bitwise equal",
              Path(pre["grad"]).read_bytes() == Path(post["grad"]).read_bytes())
    compare_fp(rep, "V1", pre, post, ("initial_weights_sha256", "eval_batches_sha256",
                                      "final_weights_sha256", "batch_starts_running_sha256"))
    rep.check("V1", "in-training sample text bitwise equal",
              Path(pre["samples"]).read_bytes() == Path(post["samples"]).read_bytes())
    rep.check("V1", "generation draw counts equal",
              fingerprints(pre).get("generation_draws") == fingerprints(post).get("generation_draws"),
              json.dumps(fingerprints(post).get("generation_draws")))
    data_pre = json.loads(Path(pre["manifest"]).read_text())["data"]
    data_post = json.loads(Path(post["manifest"]).read_text())["data"]
    rep.check("V1", "both runs read byte-identical data (manifest SHA256s)",
              {k: v["sha256"] for k, v in data_pre.items()} == {k: v["sha256"] for k, v in data_post.items()},
              data_post["train_tokens"]["sha256"][:16] + "…")

    print("\n[V2] default arguments, run summary")
    a, b = summary_lines(pre), summary_lines(post)
    differing = [(x, y) for x, y in zip(a, b) if x != y] + ([("<length>", "")] if len(a) != len(b) else [])
    rep.check("V2", "summaries identical apart from run name, date, log names, wall clock",
              not differing, "; ".join(f"{x!r} != {y!r}" for x, y in differing[:3]))
    for prefix in ("- model:", "- config:"):
        rep.check("V2", f"'{prefix}' line unchanged", line_starting(pre, prefix) == line_starting(post, prefix),
                  line_starting(post, prefix)[:90])

    if not args.no_mps:
        print("\n[V3] default arguments, pre vs post, MPS")
        compare_fp(rep, "V3", runs["pre_mps"], runs["post_mps"],
                   ("initial_weights_sha256", "eval_batches_sha256", "batch_starts_running_sha256"))
        mps_pre, mps_post = read_log(runs["pre_mps"]["grad"]), read_log(runs["post_mps"]["grad"])
        rep.check("V3", "per-step batch starts equal at all steps",
                  first_difference(mps_pre, mps_post, "batch_starts_sha256") is None)
        losses_differ = sum(x["loss"] != y["loss"] for x, y in zip(mps_pre, mps_post))
        print(f"  info  MPS per-step loss differs at {losses_differ} of {len(mps_pre)} steps "
              "(MPS is not bit-reproducible run to run; not a pass condition)")

    print("\n[V4] warmup_cosine vs constant, same seed, CPU")
    treat = runs["treat_cpu"]
    treat_log = read_log(treat["grad"])
    rep.check("V4", "per-step batch starts equal at all steps",
              first_difference(post_log, treat_log, "batch_starts_sha256") is None)
    compare_fp(rep, "V4", post, treat, ("initial_weights_sha256", "eval_batches_sha256",
                                         "batch_starts_running_sha256"))
    rep.check("V4", "step-1 loss equal", post_log[0]["loss"] == treat_log[0]["loss"], treat_log[0]["loss"])
    rep.check("V4", "step-1 gradient norm equal", post_log[0]["grad_norm"] == treat_log[0]["grad_norm"],
              treat_log[0]["grad_norm"])
    lr_rows = read_log(treat["lr"])
    mismatches = [r["step"] for r in lr_rows if float(r["lr"]) != section_2_1(int(r["step"]))]
    rep.check("V4", f"per-step LR file equals section 2.1 at all {n} steps",
              len(lr_rows) == n and not mismatches and [int(r["step"]) for r in lr_rows] == list(range(1, n + 1)),
              f"{len(lr_rows)} rows, {len(mismatches)} mismatches")
    step = first_difference(post_log, treat_log, "loss")
    print(f"  info  loss first differs from the constant run at step {step} (expected: step 2)")

    print("\n[V5] summary lines at non-default arguments")
    rep.check("V5", "warmup_cosine run states its schedule",
              COSINE_LINE in line_starting(runs["v5_cosine"], "- config:"),
              line_starting(runs["v5_cosine"], "- config:")[:120])
    rep.check("V5", "d352 run states its architecture and 10,537,472 parameters",
              D352_LINE in line_starting(runs["v5_d352"], "- model:"),
              line_starting(runs["v5_d352"], "- model:"))
    for label, extra in (("--resume", ["--resume", "checkpoints/any.pt"]),
                         ("--steps 10001", ["--steps", "10001"])):
        r = subprocess.run([sys.executable, "train_transformer.py", "--lr-schedule", "warmup_cosine",
                            "--results-file", str(work / "guard.md"), *extra],
                           cwd=REPO, capture_output=True, text=True)
        rep.check("V5", f"warmup_cosine with {label} is refused before anything runs",
                  r.returncode != 0 and "warmup_cosine" in (r.stderr + r.stdout)
                  and not (work / "guard.provenance.json").exists(),
                  (r.stderr + r.stdout).strip().splitlines()[-1][:100] if (r.stderr + r.stdout).strip() else "")

    # --- evidence -------------------------------------------------------------------
    out.mkdir(parents=True)
    for label, r in runs.items():
        for kind, suffix in (("grad", "_grad.csv"), ("samples", "_samples.txt"), ("lr", "_lr.csv"),
                             ("summary", ".md"), ("manifest", ".provenance.json")):
            src = Path(r[kind])
            if src.exists() and not (label.startswith("v5_") and kind in ("grad", "samples")):
                shutil.copy2(src, out / f"eq63_{label}{suffix}")
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "platform": platform.platform(), "mps_available": mps}
    verdict = "ALL CHECKS PASSED" if rep.passed else "FAILED"
    lines = [
        "# Stage 6.3 Amendment 1 — equivalence evidence" + (" (SMOKE TEST — NOT EVIDENCE)" if args.smoke else ""),
        "",
        f"Checks V1–V5 of `eval/stage6_3_preregistration.md` section 10, produced by "
        f"`scripts/equivalence_6_3.py` on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        f"**Result: {verdict}.**",
        "",
        f"- post-change code: HEAD `{head}`, `train_transformer.py` `{hashes['post train_transformer.py']}`",
        f"- pre-change code: `{PRE_REF}` (git worktree), `train_transformer.py` `{hashes['pre train_transformer.py']}`",
        f"- `model/transformer.py` `{hashes['model/transformer.py']}`",
        f"- training stream `{hashes['data/stage6_1_train_tokens.npy']}`, "
        f"monitoring stream `{hashes['data/stage3_val_tokens.npy']}`",
        f"- {n} steps, seed {SEED}; environment: {json.dumps(env)}",
        "",
        "| check | item | result | detail |",
        "|---|---|---|---|",
    ] + [f"| {g} | {name} | {'PASS' if ok else '**FAIL**'} | {d.replace('|', '/')} |"
         for g, name, ok, d in rep.rows] + [
        "",
        "## Runs",
        "",
    ] + [f"- `{label}`: `{' '.join(r['cmd'])}`" for label, r in runs.items()] + [
        "",
        "Files: `eq63_<run>_grad.csv` (per-step loss, gradient norm, batch-start hash), "
        "`_samples.txt`, `_lr.csv` (treatment only), `.md` (run summary), `.provenance.json`.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{verdict}: {sum(ok for *_, ok, _ in rep.rows)}/{len(rep.rows)} checks passed")
    print(f"evidence: {out}\nwork directory (scratch, safe to delete): {work}")
    sys.exit(0 if rep.passed else 1)


if __name__ == "__main__":
    main()
