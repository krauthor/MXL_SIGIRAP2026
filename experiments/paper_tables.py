#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd

RESULTS_BASE = "data/results"
OUT_DIR = "data/paper_tables"
MIN_SEED = 100
N_BOOT = 10_000
BOOTSTRAP_SEED = 20261017
MIN_PAIRED = 3
ALPHA_STAR = 0.01
ALPHA_DAGGER = 0.05
MARK_STAR = r"$^{*}$"
MARK_DAGGER = r"$^{\dagger}$"
MISSING = "--"

LOWER_IS_BETTER = {"nll", "slate_nll"}

REAL_METRICS = {"nll": ("nll", 3), "ndcg": ("nDCG", 3), "slate_nll": ("slate-nll", 3)}
REAL_DATASETS = [
    ("ContentWise", "ContentWise_LastActionSplit", ["nll", "ndcg", "slate_nll"]),
    ("DHRD", "Stockholm_LastActionSplit", ["nll", "ndcg"]),
]
REAL_BLOCKS = [
    ("Explicit user embeddings", "MNL", [
        ("Baselines", 5, [("Random", "Random"), ("MostPop", "MostPop"), ("Intercept", "Intercept"),
                          ("MNL", "MNL"), ("MNL-Multi", "MNL_Max")]),
        ("Our", 4, [("Disc. MXL", "MNL_Multi"), ("Diag. MXL", "MXL"),
                    ("Pers-Cov-MXL", "MXL_Corr"), ("Shared-Cov-MXL", "MXL_Corr_Shared")]),
    ]),
    ("SASRec", "MNL_SASRec", [
        ("", 3, [("MNL", "MNL_SASRec"), ("MNL-Multi", "MNL_Max_SASRec")]),
        ("", 4, [("Disc. MXL", "MNL_Multi_SASRec"), ("Diag. MXL", "MXL_SASRec"),
                 ("Pers-Cov-MXL", "MXL_Corr_SASRec"), ("Shared-Cov-MXL", "MXL_Corr_Shared_SASRec")]),
    ]),
]

SEMI_SLATE_METRICS = [
    ("slate_acceptance_rate", r"Choice\% $\uparrow$", 3),
    ("item_coverage", r"Agg.Div. $\uparrow$", 3),
    ("embedding_diversity_percentile", r"IID\% $\uparrow$", 1),
]
SEMI_TOPK_METRICS = [("nll", r"NLL $\downarrow$", 3), ("ndcg", r"NDCG $\uparrow$", 3)]
SEMI_POLICIES = [
    ("semisynthetic_top_k", "Top-k"),
    ("semisynthetic_diverse_embeddings", "Pref. Top-1"),
    ("semisynthetic_greedy", "Greedy"),
    ("semisynthetic_top_k_greedy", "Top-k-Greedy"),
]
BASELINE_POLICY = "semisynthetic_top_k"
SEMI_MODEL_POLICIES = {"MNL": ["semisynthetic_top_k"],
                       "MNL_Max": ["semisynthetic_top_k", "semisynthetic_diverse_embeddings"]}
SEMI_BLOCKS = [
    ("Explicit user embeddings, repeated choice of the same item allowed", "", "MNL",
     {"MNL": "*_synthetic_MNL_2*", "Diag. MXL": "*_synthetic_MXL_2*"}),
    ("SASRec, repeated choice of the same item prohibited", "_SASRec", "MNL_SASRec",
     {"MNL": "*_synthetic_MNL_SASRec_*", "Diag. MXL": "*_synthetic_MXL_SASRec_*"}),
]
SEMI_ROWS = [
    ("MNL", "MNL", True), ("MNL-Multi", "MNL_Max", True), (r"Disc.\ MXL (ours)", "MNL_Multi", True),
    (r"Diag.\ MXL (ours)", "MXL", True), ("Oracle", "Oracle", False),
]


class Dataset:

    def __init__(self, name: str, results_base: str = RESULTS_BASE, min_seed: int = MIN_SEED,
                 run_filter=None):
        self.name = name
        by_dir: dict[str, str] = {}
        for path in glob(os.path.join(results_base, name, "*", "results.json")):
            by_dir[os.path.dirname(path)] = path
        for path in glob(os.path.join(results_base, name, "*", "updated_results.json")):
            by_dir[os.path.dirname(path)] = path
        if not by_dir:
            raise FileNotFoundError(f"no results under {results_base}/{name}/*/")
        records = []
        for path in sorted(by_dir.values()):
            with open(path) as fh:
                records.append(json.load(fh))
        df = pd.json_normalize(records)
        df = df[(df["hyperparams.dataset_name"] == name) & (df["hyperparams.seed"] > min_seed)].copy()
        is_oracle = df["hyperparams.results_dir"].astype(str).str.contains("Oracle")
        df.loc[is_oracle, "model_name"] = "Oracle"
        if run_filter is not None:
            df = df[run_filter(df)].copy()

        metric_cols = [c for c in df.columns if c.startswith("results.")
                       and "performance_by_train_obs" not in c
                       and pd.api.types.is_numeric_dtype(df[c])]
        self.values = df.groupby("model_name")[metric_cols].mean()
        self.values.columns = [c[len("results."):] for c in metric_cols]
        self.n_runs = df.groupby("model_name").size()
        self.n_seeds = df.groupby("model_name")["hyperparams.seed"].nunique()
        reps = df.groupby(["model_name", "hyperparams.seed"])[metric_cols].mean()
        reps.columns = self.values.columns
        self.reps = reps

    def value(self, model: str, column: str) -> float:
        if model in self.values.index and column in self.values.columns:
            return float(self.values.at[model, column])
        return float("nan")

    def testable(self, model: str) -> bool:
        return model in self.n_runs.index and self.n_seeds[model] == self.n_runs[model]

    def replicates(self, model: str, column: str) -> pd.Series:
        if not self.testable(model) or column not in self.reps.columns:
            return pd.Series(dtype=float)
        return self.reps.xs(model, level="model_name")[column].dropna()


def find_dataset(results_base: str, pattern: str, override: str | None) -> str:
    if override:
        return override
    hits = sorted(os.path.basename(p.rstrip("/")) for p in glob(os.path.join(results_base, pattern)))
    hits = [h for h in hits if os.path.isdir(os.path.join(results_base, h))]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one results dir matching {pattern!r} under "
                         f"{results_base}, found {hits}; pass it explicitly via the CLI flags")
    return hits[0]


class Cell:

    def __init__(self, table, block, row, column, ds: Dataset | None, model, metric_col,
                 decimals, baseline_model=None, baseline_col=None, ranked=True, rank_key=None, **info):
        self.table, self.block, self.row, self.column = table, block, row, column
        self.rank_key = rank_key if rank_key is not None else column
        self.ds, self.model, self.metric_col, self.decimals = ds, model, metric_col, decimals
        self.baseline_model, self.baseline_col, self.ranked = baseline_model, baseline_col, ranked
        self.info = info
        self.value = ds.value(model, metric_col) if ds is not None else float("nan")
        self.rank = None
        self.stats: dict | None = None
        self.marker = ""

    @property
    def missing(self) -> bool:
        return not np.isfinite(self.value)

    @property
    def is_baseline(self) -> bool:
        return self.model == self.baseline_model and self.metric_col == self.baseline_col

    @property
    def text(self) -> str:
        return MISSING if self.missing else f"{self.value:.{self.decimals}f}"

    @property
    def direction(self) -> int:
        return -1 if self.metric_col.split(".")[-1] in LOWER_IS_BETTER else 1

    def tex(self, markers: bool = True) -> str:
        if self.missing:
            return MISSING
        out = self.text
        if self.rank == "best":
            out = r"\textbf{" + out + "}"
        elif self.rank == "second":
            out = r"\underline{" + out + "}"
        return out + (self.marker if markers else "")


def assign_ranks(cells: list[Cell]) -> None:
    groups: dict[tuple, list[Cell]] = {}
    for c in cells:
        if c.ranked and not c.missing:
            groups.setdefault((c.table, c.block, c.rank_key), []).append(c)
    for group in groups.values():
        rounded = sorted({round(float(c.text), c.decimals) * c.direction for c in group}, reverse=True)
        best = rounded[0]
        second = rounded[1] if len(rounded) > 1 else None
        for c in group:
            r = round(float(c.text), c.decimals) * c.direction
            c.rank = "best" if r == best else "second" if r == second else None


def _weights(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot) / n


def run_bootstrap(cells: list[Cell], n_boot: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    batches: dict[tuple, list[tuple[Cell, np.ndarray]]] = {}
    for c in cells:
        if c.missing or c.is_baseline or c.baseline_model is None or c.ds is None:
            continue
        a = c.ds.replicates(c.model, c.metric_col)
        b = c.ds.replicates(c.baseline_model, c.baseline_col)
        common = a.index.intersection(b.index)
        if len(common) < MIN_PAIRED:
            continue
        d = (a.loc[common] - b.loc[common]).to_numpy(float)
        batches.setdefault((c.ds.name, tuple(common)), []).append((c, d))

    for (_, seeds), items in batches.items():
        n = len(seeds)
        W = _weights(n, n_boot, rng)
        D = np.stack([d for _, d in items], axis=1)
        boot = W @ D
        p_le = (1 + (boot <= 0).sum(axis=0)) / (n_boot + 1)
        p_ge = (1 + (boot >= 0).sum(axis=0)) / (n_boot + 1)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        for j, (c, d) in enumerate(items):
            p_better = p_le[j] if c.direction > 0 else p_ge[j]
            c.stats = {"diff": float(d.mean()), "ci_low": float(lo[j]), "ci_high": float(hi[j]),
                       "p_better": float(p_better), "p_two_sided": float(min(1.0, 2 * min(p_le[j], p_ge[j]))),
                       "n_seeds": n, "baseline_value": c.ds.value(c.baseline_model, c.baseline_col)}
            c.marker = MARK_STAR if p_better < ALPHA_STAR else MARK_DAGGER if p_better < ALPHA_DAGGER else ""


def build_real_world(datasets: dict[str, Dataset]) -> list[Cell]:
    cells = []
    for block, baseline, groups in REAL_BLOCKS:
        for _, _, rows in groups:
            for label, model in rows:
                for ds_label, ds_name, metrics in REAL_DATASETS:
                    ds = datasets[ds_name]
                    for m in metrics:
                        cells.append(Cell("real_world", block, label, f"{ds_label} {REAL_METRICS[m][0]}",
                                          ds, model, m, REAL_METRICS[m][1], baseline_model=baseline,
                                          baseline_col=m, dataset=ds_name, model_name=model,
                                          slate_type="", metric=m))
    return cells


def render_real_world(cells: list[Cell], markers: bool, header: str) -> str:
    cell = {(c.block, c.row, c.column): c for c in cells}
    columns = [f"{dl} {REAL_METRICS[m][0]}" for dl, _, ms in REAL_DATASETS for m in ms]
    lines = [header, r"\begin{tabular}{cl ccc|cc}", r"\hline",
             "& & " + " & ".join(rf"\multicolumn{{{len(ms)}}}{{c}}{{\textbf{{{dl}}}}}"
                                  for dl, _, ms in REAL_DATASETS) + r" \\",
             rf"\cline{{3-{2 + len(columns)}}}",
             "& & " + " & ".join(rf"\textbf{{{REAL_METRICS[m][0]}}}" for _, _, ms in REAL_DATASETS for m in ms) + r" \\",
             r"\hline"]
    for block, _, groups in REAL_BLOCKS:
        lines += [rf"\multicolumn{{{2 + len(columns)}}}{{l}}{{\textit{{{block}}}}} \\", r"\hline"]
        for gi, (glabel, nrows, rows) in enumerate(groups):
            if gi:
                lines.append(rf"\cline{{2-{2 + len(columns)}}}")
            lines.append(rf"\multirow{{{nrows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{glabel}}}}}")
            for label, _ in rows:
                vals = [cell[(block, label, col)].tex(markers) for col in columns]
                lines.append(f"& {label} & " + " & ".join(vals) + r" \\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def build_semisynthetic(datasets: dict[str, Dataset], ds_names: list[dict[str, str]]) -> list[Cell]:
    cells = []
    for (block, suffix, baseline, _), sources in zip(SEMI_BLOCKS, ds_names):
        for label, family, ranked in SEMI_ROWS:
            model = family if family == "Oracle" else family + suffix
            policies = SEMI_MODEL_POLICIES.get(family, [p for p, _ in SEMI_POLICIES])
            for src_label, ds_name in sources.items():
                ds = datasets[ds_name]
                common = dict(dataset=ds_name, model_name=model, baseline_model=baseline, ranked=ranked)
                for m, hdr, dec in SEMI_TOPK_METRICS:
                    cells.append(Cell("semisynthetic", block, (label, ""), f"Source: {src_label} | {hdr}",
                                      ds, model, m, dec, baseline_col=m, slate_type="", metric=m, **common))
                for prefix, ptype in SEMI_POLICIES:
                    for m, hdr, dec in SEMI_SLATE_METRICS:
                        shown = prefix in policies
                        cells.append(Cell("semisynthetic", block, (label, ptype),
                                          f"Source: {src_label} | {ptype} | {hdr}",
                                          ds if shown else None, model, f"{prefix}.{m}", dec,
                                          baseline_col=f"{BASELINE_POLICY}.{m}", slate_type=ptype,
                                          rank_key=f"Source: {src_label} | {hdr}",
                                          metric=m, **common))
    return cells


SEMI_HEADER = r"""\addtolength{\tabcolsep}{-0.1em}
{\small
\begin{tabular}{l cc cc l ccc ccc}
\toprule
% --- Header Structure ---
& \multicolumn{4}{c}{\textbf{Top-k Performance}}
&
& \multicolumn{6}{c}{\textbf{Slate Performance}} \\
\cmidrule(lr){2-5} \cmidrule(lr){7-12}

& \multicolumn{2}{c}{Source: MNL} & \multicolumn{2}{c}{Source: Diag. MXL}
&
& \multicolumn{3}{c}{Source: MNL} & \multicolumn{3}{c}{Source: Diag. MXL} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){7-9} \cmidrule(lr){10-12}

\textbf{Model} & NLL $\downarrow$ & NDCG $\uparrow$ & NLL $\downarrow$ & NDCG $\uparrow$ & \textbf{Slate Type@4} & Choice\% $\uparrow$ & Agg.Div. $\uparrow$ & IID\% $\uparrow$ & Choice\% $\uparrow$ & Agg.Div. $\uparrow$ & IID\% $\uparrow$ \\
\midrule"""
SEMI_FOOTER = r"""\bottomrule
\end{tabular}
}
\addtolength{\tabcolsep}{0.1em}"""
SEMI_ROW_COMMENTS = {"MNL": "MNL", "MNL_Max": "MNL-Multi", "MNL_Multi": "MXL-Discrete",
                     "MXL": "MXL (ours)", "Oracle": "Oracle"}


def render_semisynthetic(cells: list[Cell], markers: bool, header: str) -> str:
    cell = {(c.block, c.row, c.column): c for c in cells}
    sources = ["MNL", "Diag. MXL"]
    lines = [header, SEMI_HEADER]
    for bi, (block, _, _, _) in enumerate(SEMI_BLOCKS):
        if bi:
            lines.append("% === SASRec-based variants ===")
        lines += [rf"\multicolumn{{12}}{{l}}{{\textit{{{block}}}}} \\", r"\midrule", ""]
        for ri, (label, family, _) in enumerate(SEMI_ROWS):
            policies = SEMI_MODEL_POLICIES.get(family, [p for p, _ in SEMI_POLICIES])
            ptypes = [pt for p, pt in SEMI_POLICIES if p in policies]
            n = len(ptypes)
            wrap = (lambda s: rf"\multirow{{{n}}}{{*}}{{{s}}}") if n > 1 else (lambda s: s)
            lines.append(f"% --- {SEMI_ROW_COMMENTS[family]} ---")
            topk = [cell[(block, (label, ""), f"Source: {s} | {hdr}")].tex(markers)
                    for s in sources for _, hdr, _ in SEMI_TOPK_METRICS]
            lines.append(wrap(label))
            lines.append("& " + " & ".join(wrap(v) for v in topk))
            for pi, ptype in enumerate(ptypes):
                if pi:
                    lines.append("& & & &")
                lines.append(f"& {ptype}")
                vals = [cell[(block, (label, ptype), f"Source: {s} | {ptype} | {hdr}")].tex(markers)
                        for s in sources for _, hdr, _ in SEMI_SLATE_METRICS]
                lines.append("& " + " & ".join(vals) + r" \\")
            if not (bi == len(SEMI_BLOCKS) - 1 and ri == len(SEMI_ROWS) - 1):
                lines += [r"\midrule", ""]
    lines += ["", SEMI_FOOTER]
    return "\n".join(lines) + "\n"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _fmt_p(p: float | None) -> str:
    return "" if p is None else f"{p:.4f}"


def _p_cell(c: Cell | None) -> str:
    if c is None or c.missing:
        return ""
    if c.is_baseline:
        return "baseline"
    return _fmt_p(c.stats["p_better"]) if c.stats else ""


def write_pvalue_tables(cells: list[Cell], out_dir: str) -> None:
    lookup = {(c.table, c.block, c.row, c.column): c for c in cells}

    columns = [f"{dl} {REAL_METRICS[m][0]}" for dl, _, ms in REAL_DATASETS for m in ms]
    with open(os.path.join(out_dir, "real_world_results_pvalues.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "model"] + columns)
        for block, _, groups in REAL_BLOCKS:
            for _, _, rows in groups:
                for label, _ in rows:
                    w.writerow([block, label] + [_p_cell(lookup.get(("real_world", block, label, col)))
                                                 for col in columns])

    sources = ["MNL", "Diag. MXL"]
    topk_cols = [f"Source: {s} | {hdr}" for s in sources for _, hdr, _ in SEMI_TOPK_METRICS]
    slate_hdrs = [(s, hdr) for s in sources for _, hdr, _ in SEMI_SLATE_METRICS]
    with open(os.path.join(out_dir, "semisynthetic_results_pvalues.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "model"] + topk_cols + ["slate_type"]
                   + [f"Source: {s} | {hdr}" for s, hdr in slate_hdrs])
        for block, _, _, _ in SEMI_BLOCKS:
            for label, family, _ in SEMI_ROWS:
                policies = SEMI_MODEL_POLICIES.get(family, [p for p, _ in SEMI_POLICIES])
                name = label.replace("\\ ", " ")
                for pi, ptype in enumerate(pt for p, pt in SEMI_POLICIES if p in policies):
                    topk = [_p_cell(lookup.get(("semisynthetic", block, (label, ""), col))) if pi == 0 else ""
                            for col in topk_cols]
                    slate = [_p_cell(lookup.get(("semisynthetic", block, (label, ptype),
                                                 f"Source: {s} | {ptype} | {hdr}")))
                             for s, hdr in slate_hdrs]
                    w.writerow([block, name] + topk + [ptype] + slate)


def write_pvalues_long(cells: list[Cell], out_dir: str) -> None:
    fields = ["table", "dataset", "block", "model", "model_name", "slate_type", "metric", "column",
              "value", "baseline_model", "baseline_value", "diff", "ci_low", "ci_high",
              "p_better", "p_two_sided", "n_seeds", "marker"]
    with open(os.path.join(out_dir, "pvalues_long.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for c in cells:
            if not c.stats:
                continue
            label = c.row if isinstance(c.row, str) else c.row[0]
            w.writerow({"table": c.table, "dataset": c.info["dataset"], "block": c.block,
                        "model": label.replace("\\ ", " "), "model_name": c.info["model_name"],
                        "slate_type": c.info["slate_type"], "metric": c.info["metric"], "column": c.column,
                        "value": f"{c.value:.6f}", "baseline_model": c.baseline_model,
                        "baseline_value": f"{c.stats['baseline_value']:.6f}",
                        "diff": f"{c.stats['diff']:.6f}", "ci_low": f"{c.stats['ci_low']:.6f}",
                        "ci_high": f"{c.stats['ci_high']:.6f}", "p_better": _fmt_p(c.stats["p_better"]),
                        "p_two_sided": _fmt_p(c.stats["p_two_sided"]), "n_seeds": c.stats["n_seeds"],
                        "marker": {MARK_STAR: "*", MARK_DAGGER: "dagger"}.get(c.marker, "")})


def legend(cells: list[Cell], n_boot: int) -> str:
    seeds = {}
    for c in cells:
        if c.stats:
            seeds.setdefault(c.table, set()).add(c.stats["n_seeds"])
    n = ", ".join(f"{t.replace('_', '-')}: n = {min(v)}" + (f"-{max(v)}" if max(v) > min(v) else "")
                  for t, v in seeds.items())
    return (f"{MARK_STAR} / {MARK_DAGGER}: significantly better than MNL (same dataset and block; for "
            f"slate policies: MNL's top-k slate) at p<{ALPHA_STAR:g} / p<{ALPHA_DAGGER:g}, one-sided paired "
            f"bootstrap over training seeds ({n}; {n_boot:,} resamples, uncorrected). No mark: not "
            f"significantly better. Bold: best, underline: second best per column (Oracle excluded).\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the result tables of the paper from data/results/.")
    parser.add_argument("--results_base", default=RESULTS_BASE)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--min_seed", type=int, default=MIN_SEED)
    parser.add_argument("--no_markers", action="store_true",
                        help="write the plain tables without significance markers (no bootstrap)")
    parser.add_argument("--semisynth_explicit", nargs=2, metavar=("MNL_DS", "MXL_DS"), default=None,
                        help="results dirs of the explicit-embedding DGP datasets (default: glob)")
    parser.add_argument("--semisynth_sasrec", nargs=2, metavar=("MNL_DS", "MXL_DS"), default=None,
                        help="results dirs of the SASRec DGP datasets (default: glob)")
    args = parser.parse_args(argv)

    overrides = [args.semisynth_explicit, args.semisynth_sasrec]
    semi_names = []
    for (_, _, _, patterns), ov in zip(SEMI_BLOCKS, overrides):
        semi_names.append({src: find_dataset(args.results_base, pat, ov[i] if ov else None)
                           for i, (src, pat) in enumerate(patterns.items())})

    needed = [d for _, d, _ in REAL_DATASETS] + [d for s in semi_names for d in s.values()]
    datasets = {}
    for name in needed:
        datasets[name] = Dataset(name, args.results_base, args.min_seed)
        print(f"{name}: {int(datasets[name].n_runs.sum())} runs, seeds per model: "
              f"{datasets[name].n_seeds.to_dict()}")

    cells = build_real_world(datasets) + build_semisynthetic(datasets, semi_names)
    assign_ranks(cells)
    if not args.no_markers:
        run_bootstrap(cells, args.n_boot, args.bootstrap_seed)

    stamp = (f"% generated by experiments/paper_tables.py (commit {_git_commit()}, "
             f"{datetime.now():%Y-%m-%d %H:%M}); "
             + ("no significance markers" if args.no_markers
                else f"paired seed bootstrap, n_boot={args.n_boot}, seed={args.bootstrap_seed}"))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "real_world_results.tex"), "w") as fh:
        fh.write(render_real_world(cells, not args.no_markers, stamp))
    with open(os.path.join(args.out_dir, "semisynthetic_results.tex"), "w") as fh:
        fh.write(render_semisynthetic(cells, not args.no_markers, stamp))
    if not args.no_markers:
        write_pvalue_tables(cells, args.out_dir)
        write_pvalues_long(cells, args.out_dir)
        with open(os.path.join(args.out_dir, "legend.txt"), "w") as fh:
            fh.write(legend(cells, args.n_boot))
    print(f"wrote {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
