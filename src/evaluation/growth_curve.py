"""
Growth Curve Analysis — multi-dataset  (Option A: clean two-panel style)
=========================================================================
Plots ontology growth curves for TWO datasets on separate side-by-side axes.

  - Same color per element type (classes / properties / restrictions)
  - Left panel  = small corpus  (solid lines)
  - Right panel = large corpus  (solid lines, independent x-axis scale)
  - No fill shading — lines only with end-point dots
  - End-of-line annotations show final counts
  - Panel letters (a) / (b) in top-left corner of each panel
  - Shared y-axis label on the left; shared x-axis label centred below

Usage:
    python growth_curve.py
    python growth_curve.py \
        --dataset1_slug "10_contracts_dataset"  \
        --dataset2_slug "100_contracts_dataset" \
        --output_dir    ./growth_out
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines

from config import cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# String → doc index
# ---------------------------------------------------------------------------

def build_string_doc_index(candidate_pool: list[dict]) -> dict[str, set]:
    index: dict[str, set] = defaultdict(set)
    for c in candidate_pool:
        s    = c.get("string", "")
        docs = set(c.get("doc_list", []))
        index[s]         |= docs
        index[s.lower()] |= docs
    return index


def lookup_docs(string: str, index: dict[str, set]) -> set:
    if string in index:
        return index[string]
    lo = string.lower()
    if lo in index:
        return index[lo]
    return set()


# ---------------------------------------------------------------------------
# Supporting-doc resolution
# ---------------------------------------------------------------------------

def cluster_supporting_docs(cluster: dict, index: dict[str, set]) -> set:
    docs: set = set()
    for member in cluster.get("members", []):
        s = member.get("string") or member.get("label") or ""
        if s:
            docs |= lookup_docs(s, index)
    if not docs:
        label = cluster.get("label") or cluster.get("name") or ""
        if label:
            docs |= lookup_docs(label, index)
    return docs


def axiom_supporting_docs(axiom: dict, index: dict[str, set]) -> set:
    docs: set = set()
    for key in ("property", "owl_class", "filler", "label", "name",
                "domain", "range", "subject", "object"):
        val = axiom.get(key)
        if isinstance(val, str) and val:
            docs |= lookup_docs(val, index)
    return docs


# ---------------------------------------------------------------------------
# First-seen indices
# ---------------------------------------------------------------------------

def first_seen_indices(
    clusters_or_axioms: list[dict],
    doc_order: list[str],
    index: dict[str, set],
    resolve_fn,
) -> list[int]:
    doc_rank  = {doc: i for i, doc in enumerate(doc_order)}
    result    = []
    n_missing = 0
    for elem in clusters_or_axioms:
        docs   = resolve_fn(elem, index)
        ranked = [doc_rank[d] for d in docs if d in doc_rank]
        if ranked:
            result.append(min(ranked))
        else:
            result.append(len(doc_order))
            n_missing += 1
    if n_missing:
        log.warning(
            "%d / %d elements had no matching docs in the candidate pool.",
            n_missing, len(clusters_or_axioms)
        )
    return result


# ---------------------------------------------------------------------------
# Build growth curve
# ---------------------------------------------------------------------------

def build_curve(
    first_seen_class:    list[int],
    first_seen_property: list[int],
    first_seen_restrict: list[int],
    n_docs: int,
) -> list[dict]:
    def cumulative(indices: list[int]) -> list[int]:
        counts = [0] * (n_docs + 1)
        for idx in indices:
            if idx < n_docs:
                counts[idx] += 1
        for i in range(1, n_docs + 1):
            counts[i] += counts[i - 1]
        return counts[1:]

    cls_curve  = cumulative(first_seen_class)
    prop_curve = cumulative(first_seen_property)
    rest_curve = cumulative(first_seen_restrict)

    return [{"n_docs": 0, "classes": 0, "properties": 0, "restrictions": 0}] + [
        {
            "n_docs":       n + 1,
            "classes":      cls_curve[n],
            "properties":   prop_curve[n],
            "restrictions": rest_curve[n],
        }
        for n in range(n_docs)
    ]


# ---------------------------------------------------------------------------
# Pipeline for one dataset
# ---------------------------------------------------------------------------

def run_pipeline(
    candidate_pool_path: str,
    clusters_path: str,
    axioms_path: str,
    stage1_dir: str,
) -> list[dict]:
    log.info("Loading candidate pool …")
    candidate_pool   = load_json(Path(candidate_pool_path))
    string_doc_index = build_string_doc_index(candidate_pool)

    log.info("Loading clusters …")
    clusters_raw      = load_json(Path(clusters_path))
    class_clusters    = [c for c in clusters_raw.get("class",    []) if not c.get("is_noise")]
    property_clusters = [c for c in clusters_raw.get("property", []) if not c.get("is_noise")]
    axiom_clusters    = [c for c in clusters_raw.get("axiom",    []) if not c.get("is_noise")]

    log.info("Loading axioms …")
    ap     = Path(axioms_path)
    axioms = load_json(ap) if ap.exists() else []  # noqa: F841

    log.info("Building document order …")
    all_docs  = []
    s1_dir    = Path(stage1_dir)
    for bf in sorted(s1_dir.glob("batch_*_stage1.json")):
        with open(bf) as f:
            all_docs.extend(json.load(f))
    doc_order = [doc["source_file"] for doc in all_docs]
    log.info("  %d documents", len(doc_order))

    fs_classes = first_seen_indices(
        class_clusters, doc_order, string_doc_index, cluster_supporting_docs)
    fs_properties = first_seen_indices(
        property_clusters, doc_order, string_doc_index, cluster_supporting_docs)
    fs_restrictions = first_seen_indices(
        axiom_clusters, doc_order, string_doc_index, cluster_supporting_docs)

    return build_curve(fs_classes, fs_properties, fs_restrictions, len(doc_order))


# ---------------------------------------------------------------------------
# Plotting — Option A: two clean independent panels
# ---------------------------------------------------------------------------

COLORS = {
    "classes":      "#2A7A40",
    "properties":   "#2A6099",
    "restrictions": "#C0442A",
}

ELEMENT_LABELS = {
    "classes":      "Classes",
    "properties":   "Properties",
    "restrictions": "Restrictions",
}

ELEMS = ("classes", "properties", "restrictions")

LINE_WIDTH  = 1.4
DOT_RADIUS  = 3.5
PANEL_LETTERS = ("(a) 10-contract dataset", "(b) 100-contract dataset")


def _extract(results: list[dict], key: str) -> list:
    return [r[key] for r in results]


def plot_growth_multi(
    results_list: list[list[dict]],
    dataset_names: list[str],
    output_dir: Path,
) -> None:
    assert len(results_list) == 2 and len(dataset_names) == 2

    # ── Typography / rcParams ─────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          9,
        "axes.linewidth":     0.7,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        "xtick.minor.width":  0.4,
        "ytick.minor.width":  0.4,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "text.usetex":        False,
    })

    # ── Figure: two INDEPENDENT panels (no sharey) ───────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.5, 3.5),
        sharey=False,           # each panel gets its own y-scale
        gridspec_kw={
            "wspace":  0.5,    # enough room for both y-axis labels
            "left":    0.10,
            "right":   0.87,
            "top":     0.91,
            "bottom":  0.20,
        },
    )

    for idx, (ax, results, name, letter) in enumerate(
        zip(axes, results_list, dataset_names, PANEL_LETTERS)
    ):
        n_docs   = _extract(results, "n_docs")
        x_max    = n_docs[-1]
        is_right = (idx == 1)

        # Per-panel y bounds
        y_max_panel = max(max(_extract(results, e)) for e in ELEMS)
        y_upper     = y_max_panel * 1.14   # headroom for annotations

        # Light horizontal grid only
        ax.yaxis.grid(
            True,
            linestyle=":",
            linewidth=0.4,
            color="#C8C8C8",
            alpha=0.8,
            zorder=0,
        )
        ax.set_axisbelow(True)

        for elem in ELEMS:
            values = _extract(results, elem)
            color  = COLORS[elem]

            ax.fill_between(
                n_docs, values,
                alpha=0.12,
                color=color,
                zorder=2,
            )

            ax.plot(
                n_docs, values,
                color=color,
                linewidth=LINE_WIDTH,
                linestyle="solid",
                solid_capstyle="round",
                zorder=3,
            )

            # End-point dot
            ax.plot(
                x_max, values[-1],
                "o",
                color=color,
                markersize=DOT_RADIUS,
                zorder=4,
                clip_on=False,
            )

            # End-value annotation on both panels
            ax.annotate(
                f"{values[-1]:,}",
                xy=(x_max, values[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color=color,
                fontsize=8,
                fontweight="bold",
                fontfamily="sans-serif",
                clip_on=False,
            )

        # Panel letter — top-left corner, slightly inset
        ax.text(
            0.04, 0.96, letter,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            color="#333333",
            fontweight="normal",
        )

        # Limits, ticks, minor ticks
        ax.set_xlim(0, x_max * 1.02)
        ax.set_ylim(0, y_upper)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(axis="both", which="major", pad=3)
        ax.tick_params(axis="both", which="minor", length=2.0)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.spines["left"].set_linewidth(0.7)

        if not is_right:
            ax.set_ylabel("Ontology element count", labelpad=5, fontsize=9)

    # ── Shared x-label centred below both panels ──────────────────────────────
    fig.text(
        0.485, 0.075,
        "Documents ingested",
        ha="center", va="bottom",
        fontsize=9,
        fontfamily="serif",
    )

    # ── Legend — centred below both panels ────────────────────────────────────
    legend_handles = [
        mlines.Line2D(
            [], [],
            color=COLORS[e],
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=ELEMENT_LABELS[e],
        )
        for e in ELEMS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.485, -0.04),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.6,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    for fmt in (["png"]):
        out = output_dir / f"growth_curve.{fmt}"
        fig.savefig(out, format=fmt, bbox_inches="tight")
        log.info("Saved → %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

DATASET_SLUGS = [
    "10_contracts_dataset",
    "100_contracts_dataset",
]

DATASET_DISPLAY_NAMES = [
    "10-contract dataset",
    "100-contract dataset",
]


def paths_for_dataset(slug: str) -> dict:
    def swap(path_str: str) -> str:
        for known_slug in DATASET_SLUGS:
            if known_slug in path_str:
                return path_str.replace(known_slug, slug)
        log.warning("Could not find a dataset slug in cfg path: %s", path_str)
        return path_str

    return {
        "stage1_dir":     swap(str(cfg.STAGE1_OUT_DIR)),
        "candidate_pool": swap(str(cfg.STAGE2_CANDIDATE_POOL)),
        "clusters":       swap(str(Path(cfg.STAGE3_OUT_DIR) / "named_concept_clusters.json")),
        "axioms":         swap(str(Path(cfg.STAGE6_OUT_DIR) / "induced_axioms.json")),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ontology growth curve — two datasets (Option A)")
    p.add_argument("--dataset1_slug", default=DATASET_SLUGS[0])
    p.add_argument("--dataset2_slug", default=DATASET_SLUGS[1])
    p.add_argument("--output_dir",    default=cfg.OUT_DIR)
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slugs         = [args.dataset1_slug, args.dataset2_slug]
    display_names = [
        DATASET_DISPLAY_NAMES[DATASET_SLUGS.index(s)] if s in DATASET_SLUGS
        else s.replace("_", " ")
        for s in slugs
    ]

    results_list = []
    for slug, display in zip(slugs, display_names):
        log.info("=== Dataset: %s ===", display)
        p = paths_for_dataset(slug)
        results = run_pipeline(
            p["candidate_pool"], p["clusters"], p["axioms"], p["stage1_dir"])

        out = output_dir / f"growth_curve_{slug}.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Saved growth data → %s", out)

        results_list.append(results)

    plot_growth_multi(results_list, display_names, output_dir)


if __name__ == "__main__":
    main()