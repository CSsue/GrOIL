import os
import glob
import json
import math
from collections import defaultdict, Counter
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde

from config import cfg


GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

def parse_graphml(filepath):
    """Parse a GraphML file and return nodes/edges with their attributes."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    keys = {}
    for key in root.findall(f"{{{GRAPHML_NS}}}key"):
        keys[key.attrib["id"]] = {
            "name": key.attrib.get("attr.name", ""),
            "for": key.attrib.get("for", ""),
            "type": key.attrib.get("attr.type", "string"),
        }
    graph = root.find(f"{{{GRAPHML_NS}}}graph")
    if graph is None: return [], []
    nodes, edges = [], []
    for node in graph.findall(f"{{{GRAPHML_NS}}}node"):
        attrs = {"id": node.attrib["id"]}
        for data in node.findall(f"{{{GRAPHML_NS}}}data"):
            key_id = data.attrib.get("key", "")
            key_info = keys.get(key_id, {})
            attr_name = key_info.get("name", key_id)
            attrs[attr_name] = data.text
        nodes.append(attrs)
    for edge in graph.findall(f"{{{GRAPHML_NS}}}edge"):
        attrs = {"source": edge.attrib["source"], "target": edge.attrib["target"]}
        for data in edge.findall(f"{{{GRAPHML_NS}}}data"):
            key_id = data.attrib.get("key", "")
            key_info = keys.get(key_id, {})
            attr_name = key_info.get("name", key_id)
            attrs[attr_name] = data.text
        edges.append(attrs)
    return nodes, edges

def percentile(sorted_data, p):
    if not sorted_data: return None
    idx = (len(sorted_data) - 1) * p / 100
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_data): return sorted_data[-1]
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac

def mean(data): return sum(data) / len(data) if data else 0

def stdev(data):
    if len(data) < 2: return 0
    m = mean(data)
    return math.sqrt(sum((x - m) ** 2 for x in data) / (len(data) - 1))

def plot_scores(all_scores, output_path):
    if not all_scores:
        print("No scores to plot.")
        return

    scores = np.array(all_scores)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    color = "#2c5f8a"
    light_color = "#d0e4f2"

    ax = axes[0]
    kde = gaussian_kde(scores, bw_method="scott")
    x = np.linspace(scores.min() - 0.05, scores.max() + 0.05, 500)
    y = kde(x)
    ax.plot(x, y, color=color, linewidth=1.8)
    ax.fill_between(x, y, alpha=0.2, color=color)
    ax.set_xlabel("Score", labelpad=6)
    ax.set_ylabel("Density", labelpad=6)
    ax.set_title("Score density", pad=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    median = np.median(scores)
    ax.axvline(median, color="#888", linewidth=1, linestyle="--")
    ax.text(median + 0.01, ax.get_ylim()[1] * 0.92,
            f"median = {median:.2f}", fontsize=9, color="#555")

    ax2 = axes[1]
    bp = ax2.boxplot(
        scores,
        vert=True,
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color=color, linewidth=2),
        boxprops=dict(facecolor=light_color, color=color, linewidth=1.2),
        whiskerprops=dict(color=color, linewidth=1.2),
        capprops=dict(color=color, linewidth=1.2),
        flierprops=dict(marker="o", color=color, alpha=0.3,
                        markersize=3, linestyle="none"),
    )
    ax2.set_ylabel("Score", labelpad=6)
    ax2.set_title("Score distribution", pad=10)
    ax2.set_xticks([])
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    q1, q3 = np.percentile(scores, [25, 75])
    ax2.text(1.32, median, f"median\n{median:.2f}", va="center",
             fontsize=8.5, color="#555")
    ax2.text(1.32, q1, f"Q1 = {q1:.2f}", va="center", fontsize=8, color="#777")
    ax2.text(1.32, q3, f"Q3 = {q3:.2f}", va="center", fontsize=8, color="#777")

    fig.suptitle(
        f"Edge score distribution  ($n$ = {len(scores):,})",
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved to: {output_path}")


def analyze_and_save(input_folder, output_file_report, output_file_score_plot):
    pattern = f"{input_folder}/*.graphml"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching: {pattern}")
        return

    all_node_types, all_entity_types, all_edge_types, all_edge_labels = Counter(), Counter(), Counter(), Counter()
    all_scores, all_fact_comparison_values = [], Counter()
    per_file_stats, files_with_errors = [], []

    for filepath in files:
        fname = os.path.basename(filepath)
        try:
            nodes, edges = parse_graphml(filepath)
        except Exception as e:
            files_with_errors.append({"file": fname, "error": str(e)})
            continue

        node_types, entity_types_in_file = Counter(), Counter()
        for n in nodes:
            ntype = n.get("type", "unknown")
            node_types[ntype] += 1
            all_node_types[ntype] += 1
            if ntype == "entity":
                raw = n.get("data", "")
                if raw:
                    try:
                        d = json.loads(raw)
                        etype = d.get("type", "unknown")
                        entity_types_in_file[etype] += 1
                        all_entity_types[etype] += 1
                    except: pass

        edge_types, edge_labels, scores, fact_comparison_vals = Counter(), Counter(), [], Counter()
        for e in edges:
            et = e.get("edge_type", "unknown"); edge_types[et] += 1; all_edge_types[et] += 1
            lbl = e.get("label", None)
            if lbl: edge_labels[lbl] += 1; all_edge_labels[lbl] += 1
            score_str = e.get("score", None)
            if score_str:
                try:
                    s_val = float(score_str)
                    scores.append(s_val); all_scores.append(s_val)
                except: pass
            fc = e.get("fact_comparison", None)
            if fc: fact_comparison_vals[fc] += 1; all_fact_comparison_values[fc] += 1

        per_file_stats.append({
            "file": fname, "nodes": len(nodes), "edges": len(edges),
            "node_types": dict(node_types), "entity_types": dict(entity_types_in_file),
            "edge_types": dict(edge_types), "edge_labels": dict(edge_labels),
            "scores": scores, "fact_comparison": dict(fact_comparison_vals),
        })

    node_counts = [s["nodes"] for s in per_file_stats]
    edge_counts = [s["edges"] for s in per_file_stats]
    sorted_scores = sorted(all_scores)

    report = {
        "metadata": {"path": pattern, "total_files_found": len(files)},
        "summary": {
            "files_successfully_parsed": len(per_file_stats),
            "files_with_errors": files_with_errors,
            "total_nodes": sum(node_counts),
            "total_edges": sum(edge_counts),
            "node_stats": {"min": min(node_counts), "max": max(node_counts), "mean": mean(node_counts), "stdev": stdev(node_counts)},
            "edge_stats": {"min": min(edge_counts), "max": max(edge_counts), "mean": mean(edge_counts), "stdev": stdev(edge_counts)}
        },
        "node_analysis": {
            "all_node_types": dict(all_node_types),
            "all_entity_types": dict(all_entity_types)
        },
        "edge_analysis": {
            "all_edge_types": dict(all_edge_types),
            "all_edge_labels": dict(all_edge_labels),
            "fact_comparison": dict(all_fact_comparison_values),
            "score_distribution": {
                "count": len(all_scores),
                "min": min(all_scores) if all_scores else 0,
                "max": max(all_scores) if all_scores else 0,
                "mean": mean(all_scores),
                "median": percentile(sorted_scores, 50)
            }
        },
    }

    os.makedirs(os.path.dirname(output_file_report), exist_ok=True)
    with open(output_file_report, 'w') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"Analysis complete. Report saved to: {output_file_report}")

    plot_scores(sorted_scores, output_path=output_file_score_plot)


if __name__ == "__main__":
    analyze_and_save(
        input_folder=cfg.GRAPHS_FULL_DIR,
        output_file_report=cfg.ANALYSIS_REPORT_FULL,
        output_file_score_plot=cfg.SCORE_PLOT_FULL,
    )

    analyze_and_save(
        input_folder=cfg.GRAPHS_STEP1_DIR,
        output_file_report=cfg.ANALYSIS_REPORT_STEP1,
        output_file_score_plot=cfg.SCORE_PLOT_STEP1,
    )

    analyze_and_save(
        input_folder=cfg.GRAPHS_STEP2_DIR,
        output_file_report=cfg.ANALYSIS_REPORT_STEP2,
        output_file_score_plot=cfg.SCORE_PLOT_STEP2,
    )