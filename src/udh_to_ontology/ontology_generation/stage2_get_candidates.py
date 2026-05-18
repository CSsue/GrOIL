"""
Stage 2 — Concept Candidate Extraction
========================================
Input:  All stage1 batch JSON files from a stage1_out directory.
Output: candidate_pool.json — a flat list of candidates, each being a
        (string, origin_tag, source_type) triple with frequency counts.

Four signals are mined:
  A — Entity (norm_type, split_role) pairs   → class + subclass candidates
  B — Edge labels with src/tgt type context  → object property candidates
  C — Fact names                             → object property / axiom candidates
  D — Chunk topics                           → supplementary class candidates

Frequency filtering (MIN_DOC_FREQ) is applied per origin separately.

Usage:
    python stage2_get_candidates.py
    python stage2_get_candidates.py --stage1_dir ./stage1_out --output_dir ./stage2_out
    python stage2_get_candidates.py --min_doc_freq 3
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import cfg


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    string:      str              # the raw string label
    origin:      str              # entity_type | entity_role | edge_label | fact_name | chunk_topic
    parent:      str | None       # for entity_role: the norm_type it was seen under
    frequency:   int              # total occurrences across all documents
    doc_freq:    int              # number of distinct documents it appears in
    doc_list:    list = field(default_factory=list)
    meta:        dict = field(default_factory=dict)  # extra context (e.g. edge src/tgt types)


# ---------------------------------------------------------------------------
# Load all stage1 batch files
# ---------------------------------------------------------------------------

def load_all_doc_records(stage1_dir: Path) -> list[dict]:
    batch_files = sorted(stage1_dir.glob("batch_*_stage1.json"))
    if not batch_files:
        print(f"No stage1 batch files found in {stage1_dir}", file=sys.stderr)
        sys.exit(1)

    all_records = []
    for bf in batch_files:
        with open(bf) as f:
            all_records.extend(json.load(f))
    return all_records


# ---------------------------------------------------------------------------
# Helper: get the normalized type of a node by its id
# ---------------------------------------------------------------------------

def build_node_type_index(doc: dict) -> dict[str, str]:
    """
    Returns {node_id: type_string} for every node in the document.
    Entities get their norm type; facts and chunks get sentinel strings.
    """
    index = {}
    for ent in doc["entities_normalized"]:
        index[ent["id"]] = ent["type"]
    for fact in doc["facts"]:
        index[fact["id"]] = "__FACT__"
    for chunk in doc.get("chunks", []):
        index[chunk["id"]] = "__CHUNK__"
    return index


# ---------------------------------------------------------------------------
# Signal A — Entity (norm_type, split_role) pairs
# ---------------------------------------------------------------------------

def signal_a(all_docs: list[dict]) -> list[Candidate]:
    type_counter: dict[str, dict] = defaultdict(lambda: {"freq": 0, "docs": set()})
    role_counter: dict[tuple, dict] = defaultdict(lambda: {"freq": 0, "docs": set()})

    for doc in all_docs:
        src = doc["source_file"]
        for ent in doc["entities_normalized"]:
            t = ent["type"]
            type_counter[t]["freq"] += 1
            type_counter[t]["docs"].add(src)
            for role in ent["roles"]:
                key = (t, role)
                role_counter[key]["freq"] += 1
                role_counter[key]["docs"].add(src)

    candidates = []
    for t, stats in type_counter.items():
        candidates.append(Candidate(
            string=t, origin="entity_type", parent=None,
            frequency=stats["freq"], doc_freq=len(stats["docs"]),
            doc_list=list(stats["docs"])
        ))
    for (t, role), stats in role_counter.items():
        candidates.append(Candidate(
            string=role, origin="entity_role", parent=t,
            frequency=stats["freq"], doc_freq=len(stats["docs"]),
            doc_list=list(stats["docs"])
        ))
    return candidates


# ---------------------------------------------------------------------------
# Signal B — Edge labels with source/target type context
# ---------------------------------------------------------------------------

def signal_b(all_docs: list[dict]) -> list[Candidate]:
    label_stats: dict[str, dict] = defaultdict(lambda: {
        "src_types": Counter(), "tgt_types": Counter(), "freq": 0, "docs": set()
    })

    for doc in all_docs:
        src_file  = doc["source_file"]
        node_types = build_node_type_index(doc)

        for bucket_name, edge_list in doc["edge_buckets"].items():
            for edge in edge_list:
                label    = edge.get("label") or bucket_name
                src_type = node_types.get(edge["source"], "__UNKNOWN__")
                tgt_type = node_types.get(edge["target"], "__UNKNOWN__")

                label_stats[label]["src_types"][src_type] += 1
                label_stats[label]["tgt_types"][tgt_type] += 1
                label_stats[label]["freq"]  += 1
                label_stats[label]["docs"].add(src_file)

    candidates = []
    for label, stats in label_stats.items():
        candidates.append(Candidate(
            string=label, origin="edge_label", parent=None,
            frequency=stats["freq"], doc_freq=len(stats["docs"]),
            doc_list=list(stats["docs"]),
            meta={
                "src_types": dict(stats["src_types"].most_common(5)),
                "tgt_types": dict(stats["tgt_types"].most_common(5)),
            },
        ))
    return candidates


# ---------------------------------------------------------------------------
# Signal C — Fact names
# ---------------------------------------------------------------------------

def signal_c(all_docs: list[dict]) -> list[Candidate]:
    name_stats: dict[str, dict] = defaultdict(lambda: {"freq": 0, "docs": set()})

    for doc in all_docs:
        src = doc["source_file"]
        for fact in doc["facts"]:
            name = fact.get("name")
            if name:
                name_stats[name]["freq"] += 1
                name_stats[name]["docs"].add(src)
                name_stats[name].setdefault("sentences", []).append(fact.get("sentence", ""))

    return [
        Candidate(
            string=name, origin="fact_name", parent=None,
            frequency=stats["freq"], doc_freq=len(stats["docs"]),
            meta={"sentence": Counter(stats["sentences"]).most_common(1)[0][0]},
            doc_list=list(stats["docs"])
        )
        for name, stats in name_stats.items()
    ]


# ---------------------------------------------------------------------------
# Signal D — Chunk topics
# ---------------------------------------------------------------------------

def signal_d(all_docs: list[dict]) -> list[Candidate]:
    topic_stats: dict[str, dict] = defaultdict(lambda: {"freq": 0, "docs": set()})

    for doc in all_docs:
        src = doc["source_file"]
        for chunk in doc.get("chunks", []):
            for topic in chunk.get("topics", []):
                if topic:
                    topic_stats[topic]["freq"] += 1
                    topic_stats[topic]["docs"].add(src)
        for fact in doc["facts"]:
            for topic in fact.get("chunk_topics", []):
                if topic:
                    topic_stats[topic]["freq"] += 1
                    topic_stats[topic]["docs"].add(src)

    return [
        Candidate(
            string=topic, origin="chunk_topic", parent=None,
            frequency=stats["freq"], doc_freq=len(stats["docs"]),
            doc_list=list(stats["docs"])
        )
        for topic, stats in topic_stats.items()
    ]


# ---------------------------------------------------------------------------
# Frequency filtering — applied per origin tag
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict[str, int] = {
    "entity_type": 1,
    "entity_role": 1,
    "edge_label":  1,
    "fact_name":   1,
    "chunk_topic": 1,
}

def filter_candidates(
    candidates: list[Candidate],
    thresholds: dict[str, int],
    global_min: int,
) -> list[Candidate]:
    kept = []
    for c in candidates:
        threshold = max(thresholds.get(c.origin, global_min), global_min)
        if c.doc_freq >= threshold:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 — Concept Candidate Extraction")
    p.add_argument(
        "--stage1_dir",
        default=cfg.STAGE1_OUT_DIR,
        help="Directory containing stage1 batch JSONs (default: STAGE1_OUT_DIR from .env)",
    )
    p.add_argument(
        "--output_dir",
        default=cfg.STAGE2_OUT_DIR,
        help="Directory for stage2 output (default: STAGE2_OUT_DIR from .env)",
    )
    p.add_argument(
        "--min_doc_freq",
        type=int,
        default=1,
        help="Global minimum doc frequency floor (default 1)",
    )
    return p.parse_args()


def main():
    args       = parse_args()
    stage1_dir = Path(args.stage1_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_docs = load_all_doc_records(stage1_dir)

    raw_candidates = (
        signal_a(all_docs) +
        signal_b(all_docs) +
        signal_c(all_docs) +
        signal_d(all_docs)
    )

    filtered = filter_candidates(raw_candidates, DEFAULT_THRESHOLDS, args.min_doc_freq)

    out = [asdict(c) for c in filtered]
    out_path = output_dir / "candidate_pool.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(filtered)} candidates to {out_path} (filtered from {len(raw_candidates)} raw candidates)\n")

    origin_counts_raw      = Counter(c.origin for c in raw_candidates)
    origin_counts_filtered = Counter(c.origin for c in filtered)
    stats = {
        "total_documents":    len(all_docs),
        "candidates_before_filter": len(raw_candidates),
        "candidates_after_filter":  len(filtered),
        "by_origin": {
            origin: {
                "before": origin_counts_raw[origin],
                "after":  origin_counts_filtered[origin],
                "threshold_used": max(DEFAULT_THRESHOLDS.get(origin, args.min_doc_freq), args.min_doc_freq),
            }
            for origin in sorted(origin_counts_raw)
        },
    }
    print(json.dumps(stats, indent=2))
    with open(output_dir / "stage2_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStage 2 candidate extraction complete. Stats saved to {output_dir / 'stage2_stats.json'}\n")

if __name__ == "__main__":
    main()