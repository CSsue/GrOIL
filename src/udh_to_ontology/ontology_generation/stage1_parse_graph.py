"""
Stage 1 — Graph Parsing & Normalization
========================================
Input:  A batch of GraphML files (default batch size = 5)
Output: Typed pools (chunks, facts, entities) + edge buckets per document,
        written as one JSON file per batch to output_dir.

Type normalization is done purely in-place:
  - strip whitespace, uppercase, spaces → underscores
  - fix known typos (ENDORSAMENT, ENDORSMENT → ENDORSEMENT)
No external type_map file is read or written.

Usage:
    python stage1_parse_graph.py
    python stage1_parse_graph.py --input_dir ./graphs --output_dir ./stage1_out
    python stage1_parse_graph.py --batch_size 10
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

from config import cfg

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Edge type buckets
EDGE_BUCKETS = {"chunk_fact", "fact_entity", "entity_fact", "fact_fact", "chunk_chunk", "entity_entity"}

# Typo-only corrections (everything else normalizes to UPPER_SNAKE automatically)
TYPO_FIXES = {
    "ENDORSAMENT": "ENDORSEMENT",
    "ENDORSMENT":  "ENDORSEMENT",
}


# ---------------------------------------------------------------------------
# Type normalization — purely in-place, no file I/O
# ---------------------------------------------------------------------------

def normalize_type(raw_type: str) -> str:
    """
    Normalize a raw entity type string:
      1. Strip whitespace
      2. Uppercase
      3. Internal spaces → underscores
      4. Apply typo fixes
    """
    normalized = raw_type.strip().upper().replace(" ", "_")
    return TYPO_FIXES.get(normalized, normalized)


def split_compound_role(role: str) -> list[str]:
    """Split 'Policy Owner and Insured' or 'Primary / Contingent' into atomic roles."""
    if not role:
        return []
    parts = re.split(r"\s+and\s+|/|,", role, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# 1.1  Parse a single GraphML file
# ---------------------------------------------------------------------------

def parse_graphml(filepath: Path) -> dict:
    G = nx.read_graphml(str(filepath))

    chunks, facts, entities = [], [], []

    for node_id, node_data in G.nodes(data=True):
        raw_data  = node_data.get("data", "{}")
        node_type = node_data.get("type", "unknown").lower()

        try:
            parsed = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        except json.JSONDecodeError:
            parsed = {"_raw": raw_data}

        record = {"id": node_id, "data": parsed}

        if node_type == "chunk":
            chunks.append(record)
        elif node_type == "fact":
            facts.append(record)
        elif node_type == "entity":
            entities.append(record)

    edge_buckets: dict[str, list] = defaultdict(list)
    for src, tgt, edge_data in G.edges(data=True):
        etype = edge_data.get("edge_type", "unknown")
        bucket = etype if etype in EDGE_BUCKETS else "other"
        edge_buckets[bucket].append({
            "source":      src,
            "target":      tgt,
            "label":       edge_data.get("label"),
            "description": edge_data.get("description"),
            "score":       edge_data.get("score"),
        })

    return {
        "source_file":  filepath.name,
        "chunks":       chunks,
        "facts":        facts,
        "entities":     entities,
        "edge_buckets": dict(edge_buckets),
    }


# ---------------------------------------------------------------------------
# 1.2 + 1.3  Entity normalization (type + compound role splitting)
# ---------------------------------------------------------------------------

def normalize_entities(entities: list[dict]) -> list[dict]:
    normalized = []
    for ent in entities:
        d        = ent["data"]
        raw_type = d.get("type", "UNKNOWN")
        raw_role = d.get("role", "")
        normalized.append({
            "id":                 ent["id"],
            "name":               d.get("name", ""),
            "type":               normalize_type(raw_type),
            "roles":              split_compound_role(raw_role),
            "relational_anchors": d.get("relational_anchors", []),
        })
    return normalized


# ---------------------------------------------------------------------------
# 1.4  Reality check — warns only, never raises
# ---------------------------------------------------------------------------

def reality_check(rec: dict) -> None:
    n_e = len(rec["entities"])
    n_f = len(rec["facts"])
    n_c = len(rec["chunks"])
    if n_e < 10:
        log.warning("%s: very few entities (%d) — possibly malformed", rec["source_file"], n_e)
    if n_e > 50_000:
        log.warning("%s: unusually many entities (%d) — possibly malformed", rec["source_file"], n_e)
    if n_c > 0 and n_f < 1:
        log.warning("%s: no facts despite %d chunks — check graph structure", rec["source_file"], n_c)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(filepaths: list[Path]) -> list[dict]:
    records = []
    for fp in filepaths:
        try:
            raw = parse_graphml(fp)
        except Exception as e:
            log.error("Failed to parse %s: %s", fp.name, e)
            continue

        raw["entities_normalized"] = normalize_entities(raw["entities"])
        reality_check(raw)
        records.append(raw)
    return records


def save_batch(records: list[dict], output_dir: Path, batch_idx: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"batch_{batch_idx:03d}_stage1.json"

    summary = []
    for rec in records:
        summary.append({
            "source_file": rec["source_file"],
            "counts": {
                "chunks":       len(rec["chunks"]),
                "facts":        len(rec["facts"]),
                "entities":     len(rec["entities"]),
                "edge_buckets": {k: len(v) for k, v in rec["edge_buckets"].items()},
            },
            "entities_normalized": rec["entities_normalized"],
            "facts": [
                {"id": f["id"], "name": f["data"].get("name"), "sentence": f["data"].get("sentence"), "chunk_topics": f["data"].get("chunk_topics")}
                for f in rec["facts"]
            ],
            "edge_buckets": rec["edge_buckets"],
        })

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 — Graph Parsing & Normalization")
    p.add_argument(
        "--input_dir",
        default=cfg.GRAPHS_STEP2_DIR,
        help="Directory containing GraphML files (default: GRAPHS_STEP2_DIR from .env)",
    )
    p.add_argument(
        "--output_dir",
        default=cfg.STAGE1_OUT_DIR,
        help="Directory for stage 1 outputs (default: STAGE1_OUT_DIR from .env)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Files per batch (default 5)",
    )
    return p.parse_args()


def main():
    args       = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size

    all_files = sorted(input_dir.glob("*.graphml"))
    if not all_files:
        print(f"No .graphml files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    for batch_idx, start in enumerate(range(0, len(all_files), batch_size)):
        batch_files = all_files[start : start + batch_size]
        records = process_batch(batch_files)
        save_batch(records, output_dir, batch_idx + 1)

    print(f"Parsed {len(all_files)} files into {output_dir} in batches of {batch_size}.")

if __name__ == "__main__":
    main()