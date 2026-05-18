"""
Usage:  python stage0_clean_graphs.py step1
        python stage0_clean_graphs.py step2 --discover      # Pass A: report only, no changes
        python stage0_clean_graphs.py step2 --apply         # Pass B: apply decisions & write graphs
        python stage0_clean_graphs.py step2 --discover --sim-threshold 0.78 --min-count 3
"""

import argparse, glob, json, os, re, shutil
from collections import Counter
from xml.etree import ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from collections import defaultdict

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
#  PATHS & CONFIG  (all sourced from config.py / .env)
# ─────────────────────────────────────────────────────────────────────────────

INPUT_GLOB       = f"{cfg.GRAPHS_FULL_DIR}/*.graphml"
STEP1_OUTPUT_DIR = cfg.GRAPHS_STEP1_DIR
STEP2_OUTPUT_DIR = cfg.GRAPHS_STEP2_DIR
STEP1_REPORT     = cfg.STEP1_REPORT
STEP2_REPORT     = cfg.STEP2_REPORT
STEP2_DECISIONS  = cfg.STEP2_DECISIONS
SCORE_THRESHOLD  = 0.55

NULL_KEYS = {
    "notrelated", "notapplicable", "na", "nonerelationship",
    "independent", "none", "unrelated", "norelationship", "norelation",
}

SYNONYM_GROUPS = {
    "COMPLEMENTS":         ["complementary", "compliments", "complementing", "is complementary to", "complement to"],
    "CORROBORATES":        ["supports", "validates", "confirms", "affirms", "reinforces", "substantiates", "corroborated by"],
    "EXPLAINS":            ["clarifies", "elaborates", "describes", "illustrates", "details", "elucidates"],
    "CAUSES":              ["results in", "leads to", "produces", "triggers"],
    "DEPENDS_ON":          ["dependent on", "contingent on", "relies on", "requires"],
    "CONDITIONAL_ON":      ["conditioned on", "conditioned by", "subject to"],
    "DERIVES_FROM":        ["derived from", "originates from", "based on", "derives from"],
    "MODIFIES":            ["alters", "changes", "adjusts", "updates", "amends"],
    "INFLUENCES":          ["influenced by", "affects", "impacts"],
    "PRECEDES":            ["followed by", "comes before", "prior to"],
    "FOLLOWS":             ["comes after", "subsequent to", "succeeds"],
    "CONTRADICTS":         ["conflicts with", "opposes", "negates", "contradicted by"],
    "ENABLES":             ["allows", "permits", "facilitates"],
    "REQUIRED_FOR":        ["necessary for", "prerequisite for", "needed for"],
    "APPLIES_TO":          ["applied to", "applicable to", "relevant to"],
    "JUSTIFIES":           ["supports as reason", "provides basis for", "grounds"],
    "CONTEXTUAL_RELATION": ["contextually related", "context of", "provides context"],
    "RESULT_OF":           ["results from", "outcome of", "consequence of"],
    "REFLECTS":            ["indicates", "shows", "demonstrates", "represents"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS  (shared by step 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

NS = "http://graphml.graphdrawing.org/xmlns"
ET.register_namespace("", NS)
ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.strip().lower()) if text else ""

_SYNONYM_LOOKUP = {_norm(k): k for k in SYNONYM_GROUPS} | {
    _norm(syn): canon for canon, syns in SYNONYM_GROUPS.items() for syn in syns
}

def _tag(name):       return f"{{{NS}}}{name}"
def _find(el, name):  return el.findall(_tag(name))

def _key_map(root):
    return {k.attrib.get("attr.name"): k.attrib["id"]
            for k in _find(root, "key") if "attr.name" in k.attrib}

def _get_attr(edge, key_id):
    if not key_id: return None
    for d in _find(edge, "data"):
        if d.attrib.get("key") == key_id: return d.text
    return None

def _set_attr(edge, key_id, value):
    for d in _find(edge, "data"):
        if d.attrib.get("key") == key_id:
            d.text = value; return
    nd = ET.SubElement(edge, _tag("data"))
    nd.attrib["key"] = key_id; nd.text = value

def _to_display_label(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'[_\-]+', ' ', text)
    return re.sub(r' +', ' ', text.strip().lower())

def _load_all_labels(input_dir):
    """Return (label_counter, per_file dict) across all graphml files in dir."""
    global_counter = Counter()
    per_file = {}
    for fp in sorted(glob.glob(f"{input_dir}/*.graphml")):
        fname = os.path.basename(fp)
        try:
            tree = ET.parse(fp)
            root = tree.getroot()
            km = _key_map(root)
            graph = root.find(_tag("graph"))
            if graph is None: continue
            labels = []
            for edge in _find(graph, "edge"):
                lbl = _get_attr(edge, km.get("label"))
                if lbl:
                    labels.append(lbl.strip())
                    global_counter[lbl.strip()] += 1
            per_file[fname] = labels
        except Exception:
            continue
    return global_counter, per_file


def audit_post_normalization_dupes():
    """
    Read-only audit: simulate the remap from step2_decisions.json and count
    how many edges would become duplicates after normalization is applied.
    No files are written.
    """
    if not os.path.exists(STEP2_DECISIONS):
        print(f"  ✗  Decisions file not found: {STEP2_DECISIONS}")
        print(f"     Run step2 --discover first."); return

    with open(STEP2_DECISIONS, encoding="utf-8") as f:
        decisions = json.load(f)

    remove_set, remap = _build_step2_lookup(decisions)

    files = sorted(glob.glob(f"{STEP1_OUTPUT_DIR}/*.graphml"))
    if not files:
        print(f"  No step1-cleaned files found in {STEP1_OUTPUT_DIR}"); return

    print(f"\n{'='*70}")
    print(f"  DUPE AUDIT (post-normalization simulation)  —  {len(files)} files")
    print(f"  Remap entries : {len(remap)}   Remove set : {len(remove_set)}")
    print(f"{'='*70}\n")

    global_dupes   = Counter()
    per_file_dupes = {}
    total_edges    = 0
    total_dupes    = 0

    for fp in sorted(files):
        fname = os.path.basename(fp)
        try:
            tree = ET.parse(fp)
            root = tree.getroot()
        except Exception as e:
            print(f"  ✗  {fname}: {e}"); continue

        km    = _key_map(root)
        graph = root.find(_tag("graph"))
        if graph is None: continue

        seen    = Counter()
        n_edges = 0

        for edge in _find(graph, "edge"):
            src   = edge.attrib.get("source", "")
            tgt   = edge.attrib.get("target", "")
            label = _get_attr(edge, km.get("label"))
            norm  = _norm(label) if label else ""

            if src == tgt:            continue
            if norm in remove_set:    continue

            effective = remap.get(norm, norm)
            triple    = (src, tgt, effective)
            seen[triple] += 1
            n_edges += 1

        file_dupes = {t: c for t, c in seen.items() if c > 1}
        n_dupes    = sum(c - 1 for c in file_dupes.values())

        total_edges += n_edges
        total_dupes += n_dupes
        global_dupes.update({t: c - 1 for t, c in file_dupes.items()})

        if n_dupes:
            per_file_dupes[fname] = n_dupes
            print(f"  {fname:<50}  edges:{n_edges:>5}  dupes:{n_dupes:>4}")

    print(f"\n{'─'*70}  SUMMARY")
    print(f"  Total edges scanned : {total_edges:,}")
    print(f"  Total dupe edges    : {total_dupes:,}  "
          f"({total_dupes/total_edges*100:.2f}% of edges)" if total_edges else "")
    print(f"  Files with dupes    : {len(per_file_dupes)} / {len(files)}")

    if global_dupes:
        print(f"\n  Top 10 duplicated triples (src → tgt via label):")
        for (src, tgt, lbl), cnt in global_dupes.most_common(10):
            print(f"    {cnt+1}×  {src[:20]:<22} →[{lbl}]→ {tgt[:20]}")

    print(f"\n  {'No duplicate edges would be introduced.' if not total_dupes else 'Run step2 --apply to remove them (fix already included).'}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1
# ─────────────────────────────────────────────────────────────────────────────

def _clean_file_step1(filepath, output_path):
    tree = ET.parse(filepath)
    root = tree.getroot()
    km   = _key_map(root)

    graph = root.find(_tag("graph"))
    if graph is None:
        shutil.copy2(filepath, output_path)
        return {"error": "no <graph> element"}

    edges  = _find(graph, "edge")
    stats  = dict(before=len(edges), null=0, score=0, norm=0,
                  null_counts=Counter(), score_by_type=Counter(), norm_counts=Counter())
    remove = []

    for edge in edges:
        label = _get_attr(edge, km.get("label"))
        etype = _get_attr(edge, km.get("edge_type"))
        score = _get_attr(edge, km.get("score"))

        if label and _norm(label) in NULL_KEYS:
            remove.append(edge); stats["null"] += 1; stats["null_counts"][label.strip()] += 1; continue

        if score is not None:
            try:
                if float(score) <= SCORE_THRESHOLD:
                    remove.append(edge); stats["score"] += 1; stats["score_by_type"][etype or "unknown"] += 1; continue
            except ValueError: pass

        if label:
            canon = _SYNONYM_LOOKUP.get(_norm(label))
            if canon and canon != label.strip():
                stats["norm_counts"][f"{label.strip()} → {canon}"] += 1
                stats["norm"] += 1
                label = canon
                _set_attr(edge, km["label"], _to_display_label(label))
            elif label:
                display = _to_display_label(label)
                if display != label.strip():
                    _set_attr(edge, km["label"], display)

    for edge in remove: graph.remove(edge)
    stats["after"] = stats["before"] - stats["null"] - stats["score"]
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    stats["null_counts"]   = dict(stats["null_counts"])
    stats["score_by_type"] = dict(stats["score_by_type"])
    stats["norm_counts"]   = dict(stats["norm_counts"])
    return stats


def run_step1():
    files = sorted(glob.glob(INPUT_GLOB))
    if not files: print(f"No files found: {INPUT_GLOB}"); return
    os.makedirs(STEP1_OUTPUT_DIR, exist_ok=True)

    all_stats, errors = {}, []
    totals = Counter()
    global_null, global_score, global_norm = Counter(), Counter(), Counter()

    print(f"{'='*70}\n  STEP 1  —  {len(files)} files  |  score threshold <= {SCORE_THRESHOLD}\n{'='*70}")

    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        try:
            s = _clean_file_step1(fp, os.path.join(STEP1_OUTPUT_DIR, fname))
        except Exception as e:
            errors.append((fname, str(e))); print(f"  [{i:>3}] ✗ {fname}: {e}"); continue

        if "error" in s:
            errors.append((fname, s["error"])); print(f"  [{i:>3}] ⚠ {fname}: {s['error']}"); continue

        all_stats[fname] = s
        totals.update({"before": s["before"], "null": s["null"], "score": s["score"], "norm": s["norm"]})
        global_null.update(s["null_counts"]); global_score.update(s["score_by_type"]); global_norm.update(s["norm_counts"])
        print(f"  [{i:>3}/{len(files)}] {fname:<45} {s['before']:>5} → {s['after']:>5}  null:{s['null']:>4}  score:{s['score']:>4}  norm:{s['norm']:>3}")

    removed = totals["null"] + totals["score"]
    pct = lambda n: f"{n/totals['before']*100:.1f}%" if totals["before"] else "0%"

    print(f"\n{'─'*70}  SUMMARY")
    print(f"  Edges before/after : {totals['before']:,} → {totals['before']-removed:,}")
    print(f"  Removed null       : {totals['null']:,} ({pct(totals['null'])})")
    print(f"  Removed low-score  : {totals['score']:,} ({pct(totals['score'])})")
    print(f"  Labels normalised  : {totals['norm']:,}")
    print(f"\n  Null labels : " + (", ".join(f"'{l}'×{c}" for l, c in sorted(global_null.items(), key=lambda x:-x[1])) or "(none)"))
    print(f"  Score types : " + (", ".join(f"{t}×{c}" for t, c in sorted(global_score.items(), key=lambda x:-x[1])) or "(none)"))

    report = {
        "step": "step1", "score_threshold": SCORE_THRESHOLD,
        "summary": dict(totals) | {"removed": removed, "after": totals["before"] - removed,
                                   "removed_pct": round(removed/totals["before"]*100, 2) if totals["before"] else 0},
        "null_label_breakdown": dict(global_null), "score_by_edge_type": dict(global_score),
        "normalization_breakdown": dict(global_norm), "per_file": all_stats, "errors": errors,
    }
    os.makedirs(os.path.dirname(STEP1_REPORT), exist_ok=True)
    with open(STEP1_REPORT, "w", encoding="utf-8") as f: json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report  → {STEP1_REPORT}\n  Cleaned → {STEP1_OUTPUT_DIR}/\n{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — DISCOVERY (Pass A)
# ─────────────────────────────────────────────────────────────────────────────

def _discover(label_counter, sim_threshold=0.75, min_count=3):
    all_labels  = list(label_counter.keys())
    label_norms = [_norm(l) for l in all_labels]

    print(f"\n  Encoding {len(all_labels)} unique labels …")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(all_labels, normalize_embeddings=True, show_progress_bar=True)

    print("  Clustering …")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - sim_threshold,
        metric="cosine",
        linkage="average",
    )
    cluster_ids = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    for label, cid in zip(all_labels, cluster_ids):
        clusters[cid].append(label)

    semantic_clusters = []
    for cid, members in clusters.items():
        if len(members) < 2:
            continue
        total_freq = sum(label_counter[m] for m in members)
        suggested_canon = max(members, key=lambda m: label_counter[m])
        semantic_clusters.append({
            "members":          sorted(members),
            "total_freq":       total_freq,
            "suggested_canon":  suggested_canon,
            "action":           "normalize",
            "canonical":        suggested_canon,
        })
    semantic_clusters.sort(key=lambda x: -x["total_freq"])

    null_embeddings = model.encode(list(NULL_KEYS), normalize_embeddings=True)
    sims     = embeddings @ null_embeddings.T
    max_sims = sims.max(axis=1)

    suspicious_nulls = []
    for label, sim, norm_label in zip(all_labels, max_sims, label_norms):
        if sim > sim_threshold and norm_label not in NULL_KEYS:
            suspicious_nulls.append({
                "label":    label,
                "freq":     label_counter[label],
                "null_sim": round(float(sim), 4),
                "action":   "remove",
            })
    suspicious_nulls.sort(key=lambda x: -x["null_sim"])

    rare_labels = [
        {"label": l, "freq": c, "action": "keep_as_is"}
        for l, c in label_counter.items()
        if c < min_count
    ]
    rare_labels.sort(key=lambda x: x["freq"])

    decisions = {
        "_instructions": (
            "Review each section. For semantic_clusters: set 'action' to "
            "'normalize' (and set 'canonical') or 'keep_as_is'. "
            "For suspicious_nulls and rare_labels: set 'action' to 'remove' or 'keep'."
        ),
        "config": {
            "sim_threshold": sim_threshold,
            "min_count":     min_count,
        },
        "semantic_clusters":  semantic_clusters,
        "suspicious_nulls":   suspicious_nulls,
        "rare_labels":        rare_labels,
    }
    return decisions


def run_step2_discover(sim_threshold=0.75, min_count=3):
    files = sorted(glob.glob(f"{STEP1_OUTPUT_DIR}/*.graphml"))
    if not files:
        print(f"  No step1-cleaned files found in {STEP1_OUTPUT_DIR}"); return

    print(f"{'='*70}\n  STEP 2 — DISCOVERY  ({len(files)} files)\n{'='*70}")
    label_counter, _ = _load_all_labels(STEP1_OUTPUT_DIR)
    print(f"  Total unique labels: {len(label_counter)}")
    print(f"  Total label occurrences: {sum(label_counter.values()):,}")

    decisions = _discover(label_counter, sim_threshold=sim_threshold, min_count=min_count)

    os.makedirs(os.path.dirname(STEP2_DECISIONS), exist_ok=True)
    with open(STEP2_DECISIONS, "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, ensure_ascii=False)

    n_clusters = len(decisions["semantic_clusters"])
    n_nulls    = len(decisions["suspicious_nulls"])
    n_rare     = len(decisions["rare_labels"])
    print(f"\n  Found {n_clusters} semantic clusters to review")
    print(f"  Found {n_nulls} suspicious null-like labels")
    print(f"  Found {n_rare} rare labels (freq < {min_count})")
    print(f"\n  ➜  Review and edit: {STEP2_DECISIONS}")
    print(f"     Then run:  python stage0_clean_graphs.py step2 --apply\n{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — APPLY (Pass B)
# ─────────────────────────────────────────────────────────────────────────────

def _build_step2_lookup(decisions):
    remove_set = set()
    remap      = {}

    for entry in decisions.get("suspicious_nulls", []):
        if entry.get("action") == "remove":
            remove_set.add(_norm(entry["label"]))

    for entry in decisions.get("rare_labels", []):
        if entry.get("action") == "remove":
            remove_set.add(_norm(entry["label"]))

    for cluster in decisions.get("semantic_clusters", []):
        action = cluster.get("action", "keep_as_is")
        if action == "normalize":
            canon = _to_display_label(cluster.get("canonical", ""))
            for member in cluster.get("members", []):
                n = _norm(member)
                if n not in remove_set:
                    remap[n] = canon
        elif action == "remove":
            for member in cluster.get("members", []):
                remove_set.add(_norm(member))

    return remove_set, remap


def _clean_file_step2(filepath, output_path, remove_set, remap):
    tree = ET.parse(filepath)
    root = tree.getroot()
    km   = _key_map(root)

    graph = root.find(_tag("graph"))
    if graph is None:
        shutil.copy2(filepath, output_path)
        return {"error": "no <graph> element"}

    edges = _find(graph, "edge")
    stats = dict(
        before=len(edges),
        removed_null_like=0, removed_rare=0, removed_selfloop=0, removed_duplicate=0,
        normalized=0,
    )
    remove = []
    seen_triples = set()

    for edge in edges:
        src   = edge.attrib.get("source", "")
        tgt   = edge.attrib.get("target", "")
        label = _get_attr(edge, km.get("label"))
        norm_label = _norm(label) if label else ""

        if src == tgt:
            remove.append(edge); stats["removed_selfloop"] += 1; continue

        if norm_label in remove_set:
            remove.append(edge); stats["removed_null_like"] += 1; continue

        triple = (src, tgt, norm_label)
        if triple in seen_triples:
            remove.append(edge); stats["removed_duplicate"] += 1; continue
        seen_triples.add(triple)

        if norm_label in remap:
            new_label = remap[norm_label]
            if new_label != (label or "").strip():
                _set_attr(edge, km["label"], new_label)
                stats["normalized"] += 1

    for edge in remove:
        graph.remove(edge)

    total_removed = (stats["removed_null_like"] + stats["removed_rare"]
                     + stats["removed_selfloop"] + stats["removed_duplicate"])
    stats["after"] = stats["before"] - total_removed

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return stats


def run_step2_apply():
    if not os.path.exists(STEP2_DECISIONS):
        print(f"  ✗  Decisions file not found: {STEP2_DECISIONS}")
        print(f"     Run --discover first.\n"); return

    with open(STEP2_DECISIONS, encoding="utf-8") as f:
        decisions = json.load(f)

    remove_set, remap = _build_step2_lookup(decisions)
    print(f"  Labels to remove : {len(remove_set)}")
    print(f"  Labels to remap  : {len(remap)}")

    files = sorted(glob.glob(f"{STEP1_OUTPUT_DIR}/*.graphml"))
    if not files:
        print(f"  No step1-cleaned files found in {STEP1_OUTPUT_DIR}"); return

    os.makedirs(STEP2_OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*70}\n  STEP 2 — APPLY  ({len(files)} files)\n{'='*70}")

    all_stats, errors = {}, []
    totals = Counter()

    for i, fp in enumerate(files, 1):
        fname = os.path.basename(fp)
        try:
            s = _clean_file_step2(fp, os.path.join(STEP2_OUTPUT_DIR, fname), remove_set, remap)
        except Exception as e:
            errors.append((fname, str(e)))
            print(f"  [{i:>3}] ✗ {fname}: {e}"); continue

        if "error" in s:
            errors.append((fname, s["error"]))
            print(f"  [{i:>3}] ⚠ {fname}: {s['error']}"); continue

        all_stats[fname] = s
        totals.update({k: v for k, v in s.items() if isinstance(v, int) and k != "after"})
        print(
            f"  [{i:>3}/{len(files)}] {fname:<45} "
            f"{s['before']:>5} → {s['after']:>5}  "
            f"null-like:{s['removed_null_like']:>3}  "
            f"dup:{s['removed_duplicate']:>3}  "
            f"loop:{s['removed_selfloop']:>2}  "
            f"norm:{s['normalized']:>3}  "
        )

    total_removed = (totals["removed_null_like"] + totals.get("removed_rare", 0)
                     + totals["removed_selfloop"] + totals["removed_duplicate"])
    total_before  = totals["before"]
    pct = lambda n: f"{n/total_before*100:.1f}%" if total_before else "0%"

    print(f"\n{'─'*70}  SUMMARY")
    print(f"  Edges before/after   : {total_before:,} → {total_before - total_removed:,}")
    print(f"  Removed null-like    : {totals['removed_null_like']:,} ({pct(totals['removed_null_like'])})")
    print(f"  Removed duplicates   : {totals['removed_duplicate']:,} ({pct(totals['removed_duplicate'])})")
    print(f"  Removed self-loops   : {totals['removed_selfloop']:,} ({pct(totals['removed_selfloop'])})")
    print(f"  Labels remapped      : {totals['normalized']:,}")

    report = {
        "step": "step2",
        "decisions_file": STEP2_DECISIONS,
        "summary": dict(totals) | {
            "total_removed": total_removed,
            "after": total_before - total_removed,
            "removed_pct": round(total_removed / total_before * 100, 2) if total_before else 0,
        },
        "per_file": all_stats,
        "errors": errors,
    }
    os.makedirs(os.path.dirname(STEP2_REPORT), exist_ok=True)
    with open(STEP2_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report  → {STEP2_REPORT}\n  Cleaned → {STEP2_OUTPUT_DIR}/\n{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["step1", "step2"])
    p.add_argument("--discover",       action="store_true", help="Step 2 Pass A: discover & write decisions file")
    p.add_argument("--apply",          action="store_true", help="Step 2 Pass B: apply reviewed decisions")
    p.add_argument("--sim-threshold",  type=float, default=0.9, help="Cosine similarity threshold (default 0.9)")
    p.add_argument("--min-count",      type=int,   default=3,    help="Min label frequency before flagging as rare (default 3)")
    p.add_argument("--audit-dupes", action="store_true", help="Dry-run: count post-normalization duplicates without writing files")

    args = p.parse_args()

    if args.step == "step1":
        run_step1()
    else:
        if args.discover:
            run_step2_discover(sim_threshold=args.sim_threshold, min_count=args.min_count)
        elif args.apply:
            run_step2_apply()
        elif args.audit_dupes:
            audit_post_normalization_dupes()
        else:
            print("  Step 2 requires --discover or --apply. See --help.")