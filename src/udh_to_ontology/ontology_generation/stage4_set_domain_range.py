"""
Stage 4 — Domain & Range Assignment
=====================================
Input:  named_concept_clusters.json  (Stage 3 output)
        All stage1 batch JSON files  (Stage 1 output)
Output: property_clusters_with_dr.json
        flagged_for_review.json
        stage4_stats.json

Pipeline:
  4.1  Class vocabulary extracted from Stage 3 class-stream clusters
  4.2  Stage 1 documents loaded from batch_*_stage1.json files
  4.3  Four graph signals computed per cluster:
         S1 — edge topology: source/target entity-type distributions over
              labeled edges that match the cluster (min 3 edges to fire)
         S2 — edge direction: entity types on entity_fact (subject) and
              fact_entity (object) edges matching the cluster
         S3 — raw sentences: text from facts whose name matches the cluster
         S4 — chunk topics: co-occurring topics from matching facts,
              ranked by frequency
  4.4  Clusters grouped by owl_type (ObjectProperty / DatatypeProperty);
       domain vocab is always the ontology class list; range vocab is class
       list for ObjectProperty, XSD datatypes for DatatypeProperty
  4.5  LLM assigns domain, range, confidence, and reasoning per cluster,
       constrained via guided decoding (vLLM + outlines) to the allowed
       vocabularies; processed in batches of 8 with up to 3 retries
  4.6  Low-confidence assignments flagged for human review and written to
       flagged_for_review.json; all annotated clusters written to
       property_clusters_with_dr.json; streams not in --streams passed
       through unchanged

Domain  : always an ontology class (owl_name from class stream, or owl:Thing)
Range   : ontology class for ObjectProperty, XSD datatype for DatatypeProperty
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import cfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OWL_THING = "owl:Thing"
S1_MIN_EDGE_COUNT = 3
BATCH_SIZE = 8
MAX_RETRIES = 3
RETRY_DELAY = 5.0

XSD_DATATYPES = [
    "xsd:string", "xsd:integer", "xsd:decimal", "xsd:float",
    "xsd:boolean", "xsd:date", "xsd:dateTime", "xsd:gYear",
    "xsd:duration", "xsd:anyURI", "rdfs:Literal",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    signal_id: str
    evidence:  list[str] = field(default_factory=list)


@dataclass
class DomainRangeResult:
    owl_name:    str
    owl_type:    str
    domain:      str = OWL_THING
    range_:      str = OWL_THING
    confidence:  str = "low"
    reasoning:   str = ""
    signals:     list = field(default_factory=list)
    flagged:     bool = False
    flag_reason: str  = ""


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def extract_class_vocab(all_cluster_data: dict) -> list[str]:
    vocab = []
    for cluster in all_cluster_data.get("class", []):
        if not cluster.get("is_noise", False):
            name = cluster.get("owl_name", "").strip()
            if name and name not in vocab:
                vocab.append(name)
    if OWL_THING not in vocab:
        vocab.append(OWL_THING)
    return sorted(vocab)


def range_vocab_for(owl_type: str, class_vocab: list[str]) -> list[str]:
    if owl_type == "owl:ObjectProperty":
        return class_vocab
    elif owl_type == "owl:DatatypeProperty":
        return XSD_DATATYPES + [OWL_THING]
    else:
        return class_vocab + XSD_DATATYPES


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_node_type_index(doc: dict) -> dict[str, str]:
    index: dict[str, str] = {}
    for ent in doc.get("entities_normalized", []):
        index[ent["id"]] = ent["type"]
    for fact in doc.get("facts", []):
        index[fact["id"]] = "__FACT__"
    for chunk in doc.get("chunks", []):
        if isinstance(chunk, dict):
            index[chunk.get("id", "")] = "__CHUNK__"
    return index


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def cluster_label_set(cluster: dict) -> set[str]:
    labels = set()
    if cluster.get("owl_name"):
        labels.add(cluster["owl_name"].lower())
    for m in cluster.get("members", []):
        s = m.get("string", "").strip().lower()
        if s:
            labels.add(s)
    return labels


def _label_matches(text: str, label_set: set[str]) -> bool:
    """Loose match: any label is a substring of text or vice versa."""
    text = text.strip().lower()
    return any(label in text or text in label for label in label_set)


def compute_s1(all_docs: list[dict], label_set: set[str]) -> Signal:
    """
    Edge topology — looks for edges whose label matches the cluster,
    then counts the norm types of source/target nodes.
    Only fires on edges that have a non-null label.
    """
    src_counter: Counter = Counter()
    tgt_counter: Counter = Counter()
    total = 0

    for doc in all_docs:
        node_types = build_node_type_index(doc)
        for bucket_name, edge_list in doc.get("edge_buckets", {}).items():
            for edge in edge_list:
                label = (edge.get("label") or "").strip().lower()
                if not label or not _label_matches(label, label_set):
                    continue
                src = node_types.get(edge["source"])
                tgt = node_types.get(edge["target"])
                if src and not src.startswith("__"):
                    src_counter[src] += 1
                if tgt and not tgt.startswith("__"):
                    tgt_counter[tgt] += 1
                total += 1

    evidence = []
    if total >= S1_MIN_EDGE_COUNT:
        if src_counter:
            top_src, top_src_count = src_counter.most_common(1)[0]
            evidence.append(
                f"Most frequent source type: {top_src} ({top_src_count} edges)"
            )
        if tgt_counter:
            top_tgt, top_tgt_count = tgt_counter.most_common(1)[0]
            evidence.append(
                f"Most frequent target type: {top_tgt} ({top_tgt_count} edges)"
            )
    else:
        evidence.append(
            f"Insufficient labeled edges (found {total}, need {S1_MIN_EDGE_COUNT})"
        )

    return Signal(signal_id="S1", evidence=evidence)


def compute_s2(all_docs: list[dict], label_set: set[str]) -> Signal:
    """
    Edge direction — entity_fact edges suggest subject type (domain),
    fact_entity edges suggest object type (range).
    """
    domain_counter: Counter = Counter()
    range_counter:  Counter = Counter()

    for doc in all_docs:
        node_types = build_node_type_index(doc)
        for edge in doc.get("edge_buckets", {}).get("entity_fact", []):
            label = (edge.get("label") or "entity_fact").strip().lower()
            if not _label_matches(label, label_set):
                continue
            src = node_types.get(edge["source"])
            if src and not src.startswith("__"):
                domain_counter[src] += 1

        for edge in doc.get("edge_buckets", {}).get("fact_entity", []):
            label = (edge.get("label") or "fact_entity").strip().lower()
            if not _label_matches(label, label_set):
                continue
            tgt = node_types.get(edge["target"])
            if tgt and not tgt.startswith("__"):
                range_counter[tgt] += 1

    evidence = []
    if domain_counter:
        top_d, top_d_count = domain_counter.most_common(1)[0]
        evidence.append(f"Most frequent subject type: {top_d} ({top_d_count} edges)")
    if range_counter:
        top_r, top_r_count = range_counter.most_common(1)[0]
        evidence.append(f"Most frequent object type: {top_r} ({top_r_count} edges)")
    if not evidence:
        evidence.append("No entity_fact / fact_entity edges matched.")

    return Signal(signal_id="S2", evidence=evidence)


def compute_s3(all_docs: list[dict], label_set: set[str]) -> Signal:
    """
    Raw sentences — collects sentences from facts whose name loosely
    matches the cluster labels. Passed as-is to the LLM.
    """
    matched_sentences: list[str] = []

    for doc in all_docs:
        for fact in doc.get("facts", []):
            name = (fact.get("name") or "").strip()
            if not name or not _label_matches(name, label_set):
                continue
            sentence = (fact.get("sentence") or "").strip()
            if sentence and sentence not in matched_sentences:
                matched_sentences.append(sentence)

    evidence = matched_sentences[:10]  # cap to keep prompt size sane
    if not evidence:
        evidence = ["No matching fact sentences found."]

    return Signal(signal_id="S3", evidence=evidence)


def compute_s4(all_docs: list[dict], label_set: set[str]) -> Signal:
    """
    Chunk topics — collects raw chunk_topics strings from facts whose name
    loosely matches the cluster labels. Top topics by frequency passed to LLM.
    """
    matched_topics: Counter = Counter()

    for doc in all_docs:
        for fact in doc.get("facts", []):
            name = (fact.get("name") or "").strip()
            if not name or not _label_matches(name, label_set):
                continue
            for topic in fact.get("chunk_topics", []):
                t = topic.strip()
                if t:
                    matched_topics[t] += 1

    evidence = []
    if matched_topics:
        evidence = [
            f"{topic} ({count})"
            for topic, count in matched_topics.most_common(10)
        ]
    else:
        evidence = ["No matching chunk topics found."]

    return Signal(signal_id="S4", evidence=evidence)


def collect_signals(cluster: dict, all_docs: list[dict]) -> list[Signal]:
    label_set = cluster_label_set(cluster)
    return [
        compute_s1(all_docs, label_set),
        compute_s2(all_docs, label_set),
        compute_s3(all_docs, label_set),
        compute_s4(all_docs, label_set),
    ]


# ---------------------------------------------------------------------------
# LLM prompt & schema
# ---------------------------------------------------------------------------

OWL_TYPE_GUIDANCE = {
    "owl:ObjectProperty":   "Object property — links two individuals. Range must be an ontology class.",
    "owl:DatatypeProperty": "Datatype property — links an individual to a literal. Range must be an XSD datatype.",
}

SIGNAL_DESCRIPTIONS = {
    "S1": "Edge topology — source/target entity types on labeled edges matching this property",
    "S2": "Edge direction — entity types on entity_fact (subject) and fact_entity (object) edges",
    "S3": "Raw sentences — from facts whose name matches this property",
    "S4": "Chunk topics — co-occurring topics from facts matching this property",
}


def build_prompt(
    batch: list[tuple[dict, list[Signal]]],
    class_vocab: list[str],
    range_vocab: list[str],
    owl_type: str,
) -> str:
    type_guidance = OWL_TYPE_GUIDANCE.get(
        owl_type,
        "Property type is uncertain — range may be an ontology class or XSD datatype.",
    )

    properties_block = ""
    for idx, (cluster, signals) in enumerate(batch):
        signal_lines = ""
        for s in signals:
            signal_lines += f"\n  {s.signal_id} ({SIGNAL_DESCRIPTIONS[s.signal_id]}):\n"
            for line in s.evidence:
                signal_lines += f"    - {line}\n"

        properties_block += f"""
            --- Property {idx + 1} ---
            owl_name   : {cluster.get('owl_name', 'UNKNOWN')}
            definition : {cluster.get('definition', 'not defined')}
            members    : {[m['string'] for m in cluster.get('members', [])[:5]]}

            Graph signals (use as hints only — may be incomplete or misleading):
            {signal_lines}
            """

    return f"""You are an OWL ontology engineer specialising in life insurance contracts.

            Assign domain and range to each property below.

            Property type: {owl_type}
            {type_guidance}

            RULES:
            1. domain MUST be chosen from the allowed domain vocabulary.
            2. range MUST be chosen from the allowed range vocabulary.
            3. Use owl:Thing only if no specific class fits.
            4. Trust insurance semantics over graph signals.
            5. Confidence: high=certain, medium=minor ambiguity, low=signals contradictory or absent.

            Allowed domain vocabulary:
            {", ".join(class_vocab)}

            Allowed range vocabulary:
            {", ".join(range_vocab)}

            Properties to annotate:
            {properties_block}
            Return a JSON object with an "assignments" array of exactly {len(batch)} items (same order),
            each with: owl_name, reasoning (1-2 sentences), domain, range, confidence.
            """


def build_schema(domain_vocab: list[str], range_vocab: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "owl_name":   {"type": "string"},
                        "reasoning":  {"type": "string"},
                        "domain":     {"type": "string", "enum": domain_vocab},
                        "range":      {"type": "string", "enum": range_vocab},
                        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["owl_name", "reasoning", "domain", "range", "confidence"],
                },
            }
        },
        "required": ["assignments"],
    }


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    schema: dict,
    expected_count: int,
) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an OWL ontology engineer for life insurance contracts. "
                            "/no_think "
                            "Respond only with valid JSON matching the provided schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "domain_range_batch",
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=0.1,
                extra_body={"guided_decoding_backend": "outlines"},
            )
            assignments = json.loads(response.choices[0].message.content)["assignments"]
            if len(assignments) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} assignments, got {len(assignments)}"
                )
            return assignments

        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] LLM error: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f"All {MAX_RETRIES} LLM attempts failed.")


def fallback_assignments(
    batch: list[tuple[dict, list]], owl_type: str
) -> list[dict]:
    default_range = (
        "xsd:string" if owl_type == "owl:DatatypeProperty" else OWL_THING
    )
    return [
        {
            "owl_name":   c.get("owl_name", ""),
            "reasoning":  "LLM call failed after all retries.",
            "domain":     OWL_THING,
            "range":      default_range,
            "confidence": "low",
        }
        for c, _ in batch
    ]


# ---------------------------------------------------------------------------
# Main assignment driver
# ---------------------------------------------------------------------------

def assign_domain_range(
    clusters: list[dict],
    all_docs: list[dict],
    client: OpenAI,
    model: str,
    class_vocab: list[str],
) -> list[DomainRangeResult]:

    print("  Computing graph signals ...")
    cluster_signals = [
        (cluster, collect_signals(cluster, all_docs))
        for cluster in clusters
    ]

    groups: dict[str, list[tuple[int, dict, list]]] = defaultdict(list)
    for idx, (cluster, signals) in enumerate(cluster_signals):
        groups[cluster.get("owl_type", "unknown")].append((idx, cluster, signals))

    schemas: dict[str, dict] = {}
    range_vocabs: dict[str, list[str]] = {}
    for owl_type in groups:
        rv = range_vocab_for(owl_type, class_vocab)
        range_vocabs[owl_type] = rv
        schemas[owl_type] = build_schema(class_vocab, rv)
        print(
            f"  Schema [{owl_type}]: {len(class_vocab)} domain choices, "
            f"{len(rv)} range choices"
        )

    results_by_idx: dict[int, DomainRangeResult] = {}

    for owl_type, group_items in groups.items():
        schema      = schemas[owl_type]
        range_vocab = range_vocabs[owl_type]
        n_batches   = (len(group_items) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n  [{owl_type}] {len(group_items)} clusters across {n_batches} batch(es)")

        for b_start in range(0, len(group_items), BATCH_SIZE):
            batch_items    = group_items[b_start: b_start + BATCH_SIZE]
            batch_for_prompt = [(c, s) for _, c, s in batch_items]
            b_num = b_start // BATCH_SIZE + 1
            print(f"    Batch {b_num}/{n_batches} ({len(batch_items)} clusters) ...")

            prompt = build_prompt(batch_for_prompt, class_vocab, range_vocab, owl_type)

            try:
                assignments = call_llm(client, model, prompt, schema, len(batch_items))
            except RuntimeError as e:
                print(f"    WARNING: {e} — using fallback.", file=sys.stderr)
                assignments = fallback_assignments(batch_for_prompt, owl_type)

            for (orig_idx, cluster, signals), assignment in zip(batch_items, assignments):
                confidence = assignment.get("confidence", "low")
                results_by_idx[orig_idx] = DomainRangeResult(
                    owl_name   = cluster.get("owl_name", ""),
                    owl_type   = owl_type,
                    domain     = assignment.get("domain",    OWL_THING),
                    range_     = assignment.get("range",     OWL_THING),
                    confidence = confidence,
                    reasoning  = assignment.get("reasoning", ""),
                    signals    = [
                        {"signal_id": s.signal_id, "evidence": s.evidence}
                        for s in signals
                    ],
                    flagged     = confidence == "low",
                    flag_reason = (
                        "LLM confidence is low — manual review recommended."
                        if confidence == "low" else ""
                    ),
                )

    return [results_by_idx[i] for i in range(len(clusters))]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_stage1_docs(stage1_dir: Path) -> list[dict]:
    batch_files = sorted(stage1_dir.glob("batch_*_stage1.json"))
    if not batch_files:
        print(f"No stage1 batch files found in {stage1_dir}", file=sys.stderr)
        sys.exit(1)
    all_docs: list[dict] = []
    for bf in batch_files:
        with open(bf) as f:
            all_docs.extend(json.load(f))
    print(f"  Loaded {len(all_docs)} docs from {len(batch_files)} batch file(s).")
    return all_docs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 4 — Domain & Range Assignment")
    p.add_argument(
        "--clusters_path",
        default=cfg.STAGE3_CLUSTERS,
        help="Path to Stage 3 named_concept_clusters.json",
    )
    p.add_argument(
        "--stage1_dir",
        default=cfg.STAGE1_OUT_DIR,
        help="Directory with stage1 batch JSONs",
    )
    p.add_argument(
        "--output_dir",
        default=cfg.STAGE4_OUT_DIR,
        help="Output directory",
    )
    p.add_argument(
        "--model",
        default=cfg.MODEL_NAME,
        help="vLLM model name",
    )
    p.add_argument(
        "--vllm_base_url",
        default=cfg.API_BASE,
        help="vLLM base URL",
    )
    p.add_argument("--streams", nargs="+", default=["property", "axiom"])
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Stage 3 clusters ...")
    with open(args.clusters_path) as f:
        all_cluster_data = json.load(f)

    print("Loading Stage 1 documents ...")
    all_docs = load_stage1_docs(Path(args.stage1_dir))

    print("Extracting class vocabulary ...")
    class_vocab = extract_class_vocab(all_cluster_data)
    print(f"  {len(class_vocab)} classes: {class_vocab[:8]} ...")

    client = OpenAI(base_url=args.vllm_base_url, api_key="EMPTY")

    annotated_output: dict[str, list] = {}
    stats: dict[str, dict] = {}

    for stream_name in args.streams:
        all_clusters   = all_cluster_data.get(stream_name, [])
        real_clusters  = [c for c in all_clusters if not c.get("is_noise", False)]
        noise_clusters = [c for c in all_clusters if     c.get("is_noise", False)]

        print(f"\n{'='*60}")
        print(
            f"Stream: {stream_name} — {len(real_clusters)} real, "
            f"{len(noise_clusters)} noise (skipped)"
        )

        results = assign_domain_range(
            real_clusters, all_docs, client, args.model, class_vocab,
        )

        confidence_counts: Counter = Counter()
        annotated: list[dict] = []

        for cluster, result in zip(real_clusters, results):
            annotated.append({
                **cluster,
                "domain":         result.domain,
                "range":          result.range_,
                "dr_confidence":  result.confidence,
                "dr_reasoning":   result.reasoning,
                "dr_signals":     result.signals,
                "dr_flagged":     result.flagged,
                "dr_flag_reason": result.flag_reason,
            })
            confidence_counts[result.confidence] += 1

        annotated.extend(noise_clusters)
        annotated_output[stream_name] = annotated
        stats[stream_name] = {
            "total_real_clusters":     len(real_clusters),
            "noise_clusters_skipped":  len(noise_clusters),
            "confidence_distribution": dict(confidence_counts),
            "flagged_for_review":      sum(1 for c in annotated if c.get("dr_flagged")),
        }

    # pass through any streams not in args.streams unchanged
    for stream_name, clusters in all_cluster_data.items():
        if stream_name not in annotated_output:
            annotated_output[stream_name] = clusters

    out_path = output_dir / "property_clusters_with_dr.json"
    with open(out_path, "w") as f:
        json.dump(annotated_output, f, indent=2)
    print(f"\nWritten: {out_path}")

    flagged = [
        {
            "stream":      sn,
            "owl_name":    c.get("owl_name"),
            "domain":      c.get("domain"),
            "range":       c.get("range"),
            "confidence":  c.get("dr_confidence"),
            "reasoning":   c.get("dr_reasoning"),
            "flag_reason": c.get("dr_flag_reason"),
            "signals":     c.get("dr_signals"),
            "top_members": [m["string"] for m in c.get("members", [])[:5]],
        }
        for sn in args.streams
        for c in annotated_output.get(sn, [])
        if c.get("dr_flagged")
    ]
    flagged_path = output_dir / "flagged_for_review.json"
    with open(flagged_path, "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"Written: {flagged_path}")

    with open(output_dir / "stage4_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Written: {output_dir / 'stage4_stats.json'}")

    print("\n=== Stage 4 complete ===")
    for sn, s in stats.items():
        print(f"\n  [{sn}]")
        print(f"    Real clusters   : {s['total_real_clusters']}")
        print(f"    Noise skipped   : {s['noise_clusters_skipped']}")
        print(f"    Flagged         : {s['flagged_for_review']}")
        for level in ("high", "medium", "low"):
            n = s["confidence_distribution"].get(level, 0)
            if n:
                print(f"    {level:8s}       : {n}")


if __name__ == "__main__":
    main()