"""
Stage 3 — Semantic Deduplication & Clustering
===============================================
Input:  candidate_pool.json  (output of Stage 2)
Output: named_concept_clusters.json — concept clusters partitioned into three
        streams (classes, properties, axioms), each cluster having a canonical
        OWL name, definition, parent, and member list.

Pipeline:
  3.1  Stream separation — LLM classifies fact_name candidates as 'axiom' or
       'property'; entity_type/entity_role/chunk_topic go to class stream;
       edge_label goes to property stream
  3.2  Sentence embedding with origin-aware templates using SentenceTransformers
       (all-mpnet-base-v2 by default)
  3.3  Agglomerative clustering per stream (average linkage, cosine distance,
       similarity thresholds: class=0.80, property=0.75, axiom=0.75)
  3.4  Cluster naming via local vLLM server — classes get PascalCase OWL names,
       properties get camelCase names with ObjectProperty/DatatypeProperty typing,
       axiom clusters get a pattern label and an axiom_family tag
  3.5  Graph-native parent assignment — for class-stream clusters with entity_role
       members, the most frequent parent type becomes graph_native_parent at
       confidence 'high'; all others fall back to llm_parent at confidence 'llm'

Usage:
    python stage3_dedup_clustering.py
    python stage3_dedup_clustering.py --candidate_pool path/to/candidate_pool.json
                                       --output_dir    path/to/out/
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from sklearn.cluster import AgglomerativeClustering
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import cfg


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    string:    str
    origin:    str
    parent:    Optional[str]
    frequency: int
    doc_freq:  int
    meta:      dict = field(default_factory=dict)


@dataclass
class ConceptCluster:
    cluster_id:           int
    stream:               str          # 'class' | 'property' | 'axiom'
    members:              list         # list of Candidate dicts
    owl_name:             str  = ""
    definition:           str  = ""
    owl_type:             str  = ""    # owl:Class | owl:ObjectProperty | owl:DatatypeProperty | owl:Restriction
    axiom_family:         Optional[str] = None
    llm_parent:           Optional[str] = None
    graph_native_parent:  Optional[str] = None
    parent_confidence:    str  = ""    # 'high' | 'llm'
    is_noise:             bool = False # label == -1


SIM_THRESHOLD = {}
SIM_THRESHOLD['class'] = 0.8
SIM_THRESHOLD['property'] = 0.75
SIM_THRESHOLD['axiom'] = 0.75

# ---------------------------------------------------------------------------
# 3.1  Stream separation — LLM-based axiom routing
# ---------------------------------------------------------------------------


AXIOM_CLASSIFY_PROMPT = """\
You are an OWL ontology engineer building a TBox for life insurance documents.

Below is a list of facts extracted from insurance policy corpora.
Your task: classify each as either "axiom" or "property".

A TRUE TBox AXIOM requires a logical operator to express in OWL — it cannot
be captured as a simple named class or property. It must express one of:
- An existential or universal restriction. Example: "a Policy must have at least one Beneficiary"
- A cardinality constraint. Example: "a Policy has exactly one PolicyOwner"
- A disjointness assertion. Example: "a person cannot be both PolicyOwner and Beneficiary"
  WARNING: Only classify as disjointness if the two concepts are MUTUALLY EXCLUSIVE
  by definition — i.e., no individual could ever belong to both classes simultaneously.
  Do NOT classify as disjointness if the concepts merely tend to be different, or if
  one concept is a subtype or specialisation of the other.
- A conditional rule. Example: "if premium is unpaid after grace period, policy lapses"
- A subclass condition with a restriction body. Example: "every eligible claimant must have filed within 30 days"

KEY RULE: If you can express it as a class name or property name without
losing meaning, it is NOT an axiom. Axioms require logical structure
(∃, ∀, ≤, ≥, ¬) that cannot be collapsed into a single name.

IMPORTANT:
- Default to "property" when uncertain — only classify as "axiom" when the
  logical constraint is explicit and cannot be expressed as a named term
- For disjointness specifically: default to "property" unless mutual exclusion
  is stated explicitly and unambiguously in the source text
- The key provided for each entry is the fact_name — use it exactly as the JSON key
- Return ONLY valid JSON, no explanation, no markdown

Facts to classify (format: "fact_name: sentence"):
{fact_items}

Return format:
{{"fact name 1": "axiom", "fact name 2": "property", ...}}
"""


def classify_fact_names(
    candidates: list[Candidate],
    client: OpenAI,
    model_name: str,
    batch_size: int = 20,
) -> dict[str, str]:
    fact_candidates = [c for c in candidates if c.origin == 'fact_name']
    if not fact_candidates:
        return {}

    results = {}
    names   = [c.string for c in fact_candidates]

    # Use sentence from meta if available (stored by Stage 2), fall back to name
    def get_text(c: Candidate) -> str:
        return c.meta.get("sentence") or c.string

    texts = [get_text(c) for c in fact_candidates]

    for i in range(0, len(names), batch_size):
        batch_names = names[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        # Format as "fact_name: sentence" so LLM has full context
        # but keys its response by fact_name
        fact_items = "\n".join(
            f"- {name}: {text}" if text != name else f"- {name}"
            for name, text in zip(batch_names, batch_texts)
        )

        prompt = AXIOM_CLASSIFY_PROMPT.format(fact_items=fact_items)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            parsed = json.loads(raw)
            # Key results by fact_name, not by sentence
            for name, text in zip(batch_names, batch_texts):
                # LLM may key by name or by text — check both
                label = parsed.get(name) or parsed.get(text) or 'property'
                results[name] = label
        except Exception as e:
            print(f"  [WARNING] Axiom classification failed for batch {i//batch_size + 1}: {e}. Defaulting to 'property'.")
            results.update({name: 'property' for name in batch_names})

    return results


def stream_of(candidate: Candidate, fact_classifications: dict = {}) -> str:
    if candidate.origin in ('entity_type', 'entity_role', 'chunk_topic'):
        return 'class'
    if candidate.origin == 'edge_label':
        return 'property'
    if candidate.origin == 'fact_name':
        return fact_classifications.get(candidate.string, 'property')
    return 'class'


def separate_streams(
    candidates: list[Candidate],
    fact_classifications: dict = {},
) -> dict[str, list[Candidate]]:
    streams: dict[str, list[Candidate]] = {'class': [], 'property': [], 'axiom': []}
    for c in candidates:
        streams[stream_of(c, fact_classifications)].append(c)
    print(f"  Stream sizes — class: {len(streams['class'])}, "
          f"property: {len(streams['property'])}, axiom: {len(streams['axiom'])}")
    return streams


# ---------------------------------------------------------------------------
# 3.2  Embedding with origin-aware templates
# ---------------------------------------------------------------------------
EMBEDDING_TEMPLATES = {
    'entity_type':  'insurance domain type: {s}',
    'entity_role':  'insurance role of {p}: {s}',
    'edge_label':   'relationship in insurance policy: {s}',
    'fact_name':    'insurance policy fact: {s}',
    'chunk_topic':  'insurance document topic: {s}',
}


def make_embedding_input(candidate: Candidate) -> str:
    t = EMBEDDING_TEMPLATES.get(candidate.origin, 'insurance concept: {s}')
    return t.format(s=candidate.string, p=candidate.parent or 'entity')


def embed_stream(
    candidates: list[Candidate],
    model: SentenceTransformer,
) -> np.ndarray:
    texts = [make_embedding_input(c) for c in candidates]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return embeddings


def cluster_stream(
    embeddings: np.ndarray,
    sim_threshold: float,
) -> np.ndarray:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - sim_threshold,
        metric="cosine",
        linkage="average",
    )
    return clustering.fit_predict(embeddings)


def build_clusters(
    stream_name: str,
    candidates: list[Candidate],
    labels: np.ndarray,
) -> list[ConceptCluster]:
    label_to_members: dict[int, list[Candidate]] = {}
    for cand, label in zip(candidates, labels):
        label_to_members.setdefault(int(label), []).append(cand)

    clusters = []
    for label, members in sorted(label_to_members.items()):
        is_noise = label == -1
        clusters.append(ConceptCluster(
            cluster_id=label,
            stream=stream_name,
            members=members,
            is_noise=is_noise,
        ))

    real   = sum(1 for c in clusters if not c.is_noise)
    noise  = sum(len(c.members) for c in clusters if c.is_noise)
    print(f"  [{stream_name:8s}] {real} clusters, {noise} noise candidates")
    return clusters


# ---------------------------------------------------------------------------
# 3.4  Cluster naming via local vLLM (OpenAI-compatible)
# ---------------------------------------------------------------------------
CLASS_SYSTEM_PROMPT = (
    "You are an ontology engineer specializing in insurance domain modeling."
)

CLASS_USER_TEMPLATE = """\
The following strings all appear in life insurance contracts and refer to the same concept. Provide:
1. A canonical OWL class name in PascalCase (e.g. PolicyOwner, DeathBenefit). Be appropriately generic - do not invent specific classes that are not directly supported by the strings.
2. A one-sentence definition (≤ 20 words)

Strings: {members}

Respond in JSON: {{"class": "...", "definition": "..."}}"""

PROPERTY_USER_TEMPLATE = """\
These strings are relationship labels from insurance contracts. Provide:
1. A canonical OWL property name in camelCase (e.g. hasInsured, isIssuedBy)
2. Whether it is ObjectProperty or DatatypeProperty
3. A one-sentence definition

Strings: {members}

JSON: {{"property": "...", "type": "ObjectProperty|DatatypeProperty", "definition": "..."}}"""

AXIOM_USER_TEMPLATE = """\
These strings are rule-like or conditional expressions from insurance contracts.
They describe candidate TBox axiom patterns, not final OWL axioms yet.

Provide:
1. A canonical short axiom-pattern label in PascalCase. Abstract away from specific fact names and sentence structures, but do not be too vague — the label should evoke the core logical constraint.
   (e.g. EligibilityRestriction, PolicyTerminationOnSurrender, DisjointnessOfIssuerAndBeneficiary)
2. A one-sentence definition (≤ 20 words)
3. The most appropriate semantic axiom family from:
   - existential_restriction
   - universal_restriction
   - subclass_condition
   - disjointness
   - cardinality_constraint
   - eligibility_condition
   - prohibition_or_exclusion
   - other

Rules:
- Do not invent missing classes or properties.
- Do not output a full OWL axiom.
- Choose only one axiom family.
- Use "other" when the strings are too vague.

Strings: {members}

JSON:
{{
  "axiom": "...",
  "definition": "...",
  "axiom_family": "existential_restriction|universal_restriction|subclass_condition|disjointness|cardinality_constraint|eligibility_condition|prohibition_or_exclusion|other"
}}
"""


def top_members_by_frequency(cluster: ConceptCluster, k: int = 10) -> list[str]:
    sorted_members = sorted(cluster.members, key=lambda c: c.frequency, reverse=True)
    return [c.string for c in sorted_members[:k]]


def call_llm(
    client: OpenAI,
    model_name: str,
    system: str,
    user: str,
    max_tokens: int = 256,
    retries: int = 3,
) -> Optional[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.05,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}): {e}", file=sys.stderr)
        except Exception as e:
            print(f"    LLM call error (attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
    return None


def name_class_cluster(cluster: ConceptCluster, client: OpenAI, model_name: str) -> None:
    members = top_members_by_frequency(cluster)
    user = CLASS_USER_TEMPLATE.format(members=json.dumps(members))
    result = call_llm(client, model_name, CLASS_SYSTEM_PROMPT, user)
    if result:
        cluster.owl_name   = result.get("class", "")
        cluster.definition = result.get("definition", "")
        cluster.owl_type   = "owl:Class"
    else:
        cluster.owl_name = _pascal_case(members[0]) if members else f"Concept_{cluster.cluster_id}"
        cluster.owl_type = "owl:Class"


def name_property_cluster(cluster: ConceptCluster, client: OpenAI, model_name: str) -> None:
    members = top_members_by_frequency(cluster)
    user = PROPERTY_USER_TEMPLATE.format(members=json.dumps(members))
    result = call_llm(client, model_name, "", user)

    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    if result and isinstance(result, dict):
        cluster.owl_name   = result.get("property", "")
        cluster.definition = result.get("definition", "")
        raw_type           = result.get("type", "ObjectProperty")
        cluster.owl_type   = (
            "owl:ObjectProperty" if "Object" in raw_type else "owl:DatatypeProperty"
        )
    else:
        cluster.owl_name = _camel_case(members[0]) if members else f"property_{cluster.cluster_id}"
        cluster.owl_type = "owl:ObjectProperty"


def name_axiom_cluster(cluster: ConceptCluster, client: OpenAI, model_name: str) -> None:
    members = top_members_by_frequency(cluster)
    user = AXIOM_USER_TEMPLATE.format(members=json.dumps(members))
    result = call_llm(client, model_name, "", user)

    if result:
        cluster.owl_name = result.get("axiom", "")
        cluster.definition = result.get("definition", "")
        cluster.owl_type = "owl:AxiomPattern"
        cluster.axiom_family = result.get("axiom_family", "other")
    else:
        cluster.owl_name = _pascal_case(members[0]) if members else f"Axiom_{cluster.cluster_id}"
        cluster.owl_type = "owl:AxiomPattern"
        cluster.axiom_family = "other"


def name_all_clusters(
    all_clusters: dict[str, list[ConceptCluster]],
    client: OpenAI,
    model_name: str,
) -> None:
    namers = {
        'class':    name_class_cluster,
        'property': name_property_cluster,
        'axiom':    name_axiom_cluster,
    }
    for stream_name, clusters in all_clusters.items():
        real_clusters = [c for c in clusters if not c.is_noise]
        print(f"  Naming {len(real_clusters)} {stream_name} clusters ...")
        namer = namers[stream_name]
        for i, cluster in enumerate(real_clusters):
            namer(cluster, client, model_name)
            if (i + 1) % 20 == 0:
                print(f"    ... {i+1}/{len(real_clusters)} done")


# ---------------------------------------------------------------------------
# 3.5  Graph-native parent assignment
# ---------------------------------------------------------------------------
def assign_graph_native_parents(class_clusters: list[ConceptCluster]) -> None:
    for cluster in class_clusters:
        if cluster.is_noise:
            continue
        role_members = [m for m in cluster.members if m.origin == 'entity_role']
        if role_members:
            parent_types = Counter(m.parent for m in role_members if m.parent)
            if parent_types:
                cluster.graph_native_parent = parent_types.most_common(1)[0][0]
                cluster.parent_confidence   = 'high'
            else:
                cluster.graph_native_parent = cluster.llm_parent
                cluster.parent_confidence   = 'llm'
        else:
            cluster.graph_native_parent = cluster.llm_parent
            cluster.parent_confidence   = 'llm'


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _pascal_case(s: str) -> str:
    return ''.join(w.capitalize() for w in re.split(r'[\s_\-]+', s) if w)


def _camel_case(s: str) -> str:
    words = re.split(r'[\s_\-]+', s)
    return words[0].lower() + ''.join(w.capitalize() for w in words[1:]) if words else s


def serialize_cluster(cluster: ConceptCluster) -> dict:
    return {
        "cluster_id":          cluster.cluster_id,
        "stream":              cluster.stream,
        "is_noise":            cluster.is_noise,
        "owl_name":            cluster.owl_name,
        "owl_type":            cluster.owl_type,
        "definition":          cluster.definition,
        "llm_parent":          cluster.llm_parent,
        "axiom_family":        cluster.axiom_family,
        "graph_native_parent": cluster.graph_native_parent,
        "parent_confidence":   cluster.parent_confidence,
        "member_count":        len(cluster.members),
        "members": [
            {
                "string":    m.string,
                "origin":    m.origin,
                "parent":    m.parent,
                "frequency": m.frequency,
                "doc_freq":  m.doc_freq,
            }
            for m in sorted(cluster.members, key=lambda m: m.frequency, reverse=True)
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stage 3 — Semantic Deduplication & Clustering")
    p.add_argument(
        "--candidate_pool",
        default=cfg.STAGE2_CANDIDATE_POOL,
        help="Path to Stage 2 candidate_pool.json (default: STAGE2_OUT_DIR/candidate_pool.json from .env)",
    )
    p.add_argument(
        "--output_dir",
        default=cfg.STAGE3_OUT_DIR,
        help="Directory for Stage 3 output (default: STAGE3_OUT_DIR from .env)",
    )
    p.add_argument(
        "--embedding_model",
        default="all-mpnet-base-v2",
        help="Sentence-Transformers model name for embedding",
    )
    p.add_argument(
        "--api_base",
        default=cfg.API_BASE,
        help="Base URL of local vLLM OpenAI-compatible server (default: API_BASE from .env)",
    )
    p.add_argument(
        "--model_name",
        default=cfg.MODEL_NAME,
        help="Model name as served by vLLM (default: MODEL_NAME from .env)",
    )
    p.add_argument("--skip_naming", action="store_true",
                   help="Skip LLM naming step (useful for testing clustering only)")
    return p.parse_args()


def main():
    args       = parse_args()
    pool_path  = Path(args.candidate_pool)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading candidate pool ...")
    with open(pool_path) as f:
        raw = json.load(f)
    candidates = [
        Candidate(
            string=r["string"], origin=r["origin"], parent=r["parent"],
            frequency=r["frequency"], doc_freq=r["doc_freq"], meta=r.get("meta", {}),
        )
        for r in raw
    ]
    print(f"  Loaded {len(candidates)} candidates.")

    # Client is needed for both axiom classification (3.1) and cluster naming (3.4)
    client = None
    if args.model_name:
        client = OpenAI(api_key="dummy", base_url=args.api_base)

    print("\n[3.1] Classifying fact names for axiom routing ...")
    fact_classifications: dict[str, str] = {}
    if client:
        fact_classifications = classify_fact_names(candidates, client, args.model_name)
    else:
        print("  [WARNING] No model_name set — all fact_name candidates will default to 'property' stream.")

    print("\n[3.1] Separating into streams ...")
    streams = separate_streams(candidates, fact_classifications)

    print(f"\n[3.2] Loading embedding model '{args.embedding_model}' ...")
    emb_model = SentenceTransformer(args.embedding_model)

    embeddings_by_stream: dict[str, np.ndarray] = {}
    for stream_name, stream_cands in streams.items():
        if not stream_cands:
            continue
        print(f"  Embedding {stream_name} stream ({len(stream_cands)} candidates) ...")
        embeddings_by_stream[stream_name] = embed_stream(stream_cands, emb_model)

    print("\n[3.3] Clustering...")
    all_clusters: dict[str, list[ConceptCluster]] = {}
    for stream_name, stream_cands in streams.items():
        if not stream_cands:
            all_clusters[stream_name] = []
            continue
        emb = embeddings_by_stream[stream_name]
        print("sim threshold for stream", stream_name, "is", SIM_THRESHOLD.get(stream_name))
        labels = cluster_stream(emb, sim_threshold=SIM_THRESHOLD.get(stream_name, 0.9))
        # labels = cluster_stream(emb, sim_threshold=args.sim_threshold)
        all_clusters[stream_name] = build_clusters(stream_name, stream_cands, labels)

    if not args.skip_naming:
        if not client:
            print(
                "\n[WARNING] --model_name not set and MODEL_NAME env var is empty. "
                "Skipping LLM naming. Pass --skip_naming to suppress this warning.",
                file=sys.stderr,
            )
        else:
            print(f"\n[3.4] Naming clusters via local vLLM at {args.api_base} ...")
            name_all_clusters(all_clusters, client, args.model_name)
    else:
        print("\n[3.4] Skipping LLM naming (--skip_naming).")

    print("\n[3.5] Assigning graph-native parents for class stream ...")
    assign_graph_native_parents(all_clusters.get('class', []))

    print("\nSerializing output ...")
    output: dict[str, list] = {}
    stats  = {}
    for stream_name, clusters in all_clusters.items():
        serialized = [serialize_cluster(c) for c in clusters]
        output[stream_name] = serialized
        real  = [c for c in clusters if not c.is_noise]
        noise = [c for c in clusters if c.is_noise]
        stats[stream_name] = {
            "real_clusters":    len(real),
            "noise_candidates": sum(len(c.members) for c in noise),
            "total_candidates": sum(len(c.members) for c in clusters),
        }

    out_path = output_dir / "named_concept_clusters.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Written: {out_path}")

    stats_path = output_dir / "stage3_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Written: {stats_path}")

    print("\n=== Stage 3 complete ===")
    for sname, s in stats.items():
        print(f"  {sname:8s}: {s['real_clusters']} clusters, "
              f"{s['noise_candidates']} noise candidates, "
              f"{s['total_candidates']} total")


if __name__ == "__main__":
    main()