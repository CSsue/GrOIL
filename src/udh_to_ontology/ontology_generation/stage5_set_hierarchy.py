"""
Stage 5 — Hierarchy Construction
==================================
Input:  property_clusters_with_dr.json  (Stage 4 output)
        candidate_pool.json             (Stage 2 output, for edge distribution signals)
Output: class_hierarchy.json           — every OWL class with one or more rdfs:subClassOf parents
        hierarchy_flagged.json         — low-confidence placements and cycle-broken classes
        stage5_stats.json

Pipeline:
  5.1  Load class vocabulary + Stage 2 edge-label meta distributions
  5.2  Resolve graph_native_parent (raw entity-type strings → OWL class names)
  5.3  Compute edge-distribution signals (source-profile + range-profile cosine similarity)
  5.4  Compute property-overlap signal (Jaccard over domain-assigned properties)
  5.5  Phase 1 — LLM discovers abstract top-level classes not present in corpus
  5.6  Topological pre-sort of placement order (high-signal classes first)
  5.7  Phase 2 — Per-class LLM placement with all four signals + full closed vocabulary
         (multiple parents allowed; vocabulary enforced via prompt constraints and
         post-hoc validation — invalid parent names are stripped before committing)
  5.8  Cycle resolution: per-edge forward-reachability detection (iterative passes);
         LLM identifies and removes the semantically weakest edge in each cycle.
         Two-node cycles (A ↔ B) use a dedicated prompt to determine which direction
         is correct. Falls back to confidence-based logic only if the LLM call fails.

Design principles (from paper §1.2 and §1.3):
  • Graph-derived signals are computed first; LLM resolves ambiguity under a
    hard vocabulary constraint — same pattern as Stage 4.
  • Multiple rdfs:subClassOf parents are permitted (OWL DAG, not a tree).
  • Low-confidence placements are flagged for human review and written to
    hierarchy_flagged.json; they are still included in class_hierarchy.json.
  • Abstract classes are inferred_top_level=True and sit directly under owl:Thing.
  • Cycle resolution: the LLM identifies the semantically weakest edge in each
    cycle and proposes a valid replacement parent. Two-node mutual-subclass cycles
    are handled with a dedicated "which direction is correct?" prompt to avoid
    incorrect collapse to owl:Thing.

Usage:
    python stage5_set_hierarchy.py
    python stage5_set_hierarchy.py \\
        --stage4_path  ./stage4_out/property_clusters_with_dr.json \\
        --stage2_pool  ./stage2_out/candidate_pool.json \\
        --output_dir   ./stage5_out
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

from config import cfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OWL_THING = "owl:Thing"
MAX_RETRIES = 3
RETRY_DELAY = 5.0
EDGE_SIM_THRESHOLD = 0.85
PROP_JACCARD_THRESHOLD = 0.60
ABSTRACT_MAX = 10

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ClassNode:
    owl_name:             str
    definition:           str
    members:              list
    graph_native_parent:  Optional[str] = None
    resolved_gnp:         Optional[str] = None
    parents:              list = field(default_factory=list)
    placement_confidence: str  = "unresolved"
    placement_source:     str  = "unresolved"
    flagged:              bool = False
    flag_reason:          str  = ""
    is_abstract:          bool = False
    inferred_top_level:   bool = False


@dataclass
class SignalBundle:
    owl_name:         str
    resolved_gnp:     Optional[str]  = None
    source_sim_top:   list = field(default_factory=list)
    range_sim_top:    list = field(default_factory=list)
    prop_overlap_top: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 5.1  Load inputs
# ---------------------------------------------------------------------------

def load_class_clusters(stage4_path: Path) -> list[dict]:
    with open(stage4_path) as f:
        data = json.load(f)
    classes = [c for c in data.get("class", []) if not c.get("is_noise", False)]
    print(f"  Loaded {len(classes)} real class clusters from Stage 4 output.")
    return classes


def load_stage2_pool(pool_path: Path) -> list[dict]:
    with open(pool_path) as f:
        raw = json.load(f)
    print(f"  Loaded {len(raw)} Stage 2 candidates.")
    return raw


def build_edge_meta_index(pool: list[dict]) -> dict[str, dict]:
    index: dict[str, dict] = defaultdict(lambda: {"src_types": Counter(), "tgt_types": Counter()})
    for cand in pool:
        if cand.get("origin") != "edge_label":
            continue
        label = cand.get("string", "").strip().lower()
        meta  = cand.get("meta", {})
        for etype, cnt in meta.get("src_types", {}).items():
            index[label]["src_types"][etype] += cnt
        for etype, cnt in meta.get("tgt_types", {}).items():
            index[label]["tgt_types"][etype] += cnt
    return dict(index)


# ---------------------------------------------------------------------------
# 5.2  Resolve graph_native_parent
# ---------------------------------------------------------------------------

def build_member_string_index(class_clusters: list[dict]) -> dict[str, str]:
    index: dict[str, str] = {}
    for cluster in class_clusters:
        owl_name = cluster.get("owl_name", "")
        if not owl_name:
            continue
        index[owl_name.lower()] = owl_name
        for m in cluster.get("members", []):
            s = m.get("string", "").strip().lower()
            if s:
                index[s] = owl_name
    return index


def resolve_gnp(raw_parent: Optional[str], member_index: dict[str, str]) -> Optional[str]:
    if not raw_parent:
        return None
    key = raw_parent.strip().lower()
    if key in member_index:
        return member_index[key]
    key_nospace = key.replace("_", " ")
    if key_nospace in member_index:
        return member_index[key_nospace]
    return None


def resolve_all_gnp(nodes: list[ClassNode], member_index: dict[str, str]) -> None:
    resolved = 0
    for node in nodes:
        if node.graph_native_parent:
            rp = resolve_gnp(node.graph_native_parent, member_index)
            node.resolved_gnp = rp
            if rp:
                resolved += 1
    print(f"  GNP resolved: {resolved}/{sum(1 for n in nodes if n.graph_native_parent)} "
          f"(of {len(nodes)} total classes have a GNP)")


# ---------------------------------------------------------------------------
# 5.3  Edge-distribution signals
# ---------------------------------------------------------------------------

def build_edge_profiles(
    class_clusters: list[dict],
    edge_meta: dict[str, dict],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    all_labels = sorted(edge_meta.keys())
    label_idx  = {lbl: i for i, lbl in enumerate(all_labels)}
    n_labels   = len(all_labels)

    if n_labels == 0:
        return {}, {}, []

    source_profiles: dict[str, np.ndarray] = {}
    range_profiles:  dict[str, np.ndarray] = {}

    for cluster in class_clusters:
        owl_name = cluster.get("owl_name", "")
        if not owl_name:
            continue

        class_strings = {owl_name.lower()}
        for m in cluster.get("members", []):
            s = m.get("string", "").strip().lower()
            if s:
                class_strings.add(s)

        src_vec = np.zeros(n_labels, dtype=np.float32)
        rng_vec = np.zeros(n_labels, dtype=np.float32)

        for label, meta in edge_meta.items():
            idx = label_idx[label]
            for entity_type_str, cnt in meta["src_types"].items():
                if entity_type_str.strip().lower() in class_strings:
                    src_vec[idx] += cnt
            for entity_type_str, cnt in meta["tgt_types"].items():
                if entity_type_str.strip().lower() in class_strings:
                    rng_vec[idx] += cnt

        source_profiles[owl_name] = src_vec
        range_profiles[owl_name]  = rng_vec

    return source_profiles, range_profiles, all_labels


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_edge_sim_signal(
    node_name: str,
    source_profiles: dict[str, np.ndarray],
    range_profiles:  dict[str, np.ndarray],
    top_k: int = 5,
) -> tuple[list, list]:
    src_vec = source_profiles.get(node_name)
    rng_vec = range_profiles.get(node_name)
    all_names = [n for n in source_profiles if n != node_name]

    src_top, rng_top = [], []

    if src_vec is not None and np.any(src_vec):
        sims = [(n, cosine_sim(src_vec, source_profiles[n])) for n in all_names if n in source_profiles]
        sims.sort(key=lambda x: x[1], reverse=True)
        src_top = [(n, round(s, 3)) for n, s in sims if s >= EDGE_SIM_THRESHOLD][:top_k]

    if rng_vec is not None and np.any(rng_vec):
        sims = [(n, cosine_sim(rng_vec, range_profiles[n])) for n in all_names if n in range_profiles]
        sims.sort(key=lambda x: x[1], reverse=True)
        rng_top = [(n, round(s, 3)) for n, s in sims if s >= EDGE_SIM_THRESHOLD][:top_k]

    return src_top, rng_top


# ---------------------------------------------------------------------------
# 5.4  Property-overlap signal
# ---------------------------------------------------------------------------

def build_domain_property_index(stage4_data: dict) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for stream in ("property", "axiom"):
        for cluster in stage4_data.get(stream, []):
            if cluster.get("is_noise", False):
                continue
            domain = cluster.get("domain", "").strip()
            prop   = cluster.get("owl_name", "").strip()
            if domain and domain != OWL_THING and prop:
                index[domain].add(prop)
    return dict(index)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_prop_overlap_signal(
    node_name: str,
    domain_property_index: dict[str, set[str]],
    all_owl_names: list[str],
    top_k: int = 5,
) -> list:
    my_props = domain_property_index.get(node_name, set())
    if not my_props:
        return []
    sims = []
    for other in all_owl_names:
        if other == node_name:
            continue
        other_props = domain_property_index.get(other, set())
        j = jaccard(my_props, other_props)
        if j >= PROP_JACCARD_THRESHOLD:
            sims.append((other, round(j, 3)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm_raw(
    client: OpenAI,
    model_name: str,
    messages: list[dict],
    max_tokens: int = 2048,
    retries: int = MAX_RETRIES,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"    LLM error (attempt {attempt+1}): {e} — retrying in {RETRY_DELAY}s")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    LLM failed after {retries} attempts: {e}", file=sys.stderr)
                return None


def safe_parse_json(text: Optional[str], fallback: dict) -> dict:
    if not text:
        return fallback
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return fallback


# ---------------------------------------------------------------------------
# 5.5  Phase 1 — Abstract class discovery
# ---------------------------------------------------------------------------

ABSTRACT_SYSTEM = (
    "You are an expert ontology engineer. Your task is to propose a minimal set of "
    "abstract top-level organizing classes for an insurance domain OWL ontology. "
    "These classes do NOT appear in the corpus — they are abstract organizing concepts "
    "that give the bottom-up hierarchy principled structure at the top."
)

ABSTRACT_USER_TEMPLATE = """\
Below is the full inventory of OWL classes discovered bottom-up from an insurance document corpus.
Because bottom-up extraction surfaces only the concepts explicitly named in documents, the corpus
classes lack organising abstractions at the top.

Your task: propose between 5 and {max_abstract} ABSTRACT top-level classes that together provide
a shallow, mutually-exclusive skeleton for this set of domain classes.

Rules:
- These abstract classes must NOT be directly observable in the corpus.
- They must be domain-appropriate for insurance (e.g., Party, FinancialInstrument, TemporalConcept).
- They must be mutually exclusive in intent — no two abstract classes should compete for the same children.
- Maximum {max_abstract} abstract classes.
- All abstract classes sit directly under owl:Thing.

Discovered classes (name | definition):
{class_list}

Respond in JSON:
{{
  "abstract_classes": [
    {{"name": "PascalCaseName", "definition": "One sentence definition."}},
    ...
  ]
}}
"""


def discover_abstract_classes(
    nodes: list[ClassNode],
    client: OpenAI,
    model_name: str,
    max_abstract: int = ABSTRACT_MAX,
) -> list[ClassNode]:
    class_list_str = "\n".join(
        f"  {n.owl_name} | {n.definition[:80]}" for n in nodes[:120]
    )
    prompt = ABSTRACT_USER_TEMPLATE.format(
        max_abstract=max_abstract,
        class_list=class_list_str,
    )
    messages = [
        {"role": "system", "content": ABSTRACT_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    print("  Calling LLM for abstract class discovery ...")
    raw = call_llm_raw(client, model_name, messages)
    parsed = safe_parse_json(raw, {"abstract_classes": []})

    corpus_names = {n.owl_name for n in nodes}
    
    abstract_nodes = []
    for item in parsed.get("abstract_classes", []):
        name = item.get("name", "").strip()
        defn = item.get("definition", "").strip()
        if not name:
            continue
        if name in corpus_names:
            print(f"    Skipping abstract '{name}' — already exists as corpus class.")
            continue
        node = ClassNode(
            owl_name=name,
            definition=defn,
            members=[],
            is_abstract=True,
            inferred_top_level=True,
            parents=[OWL_THING],
            placement_confidence="high",
            placement_source="abstract",
        )
        abstract_nodes.append(node)
        print(f"    Abstract: {name} — {defn[:60]}")

    print(f"  Abstract class discovery: {len(abstract_nodes)} classes proposed.")
    return abstract_nodes


# ---------------------------------------------------------------------------
# 5.6  Topological pre-sort
# ---------------------------------------------------------------------------

def score_node_for_sort(node: ClassNode) -> float:
    if node.resolved_gnp:
        return 0.0
    return 1.0


def topological_presort(nodes: list[ClassNode], signal_bundles: dict[str, SignalBundle]) -> list[ClassNode]:
    def sort_key(n: ClassNode) -> float:
        base = score_node_for_sort(n)
        sb = signal_bundles.get(n.owl_name)
        if sb:
            has_edge = bool(sb.source_sim_top or sb.range_sim_top)
            has_prop = bool(sb.prop_overlap_top)
            if has_edge or has_prop:
                base = min(base, 0.5)
        return base

    return sorted(nodes, key=sort_key)


# ---------------------------------------------------------------------------
# 5.7  Phase 2 — Per-class LLM placement
# ---------------------------------------------------------------------------

PLACEMENT_SYSTEM = (
    "You are an expert OWL ontology engineer specialising in the insurance domain. "
    "Your task is to determine the rdfs:subClassOf parent(s) for a given OWL class, "
    "using only the provided vocabulary and signals."
)

PLACEMENT_USER_TEMPLATE = """\
Place the following OWL class in the class hierarchy.

CLASS TO PLACE
  Name      : {owl_name}
  Definition: {definition}
  Top corpus strings: {members}

SIGNALS (use as evidence, not as direct assignments. pay attention to the type and strength of each signal, and use that to guide your decision making.):
{signal_block}

VOCABULARY — choose parents ONLY from this list:
{corpus_classes}
{abstract_classes}
  owl:Thing

INSTRUCTIONS
- Start by identifying the SINGLE best parent — the most relevant class from the vocabulary that this class is a subclass of. Make sure to take as much insight 
as possible from the class name, definition, members, and most importantly the signals. Use owl:Thing only as absolute last resort.
- Only add a second parent if this class genuinely inherits from two DISJOINT concepts that cannot be reduced to one. 
- A second parent is NOT justified if:
    (a) it is already an ancestor of your first parent, or
    (b) you are uncertain between two candidates — pick the better one.
- Maximum 2 parents total. If in doubt, use 1.

Respond in JSON:
{{
  "primary_parent": "BestParent",
  "additional_parents": [],
  "additional_parents_justification": "Why each additional parent represents a genuinely disjoint concept this class inherits from. Leave empty string if none.",
  "confidence": "high|medium|low",
  "reasoning": "One or two sentence justification."
}}
"""


def format_signal_block(sb: SignalBundle) -> str:
    lines = []
    if sb.resolved_gnp:
        lines.append(f"  [STRONG — graph-native parent, resolved] → {sb.resolved_gnp}")
    if sb.source_sim_top:
        top_str = ", ".join(f"{n} ({s:.2f})" for n, s in sb.source_sim_top[:3])
        lines.append(f"  [MEDIUM — edge source-profile similarity] → {top_str}")
    if sb.range_sim_top:
        top_str = ", ".join(f"{n} ({s:.2f})" for n, s in sb.range_sim_top[:3])
        lines.append(f"  [MEDIUM — edge range-profile similarity (sibling signal)] → {top_str}")
    if sb.prop_overlap_top:
        top_str = ", ".join(f"{n} (J={s:.2f})" for n, s in sb.prop_overlap_top[:3])
        lines.append(f"  [MEDIUM — property domain overlap] → {top_str}")
    if not lines:
        lines.append("  [No corpus signals available — use semantic reasoning only]")
    return "\n".join(lines)


def place_class(
    node: ClassNode,
    sb: SignalBundle,
    corpus_class_names: list[str],
    abstract_class_names: list[str],
    valid_vocab: set[str],
    client: OpenAI,
    model_name: str,
) -> None:
    members_str  = ", ".join(node.members[:8]) if node.members else "(none)"
    corpus_str   = "\n".join(f"  {n}" for n in corpus_class_names if n != node.owl_name)
    abstract_str = "\n".join(f"  {n}" for n in abstract_class_names)

    prompt = PLACEMENT_USER_TEMPLATE.format(
        owl_name=node.owl_name,
        definition=node.definition or "(no definition)",
        members=members_str,
        signal_block=format_signal_block(sb),
        corpus_classes=corpus_str or "  (none)",
        abstract_classes=abstract_str or "  (none)",
    )
    messages = [
        {"role": "system", "content": PLACEMENT_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    raw = call_llm_raw(client, model_name, messages)
    result = safe_parse_json(raw, {"parents": [OWL_THING], "confidence": "low", "reasoning": "LLM parse failed"})

    primary = result.get("primary_parent", OWL_THING)
    additional = result.get("additional_parents", [])
    justification = result.get("additional_parents_justification", "").strip()

    print(f"    LLM proposed primary parent: {primary}, additional={additional}, justification='{justification}'")

    if not justification:
        additional = []

    raw_parents = [primary] + additional

    cleaned_parents = []
    for p in raw_parents:
        if isinstance(p, str) and p.strip() in valid_vocab:
            cleaned_parents.append(p.strip())
    if not cleaned_parents:
        cleaned_parents = [OWL_THING]

    seen = set()
    final_parents = []
    for p in cleaned_parents:
        if p not in seen:
            seen.add(p)
            final_parents.append(p)

    confidence = result.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    node.parents             = final_parents
    node.placement_confidence = confidence
    node.placement_source    = "llm"
    node.flagged             = (confidence == "low")
    node.flag_reason         = result.get("reasoning", "") if confidence == "low" else ""


def run_phase2_placement(
    corpus_nodes: list[ClassNode],
    abstract_nodes: list[ClassNode],
    signal_bundles: dict[str, SignalBundle],
    client: OpenAI,
    model_name: str,
) -> None:
    corpus_names   = [n.owl_name for n in corpus_nodes]
    abstract_names = [n.owl_name for n in abstract_nodes]
    valid_vocab    = set(corpus_names) | set(abstract_names) | {OWL_THING}

    for an in abstract_nodes:
        an.parents             = [OWL_THING]
        an.placement_confidence = "high"
        an.placement_source    = "abstract"

    total = len(corpus_nodes)
    for i, node in enumerate(corpus_nodes):
        sb = signal_bundles.get(node.owl_name, SignalBundle(owl_name=node.owl_name))
        print(f"  [{i+1}/{total}] Placing {node.owl_name} ...")
        place_class(node, sb, corpus_names, abstract_names, valid_vocab, client, model_name)
        print(f"    → parents={node.parents}, conf={node.placement_confidence}")


# ---------------------------------------------------------------------------
# 5.8  Cycle resolution — graph helpers (shared by LLM and fallback paths)
# ---------------------------------------------------------------------------

_CONF_ORDER = {"high": 0, "medium": 1, "low": 2, "unresolved": 3}


def _build_parent_map(all_nodes: list[ClassNode]) -> dict[str, list[str]]:
    return {
        n.owl_name: [p for p in n.parents if p != OWL_THING]
        for n in all_nodes
    }


def _path_exists(src: str, dst: str, parent_map: dict[str, list[str]]) -> bool:
    if src == dst:
        return False
    visited = set()
    stack = list(parent_map.get(src, []))
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(parent_map.get(node, []))
    return False


def _find_cycle_edges(all_nodes: list[ClassNode]) -> list[tuple[str, str]]:
    parent_map = _build_parent_map(all_nodes)
    cycle_edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for node in all_nodes:
        child = node.owl_name
        for parent in parent_map.get(child, []):
            if (child, parent) in seen:
                continue
            if _path_exists(parent, child, parent_map):
                cycle_edges.append((child, parent))
                seen.add((child, parent))

    return cycle_edges


def _nearest_valid_ancestor(
    start: str,
    excluded: set[str],
    parent_map: dict[str, list[str]],
) -> str:
    visited = {start}
    queue = list(parent_map.get(start, []))
    while queue:
        candidate = queue.pop(0)
        if candidate in excluded:
            queue.extend(p for p in parent_map.get(candidate, []) if p not in visited)
            visited.add(candidate)
            continue
        return candidate
    return OWL_THING


def ensure_completeness(all_nodes: list[ClassNode]) -> None:
    for node in all_nodes:
        if not node.parents:
            node.parents     = [OWL_THING]
            node.flagged     = True
            node.flag_reason = (
                (node.flag_reason + " | " if node.flag_reason else "") +
                "No parent assigned after cycle resolution — defaulted to owl:Thing for review."
            )


# ---------------------------------------------------------------------------
# 5.8  Cycle resolution — LLM prompts
# ---------------------------------------------------------------------------

_CYCLE_SYSTEM = (
    "You are an expert OWL ontology engineer specialising in the insurance domain. "
    "You are given a directed cycle in an rdfs:subClassOf hierarchy — a situation "
    "where A is declared a subclass of B, and (directly or through intermediaries) "
    "B is also declared a subclass of A. This is logically invalid in OWL. "
    "Your task is to break the cycle by removing exactly one rdfs:subClassOf edge "
    "and, if necessary, re-anchoring the affected class to a semantically appropriate "
    "replacement parent."
)

_CYCLE_USER_TEMPLATE = """\
The following classes form a directed cycle that must be broken.
A directed cycle means: following rdfs:subClassOf edges, you can travel from a class
back to itself. This is invalid in OWL and must be resolved by removing one edge.

═══════════════════════════════════════════════════════
CYCLE EDGES  (each line: child ──subClassOf──▶ parent)
═══════════════════════════════════════════════════════
{cycle_edges}

═══════════════════════════════════════════════════════
FULL DETAILS FOR EACH CLASS IN THE CYCLE
═══════════════════════════════════════════════════════
{class_details}

═══════════════════════════════════════════════════════
VALID REPLACEMENT PARENTS  (choose new_parent from here)
These are already-placed classes NOT part of this cycle.
═══════════════════════════════════════════════════════
{valid_anchors}

═══════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════
1. Identify which single rdfs:subClassOf edge (child → parent) is semantically
   WEAKEST — i.e. where the subClassOf relationship is most questionable,
   most redundant given other parents, or most clearly backwards.

2. After removing that edge, the child class may need a new parent if it would
   otherwise be left without one (or only with owl:Thing). Propose the most
   semantically appropriate replacement from the VALID REPLACEMENT PARENTS list.
   If the child already has other valid parents after removal, set new_parent to null.

3. Reason explicitly:
   - Why is this edge the weakest in the cycle?
   - Is the relationship backwards (B is actually more specific than A)?
   - Or is it simply redundant (already implied by another parent)?
   - Why is your proposed new_parent (if any) a better anchor?

IMPORTANT:
- new_parent MUST be from the VALID REPLACEMENT PARENTS list, or null, or "owl:Thing".
- Do NOT choose a class that is part of the cycle as new_parent.
- Do NOT remove more than one edge.

Respond in JSON only:
{{
  "remove_edge": {{
    "child":  "ExactClassName",
    "parent": "ExactClassName"
  }},
  "new_parent": "ExactClassName or null",
  "reasoning": "2-4 sentence explanation of why this edge is weakest and why the replacement is appropriate."
}}
"""

_TWO_NODE_SYSTEM = (
    "You are an expert OWL ontology engineer. You are resolving a 'Mutual Subclass' error "
    "where two classes are incorrectly set as parents of each other. Your job is to "
    "determine which class is the broader category (Superclass) and which is the "
    "more specific concept (Subclass)."
)



_TWO_NODE_USER_TEMPLATE = """\
DIAGNOSIS: MUTUAL SUBCLASS ERROR
The following two classes are currently pointing to each other as parents. You must decide which one is the broader concept (Superclass).

CLASSES:
1. {class_a}
2. {class_b}

DEFINITIONS:
- {class_a}: {def_a}
- {class_b}: {def_b}

TASK:
Pick the most semantically sound relationship:
(Option A): {class_a} IS A TYPE OF {class_b} 
           (Result: {class_b} is the Parent; {class_a} is the Child)

(Option B): {class_b} IS A TYPE OF {class_a} 
           (Result: {class_a} is the Parent; {class_b} is the Child)

RESPOND IN JSON:
{{
  "decision": "A or B",
  "reasoning": "Explain your choice. For example: 'X is the broader category because...'"
}}
"""


# ---------------------------------------------------------------------------
# 5.8  Cycle resolution — LLM helpers
# ---------------------------------------------------------------------------

def _collect_cycle_node_names(cycle_edges: list[tuple[str, str]]) -> set[str]:
    names: set[str] = set()
    for child, parent in cycle_edges:
        names.add(child)
        names.add(parent)
    return names


def _format_class_detail(node: ClassNode) -> str:
    members_str = ", ".join(node.members[:6]) if node.members else "(none)"
    reasoning   = node.flag_reason or "(none recorded)"
    return (
        f"Class       : {node.owl_name}\n"
        f"  Definition : {node.definition or '(none)'}\n"
        f"  Members    : {members_str}\n"
        f"  Parents    : {node.parents}\n"
        f"  Confidence : {node.placement_confidence}\n"
        f"  Reasoning  : {reasoning}"
    )


def _safe_anchors(all_nodes: list[ClassNode], cycle_node_names: set[str]) -> set[str]:
    safe = {OWL_THING}
    for n in all_nodes:
        if n.owl_name not in cycle_node_names:
            safe.add(n.owl_name)
    return safe


def _extract_minimal_cycle(
    start_child: str,
    start_parent: str,
    parent_map: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """
    BFS from start_parent upward to find start_child, reconstructing the
    minimal cycle path as a list of (child, parent) edges.
    Returns [] if the path cannot be found.
    """
    queue: deque = deque()
    queue.append((start_parent, [start_parent]))
    visited = {start_parent}

    found_path: Optional[list[str]] = None
    while queue:
        current, path = queue.popleft()
        if current == start_child:
            found_path = path
            break
        for p in parent_map.get(current, []):
            if p not in visited and p != OWL_THING:
                visited.add(p)
                queue.append((p, path + [p]))

    if not found_path:
        return []

    # Edges along the upward walk: (found_path[i], found_path[i+1])
    cycle_edges = [(start_child, start_parent)]
    for i in range(len(found_path) - 1):
        cycle_edges.append((found_path[i], found_path[i + 1]))
    return cycle_edges


# ---------------------------------------------------------------------------
# 5.8  Cycle resolution — two-node cycle handler
# ---------------------------------------------------------------------------

def _resolve_two_node_cycle(
    node_a: ClassNode,
    node_b: ClassNode,
    all_nodes: list[ClassNode],
    client: OpenAI,
    model_name: str,
) -> None:
    cycle_node_names = {node_a.owl_name, node_b.owl_name}
    valid_anchors    = _safe_anchors(all_nodes, cycle_node_names)

    prompt = _TWO_NODE_USER_TEMPLATE.format(
        class_a     = node_a.owl_name,
        def_a       = node_a.definition or "(none)",
        members_a   = ", ".join(node_a.members[:6]) or "(none)",
        parents_a   = node_a.parents,
        conf_a      = node_a.placement_confidence,
        reasoning_a = node_a.flag_reason or "(none)",
        class_b     = node_b.owl_name,
        def_b       = node_b.definition or "(none)",
        members_b   = ", ".join(node_b.members[:6]) or "(none)",
        parents_b   = node_b.parents,
        conf_b      = node_b.placement_confidence,
        reasoning_b = node_b.flag_reason or "(none)",
        valid_anchors = "\n".join(f"  {a}" for a in sorted(valid_anchors - {OWL_THING})[:60]),
    )
    messages = [
        {"role": "system", "content": _TWO_NODE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    raw    = call_llm_raw(client, model_name, messages)

    result = safe_parse_json(raw, {})
    decision = result.get("decision", "").strip().upper()
    reasoning = result.get("reasoning", "")

    if decision == "A":
        # A is Child, B is Parent
        node_a.parents = [node_b.owl_name]
        node_b.parents = [p for p in node_b.parents if p != node_a.owl_name]
        applied = f"Set {node_a.owl_name} subclassOf {node_b.owl_name}"
    else:
        # B is Child, A is Parent (Default to B if LLM response is fuzzy)
        node_b.parents = [node_a.owl_name]
        node_a.parents = [p for p in node_a.parents if p != node_b.owl_name]
        applied = f"Set {node_b.owl_name} subclassOf {node_a.owl_name}"

    # Sync flags and reasoning
    for node in [node_a, node_b]:
        node.flagged = True
        node.flag_reason = f"Two-node cycle resolved: {applied}. Reasoning: {reasoning}"

    print(f"    Resolved: {applied}")

# ---------------------------------------------------------------------------
# 5.8  Cycle resolution — general LLM handler
# ---------------------------------------------------------------------------

def _resolve_general_cycle_llm(
    cycle_edges: list[tuple[str, str]],
    all_nodes: list[ClassNode],
    name_to_node: dict[str, ClassNode],
    client: OpenAI,
    model_name: str,
) -> bool:
    """
    Presents the full cycle to the LLM and applies its decision.
    Returns True if an edge was successfully removed, False to trigger fallback.
    """
    cycle_node_names = _collect_cycle_node_names(cycle_edges)
    valid_anchors    = _safe_anchors(all_nodes, cycle_node_names)

    cycle_edge_str = "\n".join(
        f"  {child}  ──subClassOf──▶  {parent}"
        for child, parent in cycle_edges
    )
    class_details_str = "\n\n".join(
        _format_class_detail(name_to_node[n])
        for n in cycle_node_names
        if n in name_to_node
    )
    sorted_anchors = sorted(valid_anchors - {OWL_THING})[:60] + [OWL_THING]
    anchor_str = "\n".join(f"  {a}" for a in sorted_anchors)

    prompt = _CYCLE_USER_TEMPLATE.format(
        cycle_edges   = cycle_edge_str,
        class_details = class_details_str,
        valid_anchors = anchor_str,
    )
    messages = [
        {"role": "system", "content": _CYCLE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]

    raw    = call_llm_raw(client, model_name, messages)
    result = safe_parse_json(raw, {})

    remove_edge    = result.get("remove_edge", {})
    child_name     = (remove_edge.get("child") or "").strip()
    parent_name    = (remove_edge.get("parent") or "").strip()
    new_parent_raw = result.get("new_parent")
    new_parent     = new_parent_raw.strip() if isinstance(new_parent_raw, str) else None
    reasoning      = result.get("reasoning", "")

    child = name_to_node.get(child_name)
    if not child:
        print(f"    LLM returned unknown child '{child_name}' — falling back.", file=sys.stderr)
        return False
    if parent_name not in child.parents:
        print(f"    LLM proposed non-existent edge {child_name} → {parent_name} — falling back.",
              file=sys.stderr)
        return False
    if new_parent and new_parent not in valid_anchors:
        print(f"    LLM proposed invalid new_parent '{new_parent}' — defaulting to owl:Thing.")
        new_parent = OWL_THING

    old_parents = list(child.parents)
    child.parents = [p for p in child.parents if p != parent_name]

    if not child.parents:
        child.parents = [new_parent] if new_parent else [OWL_THING]
    elif new_parent and new_parent not in child.parents:
        child.parents.append(new_parent)

    child.flagged     = True
    child.flag_reason = (
        f"Cycle resolved by LLM: removed edge {child_name} → {parent_name}. "
        f"New parents: {child.parents} (was: {old_parents}). "
        f"Reasoning: {reasoning}"
    )
    print(f"    LLM removed cycle edge {child_name} → {parent_name}; "
          f"{child_name} now parents={child.parents}")
    return True


# ---------------------------------------------------------------------------
# 5.8  Cycle resolution — confidence-based fallback (original logic)
# ---------------------------------------------------------------------------

def _resolve_cycle_confidence_fallback(
    cycle_edges: list[tuple[str, str]],
    name_to_node: dict[str, ClassNode],
    pass_num: int,
) -> None:
    child_name, parent_name = cycle_edges[0]
    child  = name_to_node.get(child_name)
    parent = name_to_node.get(parent_name)
    if not child or not parent:
        return

    child_conf  = _CONF_ORDER.get(child.placement_confidence, 3)
    parent_conf = _CONF_ORDER.get(parent.placement_confidence, 3)
    cycle_members = {child_name, parent_name}

    parent_map = _build_parent_map(list(name_to_node.values()))
    new_parent = _nearest_valid_ancestor(parent_name, cycle_members, parent_map)

    old_parents = list(child.parents)
    child.parents = [p for p in child.parents if p != parent_name]
    if not child.parents:
        child.parents = [new_parent]
    elif new_parent not in child.parents:
        child.parents.append(new_parent)

    child.flagged     = True
    child.flag_reason = (
        f"[FALLBACK] Cycle resolved (pass {pass_num}): removed edge "
        f"{child_name} → {parent_name} "
        f"(child_conf={child.placement_confidence}, parent_conf={parent.placement_confidence}); "
        f"elevated to {child.parents}. Was: {old_parents}."
    )
    print(f"    [FALLBACK] REMOVED cycle edge {child_name} → {parent_name}; "
          f"{child_name} elevated to {child.parents}")


# ---------------------------------------------------------------------------
# 5.8  Cycle resolution — main entry point
# ---------------------------------------------------------------------------

def resolve_cycles(
    all_nodes: list[ClassNode],
    client: OpenAI,
    model_name: str,
) -> int:
    """
    Iteratively finds directed cycles and resolves each one using the LLM.

    - Two-node mutual cycles (A ↔ B): dedicated prompt asks which direction
      is correct, avoiding the owl:Thing collapse of the original approach.
    - General cycles: LLM sees the full minimal cycle path, all class details,
      and the valid anchor vocabulary, and picks the semantically weakest edge.
    - Fallback: if the LLM call fails or returns an invalid response, the
      original confidence-based logic handles that cycle only.

    Returns the total number of resolution passes performed.
    """
    name_to_node  = {n.owl_name: n for n in all_nodes}
    total_removed = 0
    max_passes    = len(all_nodes) + 1

    for pass_num in range(max_passes):
        
        for node in all_nodes:
            if node.owl_name in node.parents:
                node.parents.remove(node.owl_name)

        cycle_edges = _find_cycle_edges(all_nodes)
        if not cycle_edges:
            break

        print(f"  Pass {pass_num + 1}: found {len(cycle_edges)} cycle edge(s) — resolving with LLM ...")

        # ── Detect two-node cycle (A → B and B → A both present) ────────────
        edge_set = set(cycle_edges)
        two_node_pair = None
        for child_name, parent_name in cycle_edges:
            if (parent_name, child_name) in edge_set:
                two_node_pair = (child_name, parent_name)
                break

        if two_node_pair:
            a_name, b_name = two_node_pair
            node_a = name_to_node.get(a_name)
            node_b = name_to_node.get(b_name)
            if node_a and node_b:
                print(f"  Detected two-node cycle: {a_name} ↔ {b_name}")
                _resolve_two_node_cycle(node_a, node_b, all_nodes, client, model_name)
                total_removed += 1
                continue

        # ── General cycle: extract minimal cycle path ────────────────────────
        first_child, first_parent = cycle_edges[0]
        parent_map = _build_parent_map(all_nodes)
        minimal_cycle = _extract_minimal_cycle(first_child, first_parent, parent_map)
        if not minimal_cycle:
            minimal_cycle = [(first_child, first_parent)]

        print(f"  Minimal cycle ({len(minimal_cycle)} edge(s)): "
              + " → ".join(f"{c}→{p}" for c, p in minimal_cycle))

        success = _resolve_general_cycle_llm(
            minimal_cycle, all_nodes, name_to_node, client, model_name
        )

        if not success:
            print("  LLM resolution failed — using confidence-based fallback.")
            _resolve_cycle_confidence_fallback(cycle_edges, name_to_node, pass_num + 1)

        total_removed += 1

    else:
        remaining = _find_cycle_edges(all_nodes)
        if remaining:
            print(
                f"  WARNING: {len(remaining)} cycle edge(s) remain after {max_passes} passes.",
                file=sys.stderr,
            )

    return total_removed


def cycle_resolution(
    all_nodes: list[ClassNode],
    client: OpenAI,
    model_name: str,
) -> None:
    """Entry point for §5.8."""
    print("  [5.8] Running LLM-based cycle resolution ...")
    n_removed = resolve_cycles(all_nodes, client, model_name)
    if n_removed == 0:
        print("  No cycles found — hierarchy is already a DAG.")
    else:
        print(f"  LLM cycle resolution complete: {n_removed} pass(es).")

    print("  [5.8] Ensuring every class has at least one parent ...")
    ensure_completeness(all_nodes)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_node(node: ClassNode) -> dict:
    return {
        "owl_name":             node.owl_name,
        "is_abstract":          node.is_abstract,
        "inferred_top_level":   node.inferred_top_level,
        "definition":           node.definition,
        "parents":              node.parents,
        "placement_confidence": node.placement_confidence,
        "placement_source":     node.placement_source,
        "graph_native_parent":  node.graph_native_parent,
        "resolved_gnp":         node.resolved_gnp,
        "flagged":              node.flagged,
        "flag_reason":          node.flag_reason,
        "top_members":          node.members[:8],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 5 — Hierarchy Construction")
    stage4_default = str(Path(cfg.STAGE4_OUT_DIR) / "property_clusters_with_dr.json")
    stage5_default = getattr(cfg, "STAGE5_OUT_DIR", "./stage5_out")

    p.add_argument("--stage4_path", default=stage4_default)
    p.add_argument("--stage2_pool", default=cfg.STAGE2_CANDIDATE_POOL)
    p.add_argument("--output_dir",  default=stage5_default)
    p.add_argument("--model",       default=cfg.MODEL_NAME)
    p.add_argument("--vllm_base_url", default=cfg.API_BASE)
    p.add_argument("--max_abstract", type=int, default=ABSTRACT_MAX)
    p.add_argument("--skip_abstract", action="store_true")
    p.add_argument("--edge_sim_threshold",   type=float, default=EDGE_SIM_THRESHOLD)
    p.add_argument("--prop_jaccard_threshold", type=float, default=PROP_JACCARD_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()

    global EDGE_SIM_THRESHOLD, PROP_JACCARD_THRESHOLD
    EDGE_SIM_THRESHOLD    = args.edge_sim_threshold
    PROP_JACCARD_THRESHOLD = args.prop_jaccard_threshold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 5.1  Load ──────────────────────────────────────────────────────────
    print("\n[5.1] Loading inputs ...")
    with open(args.stage4_path) as f:
        stage4_data = json.load(f)

    class_clusters = load_class_clusters(Path(args.stage4_path))
    pool           = load_stage2_pool(Path(args.stage2_pool))
    edge_meta      = build_edge_meta_index(pool)
    print(f"  Edge-label meta index: {len(edge_meta)} labels.")

    nodes: list[ClassNode] = []
    for c in class_clusters:
        top_members = [m["string"] for m in sorted(
            c.get("members", []), key=lambda x: x.get("frequency", 0), reverse=True
        )[:8]]
        nodes.append(ClassNode(
            owl_name=c.get("owl_name", ""),
            definition=c.get("definition", ""),
            members=top_members,
            graph_native_parent=c.get("graph_native_parent"),
        ))
    nodes = [n for n in nodes if n.owl_name]
    all_owl_names = [n.owl_name for n in nodes]
    print(f"  Total class nodes to place: {len(nodes)}")

    # ── 5.2  Resolve GNP ───────────────────────────────────────────────────
    print("\n[5.2] Resolving graph-native parents ...")
    member_index = build_member_string_index(class_clusters)
    resolve_all_gnp(nodes, member_index)

    # ── 5.3  Edge distribution signals ────────────────────────────────────
    print("\n[5.3] Computing edge-distribution profiles ...")
    source_profiles, range_profiles, edge_labels = build_edge_profiles(class_clusters, edge_meta)
    print(f"  Profiles built over {len(edge_labels)} edge labels.")

    # ── 5.4  Property overlap signal ──────────────────────────────────────
    print("\n[5.4] Computing property-overlap (Jaccard) signals ...")
    domain_prop_index = build_domain_property_index(stage4_data)
    print(f"  Domain property index: {len(domain_prop_index)} classes have assigned properties.")

    signal_bundles: dict[str, SignalBundle] = {}
    for node in nodes:
        src_top, rng_top = compute_edge_sim_signal(node.owl_name, source_profiles, range_profiles)
        prop_top = compute_prop_overlap_signal(node.owl_name, domain_prop_index, all_owl_names)
        signal_bundles[node.owl_name] = SignalBundle(
            owl_name=node.owl_name,
            resolved_gnp=node.resolved_gnp,
            source_sim_top=src_top,
            range_sim_top=rng_top,
            prop_overlap_top=prop_top,
        )

    # ── 5.5  Phase 1 — Abstract class discovery ───────────────────────────
    client = OpenAI(base_url=args.vllm_base_url, api_key="EMPTY")

    abstract_nodes: list[ClassNode] = []
    if not args.skip_abstract:
        print("\n[5.5] Phase 1 — Abstract class discovery ...")
        abstract_nodes = discover_abstract_classes(nodes, client, args.model, args.max_abstract)
    else:
        print("\n[5.5] Skipping abstract class discovery (--skip_abstract).")

    # ── 5.6  Topological pre-sort ──────────────────────────────────────────
    print("\n[5.6] Topological pre-sort ...")
    sorted_nodes = topological_presort(nodes, signal_bundles)
    n_high  = sum(1 for n in sorted_nodes if n.resolved_gnp)
    n_other = len(sorted_nodes) - n_high
    print(f"  Sort order: {n_high} high-signal first, {n_other} remaining.")

    # ── 5.7  Phase 2 — Per-class LLM placement ────────────────────────────
    print(f"\n[5.7] Phase 2 — LLM placement of {len(sorted_nodes)} classes ...")
    run_phase2_placement(sorted_nodes, abstract_nodes, signal_bundles, client, args.model)

    name_order = {n.owl_name: i for i, n in enumerate(nodes)}
    sorted_nodes.sort(key=lambda n: name_order.get(n.owl_name, 9999))

    all_nodes = sorted_nodes + abstract_nodes

    # Deduplicate: corpus class wins over abstract class on name collision
    seen_names: dict[str, ClassNode] = {}
    deduped_nodes = []
    for node in all_nodes:
        if node.owl_name not in seen_names:
            seen_names[node.owl_name] = node
            deduped_nodes.append(node)
        else:
            existing = seen_names[node.owl_name]
            # Corpus class (llm source) beats abstract class
            if existing.is_abstract and not node.is_abstract:
                # Replace abstract with corpus class
                idx = deduped_nodes.index(existing)
                deduped_nodes[idx] = node
                seen_names[node.owl_name] = node
                print(f"  DEDUP: corpus class '{node.owl_name}' replaced abstract placeholder.")
            else:
                print(f"  DEDUP: dropped duplicate '{node.owl_name}' (keeping first).")

    all_nodes = deduped_nodes

    # ── 5.8  Cycle resolution ──────────────────────────────────────────────
    print("\n[5.8] Cycle resolution ...")
    cycle_resolution(all_nodes, client, args.model)

    # ── Output ─────────────────────────────────────────────────────────────
    print("\nSerializing output ...")
    serialized = [serialize_node(n) for n in all_nodes]

    out_path = output_dir / "class_hierarchy.json"
    with open(out_path, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"  Written: {out_path}")

    flagged   = [s for s in serialized if s["flagged"]]
    flag_path = output_dir / "hierarchy_flagged.json"
    with open(flag_path, "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"  Written: {flag_path}  ({len(flagged)} flagged)")

    conf_dist    = Counter(n.placement_confidence for n in all_nodes)
    source_dist  = Counter(n.placement_source for n in all_nodes)
    multi_parent = sum(1 for n in all_nodes if len(n.parents) > 1)

    stats = {
        "total_classes":           len(all_nodes),
        "corpus_classes":          len(sorted_nodes),
        "abstract_classes":        len(abstract_nodes),
        "flagged_for_review":      len(flagged),
        "multi_parent_classes":    multi_parent,
        "confidence_distribution": dict(conf_dist),
        "placement_source":        dict(source_dist),
        "thresholds": {
            "edge_sim":     EDGE_SIM_THRESHOLD,
            "prop_jaccard": PROP_JACCARD_THRESHOLD,
        },
    }
    stats_path = output_dir / "stage5_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Written: {stats_path}")

    print("\n=== Stage 5 complete ===")
    print(f"  Total classes  : {stats['total_classes']} ")
    print(f"  Multi-parent   : {multi_parent}")
    print(f"  Flagged        : {len(flagged)}")
    for level in ("high", "medium", "low", "unresolved"):
        n = conf_dist.get(level, 0)
        if n:
            print(f"  {level:12s} : {n}")


if __name__ == "__main__":
    main()