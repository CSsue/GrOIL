"""
Stage 7 — TBox Compilation
===========================
Input:  named_concept_clusters.json     (Stage 3 output, for member strings and definitions)
        property_clusters_with_dr.json  (Stage 4 output, for annotated property clusters)
        class_hierarchy.json            (Stage 5 output, authoritative class list)
        induced_axioms.json             (Stage 6 output, OWL 2 TBox restriction axioms)
Output: tbox.ttl                        — OWL 2 TBox serialised as Turtle
        tbox.owl                        — RDF/XML serialisation (via rdflib, if installed)
        tbox_flagged.json               — classes, properties, and axioms below min_confidence
        tbox_stats.json

Pipeline:
  7.1  Load all four pipeline outputs; normalise to list/dict as required.
  7.2  Class list — Stage 5 hierarchy is authoritative; Stage 3 is used as
         fallback if Stage 5 is empty, and to back-fill missing definitions and
         member strings for Stage 5 nodes.
  7.3  Property list — collect owl:ObjectProperty and owl:DatatypeProperty
         clusters from Stage 4 property and axiom streams.
  7.4  Axiom list — collect all Stage 6 entries that carry an owl_construct.
  7.5  Emit Turtle: prefixes, ontology header, then classes, properties, and
         TBox restrictions in sorted order. Terms below min_confidence are
         withheld and written to tbox_flagged.json. Restriction axioms use
         inline anonymous blank-node syntax ([ a owl:Restriction ; ... ]) to
         avoid forward-reference issues in strict Turtle parsers.
  7.5a Emit the ontology root class (ins:LifeInsurancePolicy subClassOf
         owl:Thing) before all pipeline-derived classes. Any class whose
         parents list is empty or resolves only to owl:Thing is re-parented
         to ins:LifeInsurancePolicy, giving the ontology a single named root.
  7.6  Write tbox.ttl.
  7.7  Parse tbox.ttl with rdflib and re-serialise as RDF/XML (tbox.owl);
         skipped with an informational message if rdflib is not installed.
  7.8  Write tbox_stats.json and tbox_flagged.json.

Design principles:
  • Stage 5 is the single source of truth for the class hierarchy; its
    placement_confidence gates class inclusion under --min_confidence.
    Abstract (inferred_top_level) classes are always included regardless of
    confidence, because they are explicitly proposed by the pipeline.
  • Property domain/range triples are only emitted when the property's
    dr_confidence meets the min_confidence threshold; the property declaration
    itself (label, comment, type) is always emitted.
  • Disjointness axioms are emitted as direct owl:disjointWith triples on the
    subject class — no blank node is required.
  • All other restriction axioms (someValuesFrom, allValuesFrom, cardinality)
    are emitted as inline anonymous blank nodes to produce valid, parser-safe
    Turtle without forward references.
  • XSD datatype names may arrive with or without the "xsd:" prefix; bare
    local names (e.g. "string", "integer") are normalised before any
    class-namespace routing so they are never mistakenly prefixed as ins:string.
  • Already-prefixed names (e.g. "owl:Thing", "xsd:date") are never passed
    through _iri_safe, which would corrupt the colon into an underscore.
  • Human-readable labels prefer the most frequent member string (if it
    contains a space); otherwise PascalCase/camelCase splitting is applied.
  • Duplicate class and axiom names are deduplicated at emit time.

Usage:
    python stage7_serialize_tbox.py
    python stage7_serialize_tbox.py \\
        --stage3  ./stage3_out/named_concept_clusters.json \\
        --stage4  ./stage4_out/property_clusters_with_dr.json \\
        --stage5  ./stage5_out/class_hierarchy.json \\
        --stage6  ./stage6_out/induced_axioms.json \\
        --output_dir   ./stage7_out \\
        --base_iri     https://example.org/ontology/insurance# \\
        --prefix       ins \\
        --min_confidence medium
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from config import cfg
    _HAS_CFG = True
except ImportError:
    _HAS_CFG = False

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_IRI   = "https://example.org/ontology/insurance#"
DEFAULT_PREFIX     = "ins"
PIPELINE_VERSION   = "1.0"
OWL_THING          = "owl:Thing"
CONFIDENCE_RANK    = {"high": 2, "medium": 1, "low": 0}

# The single named root that sits between owl:Thing and every top-level class.
ONTOLOGY_ROOT_NAME  = "LifeInsurancePolicy"
ONTOLOGY_ROOT_LABEL = "Life Insurance Policy"
ONTOLOGY_ROOT_DEF   = (
    "The root concept of this ontology. Represents a life insurance policy "
    "as a whole; all domain classes are subsumed under this concept."
)

XSD_TYPES = {
    "xsd:string", "xsd:integer", "xsd:decimal", "xsd:float",
    "xsd:boolean", "xsd:date", "xsd:dateTime", "xsd:gYear",
    "xsd:duration", "xsd:anyURI", "rdfs:Literal",
}

_BARE_XSD_NAMES: dict[str, str] = {
    "string":   "xsd:string",
    "integer":  "xsd:integer",
    "int":      "xsd:integer",
    "decimal":  "xsd:decimal",
    "float":    "xsd:float",
    "double":   "xsd:float",
    "boolean":  "xsd:boolean",
    "bool":     "xsd:boolean",
    "date":     "xsd:date",
    "dateTime": "xsd:dateTime",
    "datetime": "xsd:dateTime",
    "gYear":    "xsd:gYear",
    "year":     "xsd:gYear",
    "duration": "xsd:duration",
    "anyURI":   "xsd:anyURI",
    "uri":      "xsd:anyURI",
    "url":      "xsd:anyURI",
    "Literal":  "rdfs:Literal",
    "literal":  "rdfs:Literal",
}

PREFIXES = """\
@prefix owl:     <http://www.w3.org/2002/07/owl#> .
@prefix rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix prov:    <http://www.w3.org/ns/prov#> .
@prefix skos:    <http://www.w3.org/2004/02/skos/core#> .
@prefix {prefix}: <{base_iri}> .
"""

ONTOLOGY_HEADER = """\
### =========================================================
###  Ontology header
### =========================================================

<{base_iri_no_hash}> a owl:Ontology ;
    rdfs:label "Life Insurance Policy TBox"@en ;
    dcterms:title "Life Insurance Policy TBox"@en ;
    dcterms:description "OWL TBox ontology induced from a corpus of life insurance policy documents via the Graph-Grounded Ontology Induction pipeline."@en ;
    dcterms:created "{date}"^^xsd:date ;
    owl:versionInfo "{version}"^^xsd:string ;
    dcterms:creator "Graph-Grounded Ontology Induction Pipeline (Stages 1-7)"^^xsd:string .

"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conf_ok(conf: str, min_conf: str) -> bool:
    return CONFIDENCE_RANK.get(conf, 0) >= CONFIDENCE_RANK.get(min_conf, 1)


def _ttl_str(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", "")
    return f'"{s}"'


def _ttl_lang(s: str, lang: str = "en") -> str:
    """Natural-language literal with a language tag.
    Use only for human-readable annotation properties:
    rdfs:label, rdfs:comment, skos:definition, skos:altLabel."""
    return f'{_ttl_str(s)}@{lang}'


def _human_label(owl_name: str, members: list[dict]) -> str:
    """
    Derive a human-readable rdfs:label.
    Prefers the most frequent member string if it contains a space;
    otherwise splits PascalCase/camelCase on word boundaries.
    """
    for m in members[:3]:
        s = (m.get("string") or "").strip()
        if s and " " in s:
            return s.title() if s == s.lower() else s
    label = re.sub(r'([a-z])([A-Z])', r'\1 \2', owl_name)
    label = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', label)
    return label


def _local(name: str, prefix: str) -> str:
    return f"{prefix}:{name}"


def _iri_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.]", "_", name)


def _is_xsd(name: str) -> bool:
    return name in XSD_TYPES or name.startswith("xsd:") or name.startswith("rdfs:")


def _normalise_xsd(name: str) -> str:
    """Attach 'xsd:' prefix to bare XSD local names (e.g. 'string' → 'xsd:string')."""
    return _BARE_XSD_NAMES.get(name, name)


def _safe_prefixed(name: str, prefix: str) -> str:
    """Return a prefixed Turtle name. Already-prefixed names (containing ':')
    are returned as-is without passing through _iri_safe."""
    if ":" in name:
        return name
    return _local(_iri_safe(name), prefix)


def _resolve_range(range_str: str, prefix: str) -> str:
    """Return the Turtle token for a range value.
    XSD/RDF types are returned as-is; everything else is prefixed."""
    if not range_str or range_str == OWL_THING:
        return "owl:Thing"
    range_str = _normalise_xsd(range_str)
    if _is_xsd(range_str):
        return range_str
    return _safe_prefixed(range_str, prefix)


def _resolve_class(name: str, prefix: str) -> str:
    if not name or name == OWL_THING:
        return "owl:Thing"
    return _safe_prefixed(name, prefix)


def _truncate(s: str, max_chars: int = 220) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def _close_block(lines: list[str]) -> None:
    """Find the last non-empty line in *lines*, strip any trailing ' ;' from
    it, and append ' .' to properly close the Turtle subject block."""
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            lines[i] = lines[i].rstrip().rstrip(";").rstrip() + " ."
            return


# ---------------------------------------------------------------------------
# Comment builders
# ---------------------------------------------------------------------------

def _build_class_comment(node: dict) -> str:
    definition = (node.get("definition") or "").strip().rstrip(".")
    conf        = node.get("placement_confidence", "")
    parents     = node.get("parents", [])
    gnp         = node.get("graph_native_parent") or node.get("resolved_gnp")
    member_cnt  = len(node.get("members", []))
    is_abstract = node.get("inferred_top_level", False)

    sent1 = definition if definition else f"OWL class {node.get('owl_name', '')}."
    sent1 = sent1.rstrip(".") + "."

    prov_parts: list[str] = []
    if is_abstract:
        prov_parts.append(
            "Abstract class inferred by the LLM from corpus structure "
            "(Stage 5, Phase 1); not directly attested in source documents"
        )
    else:
        if gnp:
            signal = (
                "graph-native parent (entity role co-occurrence)"
                if conf == "high" else "LLM-assigned parent"
            )
            prov_parts.append(f"Hierarchy placement via {signal}: subClassOf {gnp}")
        elif parents:
            prov_parts.append(
                f"Placed as subClassOf {', '.join(parents[:2])} "
                f"(Stage 5, confidence: {conf})"
            )
        if member_cnt:
            prov_parts.append(f"cluster of {member_cnt} surface variant(s) merged in Stage 3")

    if prov_parts:
        sent2 = "Induced from a " + "; ".join(prov_parts) + "."
    else:
        sent2 = (
            f"Placed under {', '.join(parents) if parents else OWL_THING} "
            f"with confidence {conf}."
        )

    return _truncate(f"{sent1} {sent2}")


def _build_property_comment(cluster: dict) -> str:
    definition = (cluster.get("definition") or "").strip().rstrip(".")
    reasoning  = (cluster.get("dr_reasoning") or "").strip().rstrip(".")
    signals    = cluster.get("dr_signals") or []
    conf       = cluster.get("dr_confidence", "")
    domain     = cluster.get("domain", "")
    range_     = cluster.get("range", "")
    member_cnt = len(cluster.get("members", []))

    sent1 = (
        definition if definition else f"Property {cluster.get('owl_name', '')}."
    ).rstrip(".") + "."

    sig_evidence: list[str] = []
    for sig in signals:
        if sig.get("signal_id") in ("S1", "S2"):
            sig_evidence.extend(sig.get("evidence", [])[:1])
    sig_text = sig_evidence[0] if sig_evidence else ""

    domain_range = ""
    if domain and domain != OWL_THING and range_ and range_ != OWL_THING:
        domain_range = f"domain={domain}, range={range_}"
    elif domain and domain != OWL_THING:
        domain_range = f"domain={domain}"
    elif range_ and range_ != OWL_THING:
        domain_range = f"range={range_}"

    prov_parts: list[str] = []
    if domain_range:
        prov_parts.append(domain_range)
    if reasoning:
        prov_parts.append(reasoning[:120])
    elif sig_text:
        prov_parts.append(sig_text[:120])
    if member_cnt:
        prov_parts.append(f"merged from {member_cnt} surface variant(s) in Stage 3")

    sent2 = (
        "Assigned " + "; ".join(prov_parts) + f" (Stage 4, confidence: {conf})."
        if prov_parts else ""
    )
    return _truncate(f"{sent1} {sent2}".strip())


def _build_axiom_comment(axiom: dict) -> str:
    definition = (axiom.get("definition") or "").strip().rstrip(".")
    reasoning  = (axiom.get("reasoning") or "").strip().rstrip(".")
    sentences  = axiom.get("evidence_sentences") or []
    conf       = axiom.get("confidence", "")
    source_cl  = axiom.get("source_cluster", "")

    sent1 = (definition if definition else axiom.get("axiom_str", "")).rstrip(".") + "."

    prov_parts: list[str] = []
    if reasoning:
        prov_parts.append(reasoning[:150])
    elif sentences:
        prov_parts.append(sentences[0][:150])
    if source_cl:
        prov_parts.append(f"source cluster: {source_cl}")
    prov_parts.append(f"confidence: {conf}")

    sent2 = (
        "Induced via Stage 6 restriction induction; " + "; ".join(prov_parts) + "."
        if prov_parts else ""
    )
    return _truncate(f"{sent1} {sent2}")


# ---------------------------------------------------------------------------
# Turtle emission helpers
# ---------------------------------------------------------------------------

def _emit_root_class(prefix: str) -> list[str]:
    """Emit the hand-crafted ontology root (ins:LifeInsurancePolicy subClassOf owl:Thing)."""
    local_name = _local(ONTOLOGY_ROOT_NAME, prefix)
    lines: list[str] = []
    lines.append(f"### Class: {local_name}  [ONTOLOGY ROOT]")
    lines.append(f"{local_name}")
    lines.append(f"    a owl:Class ;")
    lines.append(f"    rdfs:label {_ttl_lang(ONTOLOGY_ROOT_LABEL)} ;")
    lines.append(
        f"    rdfs:comment {_ttl_lang(ONTOLOGY_ROOT_DEF)} ;"
    )
    lines.append(
        f"    skos:definition {_ttl_lang(ONTOLOGY_ROOT_DEF)} ;"
    )
    lines.append(f"    rdfs:subClassOf owl:Thing ;")
    _close_block(lines)
    lines.append("")
    return lines


def _emit_class(node: dict, prefix: str, min_conf: str) -> list[str]:
    """Return a list of Turtle lines for one OWL class."""
    name = node.get("owl_name", "").strip()
    if not name:
        return []

    safe_name = _iri_safe(name)
    if not safe_name:
        return []

    conf = node.get("placement_confidence", "medium")
    if not _conf_ok(conf, min_conf) and not node.get("inferred_top_level", False):
        return []

    local_name  = _resolve_class(name, prefix)
    parents     = node.get("parents") or []
    definition  = (node.get("definition") or "").strip()
    comment     = _build_class_comment(node)
    members     = node.get("members") or []
    label       = _human_label(name, members)
    member_strs = [m.get("string", "") for m in members[:5] if m.get("string")]

    lines: list[str] = []
    lines.append(f"### Class: {local_name}")
    lines.append(f"{local_name}")
    lines.append(f"    a owl:Class ;")
    lines.append(f"    rdfs:label {_ttl_lang(label)} ;")
    if comment:
        lines.append(f"    rdfs:comment {_ttl_lang(comment)} ;")
    if definition:
        lines.append(f"    skos:definition {_ttl_lang(definition)} ;")

    # Resolve parent list; classes whose only parent would be owl:Thing are
    # instead placed under the ontology root (ins:LifeInsurancePolicy) so the
    # hierarchy has a single named entry point beneath owl:Thing.
    root_ref = _local(ONTOLOGY_ROOT_NAME, prefix)

    resolved_parents: list[str] = []
    for p in parents:
        rp = _resolve_class(p, prefix)
        if rp not in resolved_parents:
            resolved_parents.append(rp)

    # Drop bare owl:Thing entries — we'll replace them with the ontology root.
    resolved_parents = [rp for rp in resolved_parents if rp != "owl:Thing"]

    # If this class IS the root itself, parent stays owl:Thing.
    # Otherwise, anything left parentless points to the root.
    if name == ONTOLOGY_ROOT_NAME:
        resolved_parents = ["owl:Thing"]
    elif not resolved_parents:
        resolved_parents = [root_ref]

    for rp in resolved_parents:
        lines.append(f"    rdfs:subClassOf {rp} ;")

    for ms in member_strs[:3]:
        clean = ms.strip()
        if clean and clean.lower() != name.lower() and clean.lower() != label.lower():
            lines.append(f"    skos:altLabel {_ttl_lang(clean)} ;")

    _close_block(lines)
    lines.append("")
    return lines


def _emit_property(cluster: dict, prefix: str, min_conf: str) -> list[str]:
    """Return Turtle lines for one OWL object/datatype property."""
    name = cluster.get("owl_name", "").strip()
    if not name:
        return []

    safe_name = _iri_safe(name) if ":" not in name else name.split(":", 1)[1]
    if not safe_name:
        return []

    conf = cluster.get("dr_confidence", "medium")
    owl_type = cluster.get("owl_type", "owl:ObjectProperty")
    if owl_type not in ("owl:ObjectProperty", "owl:DatatypeProperty"):
        owl_type = "owl:ObjectProperty"

    local_name  = _safe_prefixed(name, prefix)
    members     = cluster.get("members") or []
    label       = _human_label(name, members)
    definition  = (cluster.get("definition") or "").strip()
    comment     = _build_property_comment(cluster)
    domain      = cluster.get("domain", "")
    range_      = cluster.get("range", "")
    member_strs = [m.get("string", "") for m in members[:5] if m.get("string")]

    lines: list[str] = []
    lines.append(f"### Property: {local_name}")
    lines.append(f"{local_name}")
    lines.append(f"    a {owl_type} ;")
    lines.append(f"    rdfs:label {_ttl_lang(label)} ;")
    if comment:
        lines.append(f"    rdfs:comment {_ttl_lang(comment)} ;")
    if definition:
        lines.append(f"    skos:definition {_ttl_lang(definition)} ;")

    if _conf_ok(conf, min_conf):
        if domain and domain != OWL_THING:
            lines.append(f"    rdfs:domain {_resolve_class(domain, prefix)} ;")
        if range_ and range_ != OWL_THING:
            lines.append(f"    rdfs:range {_resolve_range(range_, prefix)} ;")

    for ms in member_strs[:3]:
        clean = ms.strip()
        if clean and clean.lower() != name.lower() and clean.lower() != label.lower():
            lines.append(f"    skos:altLabel {_ttl_lang(clean)} ;")

    _close_block(lines)
    lines.append("")
    return lines


def _emit_axiom(axiom: dict, prefix: str, min_conf: str) -> list[str]:
    """Return Turtle lines for one TBox restriction axiom (Stage 6).
    Restriction axioms use inline anonymous blank-node syntax to avoid
    forward-reference issues in strict Turtle parsers such as rdflib."""
    conf = axiom.get("confidence", "medium")
    if not _conf_ok(conf, min_conf):
        return []

    construct    = axiom.get("owl_construct", "")
    subject_name = axiom.get("subject_class", "")
    prop_name    = axiom.get("property_name", "")
    filler_name  = axiom.get("filler", "")
    cardinality  = axiom.get("cardinality")
    axiom_str    = axiom.get("axiom_str", "")
    comment      = _build_axiom_comment(axiom)

    if not subject_name or not construct:
        return []

    subject_ref = _resolve_class(subject_name, prefix)

    lines: list[str] = []
    lines.append(f"### Axiom: {axiom.get('axiom_id', '')}  [{construct}]")
    if axiom_str:
        lines.append(f"# {axiom_str}")

    # ── Disjointness — no blank node needed ──────────────────────────────
    if construct == "owl:disjointWith":
        if not filler_name:
            return []
        filler_ref = _resolve_class(filler_name, prefix)
        lines.append(f"{subject_ref}")
        lines.append(f"    owl:disjointWith {filler_ref} ;")
        if comment:
            lines.append(f"    rdfs:comment {_ttl_lang(comment)} ;")
        _close_block(lines)
        lines.append("")
        return lines

    # ── All other constructs — inline anonymous blank-node restriction ────
    if not prop_name:
        return []
    prop_ref = _safe_prefixed(prop_name, prefix)

    restriction_inner: list[str] = []
    restriction_inner.append(f"        a owl:Restriction ;")
    restriction_inner.append(f"        owl:onProperty {prop_ref} ;")

    if construct in ("owl:minCardinality", "owl:maxCardinality", "owl:exactCardinality"):
        card_val = int(cardinality) if cardinality is not None else 1
        restriction_inner.append(
            f'        {construct} "{card_val}"^^xsd:nonNegativeInteger'
        )
    else:
        filler_ref = _resolve_range(filler_name, prefix) if filler_name else "owl:Thing"
        restriction_inner.append(f"        {construct} {filler_ref}")

    lines.append(f"{subject_ref}")
    lines.append(f"    rdfs:subClassOf [")
    lines.extend(restriction_inner)
    if comment:
        lines.append(f"    ] ;")
        lines.append(f"    rdfs:comment {_ttl_lang(comment)} ;")
    else:
        lines.append(f"    ] ;")
    _close_block(lines)
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> dict | list:
    if not path.exists():
        print(f"  [WARNING] {label} not found at {path} — skipping.", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main compilation logic
# ---------------------------------------------------------------------------

def compile_tbox(
    stage3_path:  Path,
    stage4_path:  Path,
    stage5_path:  Path,
    stage6_path:  Path,
    output_dir:   Path,
    base_iri:     str,
    prefix:       str,
    min_conf:     str,
    ontology_iri: Optional[str],
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("\n[7.1] Loading pipeline outputs ...")
    stage3_data: dict = _load_json(stage3_path, "Stage 3")   # type: ignore[assignment]
    stage4_data: dict = _load_json(stage4_path, "Stage 4")   # type: ignore[assignment]
    stage5_data: list = _load_json(stage5_path, "Stage 5")   # type: ignore[assignment]
    stage6_data: list = _load_json(stage6_path, "Stage 6")   # type: ignore[assignment]

    if not isinstance(stage3_data, dict):
        stage3_data = {}
    if not isinstance(stage4_data, dict):
        stage4_data = {}
    if not isinstance(stage5_data, list):
        stage5_data = []
    if not isinstance(stage6_data, list):
        stage6_data = []

    # ── 7.2  Class list — Stage 5 is authoritative ──────────────────────────
    print("[7.2] Collecting classes from Stage 5 hierarchy ...")
    class_nodes: list[dict] = [n for n in stage5_data if n.get("owl_name")]

    if not class_nodes:
        print("  [WARNING] Stage 5 empty — falling back to Stage 3 class stream.")
        class_nodes = [
            c for c in stage3_data.get("class", [])
            if not c.get("is_noise") and c.get("owl_name")
        ]

    stage3_class_by_name: dict[str, dict] = {
        c["owl_name"]: c
        for c in stage3_data.get("class", [])
        if c.get("owl_name")
    }
    for node in class_nodes:
        if not node.get("definition"):
            s3 = stage3_class_by_name.get(node["owl_name"])
            if s3:
                node["definition"] = s3.get("definition", "")
        if not node.get("members"):
            s3 = stage3_class_by_name.get(node["owl_name"])
            if s3:
                node["members"] = s3.get("members", [])

    # ── 7.3  Property list — Stage 4 annotated clusters ────────────────────
    print("[7.3] Collecting properties from Stage 4 ...")
    prop_clusters: list[dict] = []
    for stream in ("property", "axiom"):
        for c in stage4_data.get(stream, []):
            if not c.get("is_noise") and c.get("owl_name"):
                ot = c.get("owl_type", "")
                if ot in ("owl:ObjectProperty", "owl:DatatypeProperty"):
                    prop_clusters.append(c)

    # ── 7.4  Axioms — Stage 6 ────────────────────────────────────────────
    print("[7.4] Collecting axioms from Stage 6 ...")
    axioms: list[dict] = [a for a in stage6_data if a.get("owl_construct")]

    print(
        f"  Classes: {len(class_nodes)} | "
        f"Properties: {len(prop_clusters)} | "
        f"Axioms: {len(axioms)}"
    )

    # ── 7.5  Emit Turtle ────────────────────────────────────────────────────
    print("[7.5] Generating Turtle ...")
    lines: list[str] = []

    prefix_block = PREFIXES.format(prefix=prefix, base_iri=base_iri)
    lines.append(prefix_block)

    ont_iri = ontology_iri or base_iri.rstrip("#/")
    lines.append(
        ONTOLOGY_HEADER.format(
            base_iri_no_hash=ont_iri,
            date=today,
            version=PIPELINE_VERSION,
        )
    )

    # ---- Classes ----
    flagged_classes: list[dict] = []
    lines.append("### =========================================================")
    lines.append("###  Classes")
    lines.append("### =========================================================")
    lines.append("")

    # ── 7.5a  Emit the ontology root ────────────────────────────────────────
    #   ins:LifeInsurancePolicy is emitted first so every subsequent class that
    #   would otherwise dangle from owl:Thing can reference it as its parent.
    lines.extend(_emit_root_class(prefix))
    emitted_class_names: set[str] = {ONTOLOGY_ROOT_NAME}

    for node in sorted(class_nodes, key=lambda n: n.get("owl_name", "")):
        name = node.get("owl_name", "").strip()
        if not name or name in emitted_class_names:
            continue
        conf = node.get("placement_confidence", "medium")
        if not _conf_ok(conf, min_conf) and not node.get("inferred_top_level", False):
            flagged_classes.append({
                "owl_name":   name,
                "confidence": conf,
                "reason":     "low placement confidence",
                "definition": node.get("definition", ""),
            })
            continue
        cls_lines = _emit_class(node, prefix, min_conf)
        if cls_lines:
            lines.extend(cls_lines)
            emitted_class_names.add(name)

    # ---- Properties ----
    flagged_props: list[dict] = []
    lines.append("### =========================================================")
    lines.append("###  Properties")
    lines.append("### =========================================================")
    lines.append("")

    emitted_prop_names: set[str] = set()
    for cluster in sorted(prop_clusters, key=lambda c: c.get("owl_name", "")):
        name = cluster.get("owl_name", "").strip()
        if not name or name in emitted_prop_names:
            continue
        conf = cluster.get("dr_confidence", "medium")
        prop_lines = _emit_property(cluster, prefix, min_conf)
        if prop_lines:
            lines.extend(prop_lines)
            emitted_prop_names.add(name)
        elif not _conf_ok(conf, min_conf):
            flagged_props.append({
                "owl_name":   name,
                "confidence": conf,
                "reason":     "low DR confidence",
                "definition": cluster.get("definition", ""),
            })

    # ---- Restrictions ----
    flagged_axioms: list[dict] = []
    lines.append("### =========================================================")
    lines.append("###  TBox Restrictions & Disjointness Axioms")
    lines.append("### =========================================================")
    lines.append("")

    emitted_axiom_strs: set[str] = set()
    axioms_emitted = 0

    for axiom in axioms:
        conf = axiom.get("confidence", "medium")
        astr = axiom.get("axiom_str", "")
        if not _conf_ok(conf, min_conf):
            flagged_axioms.append({
                "axiom_id":   axiom.get("axiom_id"),
                "axiom_str":  astr,
                "confidence": conf,
                "reason":     "low confidence",
            })
            continue
        if astr and astr in emitted_axiom_strs:
            continue
        ax_lines = _emit_axiom(axiom, prefix, min_conf)
        if ax_lines:
            lines.extend(ax_lines)
            axioms_emitted += 1
            if astr:
                emitted_axiom_strs.add(astr)

    # ── 7.6  Write Turtle ────────────────────────────────────────────────────
    ttl_path = output_dir / "tbox.ttl"
    with open(ttl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Written: {ttl_path}")

    # ── 7.7  Write RDF/XML (using rdflib if available, else skip) ───────────
    owl_path = output_dir / "tbox.owl"
    try:
        import rdflib
        g = rdflib.Graph()
        g.parse(str(ttl_path), format="turtle")
        g.serialize(destination=str(owl_path), format="xml")
        print(f"  Written: {owl_path}  (RDF/XML via rdflib)")
    except ImportError:
        print(
            "  [INFO] rdflib not installed — skipping RDF/XML serialisation. "
            "Install with: pip install rdflib"
        )
    except Exception as e:
        print(f"  [WARNING] RDF/XML serialisation failed: {e}")

    # ── 7.8  Stats & flagged ────────────────────────────────────────────────
    stats = {
        "date":               today,
        "min_confidence":     min_conf,
        "base_iri":           base_iri,
        "classes_emitted":    len(emitted_class_names),
        "classes_flagged":    len(flagged_classes),
        "properties_emitted": len(emitted_prop_names),
        "properties_flagged": len(flagged_props),
        "axioms_emitted":     axioms_emitted,
        "axioms_flagged":     len(flagged_axioms),
        "provenance": {
            "stage3_clusters_class":    len(stage3_data.get("class", [])),
            "stage3_clusters_property": len(stage3_data.get("property", [])),
            "stage3_clusters_axiom":    len(stage3_data.get("axiom", [])),
            "stage4_annotated_props":   len(prop_clusters),
            "stage5_class_nodes":       len(class_nodes),
            "stage6_axioms_raw":        len(axioms),
        },
    }
    stats_path = output_dir / "tbox_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Written: {stats_path}")

    all_flagged = {
        "flagged_classes":    flagged_classes,
        "flagged_properties": flagged_props,
        "flagged_axioms":     flagged_axioms,
    }
    flagged_path = output_dir / "tbox_flagged.json"
    with open(flagged_path, "w") as f:
        json.dump(all_flagged, f, indent=2)
    print(f"  Written: {flagged_path}")

    _print_summary(stats)


def _print_summary(stats: dict) -> None:
    print("\n" + "=" * 60)
    print("  Stage 7 — TBox Compilation Complete")
    print("=" * 60)
    print(f"  Base IRI         : {stats['base_iri']}")
    print(f"  Min confidence   : {stats['min_confidence']}")
    print(f"  Classes emitted  : {stats['classes_emitted']}  (flagged: {stats['classes_flagged']})")
    print(f"  Props emitted    : {stats['properties_emitted']}  (flagged: {stats['properties_flagged']})")
    print(f"  Axioms emitted   : {stats['axioms_emitted']}  (flagged: {stats['axioms_flagged']})")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cfg_attr(attr: str, default: str) -> str:
    if _HAS_CFG:
        return getattr(cfg, attr, default)
    return default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 7 — TBox Compilation to OWL Turtle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    p.add_argument(
        "--stage3",
        default=str(
            Path(_cfg_attr("STAGE3_OUT_DIR", "./stage3_out")) / "named_concept_clusters.json"
        ),
        help="Path to Stage 3 named_concept_clusters.json",
    )
    p.add_argument(
        "--stage4",
        default=str(
            Path(_cfg_attr("STAGE4_OUT_DIR", "./stage4_out")) / "property_clusters_with_dr.json"
        ),
        help="Path to Stage 4 property_clusters_with_dr.json",
    )
    p.add_argument(
        "--stage5",
        default=str(
            Path(_cfg_attr("STAGE5_OUT_DIR", "./stage5_out")) / "class_hierarchy.json"
        ),
        help="Path to Stage 5 class_hierarchy.json",
    )
    p.add_argument(
        "--stage6",
        default=str(
            Path(_cfg_attr("STAGE6_OUT_DIR", "./stage6_out")) / "induced_axioms.json"
        ),
        help="Path to Stage 6 induced_axioms.json",
    )
    p.add_argument(
        "--output_dir",
        default=_cfg_attr("STAGE7_OUT_DIR", "./stage7_out"),
        help="Output directory (default: STAGE7_OUT_DIR from .env or ./stage7_out)",
    )
    p.add_argument(
        "--base_iri",
        default=DEFAULT_BASE_IRI,
        help=f"Base ontology IRI (default: {DEFAULT_BASE_IRI})",
    )
    p.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"Turtle prefix for the ontology namespace (default: {DEFAULT_PREFIX})",
    )
    p.add_argument(
        "--min_confidence",
        choices=["high", "medium", "low"],
        default="medium",
        help="Minimum pipeline confidence to include a term in the TBox (default: medium)",
    )
    p.add_argument(
        "--ontology_iri",
        default=None,
        help="Override ontology IRI in the OWL header (default: derived from --base_iri)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    compile_tbox(
        stage3_path  = Path(args.stage3),
        stage4_path  = Path(args.stage4),
        stage5_path  = Path(args.stage5),
        stage6_path  = Path(args.stage6),
        output_dir   = Path(args.output_dir),
        base_iri     = args.base_iri,
        prefix       = args.prefix,
        min_conf     = args.min_confidence,
        ontology_iri = args.ontology_iri,
    )


if __name__ == "__main__":
    main()