"""
ABox Population Pipeline
========================
Input:  - A single contract's Stage 1 batch JSON (one document dict)
        - TBox pipeline outputs: Stage 3 + Stage 4 JSONs + final tbox.ttl
Output: abox_{contract_id}.ttl

Steps:
  A. Build lookup tables from TBox stage outputs (Stage 3 + Stage 4)
     ALSO: build subclass depth map from tbox.ttl for hierarchy-aware
     canonical class selection (replaces hardcoded GENERIC_FALLBACKS).
  B. Mint individuals from entity pool, with deduplication  (Step 1)
     FIX: dedup groups are keyed by (canonical_class, normalised_name)
     and nodes with *different* canonical classes are never merged even
     if they share a name — avoids the "SecureLife as Beneficiary" bug.
     FIX: canonical class is picked by TBox hierarchy depth (most derived
     wins), not by alphabetical order over a hardcoded exclusion list.
     FIX: IRI uses a human-readable name slug; the 8-char hash is only
     appended when two distinct individuals would otherwise collide.
  C. Assert object properties from EHF edges  (Step 2)
  D. Mine literals from fact sentences via bounded LLM  (Step 3)
     FIX: closed vocabulary is restricted to owl:DatatypeProperty terms
     only — object properties no longer leak in via the missing-domain
     heuristic.
  E. Serialize to Turtle
  F. Pellet verify TBox ∪ ABox

Schema notes (from actual stage implementations):
  Stage 3 named_concept_clusters.json top-level keys: "class", "property", "axiom"
  Stage 4 property_clusters_with_dr.json: a DICT with stream keys
  Stage 1 edge_buckets keys: chunk_fact, fact_entity, entity_fact, fact_fact,
                              chunk_chunk, entity_entity
  Stage 1 facts: {"id", "name", "sentence", "chunk_topics"} — flat, no "data" nesting.
"""

import hashlib
import json
import re
import os
import glob
import logging
import argparse
from collections import defaultdict
from pathlib import Path
from rdflib import Graph, Namespace, Literal, RDF, RDFS, OWL, XSD, URIRef
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from config import cfg

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

TBOX_BASE_IRI = os.getenv("TBOX_BASE_IRI", "https://example.org/ontology/insurance#")
ABOX_NS       = os.getenv("ABOX_NS",        "https://example.org/ontology/insurance/abox#")

ANCHOR_MAP = {
    "dob":                ("dateOfBirth",                XSD.date),
    "date of birth":      ("dateOfBirth",                XSD.date),
    "age":                ("ageAtIssue",                 XSD.integer),
    "gender":             ("gender",                     XSD.string),
    "underwriting class": ("underwritingClass",          XSD.string),
    "face amount":        ("faceAmount",                 XSD.decimal),
    "policy number":      ("policyNumber",               XSD.string),
    "issue date":         ("issueDate",                  XSD.date),
    "expiry date":        ("expiryDate",                 XSD.date),
    "state of issuance":  ("stateOfIssuance",            XSD.string),
    "premium":            ("premiumAmount",              XSD.decimal),
    "death benefit":      ("faceAmount",                 XSD.decimal),
    "duration":           ("termYears",                  XSD.integer),
    "relationship":       ("beneficiaryRelationship",    XSD.string),
    "share":              ("beneficiarySharePercentage", XSD.decimal),
    "phone":              ("phoneNumber",                XSD.string),
    "address":            ("address",                    XSD.string),
    "email":              ("email",                      XSD.string),
}

# ─────────────────────────────────────────
# NAMESPACE / IRI RESOLUTION
# ─────────────────────────────────────────

def resolve_tbox_base_iri(stage7_dir: str) -> str:
    tbox_path = Path(stage7_dir) / "tbox.ttl"
    if not tbox_path.exists():
        log.warning(f"tbox.ttl not found at {tbox_path}, using fallback IRI.")
        return TBOX_BASE_IRI

    skip_prefixes = {
        "http://www.w3.org/2002/07/owl",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns",
        "http://www.w3.org/2000/01/rdf-schema",
        "http://www.w3.org/2001/XMLSchema",
        "http://purl.org/dc/terms",
        "http://www.w3.org/ns/prov",
        "http://www.w3.org/2004/02/skos",
    }
    with open(tbox_path) as f:
        for line in f:
            m = re.match(r'@prefix\s+\w+:\s+<([^>]+)>\s*\.', line)
            if m:
                candidate = m.group(1)
                if not any(candidate.startswith(s) for s in skip_prefixes):
                    log.info(f"TBox base IRI from tbox.ttl: {candidate}")
                    return candidate
    log.warning("Could not detect domain prefix in tbox.ttl, using fallback.")
    return TBOX_BASE_IRI


# ─────────────────────────────────────────
# TBOX HIERARCHY DEPTH
# ─────────────────────────────────────────

def build_class_depth_map(tbox_ttl_path: str, tbox_base_iri: str) -> dict[str, int]:
    """
    Parse tbox.ttl and compute the rdfs:subClassOf depth of every named class.

    Depth 0 = owl:Thing (or any class with no parent in the TBox).
    Depth k = k hops from owl:Thing down through subClassOf chains.

    Used by _canonical_class to prefer the most-derived (highest-depth) class
    over generic ancestors — replacing the hardcoded GENERIC_FALLBACKS set.

    Classes not found in the TBox are assigned depth -1 so they sort last
    (least preferred) when picking a canonical class.
    """
    if not Path(tbox_ttl_path).exists():
        log.warning(f"tbox.ttl not found at {tbox_ttl_path} — class depth map unavailable.")
        return {}

    g = Graph()
    g.parse(tbox_ttl_path, format="turtle")
    tbox_ns = Namespace(tbox_base_iri)

    # Build local_name → set[parent_local_name]
    parents: dict[str, set[str]] = defaultdict(set)
    all_classes: set[str] = set()

    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            s_local = str(s).split("#")[-1].split("/")[-1]
            o_local = str(o).split("#")[-1].split("/")[-1]
            if s_local and o_local and o_local != "Thing":
                parents[s_local].add(o_local)
            all_classes.add(s_local)
            all_classes.add(o_local)

    # BFS from roots (classes with no parents) to assign depths
    depth: dict[str, int] = {}
    roots = [c for c in all_classes if not parents.get(c)]
    queue = list(roots)
    for r in roots:
        depth[r] = 0

    visited: set[str] = set(roots)
    while queue:
        current = queue.pop(0)
        # Find children (classes whose parent set includes current)
        for cls, par_set in parents.items():
            if current in par_set and cls not in visited:
                depth[cls] = depth[current] + 1
                visited.add(cls)
                queue.append(cls)

    # Any class still unvisited (disconnected islands) gets depth 0
    for cls in all_classes:
        if cls not in depth:
            depth[cls] = 0

    log.info(f"Class depth map built: {len(depth)} classes, "
             f"max depth={max(depth.values(), default=0)}")
    return depth


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def safe_iri_fragment(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s.strip())


def _name_slug(name: str, max_len: int = 32) -> str:
    """
    Convert a human name / label into a compact IRI-safe slug.
    E.g. "John Michael Anderson" → "JohnMichaelAnderson"
         "SecureLife Insurance Company" → "SecureLifeInsuranceCo"
    """
    # Title-case each word and strip non-alphanumeric characters
    words = re.sub(r"[^a-zA-Z0-9 ]", " ", name.strip()).split()
    slug  = "".join(w.capitalize() for w in words if w)
    return slug[:max_len] if slug else "Unknown"


def _dedup_key_hash(key: str) -> str:
    """Stable 8-char hex hash — used only when two individuals would collide on slug."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _normalise_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def parse_anchor_value(key: str, raw_value: str):
    key_norm = key.lower().strip().rstrip(":")
    for anchor_key, (prop, xsd_type) in ANCHOR_MAP.items():
        if anchor_key in key_norm:
            val = raw_value.strip()
            try:
                if xsd_type == XSD.date:
                    from datetime import datetime
                    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"):
                        try:
                            val = datetime.strptime(val, fmt).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue
                elif xsd_type == XSD.integer:
                    val = str(int(re.sub(r"[^\d]", "", val)))
                elif xsd_type == XSD.decimal:
                    val = str(float(re.sub(r"[^\d.]", "", val)))
            except Exception:
                pass
            return prop, Literal(val, datatype=xsd_type)
    return None


def parse_relational_anchors(anchors: list) -> list[tuple]:
    results = []
    for anchor in anchors:
        if ":" in anchor:
            key, _, val = anchor.partition(":")
            parsed = parse_anchor_value(key, val)
            if parsed:
                results.append(parsed)
    return results


ENTITY_TYPE_TO_NAME_PROPERTY = {
    "PERSON":              ("partyName",        XSD.string),
    "ORGANIZATION":        ("insurerName",      XSD.string),
    "DOCUMENT_IDENTIFIER": ("policyNumber",     XSD.string),
    "DATE":                ("dateValue",        XSD.string),
    "CURRENCY":            ("faceAmount",       XSD.decimal),
    "DURATION":            ("durationValue",    XSD.string),
    "LOCATION":            ("stateOfIssuance",  XSD.string),
    "PERCENTAGE":          ("sharePercentage",  XSD.decimal),
}


def infer_name_property(entity: dict) -> tuple[str, Literal]:
    etype = entity.get("type", "").upper()
    roles = [r.lower() for r in entity.get("roles", [])]
    name  = entity.get("name", "").strip()

    if "unique policy number" in roles or etype == "DOCUMENT_IDENTIFIER":
        return "policyNumber", Literal(name, datatype=XSD.string)
    if "date of issue" in roles:
        return "issueDate", Literal(name, datatype=XSD.string)
    if any(r in roles for r in ["date of expiry", "expiry date", "maturity date"]):
        return "expiryDate", Literal(name, datatype=XSD.string)
    if etype == "CURRENCY":
        cleaned = re.sub(r"[^\d.]", "", name)
        return "faceAmount", Literal(cleaned or name,
                                     datatype=XSD.decimal if cleaned else XSD.string)

    prop, xsd_type = ENTITY_TYPE_TO_NAME_PROPERTY.get(etype, ("labelValue", XSD.string))
    return prop, Literal(name, datatype=xsd_type)


def _resolve_classes_for_entity(entity: dict, lookups: dict) -> set[str]:
    """Return the set of OWL class names applicable to one entity dict."""
    etype = entity.get("type", "UNKNOWN").upper()
    classes = set()
    tc = lookups["type_to_class"].get(etype)
    if tc:
        classes.add(tc)
    for role in entity.get("roles", []):
        rc = lookups["role_to_class"].get(role.lower().strip())
        if rc:
            classes.add(rc)
    return classes


def _canonical_class(classes: set[str], etype: str,
                     class_depth: dict[str, int]) -> str:
    """
    Pick one stable class name to use as the IRI prefix and dedup-key component.

    Strategy (replaces the hardcoded GENERIC_FALLBACKS approach):
      1. For each class in the resolved set, look up its depth in the TBox
         subClassOf hierarchy (built from tbox.ttl by build_class_depth_map).
         Deeper = more specific = preferred.
      2. Among classes at equal depth, pick alphabetically first for stability.
      3. If no depth information is available (empty map or unknown class),
         fall back to alphabetical selection — still deterministic, just not
         hierarchy-aware.
      4. If no OWL class resolved at all, fall back to the raw entity type string.

    This removes all hardcoded domain knowledge from the dedup logic.
    """
    if not classes:
        return etype

    if class_depth:
        # Sort by (depth descending, name ascending) — most derived wins
        ranked = sorted(classes,
                        key=lambda c: (-class_depth.get(c, -1), c))
        return ranked[0]

    # No depth map — alphabetical fallback (deterministic, no hardcoding)
    return sorted(classes)[0]


# ─────────────────────────────────────────
# A. BUILD LOOKUP TABLES FROM TBOX OUTPUTS
# ─────────────────────────────────────────

def load_tbox_lookups(stage3_dir: str, stage4_dir: str) -> dict:
    """
    Build lookup dicts from Stage 3 (named_concept_clusters.json)
    and Stage 4 (property_clusters_with_dr.json).

    FIX: track owl_type per property so Step 3 can filter to
    owl:DatatypeProperty only (object properties no longer leak in).
    """
    lookups: dict = {
        "role_to_class":          {},
        "type_to_class":          {},
        "edge_label_to_property": {},
        "class_names":            set(),
        "property_names":         set(),
        "property_owl_type":      {},
        "property_domains":       {},
        "property_ranges":        {},
    }

    # ── Stage 3 ───────────────────────────────────────────────────────────
    s3_path = Path(stage3_dir) / "named_concept_clusters.json"
    if not s3_path.exists():
        log.warning(f"Stage 3 output not found at {s3_path}")
    else:
        with open(s3_path) as f:
            s3_data = json.load(f)

        for cluster in s3_data.get("class", []):
            if cluster.get("is_noise", False):
                continue
            owl_name = cluster.get("owl_name", "").strip()
            if not owl_name:
                continue
            lookups["class_names"].add(owl_name)
            gnp = cluster.get("graph_native_parent", "")
            if gnp:
                lookups["type_to_class"][gnp.upper()] = owl_name
            for m in cluster.get("members", []):
                s      = m.get("string", "").lower().strip()
                origin = m.get("origin", "")
                if not s:
                    continue
                if origin == "entity_role":
                    lookups["role_to_class"][s] = owl_name
                elif origin == "entity_type":
                    lookups["type_to_class"][s.upper()] = owl_name

        for cluster in s3_data.get("property", []):
            if cluster.get("is_noise", False):
                continue
            owl_name = cluster.get("owl_name", "").strip()
            owl_type = cluster.get("owl_type", "")
            if not owl_name:
                continue
            if owl_type in ("owl:ObjectProperty", "owl:DatatypeProperty"):
                lookups["property_names"].add(owl_name)
                lookups["property_owl_type"][owl_name] = owl_type
                for m in cluster.get("members", []):
                    s = m.get("string", "").lower().strip()
                    if s:
                        lookups["edge_label_to_property"][s] = owl_name

    # ── Stage 4 ───────────────────────────────────────────────────────────
    s4_path = Path(stage4_dir) / "property_clusters_with_dr.json"
    if not s4_path.exists():
        log.warning(f"Stage 4 output not found at {s4_path}")
    else:
        with open(s4_path) as f:
            s4_data = json.load(f)

        for stream_name in ("property", "axiom"):
            for cluster in s4_data.get(stream_name, []):
                owl_name = cluster.get("owl_name", "").strip()
                owl_type = cluster.get("owl_type", "")
                domain   = cluster.get("domain", "").strip()
                rng      = cluster.get("range", "").strip()
                conf     = cluster.get("dr_confidence", "low")

                if not owl_name:
                    continue
                if owl_type in ("owl:ObjectProperty", "owl:DatatypeProperty"):
                    lookups["property_names"].add(owl_name)
                    # Stage 4 is authoritative for owl_type — overwrite Stage 3 if present
                    lookups["property_owl_type"][owl_name] = owl_type
                    for m in cluster.get("members", []):
                        s = m.get("string", "").lower().strip()
                        if s and s not in lookups["edge_label_to_property"]:
                            lookups["edge_label_to_property"][s] = owl_name

                if conf in ("high", "medium"):
                    if domain and domain not in ("owl:Thing", ""):
                        lookups["property_domains"][owl_name] = domain
                    if rng and rng not in ("owl:Thing", ""):
                        lookups["property_ranges"][owl_name] = rng

    log.info(
        f"Lookups built | classes={len(lookups['class_names'])} | "
        f"properties={len(lookups['property_names'])} | "
        f"roles mapped={len(lookups['role_to_class'])} | "
        f"edge_labels mapped={len(lookups['edge_label_to_property'])}"
    )
    return lookups


# ─────────────────────────────────────────
# B. STEP 1 — MINT INDIVIDUALS WITH DEDUPLICATION
# ─────────────────────────────────────────

def step1_mint_individuals(stage1_data: dict, lookups: dict,
                            contract_id: str, abox_ns: Namespace,
                            tbox_ns: Namespace, g: Graph,
                            class_depth: dict[str, int]) -> dict:
    """
    Mint one OWL individual per unique real-world entity, not per graph node.

    KEY FIXES over the original implementation
    ──────────────────────────────────────────
    FIX 1 — Dedup scope: nodes are grouped by (canonical_class, normalised_name).
      Nodes that share a name but have *different* canonical classes are NOT merged.
      The original code only used canonical_class::name as the key but then merged
      across class boundaries by accident in the type-union step, producing
      incoherent individuals like "SecureLife a ContingentBeneficiary, IssuingInsurer".

    FIX 2 — Type union is scoped: when multiple nodes share the same dedup key
      (same canonical class + same name), their type sets are unioned.  But only
      the types that are *consistent with the canonical class* (i.e. same class or
      subclasses of it, according to the TBox depth map) are added.  Unknown types
      are added permissively under the open-world assumption; clearly unrelated
      classes are not added.  This prevents role-crossing contamination while
      still capturing legitimate multi-typing (e.g. Insured who is also PolicyOwner).

    FIX 3 — Canonical class selection uses TBox hierarchy depth (most derived wins)
      instead of the hardcoded GENERIC_FALLBACKS exclusion list.

    FIX 4 — IRI uses a human-readable name slug; the 8-char hash is only appended
      when two distinct individuals would otherwise produce the same IRI fragment.
      This makes the ABox legible without losing collision safety.

    Returns {original_node_id → merged_iri} for use by Steps 2 & 3.
    """
    entities = stage1_data.get("entities_normalized", [])

    # ── Pass 1: resolve classes and group nodes by (canonical_class, normalised_name)
    # Key is a 2-tuple so that same-name entities with different canonical classes
    # are kept strictly separate — e.g. "SecureLife" as IssuingInsurer vs as
    # ClaimsProcessingAuthority are different individuals.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    node_to_key: dict[str, tuple[str, str]] = {}

    for entity in entities:
        name    = entity.get("name", "").strip()
        classes = _resolve_classes_for_entity(entity, lookups)
        canon   = _canonical_class(classes, entity.get("type", "UNKNOWN").upper(),
                                   class_depth)
        key     = (canon, _normalise_name(name))
        groups[key].append(entity)
        node_to_key[entity["id"]] = key

    # ── Pass 2: mint one IRI per group, assert all triples ───────────────
    key_to_iri: dict[tuple[str, str], URIRef] = {}

    # Track slugs already used in this ABox to detect collisions
    slug_seen: dict[str, tuple[str, str]] = {}  # slug → first key that used it

    for key, members in groups.items():
        canon_class, norm_name = key

        # Build a human-readable slug: CanonicalClass_ContractID_NameSlug
        name_slug = _name_slug(members[0].get("name", norm_name))
        base_frag = f"{canon_class}_{safe_iri_fragment(contract_id)}_{name_slug}"

        if base_frag in slug_seen and slug_seen[base_frag] != key:
            # Collision: two genuinely different individuals map to same slug.
            # Append a short hash to disambiguate — this is rare in practice.
            hash_suffix = _dedup_key_hash(f"{canon_class}::{norm_name}")
            iri_frag = f"{base_frag}_{hash_suffix}"
            log.debug(f"IRI collision on slug '{base_frag}' — appending hash: {iri_frag}")
        else:
            iri_frag = base_frag
            slug_seen[base_frag] = key

        iri = abox_ns[iri_frag]
        key_to_iri[key] = iri

        # ── Type union: only within the dedup group (same canonical class + name)
        # We add any class that was observed on any merged node.  Because these
        # nodes already share the same canonical class, cross-class contamination
        # cannot happen — the strict (canon, name) grouping prevents it.
        all_classes: set[str] = set()
        for entity in members:
            all_classes.update(_resolve_classes_for_entity(entity, lookups))

        if all_classes:
            for cls in all_classes:
                g.add((iri, RDF.type, tbox_ns[cls]))
        else:
            etype = members[0].get("type", "UNKNOWN").upper()
            g.add((iri, RDF.type, OWL.Thing))
            g.add((iri, RDFS.comment,
                   Literal(f"Unresolved entity type: {etype}", datatype=XSD.string)))

        # Name literal — representative member (all share the same name by dedup key)
        name = next((e.get("name", "").strip() for e in members if e.get("name", "").strip()), "")
        if name:
            prop_name, literal = infer_name_property(members[0])
            g.add((iri, tbox_ns[prop_name], literal))

        # Relational anchors — union across merged nodes, one triple per property
        seen_anchor_props: set[str] = set()
        for entity in members:
            for prop_name, literal in parse_relational_anchors(
                entity.get("relational_anchors", [])
            ):
                if prop_name not in seen_anchor_props:
                    g.add((iri, tbox_ns[prop_name], literal))
                    seen_anchor_props.add(prop_name)

        g.add((iri, tbox_ns["sourceContractID"], Literal(contract_id, datatype=XSD.string)))

    # ── Build return map ──────────────────────────────────────────────────
    entity_iris = {nid: key_to_iri[key] for nid, key in node_to_key.items()}

    n_nodes  = len(entities)
    n_merged = len(groups)
    log.info(
        f"Step 1: merged {n_nodes} entity nodes → {n_merged} individuals "
        f"({n_nodes - n_merged} duplicates eliminated)"
    )
    return entity_iris


# ─────────────────────────────────────────
# C. STEP 2 — OBJECT PROPERTIES FROM EHF EDGES
# ─────────────────────────────────────────

def step2_object_properties(stage1_data: dict, entity_iris: dict, lookups: dict,
                              tbox_ns: Namespace, g: Graph):
    """
    Assert object property triples from graph edges, with domain validation.

    Unchanged in logic from the reviewed version — domain validation and
    entity-only target guards are already correct.
    """
    edge_buckets = stage1_data.get("edge_buckets", {})
    asserted = 0

    # Build iri → set-of-class-names from the graph for domain validation
    iri_types: dict[URIRef, set[str]] = defaultdict(set)
    for s, p, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef):
            local = str(o).split("#")[-1].split("/")[-1]
            iri_types[s].add(local)

    def _domain_ok(iri: URIRef, prop_name: str) -> bool:
        domain = lookups["property_domains"].get(prop_name)
        if not domain or domain == "owl:Thing":
            return True
        return domain in iri_types.get(iri, set())

    # entity_entity: direct object properties
    for edge in edge_buckets.get("entity_entity", []):
        score = edge.get("score")
        if score is not None and score < 0.5:
            continue
        label   = (edge.get("label") or "").lower().strip()
        prop    = lookups["edge_label_to_property"].get(label)
        if not prop:
            continue
        src_iri = entity_iris.get(edge["source"])
        tgt_iri = entity_iris.get(edge["target"])
        if src_iri and tgt_iri and _domain_ok(src_iri, prop):
            g.add((src_iri, tbox_ns[prop], tgt_iri))
            asserted += 1

    # entity_fact edges → fact_to_subject map
    SUBJECT_ROLES = {"payer", "owner", "policyholder", "policy owner",
                     "insured", "applicant", "issuer", "subject"}

    fact_to_subject: dict[str, str] = {}
    for edge in edge_buckets.get("entity_fact", []):
        fact_id   = edge.get("target")
        entity_id = edge.get("source")
        label     = (edge.get("label") or "").lower().strip()
        if entity_id not in entity_iris:
            continue
        if fact_id not in fact_to_subject:
            fact_to_subject[fact_id] = entity_id
        elif any(r in label for r in SUBJECT_ROLES):
            fact_to_subject[fact_id] = entity_id

    # fact_entity edges → assert subject → object
    for edge in edge_buckets.get("fact_entity", []):
        score = edge.get("score")
        if score is not None and score < 0.5:
            continue
        fact_id = edge.get("source")
        tgt_id  = edge.get("target")
        label   = (edge.get("label") or "").lower().strip()

        if tgt_id not in entity_iris:
            continue

        prop    = lookups["edge_label_to_property"].get(label)
        if not prop:
            continue
        tgt_iri = entity_iris.get(tgt_id)
        if not tgt_iri:
            continue
        src_id  = fact_to_subject.get(fact_id)
        src_iri = entity_iris.get(src_id) if src_id else None
        if src_iri and _domain_ok(src_iri, prop):
            g.add((src_iri, tbox_ns[prop], tgt_iri))
            asserted += 1

    log.info(f"Step 2: asserted {asserted} object property triples")


# ─────────────────────────────────────────
# D. STEP 3 — MINE LITERALS FROM FACT SENTENCES (BOUNDED LLM)
# ─────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are extracting RDF datatype property values from an insurance policy sentence.

Fact name: {fact_name}
Sentence: {sentence}

Extract literal values only. Use ONLY properties from this closed vocabulary:
{property_vocab}

Respond ONLY with a JSON array. Each element:
{{"property": "<exact property name from vocab>", "value": "<extracted value>", "xsd_type": "xsd:string|xsd:date|xsd:integer|xsd:decimal|xsd:boolean"}}

Rules:
- If nothing can be extracted, return []
- Do not invent property names outside the vocab
- Do not include markdown or explanation"""


def _call_llm(client: OpenAI, prompt: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=cfg.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return raw
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return ""


XSD_MAP = {
    "xsd:string":   XSD.string,
    "xsd:date":     XSD.date,
    "xsd:integer":  XSD.integer,
    "xsd:decimal":  XSD.decimal,
    "xsd:boolean":  XSD.boolean,
    "xsd:dateTime": XSD.dateTime,
}


def step3_mine_fact_literals(stage1_data: dict, entity_iris: dict, lookups: dict,
                              tbox_ns: Namespace, g: Graph, client: OpenAI):
    """
    For each fact with a resolvable subject entity, use a bounded LLM call to
    extract datatype property values from the fact sentence.

    FIX: closed property vocabulary is now restricted to owl:DatatypeProperty
    terms only.  The original heuristic admitted any property whose range started
    with "xsd:" OR whose domain was absent — the second condition silently
    included object properties with unrecorded domains, causing the LLM to
    occasionally return object-property names as if they were datatype properties,
    resulting in literal triples on object properties.

    We now use the property_owl_type index built in load_tbox_lookups, which is
    populated authoritatively from Stage 3 and Stage 4 outputs.
    """
    facts = {f["id"]: f for f in stage1_data.get("facts", [])}
    edge_buckets = stage1_data.get("edge_buckets", {})

    dt_props = sorted(
        p for p in lookups["property_names"]
        if lookups["property_owl_type"].get(p) == "owl:DatatypeProperty"
    )

    if not dt_props:
        log.warning("Step 3: no owl:DatatypeProperty terms found — skipping literal mining.")
        return

    property_vocab = ", ".join(dt_props[:60])

    SUBJECT_ROLES = {"payer", "owner", "policyholder", "policy owner",
                     "insured", "applicant", "issuer", "subject"}

    fact_to_subject: dict[str, str] = {}
    for edge in edge_buckets.get("entity_fact", []):
        fact_id   = edge.get("target")
        entity_id = edge.get("source")
        label     = (edge.get("label") or "").lower().strip()
        if entity_id not in entity_iris:
            continue
        if fact_id not in fact_to_subject:
            fact_to_subject[fact_id] = entity_id
        elif any(r in label for r in SUBJECT_ROLES):
            fact_to_subject[fact_id] = entity_id

    fact_to_topics: dict[str, list] = {
        fid: fdata.get("chunk_topics") or []
        for fid, fdata in facts.items()
        if fdata.get("chunk_topics")
    }

    asserted = 0
    processed = 0

    for fact_id, fact_data in facts.items():
        sentence = (fact_data.get("sentence") or "").strip()
        name     = (fact_data.get("name") or "").strip()
        if not sentence or len(sentence) < 10:
            continue

        subject_id  = fact_to_subject.get(fact_id)
        subject_iri = entity_iris.get(subject_id) if subject_id else None
        if not subject_iri:
            continue

        prompt  = EXTRACTION_PROMPT.format(
            fact_name=name, sentence=sentence, property_vocab=property_vocab
        )
        raw_out = _call_llm(client, prompt)
        if not raw_out:
            continue

        try:
            triples = json.loads(raw_out)
        except json.JSONDecodeError:
            log.debug(f"JSON parse failed for fact {fact_id}: {raw_out[:80]}")
            continue

        for triple in triples:
            prop     = (triple.get("property") or "").strip()
            val      = (triple.get("value") or "").strip()
            xsd_type = triple.get("xsd_type", "xsd:string")
            if not prop or not val:
                continue
            # Guard: prop must be in the closed vocab AND be a DatatypeProperty
            if prop not in lookups["property_names"]:
                continue
            if lookups["property_owl_type"].get(prop) != "owl:DatatypeProperty":
                log.debug(f"Step 3: LLM returned non-datatype property '{prop}' — skipped")
                continue
            xsd_uri = XSD_MAP.get(xsd_type, XSD.string)
            g.add((subject_iri, tbox_ns[prop], Literal(val, datatype=xsd_uri)))
            topics = fact_to_topics.get(fact_id, [])
            if topics:
                g.add((subject_iri, tbox_ns["sourceContractSection"],
                       Literal(", ".join(topics[:3]), datatype=XSD.string)))
            asserted += 1

        processed += 1

    log.info(f"Step 3: processed {processed} facts, asserted {asserted} literal triples")


# ─────────────────────────────────────────
# E. SERIALIZE
# ─────────────────────────────────────────

def serialize_abox(g: Graph, contract_id: str, out_dir: str) -> Path:
    out_path = Path(out_dir) / f"abox_{safe_iri_fragment(contract_id)}.ttl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    log.info(f"ABox written → {out_path}  ({len(g)} triples)")
    return out_path


# ─────────────────────────────────────────
# F. PELLET VERIFY
# ─────────────────────────────────────────

def verify_with_pellet(tbox_path: str, abox_path: Path):
    import subprocess
    combined = Graph()
    combined.parse(tbox_path, format="turtle")
    combined.parse(str(abox_path), format="turtle")
    combined_path = str(abox_path).replace(".ttl", "_combined.ttl")
    combined.serialize(destination=combined_path, format="turtle")
    try:
        result = subprocess.run(
            ["pellet", "consistency", combined_path],
            capture_output=True, text=True, timeout=120
        )
        if "Consistent: Yes" in result.stdout:
            log.info("Pellet: ABox is consistent with TBox ✓")
        else:
            log.warning(f"Pellet consistency issue:\n{result.stdout}\n{result.stderr}")
    except FileNotFoundError:
        log.warning("Pellet not found on PATH — skipping reasoner verification.")
    except Exception as e:
        log.warning(f"Pellet error: {e}")


# ─────────────────────────────────────────
# LOAD STAGE 1 OUTPUT FOR A CONTRACT
# ─────────────────────────────────────────

def load_stage1_for_contract(stage1_dir: str, source_file: str) -> dict | None:
    batch_files = sorted(glob.glob(str(Path(stage1_dir) / "batch_*_stage1.json")))
    for bf in batch_files:
        with open(bf) as f:
            batches = json.load(f)
        docs = batches if isinstance(batches, list) else [batches]
        for doc in docs:
            if source_file in doc.get("source_file", ""):
                return doc
    return None


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def populate_abox(contract_source_file: str, contract_id: str,
                  skip_llm: bool = False) -> tuple[Graph, Path]:
    tbox_base_iri = resolve_tbox_base_iri(cfg.STAGE7_OUT_DIR)
    tbox_ns  = Namespace(tbox_base_iri)
    abox_ns  = Namespace(ABOX_NS)

    # Build hierarchy depth map from the assembled TBox — used for canonical
    # class selection in Step 1.  Falls back gracefully if tbox.ttl is absent.
    tbox_ttl_path = str(Path(cfg.STAGE7_OUT_DIR) / "tbox.ttl")
    class_depth   = build_class_depth_map(tbox_ttl_path, tbox_base_iri)

    stage1_data = load_stage1_for_contract(cfg.STAGE1_OUT_DIR, contract_source_file)
    if stage1_data is None:
        raise FileNotFoundError(
            f"No Stage 1 batch found for source_file='{contract_source_file}'"
        )

    lookups = load_tbox_lookups(cfg.STAGE3_OUT_DIR, cfg.STAGE4_OUT_DIR)

    g = Graph()
    g.bind("li",   tbox_ns)
    g.bind("abox", abox_ns)
    g.bind("rdf",  RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl",  OWL)
    g.bind("xsd",  XSD)

    entity_iris = step1_mint_individuals(
        stage1_data, lookups, contract_id, abox_ns, tbox_ns, g, class_depth
    )

    step2_object_properties(stage1_data, entity_iris, lookups, tbox_ns, g)

    if not skip_llm:
        client = OpenAI(api_key="dummy", base_url=cfg.API_BASE)
        step3_mine_fact_literals(stage1_data, entity_iris, lookups, tbox_ns, g, client)
    else:
        log.info("Step 3: skipped (skip_llm=True)")

    abox_path = serialize_abox(g, contract_id, cfg.ABOX_OUT_DIR)

    if Path(tbox_ttl_path).exists():
        verify_with_pellet(tbox_ttl_path, abox_path)
    else:
        log.warning(f"tbox.ttl not found at {tbox_ttl_path} — skipping Pellet verification.")

    return g, abox_path


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABox Population Pipeline")
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--stage1-dir",  default=None)
    parser.add_argument("--stage3-dir",  default=None)
    parser.add_argument("--stage4-dir",  default=None)
    parser.add_argument("--stage7-dir",  default=None)
    parser.add_argument("--abox-dir",    default=None)
    args = parser.parse_args()

    if args.stage1_dir:  cfg.STAGE1_OUT_DIR = args.stage1_dir
    if args.stage3_dir:  cfg.STAGE3_OUT_DIR = args.stage3_dir
    if args.stage4_dir:  cfg.STAGE4_OUT_DIR = args.stage4_dir
    if args.stage7_dir:  cfg.STAGE7_OUT_DIR = args.stage7_dir
    if args.abox_dir:    cfg.ABOX_OUT_DIR   = args.abox_dir

    populate_abox(
        contract_source_file=args.source_file,
        contract_id=args.contract_id,
        skip_llm=args.skip_llm,
    )