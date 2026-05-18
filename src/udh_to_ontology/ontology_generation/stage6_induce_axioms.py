"""
Stage 6 — Restriction Axiom Induction
======================================
Input:  property_clusters_with_dr.json  (Stage 4 output)
        class_hierarchy.json            (Stage 5 output)
Output: induced_axioms.json             — committed OWL 2 TBox axioms
        axioms_manchester_summary.txt   — human-readable Manchester syntax summary
        stage6_review.json              — low-confidence formalizations queued for human review (not committed)
        stage6_disjoint_flagged.json    — disjointness axioms removed by hierarchy conflict filter
        stage6_dropped_candidates.json  — candidates dropped before LLM formalization
        stage6_stats.json

Pipeline:
  6.1  Load Stage 4 axiom-stream clusters and Stage 5 class hierarchy; build
         class vocab, property vocab, direct-parent map, and transitive
         ancestor sets (memoised DFS, cycle-safe).
  6.2  Build axiom candidates: resolve property via three-tier fuzzy matching
         (exact → token-Jaccard on owl_name → token-Jaccard on members → token-
         Jaccard on definition); resolve filler from range or definition text;
         extract cardinality hint from member strings and S3 sentences;
         collect S1 edge topology, S3 raw sentences, S4 chunk topics as LLM signals.
  6.3  Filter candidates: drop if subject_class is absent from vocabulary,
         cardinality-family candidate has no resolved property, disjointness-family
         candidate has no valid filler class, or candidate has no evidence at all.
  6.4  LLM formalization in batches of {BATCH_SIZE}: prompt supplies full class
         and property vocabularies, the Stage 5 hierarchy block, and all four
         graph signals; JSON schema is enforced via guided decoding (outlines
         backend). Low-confidence results are written to stage6_review.json and
         NOT committed to induced_axioms.json.
  6.5  Coherence check (five passes, in order):
         (1) Vocabulary validation — drop axioms whose subject, property, or
             filler fall outside the known vocabularies.
         (2) Hierarchy-aware disjointness filter — remove owl:disjointWith
             axioms where subject and filler stand in an ancestor/descendant
             relationship (which would make one or both classes unsatisfiable).
         (3) Deduplication by axiom_str — keep the highest-confidence instance
             when multiple axioms share the same Manchester string.
         (4) Cardinality contradiction check — drop conflicting min/max pairs;
             prefer owl:exactCardinality when present.
         (5) allValuesFrom downgrade — medium-confidence universal restrictions
             are demoted to owl:someValuesFrom (open-world assumption).
  6.6  Serialize: induced_axioms.json, Manchester syntax summary grouped by
         OWL construct, and stage6_stats.json.

Design principles:
  • Axiom family (set in Stage 4) pre-constrains the allowed OWL constructs
    per candidate; the LLM must select within that set.
  • Property resolution is fuzzy (token-level Jaccard over camelCase/PascalCase-
    tokenised strings) with a configurable threshold (default {PROP_MATCH_THRESHOLD}).
  • The Stage 5 hierarchy is surfaced both in the LLM prompt (for disjointness
    safety reasoning) and enforced post-hoc by the coherence filter — two
    independent layers of protection against unsatisfiable disjointness assertions.
  • Low-confidence formalizations are quarantined for human review and never
    committed, unlike Stage 5 where low-confidence placements were committed
    and merely flagged.
  • Guided decoding (outlines backend) enforces the JSON schema so that
    subject_class, property_name, filler, and owl_construct are always drawn
    from the known vocabularies.

Usage:
    python stage6_induce_axioms.py
    python stage6_induce_axioms.py \\
        --stage4_path  ./stage4_out/property_clusters_with_dr.json \\
        --stage5_path  ./stage5_out/class_hierarchy.json \\
        --output_dir   ./stage6_out \\
        --prop_threshold 0.35
"""


import argparse
import json
import re
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

OWL_THING    = "owl:Thing"
MAX_RETRIES  = 3
RETRY_DELAY  = 5.0
BATCH_SIZE   = 6

OWL_CONSTRUCTS = [
    "owl:someValuesFrom",
    "owl:allValuesFrom",
    "owl:minCardinality",
    "owl:maxCardinality",
    "owl:exactCardinality",
    "owl:disjointWith",
]

CARDINALITY_CONSTRUCTS = {
    "owl:minCardinality",
    "owl:maxCardinality",
    "owl:exactCardinality",
}

# Maps Stage 3 axiom_family → allowed OWL constructs.
AXIOM_FAMILY_TO_CONSTRUCTS: dict[str, list[str]] = {
    "existential_restriction":  ["owl:someValuesFrom"],
    "universal_restriction":    ["owl:allValuesFrom", "owl:someValuesFrom"],
    "subclass_condition":       ["owl:someValuesFrom"],
    "cardinality_constraint":   ["owl:exactCardinality", "owl:minCardinality", "owl:maxCardinality"],
    "eligibility_condition":    ["owl:someValuesFrom", "owl:minCardinality"],
    "prohibition_or_exclusion": ["owl:disjointWith", "owl:someValuesFrom"],
    "disjointness":             ["owl:disjointWith"],
    "other":                    ["owl:someValuesFrom"],
}

# Minimum token overlap fraction to accept a fuzzy property match
PROP_MATCH_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AxiomCandidate:
    candidate_id:       str
    subject_class:      Optional[str]
    property_name:      Optional[str]
    filler_class:       Optional[str]
    cardinality_hint:   Optional[str]    # advisory string, not pre-committed int
    axiom_family:       str
    allowed_constructs: list[str]
    dr_confidence:      str
    dr_reasoning:       str
    evidence_sentences: list[str] = field(default_factory=list)
    member_strings:     list[str] = field(default_factory=list)
    chunk_topics:       list[str] = field(default_factory=list)    # S4 signal
    edge_topology:      str       = ""                             # S1 signal (formatted)
    owl_name:           str       = ""
    definition:         str       = ""


@dataclass
class FormalAxiom:
    axiom_id:           str
    owl_construct:      str
    subject_class:      str
    property_name:      Optional[str]
    filler:             Optional[str]
    cardinality:        Optional[int]
    axiom_str:          str
    definition:         str
    confidence:         str
    reasoning:          str
    evidence_sentences: list[str]
    axiom_family:       str
    source_cluster:     str


# ---------------------------------------------------------------------------
# 6.1  Load inputs
# ---------------------------------------------------------------------------

def load_stage4(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_stage5(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_class_vocab(stage4: dict, hierarchy: list[dict]) -> list[str]:
    vocab: list[str] = []
    seen:  set[str]  = set()
    for c in stage4.get("class", []):
        if not c.get("is_noise") and (name := c.get("owl_name", "").strip()):
            if name not in seen:
                vocab.append(name)
                seen.add(name)
    for node in hierarchy:
        if (name := node.get("owl_name", "").strip()) and name not in seen:
            vocab.append(name)
            seen.add(name)
    return sorted(vocab)


def build_property_vocab(stage4: dict) -> list[str]:
    vocab: list[str] = []
    seen:  set[str]  = set()
    for stream in ("property", "axiom"):
        for c in stage4.get(stream, []):
            if not c.get("is_noise") and (name := c.get("owl_name", "").strip()):
                if name not in seen:
                    vocab.append(name)
                    seen.add(name)
    return sorted(vocab)


# ---------------------------------------------------------------------------
# Hierarchy helpers
# ---------------------------------------------------------------------------

def build_class_parents(hierarchy: list[dict]) -> dict[str, list[str]]:
    """
    Build a direct-parents map from the Stage 5 hierarchy nodes.
    { "PolicyOwner": ["Person"], "PrimaryBeneficiary": ["Beneficiary", "Person"], ... }
    """
    return {
        node["owl_name"]: list(node.get("parents") or [])
        for node in hierarchy
        if node.get("owl_name")
    }


def build_ancestor_sets(class_parents: dict[str, list[str]]) -> dict[str, set[str]]:
    """
    Compute the transitive ancestor set for every class using memoised DFS.
    Handles cycles defensively (skips already-visited nodes).
    """
    cache: dict[str, set[str]] = {}

    def _ancestors(name: str, visiting: set[str]) -> set[str]:
        if name in cache:
            return cache[name]
        if name in visiting:
            return set()          # cycle guard
        visiting = visiting | {name}
        result: set[str] = set()
        for parent in class_parents.get(name, []):
            result.add(parent)
            result |= _ancestors(parent, visiting)
        cache[name] = result
        return result

    for name in class_parents:
        _ancestors(name, set())

    return cache


def _build_hierarchy_block(class_parents: dict[str, list[str]]) -> str:
    """
    Render the hierarchy as a compact, prompt-friendly block.
    Only includes classes that have at least one non-owl:Thing parent,
    to keep the prompt concise.
    """
    lines = []
    for cls in sorted(class_parents):
        parents = [p for p in class_parents[cls] if p and p != OWL_THING]
        if parents:
            lines.append(f"  {cls} subClassOf {', '.join(parents)}")
    return "\n".join(lines) if lines else "  (not provided)"


# ---------------------------------------------------------------------------
# Fuzzy property matching
# ---------------------------------------------------------------------------

def _tokenise(s: str) -> set[str]:
    """Split a camelCase/PascalCase/snake_case string into lowercase tokens."""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', s)
    s = re.sub(r'[^a-zA-Z]+', ' ', s)
    return {t.lower() for t in s.split() if len(t) > 1}


def fuzzy_match_property(
    query_tokens: set[str],
    property_vocab: list[str],
    threshold: float = PROP_MATCH_THRESHOLD,
) -> Optional[str]:
    best_score = 0.0
    best_name  = None
    for name in property_vocab:
        name_tokens = _tokenise(name)
        if not name_tokens:
            continue
        intersection = len(query_tokens & name_tokens)
        union        = len(query_tokens | name_tokens)
        score        = intersection / union if union else 0.0
        if score > best_score:
            best_score = score
            best_name  = name
    return best_name if best_score >= threshold else None


def resolve_property(
    cluster: dict,
    property_vocab: list[str],
    property_vocab_set: set[str],
    axiom_family: str,
) -> Optional[str]:
    if axiom_family == "disjointness":
        return None

    owl_name   = cluster.get("owl_name", "").strip()
    definition = cluster.get("definition", "").strip()

    if owl_name and owl_name in property_vocab_set:
        return owl_name

    name_tokens = _tokenise(owl_name) if owl_name else set()
    if name_tokens:
        match = fuzzy_match_property(name_tokens, property_vocab)
        if match:
            return match

    member_tokens: set[str] = set()
    for m in cluster.get("members", [])[:10]:
        member_tokens |= _tokenise(m.get("string", ""))
    if member_tokens:
        match = fuzzy_match_property(member_tokens, property_vocab)
        if match:
            return match

    if definition:
        def_tokens = _tokenise(definition)
        match = fuzzy_match_property(def_tokens, property_vocab)
        if match:
            return match

    return None


# ---------------------------------------------------------------------------
# Filler resolution for disjointness families
# ---------------------------------------------------------------------------

def _extract_class_from_text(text: str, class_vocab_set: set[str]) -> Optional[str]:
    for cls in class_vocab_set:
        if cls.lower() in text.lower():
            return cls
    return None


def resolve_filler(
    cluster: dict,
    range_: str,
    axiom_family: str,
    class_vocab_set: set[str],
) -> Optional[str]:
    if range_ and range_ != OWL_THING and range_ in class_vocab_set:
        return range_

    if axiom_family in ("disjointness", "prohibition_or_exclusion"):
        definition = cluster.get("definition", "")
        if definition:
            match = _extract_class_from_text(definition, class_vocab_set)
            if match:
                return match
        for m in cluster.get("members", [])[:8]:
            match = _extract_class_from_text(m.get("string", ""), class_vocab_set)
            if match:
                return match

    if range_ and range_ != OWL_THING:
        if range_.startswith("xsd:"):
            return range_

    return None


# ---------------------------------------------------------------------------
# Signal assembly helpers
# ---------------------------------------------------------------------------

def _collect_s4_topics(cluster: dict) -> list[str]:
    topics = []
    for s in cluster.get("dr_signals", []):
        if s.get("signal_id") == "S4":
            topics.extend(e for e in s.get("evidence", []) if e and "No matching" not in e)
    return topics[:15]


def _collect_s1_topology(cluster: dict) -> str:
    for s in cluster.get("dr_signals", []):
        if s.get("signal_id") == "S1":
            lines = [e for e in s.get("evidence", []) if e and "No matching" not in e]
            return "; ".join(lines[:6]) if lines else ""
    return ""


def _collect_s3_sentences(cluster: dict) -> list[str]:
    return [
        e for s in cluster.get("dr_signals", [])
        if s.get("signal_id") == "S3"
        for e in s.get("evidence", [])
        if e and "No matching" not in e
    ]


def _extract_cardinality_hint(cluster: dict) -> Optional[str]:
    member_strings = [m.get("string", "") for m in cluster.get("members", [])[:12]]
    sentences      = _collect_s3_sentences(cluster)

    number_pattern = re.compile(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', re.I)
    cardinal_words = {"one", "two", "three", "four", "five", "six",
                      "seven", "eight", "nine", "ten"}

    candidates = []
    for text in member_strings + sentences:
        nums = number_pattern.findall(text)
        for n in nums:
            if n.lower() in cardinal_words or n.isdigit():
                candidates.append(f"{n} (from: '{text[:80]}')")
        if candidates:
            break

    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# 6.2  Build candidates
# ---------------------------------------------------------------------------

def build_candidates(
    stage4: dict,
    class_vocab_set: set[str],
    property_vocab: list[str],
    property_vocab_set: set[str],
) -> list[AxiomCandidate]:
    candidates: list[AxiomCandidate] = []
    cid = 0

    for cluster in stage4.get("axiom", []):
        if cluster.get("is_noise"):
            continue

        owl_name     = cluster.get("owl_name",     "").strip()
        definition   = cluster.get("definition",   "").strip()
        axiom_family = cluster.get("axiom_family", "other") or "other"
        domain       = cluster.get("domain",  OWL_THING)
        range_       = cluster.get("range",   OWL_THING)
        dr_conf      = cluster.get("dr_confidence", "low") or "low"
        dr_reasoning = cluster.get("dr_reasoning",  "") or ""

        if domain == OWL_THING or domain not in class_vocab_set:
            continue

        prop_name = resolve_property(cluster, property_vocab, property_vocab_set, axiom_family)
        filler    = resolve_filler(cluster, range_, axiom_family, class_vocab_set)

        sentences     = _collect_s3_sentences(cluster)[:10]
        chunk_topics  = _collect_s4_topics(cluster)
        edge_topology = _collect_s1_topology(cluster)

        member_strings = [
            m.get("string", "") for m in cluster.get("members", [])
            if m.get("string", "").strip()
        ][:12]

        card_hint = _extract_cardinality_hint(cluster) if axiom_family == "cardinality_constraint" else None
        allowed   = AXIOM_FAMILY_TO_CONSTRUCTS.get(axiom_family, ["owl:someValuesFrom"])

        candidates.append(AxiomCandidate(
            candidate_id=f"A_{cid:04d}",
            subject_class=domain,
            property_name=prop_name,
            filler_class=filler,
            cardinality_hint=card_hint,
            axiom_family=axiom_family,
            allowed_constructs=allowed,
            dr_confidence=dr_conf,
            dr_reasoning=dr_reasoning,
            evidence_sentences=sentences,
            member_strings=member_strings,
            chunk_topics=chunk_topics,
            edge_topology=edge_topology,
            owl_name=owl_name,
            definition=definition,
        ))
        cid += 1

    print(f"  Built {len(candidates)} candidates from axiom-stream clusters.")
    family_counts = Counter(c.axiom_family for c in candidates)
    for family, count in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"    {family:35s}: {count}")
    return candidates


# ---------------------------------------------------------------------------
# 6.3  Filter candidates
# ---------------------------------------------------------------------------

def filter_candidates(
    candidates: list[AxiomCandidate],
    class_vocab_set: set[str],
) -> tuple[list[AxiomCandidate], list[AxiomCandidate]]:
    kept, dropped = [], []

    for c in candidates:
        if not c.subject_class or c.subject_class not in class_vocab_set:
            dropped.append(c)
            continue

        if (set(c.allowed_constructs) & CARDINALITY_CONSTRUCTS) and not c.property_name:
            dropped.append(c)
            continue

        if c.axiom_family in ("disjointness", "prohibition_or_exclusion"):
            if not c.filler_class or c.filler_class not in class_vocab_set:
                dropped.append(c)
                continue

        if not c.evidence_sentences and not c.member_strings and not c.chunk_topics:
            dropped.append(c)
            continue

        kept.append(c)

    print(f"  Filter: {len(kept)} kept, {len(dropped)} dropped.")
    return kept, dropped


# ---------------------------------------------------------------------------
# 6.4  LLM formalization
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an OWL 2 ontology engineer specializing in life insurance contracts. "
    "Respond only with valid JSON matching the provided schema. "
    "Use only class and property names from the provided vocabulary. "
    "Do not invent terms not present in the vocabulary."
)


def _build_prompt(
    batch: list[AxiomCandidate],
    class_vocab: list[str],
    property_vocab: list[str],
    class_parents: dict[str, list[str]],
) -> str:
    class_vocab_str  = ", ".join(class_vocab)
    prop_vocab_str   = ", ".join(property_vocab) if property_vocab else "(none)"
    hierarchy_block  = _build_hierarchy_block(class_parents)

    blocks = ""
    for i, c in enumerate(batch):
        sents      = "\n".join(f"  - {s}" for s in c.evidence_sentences[:8]) or "  (none)"
        members    = ", ".join(f'"{s}"' for s in c.member_strings[:10])       or "(none)"
        topics     = ", ".join(c.chunk_topics[:10])                            or "(none)"
        topology   = c.edge_topology or "(none)"

        if len(c.allowed_constructs) == 1:
            construct_guidance = (
                f"axiom_family='{c.axiom_family}'. "
                f"You MUST use owl_construct='{c.allowed_constructs[0]}'."
            )
        else:
            construct_guidance = (
                f"axiom_family='{c.axiom_family}'. "
                f"Choose the most appropriate owl_construct from: {', '.join(c.allowed_constructs)}."
            )

        card_hint_str = (
            f"Cardinality hint : {c.cardinality_hint}"
            if c.cardinality_hint
            else "Cardinality hint : (none — infer from evidence if applicable)"
        )

        blocks += f"""
            --- Candidate {i + 1} (id: {c.candidate_id}) ---
            Cluster name     : {c.owl_name or '(unnamed)'}
            Definition       : {c.definition or '(none)'}
            Construct rule   : {construct_guidance}
            Subject class    : {c.subject_class or 'unknown'}
            Property         : {c.property_name or 'unknown — infer from evidence or pick closest from property vocab'}
            Filler class     : {c.filler_class  or 'unknown — infer from evidence or pick closest from class vocab'}
            {card_hint_str}
            DR confidence    : {c.dr_confidence}
            DR reasoning     : {c.dr_reasoning or '(none)'}
            S1 edge topology : {topology}
            S4 chunk topics  : {topics}
            Member strings   : {members}
            S3 sentences:
            {sents}
            """

    return f"""You are an OWL 2 ontology engineer for a life insurance domain ontology.

            Task: formalize each axiom candidate into a precise OWL 2 TBox axiom.
            Each candidate has an axiom_family confirmed by a prior LLM step — use this
            to select the correct OWL construct as instructed per candidate.
            Four graph signals (S1 edge topology, S3 raw sentences, S4 chunk topics,
            member strings) are provided to help you infer missing property and filler values.

            ALLOWED class names   : {class_vocab_str}
            ALLOWED property names: {prop_vocab_str}

            KNOWN CLASS HIERARCHY (subClassOf relations from Stage 5):
            {hierarchy_block}

            RULES:
            1. subject_class MUST come from the allowed class list.
            2. property_name MUST come from the allowed property list, or null for disjointWith.
            If the candidate lists 'unknown', infer the closest matching property from
            the evidence and the allowed property list — do NOT leave it null unless the
            family is disjointness.
            3. filler MUST be from the class list, an XSD datatype (xsd:string, xsd:integer,
            xsd:decimal, xsd:boolean, xsd:date, xsd:dateTime, xsd:gYear, xsd:duration), or null.
            If the candidate lists 'unknown', infer the filler from the evidence sentences
            and member strings, picking the most semantically appropriate class from the
            allowed class list.
            4. For owl:disjointWith — HIERARCHY SAFETY CHECK (mandatory before asserting):
            Using the KNOWN CLASS HIERARCHY above, verify that:
                (a) neither class is a direct or indirect subclass of the other, AND
                (b) they share no common subclass already in the hierarchy.
            If either condition is violated, do NOT assert owl:disjointWith. Instead:
                - reformulate as owl:someValuesFrom or owl:allValuesFrom if the evidence
                supports a restriction reading, OR
                - set confidence=low with reasoning that explains the hierarchy conflict.
            property_name is null for disjointWith; filler is the second class name.
            5. For owl:allValuesFrom: use only when the evidence clearly states a universal
            constraint ("all X must be Y", "only Y", "exclusively"). Default to
            owl:someValuesFrom when uncertain (open-world assumption).
            6. For cardinality constructs: cardinality must be a positive integer; interpret
            the cardinality hint critically — ignore numbers that clearly refer to amounts,
            days, ages, or percentages rather than counts of related individuals.
            7. For owl:someValuesFrom / owl:allValuesFrom: cardinality is null.
            8. axiom_str must use pure Manchester restriction syntax:
                owl:someValuesFrom   → "Policy hasInsured some Person"
                owl:allValuesFrom    → "Policy hasRider all Rider"
                owl:exactCardinality → "Policy hasIdentifier exactly 1 xsd:string"
                owl:minCardinality   → "Policy hasRider min 1 Rider"
                owl:maxCardinality   → "Policy hasOwner max 1 PolicyOwner"
                owl:disjointWith     → "PolicyOwner DisjointWith Beneficiary"
            9. confidence: high = strongly evidenced by sentences and signals,
                        medium = reasonable inference from evidence,
                        low = weak or genuinely ambiguous.
            10. Even if property or filler was listed as 'unknown', make your best inference
                from the provided evidence. Only set confidence=low if the evidence is truly
                insufficient to determine the axiom.

            {blocks}

            Return a JSON object with a "formalizations" array of exactly {len(batch)} items
            in the same order as the candidates above.
            Each item: candidate_id, owl_construct, subject_class, property_name (or null),
            filler (or null), cardinality (integer or null), axiom_str,
            definition (1 sentence ≤ 20 words), confidence, reasoning (1-2 sentences).
            """


def _build_schema(class_vocab: list[str], property_vocab: list[str]) -> dict:
    xsd_types = [
        "xsd:string", "xsd:integer", "xsd:decimal", "xsd:float",
        "xsd:boolean", "xsd:date", "xsd:dateTime", "xsd:gYear",
        "xsd:duration", "xsd:anyURI",
    ]
    filler_vocab = class_vocab + xsd_types
    return {
        "type": "object",
        "properties": {
            "formalizations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id":  {"type": "string"},
                        "owl_construct": {"type": "string", "enum": OWL_CONSTRUCTS},
                        "subject_class": {"type": "string", "enum": class_vocab},
                        "property_name": {"anyOf": [
                            {"type": "string", "enum": property_vocab + [""]},
                            {"type": "null"},
                        ]},
                        "filler": {"anyOf": [
                            {"type": "string", "enum": filler_vocab},
                            {"type": "null"},
                        ]},
                        "cardinality": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "axiom_str":   {"type": "string"},
                        "definition":  {"type": "string"},
                        "confidence":  {"type": "string", "enum": ["high", "medium", "low"]},
                        "reasoning":   {"type": "string"},
                    },
                    "required": [
                        "candidate_id", "owl_construct", "subject_class",
                        "property_name", "filler", "cardinality",
                        "axiom_str", "definition", "confidence", "reasoning",
                    ],
                },
            }
        },
        "required": ["formalizations"],
    }


def _call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    schema: dict,
    expected: int,
) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "axiom_formalization",
                        "schema": schema,
                        "strict": True,
                    },
                },
                temperature=0.1,
                extra_body={"guided_decoding_backend": "outlines"},
            )
            data   = json.loads(resp.choices[0].message.content)
            result = data.get("formalizations", [])
            if len(result) != expected:
                raise ValueError(f"Expected {expected} formalizations, got {len(result)}")
            return result
        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] LLM error: {e}", file=sys.stderr)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return []


def formalize(
    candidates: list[AxiomCandidate],
    client: OpenAI,
    model: str,
    class_vocab: list[str],
    property_vocab: list[str],
    class_parents: dict[str, list[str]],
) -> tuple[list[FormalAxiom], list[dict]]:
    """
    Returns (committed_axioms, review_items).
    review_items are low-confidence formalizations saved for human inspection.
    """
    axioms:  list[FormalAxiom] = []
    review:  list[dict]        = []
    schema    = _build_schema(class_vocab, property_vocab)
    n_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE
    ax_id     = 0

    for b_start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[b_start : b_start + BATCH_SIZE]
        b_num = b_start // BATCH_SIZE + 1
        print(f"  Batch {b_num}/{n_batches} ({len(batch)} candidates) ...")

        prompt  = _build_prompt(batch, class_vocab, property_vocab, class_parents)
        results = _call_llm(client, model, prompt, schema, len(batch))

        if not results:
            print(f"  [WARNING] All LLM attempts failed for batch {b_num} — dropping batch.")
            continue

        for cand, res in zip(batch, results):
            conf = res.get("confidence", "low")

            formal = FormalAxiom(
                axiom_id=f"AX_{ax_id:04d}",
                owl_construct=res.get("owl_construct", "owl:someValuesFrom"),
                subject_class=res.get("subject_class", cand.subject_class or OWL_THING),
                property_name=res.get("property_name") or None,
                filler=res.get("filler") or None,
                cardinality=res.get("cardinality"),
                axiom_str=res.get("axiom_str", ""),
                definition=res.get("definition", ""),
                confidence=conf,
                reasoning=res.get("reasoning", ""),
                evidence_sentences=cand.evidence_sentences,
                axiom_family=cand.axiom_family,
                source_cluster=cand.owl_name,
            )
            ax_id += 1

            if conf == "low":
                review.append({
                    **_serialize(formal),
                    "candidate_property_hint": cand.property_name,
                    "candidate_filler_hint":   cand.filler_class,
                    "dr_confidence":           cand.dr_confidence,
                    "member_strings":          cand.member_strings[:5],
                })
                print(f"  [review] {formal.axiom_id} ({cand.owl_name}): low confidence — queued for review.")
            else:
                axioms.append(formal)

    return axioms, review


# ---------------------------------------------------------------------------
# 6.5  Coherence check (updated with hierarchy-aware disjointness filter)
# ---------------------------------------------------------------------------

def _filter_contradictory_disjointness(
    axioms: list[FormalAxiom],
    ancestor_sets: dict[str, set[str]],
) -> tuple[list[FormalAxiom], list[FormalAxiom]]:
    """
    Remove owl:disjointWith axioms where subject and filler are in an
    ancestor/descendant relationship in the Stage 5 hierarchy.

    Such assertions make one or both classes unsatisfiable (owl:Nothing) in
    OWL DL reasoning, causing a cascade of 'unsatisfiable class' errors.

    Returns (kept, flagged).
    """
    kept:    list[FormalAxiom] = []
    flagged: list[FormalAxiom] = []

    for ax in axioms:
        if ax.owl_construct != "owl:disjointWith":
            kept.append(ax)
            continue

        a = ax.subject_class or ""
        b = ax.filler        or ""

        if not a or not b:
            flagged.append(ax)
            continue

        a_ancestors = ancestor_sets.get(a, set())
        b_ancestors = ancestor_sets.get(b, set())

        conflict = (
            a == b
            or b in a_ancestors
            or a in b_ancestors
        )

        if conflict:
            reason = (
                f"Hierarchy conflict: {a} and {b} are in a subClassOf relationship "
                f"(ancestors of {a}: {{{', '.join(sorted(a_ancestors)[:5])}}}; "
                f"ancestors of {b}: {{{', '.join(sorted(b_ancestors)[:5])}}}) — "
                f"asserting disjointWith would make one or both classes unsatisfiable."
            )
            print(f"  [disjoint-filter] Dropping {ax.axiom_id}: {reason}")
            # Attach reason for the flagged output
            ax.reasoning = ax.reasoning + " | FLAGGED: " + reason
            flagged.append(ax)
        else:
            kept.append(ax)

    if flagged:
        print(f"  Disjointness filter: {len(flagged)} contradictory axiom(s) removed, "
              f"{len(kept)} kept.")
    return kept, flagged


def coherence_check(
    axioms: list[FormalAxiom],
    class_vocab_set: set[str],
    property_vocab_set: set[str],
    ancestor_sets: dict[str, set[str]],
) -> tuple[list[FormalAxiom], list[FormalAxiom]]:
    """
    Returns (clean_axioms, flagged_disjointness_axioms).
    The flagged list is written to stage6_disjoint_flagged.json for inspection.
    """

    # 1. Vocabulary validation
    clean = []
    for ax in axioms:
        issues = []
        if ax.subject_class and ax.subject_class not in class_vocab_set:
            issues.append(f"subject_class '{ax.subject_class}' not in vocabulary")
        if ax.property_name and ax.property_name not in property_vocab_set:
            issues.append(f"property_name '{ax.property_name}' not in vocabulary")
        if (ax.filler
                and not ax.filler.startswith("xsd:")
                and ax.filler not in class_vocab_set
                and ax.filler != OWL_THING):
            issues.append(f"filler '{ax.filler}' not in vocabulary")
        if issues:
            print(f"  Dropping {ax.axiom_id}: {'; '.join(issues)}")
        else:
            clean.append(ax)
    axioms = clean

    # 2. Hierarchy-aware disjointness filter
    axioms, flagged_disjoint = _filter_contradictory_disjointness(axioms, ancestor_sets)

    # 3. Duplicate axiom_str — keep highest-confidence instance
    conf_rank = {"high": 3, "medium": 2, "low": 1}
    seen: dict[str, FormalAxiom] = {}
    for ax in axioms:
        key = ax.axiom_str.strip()
        if not key:
            continue
        if key in seen:
            existing = seen[key]
            winner   = ax if conf_rank.get(ax.confidence, 0) > conf_rank.get(existing.confidence, 0) else existing
            loser    = existing if winner is ax else ax
            print(f"  Dropping duplicate {loser.axiom_id} (kept {winner.axiom_id}): '{key}'")
            seen[key] = winner
        else:
            seen[key] = ax
    axioms = list(seen.values())

    # 4. Cardinality contradiction check
    card_groups: dict[tuple, list[FormalAxiom]] = defaultdict(list)
    for ax in axioms:
        if ax.owl_construct in CARDINALITY_CONSTRUCTS and ax.subject_class and ax.property_name:
            card_groups[(ax.subject_class, ax.property_name)].append(ax)

    to_drop: set[str] = set()
    for (subj, prop), group in card_groups.items():
        mins  = [(a, a.cardinality) for a in group if a.owl_construct == "owl:minCardinality" and a.cardinality is not None]
        maxs  = [(a, a.cardinality) for a in group if a.owl_construct == "owl:maxCardinality" and a.cardinality is not None]
        exact = [a for a in group if a.owl_construct == "owl:exactCardinality"]

        if mins and maxs and max(v for _, v in mins) > min(v for _, v in maxs):
            if exact:
                for a, _ in mins:
                    print(f"  Dropping {a.axiom_id}: min/max contradiction on ({subj},{prop}), exactCardinality kept.")
                    to_drop.add(a.axiom_id)
                for a, _ in maxs:
                    print(f"  Dropping {a.axiom_id}: min/max contradiction on ({subj},{prop}), exactCardinality kept.")
                    to_drop.add(a.axiom_id)
            else:
                for a in group:
                    print(f"  Dropping {a.axiom_id}: cardinality contradiction on ({subj},{prop}).")
                    to_drop.add(a.axiom_id)

    axioms = [a for a in axioms if a.axiom_id not in to_drop]

    # 5. allValuesFrom sanity: downgrade to someValuesFrom if medium confidence
    for ax in axioms:
        if ax.owl_construct == "owl:allValuesFrom" and ax.confidence == "medium":
            print(f"  Downgrading {ax.axiom_id} from allValuesFrom to someValuesFrom (medium confidence, open-world).")
            ax.owl_construct = "owl:someValuesFrom"
            ax.axiom_str     = ax.axiom_str.replace(" all ", " some ")

    print(f"  Coherence check complete: {len(axioms)} axioms committed.")
    return axioms, flagged_disjoint


# ---------------------------------------------------------------------------
# 6.6  Serialize & Manchester summary
# ---------------------------------------------------------------------------

def _serialize(ax: FormalAxiom) -> dict:
    return {
        "axiom_id":           ax.axiom_id,
        "owl_construct":      ax.owl_construct,
        "axiom_family":       ax.axiom_family,
        "subject_class":      ax.subject_class,
        "property_name":      ax.property_name,
        "filler":             ax.filler,
        "cardinality":        ax.cardinality,
        "axiom_str":          ax.axiom_str,
        "definition":         ax.definition,
        "confidence":         ax.confidence,
        "reasoning":          ax.reasoning,
        "evidence_sentences": ax.evidence_sentences[:5],
        "source_cluster":     ax.source_cluster,
    }


# Keep backward-compatible public name
serialize = _serialize


def write_manchester_summary(axioms: list[FormalAxiom], output_dir: Path) -> None:
    lines = ["# Induced TBox Axioms — Manchester Syntax Summary", ""]

    CONSTRUCT_ORDER = [
        "owl:someValuesFrom",
        "owl:allValuesFrom",
        "owl:exactCardinality",
        "owl:minCardinality",
        "owl:maxCardinality",
        "owl:disjointWith",
    ]

    by_construct: dict[str, list[FormalAxiom]] = defaultdict(list)
    for ax in axioms:
        by_construct[ax.owl_construct].append(ax)

    conf_marker = {"high": "✓", "medium": "~"}

    for construct in CONSTRUCT_ORDER:
        group = by_construct.get(construct, [])
        if not group:
            continue
        lines.append(f"## {construct}  ({len(group)} axioms)")
        lines.append("")
        for ax in sorted(group, key=lambda a: (a.subject_class or "", a.confidence)):
            marker = conf_marker.get(ax.confidence, "~")
            lines.append(f"  [{marker}] {ax.axiom_str}")
            if ax.definition:
                lines.append(f"      # {ax.definition}")
        lines.append("")

    path = output_dir / "axioms_manchester_summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Written: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stage 6 — Axiom Induction (revised)")

    stage4_default = str(Path(cfg.STAGE4_OUT_DIR) / "property_clusters_with_dr.json")
    stage5_default = str(Path(getattr(cfg, "STAGE5_OUT_DIR", "./stage5_out")) / "class_hierarchy.json")
    stage6_default = getattr(cfg, "STAGE6_OUT_DIR", "./stage6_out")

    p.add_argument("--stage4_path",    default=stage4_default)
    p.add_argument("--stage5_path",    default=stage5_default)
    p.add_argument("--output_dir",     default=stage6_default)
    p.add_argument("--model",          default=cfg.MODEL_NAME)
    p.add_argument("--vllm_base_url",  default=cfg.API_BASE)
    p.add_argument("--prop_threshold", type=float, default=PROP_MATCH_THRESHOLD,
                   help="Jaccard threshold for fuzzy property matching (default: 0.35)")
    p.add_argument("--skip_llm", action="store_true",
                   help="Skip LLM formalization — output raw candidates for inspection.")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global PROP_MATCH_THRESHOLD
    PROP_MATCH_THRESHOLD = args.prop_threshold

    # ── 6.1  Load ──────────────────────────────────────────────────────────
    print("\n[6.1] Loading inputs ...")
    stage4    = load_stage4(Path(args.stage4_path))
    hierarchy = load_stage5(Path(args.stage5_path))

    class_vocab    = build_class_vocab(stage4, hierarchy)
    property_vocab = build_property_vocab(stage4)
    class_vocab_set    = set(class_vocab)
    property_vocab_set = set(property_vocab)

    # Build hierarchy maps once — reused by prompt builder and coherence check
    class_parents  = build_class_parents(hierarchy)
    ancestor_sets  = build_ancestor_sets(class_parents)

    print(f"  Classes   : {len(class_vocab)}")
    print(f"  Properties: {len(property_vocab)}")
    print(f"  Hierarchy nodes: {len(class_parents)}")

    # ── 6.2  Build candidates ──────────────────────────────────────────────
    print("\n[6.2] Building axiom candidates ...")
    candidates = build_candidates(stage4, class_vocab_set, property_vocab, property_vocab_set)

    # ── 6.3  Filter ────────────────────────────────────────────────────────
    print("\n[6.3] Filtering candidates ...")
    kept, dropped = filter_candidates(candidates, class_vocab_set)

    import dataclasses
    with open(output_dir / "stage6_dropped_candidates.json", "w") as f:
        json.dump([dataclasses.asdict(c) for c in dropped], f, indent=2)

    if args.skip_llm:
        cand_path = output_dir / "stage6_candidates.json"
        with open(cand_path, "w") as f:
            json.dump([dataclasses.asdict(c) for c in kept], f, indent=2)
        print(f"\n[skip_llm] Written {len(kept)} candidates to {cand_path}")
        return

    # ── 6.4  LLM formalization ─────────────────────────────────────────────
    print(f"\n[6.4] Formalizing {len(kept)} candidates ...")
    client = OpenAI(base_url=args.vllm_base_url, api_key="EMPTY")
    axioms, review_items = formalize(
        kept, client, args.model, class_vocab, property_vocab, class_parents
    )

    with open(output_dir / "stage6_review.json", "w") as f:
        json.dump(review_items, f, indent=2)
    print(f"  Written: stage6_review.json  ({len(review_items)} items for human review)")

    # ── 6.5  Coherence check ───────────────────────────────────────────────
    print("\n[6.5] Coherence check ...")
    axioms, flagged_disjoint = coherence_check(
        axioms, class_vocab_set, property_vocab_set, ancestor_sets
    )

    # Write flagged disjointness axioms for inspection
    with open(output_dir / "stage6_disjoint_flagged.json", "w") as f:
        json.dump([_serialize(a) for a in flagged_disjoint], f, indent=2)
    print(f"  Written: stage6_disjoint_flagged.json  ({len(flagged_disjoint)} axioms)")

    # ── 6.6  Serialize ─────────────────────────────────────────────────────
    print("\n[6.6] Writing output ...")
    serialized = [_serialize(a) for a in axioms]

    with open(output_dir / "induced_axioms.json", "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"  Written: induced_axioms.json  ({len(serialized)} axioms)")

    write_manchester_summary(axioms, output_dir)

    conf_dist   = Counter(a.confidence    for a in axioms)
    cons_dist   = Counter(a.owl_construct for a in axioms)
    family_dist = Counter(a.axiom_family  for a in axioms)

    stats = {
        "total_axioms":                   len(axioms),
        "review_items":                   len(review_items),
        "flagged_disjoint_axioms":        len(flagged_disjoint),
        "candidates_built":               len(candidates),
        "candidates_kept":                len(kept),
        "candidates_dropped":             len(dropped),
        "confidence_distribution":        dict(conf_dist),
        "construct_distribution":         dict(cons_dist),
        "axiom_family_distribution":      dict(family_dist),
    }
    with open(output_dir / "stage6_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Stage 6 complete ===")
    print(f"  Axioms committed         : {stats['total_axioms']}")
    print(f"  Queued for review        : {stats['review_items']}")
    print(f"  Disjointness flagged     : {stats['flagged_disjoint_axioms']}")
    print(f"  By construct:")
    for construct, count in sorted(cons_dist.items(), key=lambda x: -x[1]):
        print(f"    {construct:35s}: {count}")
    print(f"  By axiom family:")
    for family, count in sorted(family_dist.items(), key=lambda x: -x[1]):
        print(f"    {family:35s}: {count}")


if __name__ == "__main__":
    main()