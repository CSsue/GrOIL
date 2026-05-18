"""
Ontology Coverage Analysis
==========================
Evaluates how well an OWL ontology (TTL) covers the vocabulary of a corpus of insurance contracts.
"""

import os
import glob
import re
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from rdflib import Graph, RDFS, OWL, RDF

from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz


def load_contracts(directory: str) -> dict[str, str]:
    contracts = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        name = Path(path).stem
        with open(path, "r", encoding="utf-8") as f:
            contracts[name] = f.read()
    print(f"Loaded {len(contracts)} contracts: {list(contracts.keys())}")
    return contracts


def extract_keyphrases(phrases_path: str) -> list[str]:
    with open(phrases_path, "r", encoding="utf-8") as f:
        terms_by_contract = json.load(f)
    
    keyphrases = []
    seen = set()
    for contract_terms in terms_by_contract.values():
        for term in contract_terms:
            term = term.lower().strip()
            if term not in seen:
                keyphrases.append(term)
                seen.add(term)
    
    print(f" Extracted {len(keyphrases)} unique keyphrases from the contracts")
    return keyphrases


def load_ontology_labels(ttl_path: str) -> set[str]:
    """
    Extract human-readable labels from the ontology.
    """
    g = Graph()
    g.parse(ttl_path, format="turtle")
    labels = set()

    def camel_to_words(text: str) -> str:
        s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        s2 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s1)
        return s2.replace("_", " ").strip().lower()

    for _, _, label in g.triples((None, RDFS.label, None)):
        labels.add(str(label).lower().strip())

    for rdf_type in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty,
                     OWL.NamedIndividual, RDF.Property):
        for subj, _, _ in g.triples((None, RDF.type, rdf_type)):
            local = str(subj).split("#")[-1].split("/")[-1]
            if local:
                labels.add(camel_to_words(local))

    print(f"Loaded {len(labels)} ontology labels/concepts")
    return labels


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


def map_keyphrases(keyphrases: list[str],
                   ontology_labels: set[str],
                   sem_model: SentenceTransformer,
                   ontology_embeddings,
                   ont_labels_list: list[str]) -> pd.DataFrame:
    norm_labels = {normalize(l) for l in ontology_labels}
    rows = []
    
    for phrase in keyphrases:
        norm_phrase = normalize(phrase)
        phrase_words = set(norm_phrase.split())

        matched_labels_substring = sorted(
            [
                lbl for lbl in norm_labels
                if all(word in set(lbl.split()) for word in phrase_words)
            ],
            key=lambda x: len(x.split())
        )
        
        matched_labels_fuzzy = [lbl for lbl in norm_labels if fuzz.token_sort_ratio(norm_phrase, lbl) >= 70]
        
        phrase_emb = sem_model.encode(phrase, convert_to_tensor=True)
        sim_scores = util.cos_sim(phrase_emb, ontology_embeddings)[0]

        matched_indices = (sim_scores >= 0.70).nonzero(as_tuple=True)[0]
        matched_labels_semantic = []
        if len(matched_indices) > 0:
            matched_labels_semantic = sorted(
                [(ont_labels_list[i], float(sim_scores[i])) for i in matched_indices],
                key=lambda x: x[1], 
                reverse=True
            )

        is_substring = len(matched_labels_substring) > 0
        is_fuzzy = len(matched_labels_fuzzy) > 0
        is_semantic = len(matched_labels_semantic) > 0

        match_label = None
        if is_substring: match_label = matched_labels_substring[0]
        elif is_fuzzy: match_label = matched_labels_fuzzy[0]
        elif is_semantic: match_label = matched_labels_semantic[0][0]

        rows.append({
            "keyphrase": phrase,
            "is_substring": is_substring,
            "is_fuzzy": is_fuzzy and not is_substring,
            "is_semantic": is_semantic and not (is_substring or is_fuzzy),
            "mapped": is_substring or is_fuzzy or is_semantic,
            "matched_ontology_label": match_label
        })

    df = pd.DataFrame(rows)
    return df


def get_mapping_examples(df: pd.DataFrame) -> dict:
    examples = {}
    def format_tuple(row):
        return f"('{row['keyphrase']}',\n'{row['matched_ontology_label']}')"

    sub = df[df["is_substring"] == True]
    if not sub.empty: examples['substring'] = format_tuple(sub.iloc[min(4, len(sub)-1)])
    
    fuz = df[df["is_fuzzy"] == True]
    if not fuz.empty: examples['fuzzy'] = format_tuple(fuz.iloc[min(24, len(fuz)-1)])

    sem = df[df["is_semantic"] == True]
    if not sem.empty: examples['semantic'] = format_tuple(sem.iloc[min(7, len(sem)-1)])

    unm = df[df["mapped"] == False]
    if not unm.empty: examples['unmapped'] = f"Unmapped:\n'{unm.iloc[min(1, len(unm)-1)]['keyphrase']}'"

    return examples


# Modified to accept a specific Matplotlib axis (ax)
def plot_ontology_coverage_subplot(df: pd.DataFrame, dataset: str, ax: plt.Axes):
    exact_count = df["is_substring"].sum()
    fuzzy_count = df["is_fuzzy"].sum()
    semantic_count = df["is_semantic"].sum()
    total_mapped = df["mapped"].sum()
    unmapped = len(df) - total_mapped
    total_phrases = len(df)
    coverage_pct = (total_mapped / total_phrases) * 100 if total_phrases > 0 else 0

    categories = ['Exact / \nSubstring', 'Fuzzy \nMatch', 'Semantic \n(cosine≥0.70)', 'Total \nMapped', 'Unmapped']
    counts = [exact_count, fuzzy_count, semantic_count, total_mapped, unmapped]
    colors = ['#64a5f2', '#7db973', '#a452bd', '#3262b9', '#de6255']

    # Draw barplot onto the specific sub-axis
    sns.barplot(x=categories, y=counts, palette=colors, width=0.8, ax=ax)

    # Dynamic step sizing for annotations based on dataset scale
    y_buffer = max(5, total_phrases * 0.03)

    for i, v in enumerate(counts):
        ax.text(i, v + y_buffer, str(v), ha='center', fontweight='bold', fontsize=16)

    ax.axhline(y=total_phrases, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(4.2, total_phrases + y_buffer, f"{total_phrases} total", color='grey', fontsize=15)

    # Placing coverage badge dynamically above the 'Total Mapped' bar
    ax.text(3, total_mapped + (y_buffer * 3), f'{coverage_pct:.1f}% Coverage',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f0f7ff', ec='#3262b9', lw=1),
            fontsize=16, color='#3262b9', fontweight='bold', ha='center')

    ax.set_title(dataset, fontsize=20, fontweight='bold', pad=15)
    ax.set_xticklabels(categories, rotation=30, ha='right', fontsize=16)
    ax.set_ylabel("Number of Keyphrases", fontsize=16)
    ax.set_ylim(0, total_phrases + (y_buffer * 6)) 
    sns.despine(ax=ax)


# --- MAIN PIPELINE EXECUTIVE LOOP ---

DATASETS = [
    "abid_et_al_sentinel",
    "abid_et_al_equivita",
    "10_contracts_dataset",
    "100_contracts_dataset"
]

# Set style globally
sns.set_style("ticks")

# Create a 2x2 grid figure
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes_flat = axes.flatten()

# Initialize model once
sem_model = SentenceTransformer("all-MiniLM-L6-v2")

for idx, dataset in enumerate(DATASETS):
    print(f"\nProcessing Dataset {idx+1}/{len(DATASETS)}: {dataset}...")
    
    EXTRACTED_PHRASES_PATH = f"data/{dataset}/extracted_keyphrases.json"
    ONTOLOGY_PATH = f"out/ontology_construction/{dataset}/stage7_serialized_ontology/tbox.ttl"
    
    if not os.path.exists(EXTRACTED_PHRASES_PATH) or not os.path.exists(ONTOLOGY_PATH):
        print(f"Skipping {dataset} - Target files do not exist.")
        continue
        
    keyphrases = extract_keyphrases(EXTRACTED_PHRASES_PATH)
    ontology_labels = load_ontology_labels(ONTOLOGY_PATH)
    
    ontology_embeddings = sem_model.encode(list(ontology_labels), convert_to_tensor=True, show_progress_bar=False)
    ont_labels_list = list(ontology_labels)
    
    results_df = map_keyphrases(keyphrases, ontology_labels, sem_model, ontology_embeddings, ont_labels_list)
    
    # Generate the plot directly onto its designated quadrant grid location
    plot_ontology_coverage_subplot(results_df, dataset, axes_flat[idx])

# Final configuration cleanup and saving the combined canvas
plt.tight_layout()
os.makedirs("out/ontology_coverage", exist_ok=True)
plt.savefig("out/ontology_coverage/ontology_coverage_combined_2x2.png", dpi=300, bbox_inches="tight")
plt.show()