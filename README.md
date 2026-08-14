# GrOIL
Graph-Grounded Domain Ontology Induction with Constrained LLM Mediation

---

## Setup

```bash
conda env create -f environment.yml
conda activate groil
```

Create a `.env` file in the project root with the following contents:

```bash
API_BASE="http://localhost:8000/v1"
MODEL_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

# Maximum output tokens for a single extraction (not context window)
MAX_TOKENS=4096

# Dummy key for LiteLLM/OpenAI compatibility
OPENAI_API_KEY="sk-no-key-needed"

DATA_DIR="data"
OUT_DIR="out"

# DATASET="10_contracts_dataset"
DATASET="100_contracts_dataset"
# DATASET="abid_et_al_equivita"
# DATASET="abid_et_al_sentinel"

GRAPHS_FULL_DIR="${DATA_DIR}/${DATASET}/graphs/checkpoints/full"
GRAPHS_STEP1_DIR="${DATA_DIR}/${DATASET}/graphs/checkpoints/step1_cleaned"
GRAPHS_STEP2_DIR="${DATA_DIR}/${DATASET}/graphs/checkpoints/step2_cleaned"

ANALYSIS_REPORT_FULL="${OUT_DIR}/analysis/${DATASET}/analysis_report_full_graphs.json"
ANALYSIS_REPORT_STEP1="${OUT_DIR}/analysis/${DATASET}/analysis_report_step1_cleaned.json"
ANALYSIS_REPORT_STEP2="${OUT_DIR}/analysis/${DATASET}/analysis_report_step2_cleaned.json"

SCORE_PLOT_FULL="${OUT_DIR}/analysis/${DATASET}/score_distribution_full_graphs.png"
SCORE_PLOT_STEP1="${OUT_DIR}/analysis/${DATASET}/score_distribution_step1_cleaned.png"
SCORE_PLOT_STEP2="${OUT_DIR}/analysis/${DATASET}/score_distribution_step2_cleaned.png"

STEP1_REPORT="${OUT_DIR}/cleaning/${DATASET}/step1_cleaning_report.json"
STEP2_REPORT="${OUT_DIR}/cleaning/${DATASET}/step2_cleaning_report.json"
STEP2_DECISIONS="${OUT_DIR}/cleaning/${DATASET}/step2_decisions.json"

ONTOLOGY_BASE_DIR="${OUT_DIR}/ontology_construction/${DATASET}"

STAGE1_OUT_DIR="${ONTOLOGY_BASE_DIR}/stage1_parsed_graphs"
STAGE2_OUT_DIR="${ONTOLOGY_BASE_DIR}/stage2_candidates"
STAGE3_OUT_DIR="${ONTOLOGY_BASE_DIR}/stage3_clusters"
STAGE4_OUT_DIR="${ONTOLOGY_BASE_DIR}/stage4_domain_range"
STAGE5_OUT_DIR="${ONTOLOGY_BASE_DIR}/stage5_hierarchy"
STAGE6_OUT_DIR="${ONTOLOGY_BASE_DIR}/stage6_axioms"
STAGE7_OUT_DIR="${ONTOLOGY_BASE_DIR}/stage7_serialized_ontology"
ABOX_OUT_DIR="${ONTOLOGY_BASE_DIR}/abox"
```

We ran all experiments using [vLLM](https://github.com/vllm-project/vllm) to serve the model locally and [LiteLLM](https://github.com/BerriAI/litellm) as a proxy. Any OpenAI-compatible endpoint will work.

---

## Data

Graphs need to be generated for the 100-contract dataset using `src/udh_to_ontology/ontology_generation/create_graph.py`, for other datasets, the pre-generated and cleaned graphs are available in the repo. Each dataset follows the same structure:

```
data/{DATASET}/
├── contracts/
├── graphs/checkpoints/
│   ├── full/
│   ├── step1_cleaned/
│   └── step2_cleaned/        ← input to the pipeline
├── competency_questions.json
└── extracted_keyphrases.json
```

---

## Reproducibility

### Graph Cleaning (`stage0_clean_graphs.py`)

If starting from `full/` graphs, run cleaning before anything else. The script supports two steps and several modes — run it with `--help` for full usage details.

### Ontology Induction (`stage1` through `stage7`)

Run stages 1–7 in order from `src/udh_to_ontology/ontology_generation/`. Each stage reads from the previous stage's output directory as configured in `.env`. The final TBox (Turtle + RDF/XML) is written to `STAGE7_OUT_DIR`.

If starting from the provided `step2_cleaned/` graphs, you can skip `stage0` and begin directly at `stage1_parse_graph.py`.

### ABox Population (`abox_population.py`)

Run after stage 7. Takes a source GraphML file and contract ID as arguments — run with `--help` for usage details.

#### The scripts are sufficiently documented. That should help if you want to change certain parameters (i.e., threshold for cosine similarity, etc) or do ablations for experimental purposes.

---

## Evaluation

All evaluation scripts are in `src/evaluation/`.

| Script | What it reproduces |
|---|---|
| `ontology_coverage_analysis.py` | Keyphrase coverage results |
| `cq_check.py` | CQ coverage evaluation |
| `mridul_et_al/` | Gap and overlap analysis Benchmark [1] |
| `abid_et_al/` | Automated ontology generation benchmark [2] |
| `growth_curve.py` | Ontology growth curves |
| `contract_diversity.py` | UMAP contract diversity figure |


---

## Switching Datasets

Change the `DATASET` variable in `.env` and re-run. All paths are derived from it automatically.


## Citation

If you use GrOIL in your research, please cite our paper:

```bibtex
@inproceedings{mridul2026groil,
  title     = {GrOIL: Graph-Grounded Domain Ontology Induction with Constrained LLM Mediation},
  author    = {Mridul, Maruf Ahmed and Talukder, Abid and Seneviratne, Oshani},
  year      = {2026},
  publisher = {ACM},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)}
}
```



---

### References
[1] Mridul, M. A., Kapa, R., & Seneviratne, O. (2026). A Benchmark for Gap and Overlap Analysis as a Test of KG Task Readiness. arXiv preprint arXiv:2604.10853.

[2] Talukder, A., Mridul, M. A., & Seneviratne, O. (2026). Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach. arXiv preprint arXiv:2604.23090.
