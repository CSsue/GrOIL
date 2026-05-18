"""
config.py — central path & settings loader for the ontology-construction pipeline.

All scripts do:
    from config import cfg

and then use e.g. cfg.STAGE1_OUT_DIR, cfg.GRAPHS_STEP2_DIR, etc.

Values are read from the .env file (searched upward from this file's location)
and then from the environment.  Every key has a sensible default so the scripts
remain runnable even without a .env present.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env starting from this file's directory, then parent dirs
_here = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_here / ".env", override=False)
# Also try one and two levels up (common project layouts)
for _candidate in (_here.parent / ".env", _here.parent.parent / ".env"):
    if _candidate.exists():
        load_dotenv(dotenv_path=_candidate, override=False)
        break


def _e(key: str, default: str) -> str:
    """Return env var or default, with ~ expansion."""
    return os.path.expandvars(os.environ.get(key, default))


class _Config:
    API_BASE    = _e("API_BASE",    "http://localhost:8000/v1")
    MODEL_NAME  = _e("MODEL_NAME",  "")
    MAX_TOKENS  = int(_e("MAX_TOKENS", "4096"))

    DATA_DIR = _e("DATA_DIR", "data")
    OUT_DIR  = _e("OUT_DIR",  "out")

    DATASET = _e("DATASET", "data/10_contracts_dataset")

    GRAPHS_FULL_DIR  = _e("GRAPHS_FULL_DIR", "")
    GRAPHS_STEP1_DIR = _e("GRAPHS_STEP1_DIR", "")
    GRAPHS_STEP2_DIR = _e("GRAPHS_STEP2_DIR", "")

    ANALYSIS_REPORT_FULL  = _e("ANALYSIS_REPORT_FULL",  "")
    ANALYSIS_REPORT_STEP1 = _e("ANALYSIS_REPORT_STEP1", "")
    ANALYSIS_REPORT_STEP2 = _e("ANALYSIS_REPORT_STEP2", "")

    SCORE_PLOT_FULL  = _e("SCORE_PLOT_FULL",  "")
    SCORE_PLOT_STEP1 = _e("SCORE_PLOT_STEP1", "")
    SCORE_PLOT_STEP2 = _e("SCORE_PLOT_STEP2", "")

    STEP1_REPORT     = _e("STEP1_REPORT",     "")
    STEP2_REPORT     = _e("STEP2_REPORT",     "")
    STEP2_DECISIONS  = _e("STEP2_DECISIONS",  "")

    ONTOLOGY_BASE_DIR = _e("ONTOLOGY_BASE_DIR", "")

    STAGE1_OUT_DIR = _e("STAGE1_OUT_DIR", "")
    STAGE2_OUT_DIR = _e("STAGE2_OUT_DIR", "")
    STAGE3_OUT_DIR = _e("STAGE3_OUT_DIR", "")
    STAGE4_OUT_DIR = _e("STAGE4_OUT_DIR", "")
    STAGE5_OUT_DIR = _e("STAGE5_OUT_DIR", "")
    STAGE6_OUT_DIR = _e("STAGE6_OUT_DIR", "")
    STAGE7_OUT_DIR = _e("STAGE7_OUT_DIR", "")
    ABOX_OUT_DIR = _e("ABOX_OUT_DIR", "")

    @property
    def STAGE2_CANDIDATE_POOL(self) -> str:
        return str(Path(self.STAGE2_OUT_DIR) / "candidate_pool.json")

    @property
    def STAGE3_CLUSTERS(self) -> str:
        return str(Path(self.STAGE3_OUT_DIR) / "named_concept_clusters.json")


cfg = _Config()