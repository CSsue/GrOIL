"""
Full Ontology Pipeline Runner
==============================
Runs stages 1-7 sequentially, stopping on any failure.

Usage:
    python run_pipeline.py
    python run_pipeline.py --start_from 3      # resume from a specific stage
    python run_pipeline.py --only 1 2          # run only specific stages
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

PATH = "src/udh_to_ontology/ontology_generation/"

STAGES = [
    (1, PATH+"stage1_parse_graph.py"),
    (2, PATH+"stage2_get_candidates.py"),
    (3, PATH+"stage3_dedup_clustering.py"),
    (4, PATH+"stage4_set_domain_range.py"),
    (5, PATH+"stage5_set_hierarchy.py"),
    (6, PATH+"stage6_induce_axioms.py"),
    (7, PATH+"stage7_serialize_tbox.py"),
]


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_stage(stage_num: int, script: str) -> tuple[bool, float]:
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}: {script}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] Stage {stage_num} exited with code {result.returncode} after {fmt_duration(elapsed)}")
        return False, elapsed

    print(f"\n[OK] Stage {stage_num} completed in {fmt_duration(elapsed)}")
    return True, elapsed


def parse_args():
    p = argparse.ArgumentParser(description="Run ontology pipeline stages sequentially")
    p.add_argument("--start_from", type=int, default=1,
                   help="Resume from this stage number (default: 1)")
    p.add_argument("--only", type=int, nargs="+",
                   help="Run only these stage numbers e.g. --only 1 2")
    return p.parse_args()


def main():
    args = parse_args()

    if args.only:
        stages_to_run = [(n, s) for n, s in STAGES if n in args.only]
    else:
        stages_to_run = [(n, s) for n, s in STAGES if n >= args.start_from]

    if not stages_to_run:
        print("No stages selected. Check --start_from or --only arguments.")
        sys.exit(1)

    print(f"\nPipeline starting — {len(stages_to_run)} stage(s) to run")
    print(f"Stages: {[n for n, _ in stages_to_run]}")

    total_start = time.time()
    timings = []

    for stage_num, script in stages_to_run:
        ok, elapsed = run_stage(stage_num, script)
        timings.append((stage_num, script, elapsed, ok))
        if not ok:
            print("\n[PIPELINE ABORTED] Fix the error above and re-run with:")
            print(f"    python run_pipeline.py --start_from {stage_num}")
            sys.exit(1)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    for stage_num, script, elapsed, ok in timings:
        status = "OK" if ok else "FAILED"
        print(f"  Stage {stage_num}: {script:<35} {fmt_duration(elapsed):>10}  [{status}]")
    print(f"  {'TOTAL':<40} {fmt_duration(total_elapsed):>10}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()