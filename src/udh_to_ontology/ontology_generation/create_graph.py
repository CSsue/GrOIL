import os
import queue
import subprocess
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from sentence_transformers import SentenceTransformer

from src.dispergraph_to_ontology.dispergraph.deduplication.entity_deduplication import GraphDeduplicator
from src.dispergraph_to_ontology.dispergraph.index.construction.construct_edges import (
    construct_all_edges,
    construct_edges_during_merge,
)
from src.dispergraph_to_ontology.dispergraph.index.construction.construct_nodes import initialize_graph_from_text
from src.dispergraph_to_ontology.dispergraph.models.Graph import Graph
from src.dispergraph_to_ontology.dispergraph.utils.config import configure_dspy

warnings.filterwarnings("ignore")

_DEDUP_DONE = object()


# ──────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ──────────────────────────────────────────────────────────────────────────────

def _query_gpu_free_mib() -> dict[int, int]:
    """
    Returns {gpu_index: free_mib} for all GPUs via nvidia-smi.
    Uses strip() on each field so the result is robust to any whitespace
    variation in the CSV output (e.g. "0, 5000" or "0,5000").
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    out = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(",")
        out[int(parts[0].strip())] = int(parts[1].strip())
    return out


def pick_embedding_device(min_free_mib: int = 4_000) -> str:
    """
    Picks the GPU with the most free memory above min_free_mib.
    Falls back to CPU gracefully if nvidia-smi fails or no GPU qualifies.

    Why sort by most-free instead of hardcoding cuda:1?
    VLLM reserves memory upfront on its GPU, so that GPU naturally has less
    free memory and loses the sort — no hardcoding needed, and this works
    correctly even if GPU assignments change between runs.
    """
    try:
        free_by_gpu = _query_gpu_free_mib()
        candidates = [
            (free_mib, idx)
            for idx, free_mib in free_by_gpu.items()
            if free_mib >= min_free_mib
        ]
        if not candidates:
            print("  [gpu] No GPU with enough free memory — falling back to CPU")
            return "cpu"
        candidates.sort(reverse=True)          # most free memory first
        free_mib, chosen = candidates[0]
        print(f"  [gpu] Embedding device: cuda:{chosen} ({free_mib} MiB free)")
        return f"cuda:{chosen}"
    except Exception as e:
        print(f"  [gpu] nvidia-smi query failed ({e}) — falling back to CPU")
        return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint path helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_stem(filepath: str, base_dir: str) -> str:
    """
    Creates a checkpoint stem from just the filename (no subdirectory prefix).
    Note: if two subdirectories contain files with the same name, they will
    collide. Ensure filenames are unique across the dataset.
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def full_ckpt_path(checkpoint_dir: str, filename: str) -> str:
    return os.path.join(checkpoint_dir, "full", filename)

def dedup_ckpt_path(checkpoint_dir: str, filename: str) -> str:
    return os.path.join(checkpoint_dir, "deduplicated", filename)

def merged_ckpt_path(checkpoint_dir: str, filename: str) -> str:
    return os.path.join(checkpoint_dir, "merged", filename)


# ──────────────────────────────────────────────────────────────────────────────
# Graph I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_graph(graph: Graph, path: str) -> None:
    """
    Atomic save: write to <path>.tmp then rename to <path>.

    os.replace() is rename(2) on Linux — a single syscall. A crash at any
    point leaves either the previous valid checkpoint or the new complete file,
    never a partial write.

    Thread safety: safe to call concurrently from multiple threads as long as
    each thread writes to a unique path (guaranteed by _make_stem uniqueness).
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp = path + ".tmp"
    graph.export_graph(tmp)
    os.replace(tmp, path)
    print(
        f"  [saved]  {path} | "
        f"nodes={graph.network.number_of_nodes()} "
        f"edges={graph.network.number_of_edges()} "
        f"entities={len(graph.entities)}"
    )


def load_graph(path: str) -> Graph:
    g = Graph()
    g.load_graph(path)
    print(
        f"  [loaded] {path} | "
        f"nodes={g.network.number_of_nodes()} "
        f"edges={g.network.number_of_edges()} "
        f"entities={len(g.entities)}"
    )
    return g


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def process_text(
    text: str,
    chunk_size: int,
    overlap: int,
    edge_threshold: float,
    max_workers: int,
) -> Graph:
    """
    Two phases:
      1. initialize_graph_from_text  — LLM calls via VLLM (sequential per chunk)
      2. construct_all_edges         — CPU/embedding similarity (parallel, max_workers)
    max_workers applies only to phase 2.
    """
    graph = initialize_graph_from_text(text, chunk_size=chunk_size, overlap=overlap)
    construct_all_edges(graph, threshold=edge_threshold, max_workers=max_workers)
    return graph


def _merge_graphs(graphs: list[Graph], edge_threshold: float, max_workers: int) -> Graph:
    merged = Graph()
    for g in graphs:
        merged.merge(g)
    construct_edges_during_merge(merged, threshold=edge_threshold, max_workers=max_workers)
    merged.set_graph(merged.network)
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ──────────────────────────────────────────────────────────────────────────────
#
# Architecture:
#
#   Files ──▶ [ThreadPoolExecutor, N workers] ──dedup_queue──▶ [NUM_DEDUP_WORKERS dedup threads]
#                 LLM via VLLM (GPU 0)                          each with its own SentenceTransformer
#                                                               (GPU with most free mem)
#
# Termination — sentinel pattern (provably race-free):
#
#   1. Pre-seed dedup_queue with dedup-only items.
#   2. Start dedup threads (begin working on pre-seeded items immediately).
#   3. Run extraction via ThreadPoolExecutor; each success puts item on queue.
#   4. executor.__exit__() is a hard barrier: all futures resolved before we proceed.
#   5. Push one _DEDUP_DONE sentinel per dedup worker onto queue.
#   6. Each dedup thread drains remaining items, hits its sentinel, exits.
#   7. Main thread joins all dedup threads.
#
# Key design decisions:
#   • Each dedup worker owns its own SentenceTransformer instance — sharing a
#     single model across threads serializes inference at the PyTorch level,
#     negating all parallelism. Per-worker models run truly concurrently.
#   • completed counter is protected by a lock — multiple dedup threads
#     increment it concurrently so += is not safe without one.
#   • Sentinel is pushed only after the executor barrier, so no extraction
#     thread can add items after it — provably race-free termination.
#
# Concurrency sizing (tune based on VLLM logs):
#   • Target: Running≈N, Waiting=0 in VLLM logs.
#   • num_extraction_workers + NUM_DEDUP_WORKERS combined drive VLLM load.
#   • max_workers_per_file applies to construct_all_edges (CPU/embedding, not VLLM).
#     Keep at 4 — intra-chunk comparisons have thousands of items but are
#     bottlenecked by GPU throughput, not thread count.
# ──────────────────────────────────────────────────────────────────────────────

def build_index_from_directory(
        directory: str,
        chunk_size: int = 500,
        overlap: int = 50,
        edge_threshold: float = 0.4,
        num_extraction_workers: int = 10,  # concurrent files → drives VLLM load
        max_workers_per_file: int = 4,     # workers inside construct_all_edges (CPU/embed)
        num_dedup_workers: int = 5,        # concurrent dedup threads, each with own model
        checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Processes every .txt file in *directory* into a full + deduplicated graph.

    All writes are atomic checkpoints. The run can be interrupted (Ctrl-C,
    crash, OOM) and resumed at any point without re-doing completed work:
      - dedup checkpoint exists        → skip entirely
      - only full checkpoint exists    → skip extraction, go straight to dedup
      - neither checkpoint exists      → extract then dedup
    """

    # ── Pick embedding device at runtime ─────────────────────────────────────
    # Only used to determine which GPU to target — each worker creates its own
    # model instance to avoid serialization on shared PyTorch model state.
    embed_device = pick_embedding_device(min_free_mib=4_000)

    # ── Discover .txt files ───────────────────────────────────────────────────
    filepaths: list[str] = []
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            if filename.endswith(".txt"):
                filepaths.append(os.path.join(root, filename))

    if not filepaths:
        print(f"No .txt files found in {directory}")
        return

    # ── Classify each file by its checkpoint state ────────────────────────────
    #   Category A — dedup checkpoint exists      → skip entirely
    #   Category B — full checkpoint exists only  → push straight to dedup_queue
    #   Category C — no checkpoint at all         → extract then dedup
    needs_extraction: list[tuple] = []   # (filepath, stem, full_path, dedup_path)
    needs_dedup_only: list[tuple] = []   # (stem, full_path, dedup_path)
    already_done = 0

    for fp in filepaths:
        stem       = _make_stem(fp, directory)
        full_path  = full_ckpt_path(checkpoint_dir, f"{stem}.graphml")
        dedup_path = dedup_ckpt_path(checkpoint_dir, f"{stem}.graphml")

        if os.path.exists(dedup_path):
            already_done += 1
        elif os.path.exists(full_path):
            needs_dedup_only.append((stem, full_path, dedup_path))
        else:
            needs_extraction.append((fp, stem, full_path, dedup_path))

    total_pending = len(needs_extraction) + len(needs_dedup_only)
    print(
        f"\nFiles: {already_done} already done | "
        f"{len(needs_extraction)} need extraction | "
        f"{len(needs_dedup_only)} need dedup only\n"
    )

    if total_pending == 0:
        print("All files already processed.")
        return

    # ── Shared queue and thread-safe counter ──────────────────────────────────
    dedup_queue: queue.Queue = queue.Queue()
    completed = [0]
    completed_lock = threading.Lock()

    # Pre-seed Category B files so dedup threads start immediately.
    for item in needs_dedup_only:
        dedup_queue.put(item)

    # ── Dedup workers ─────────────────────────────────────────────────────────
    # Each worker creates its own SentenceTransformer and GraphDeduplicator.
    # Sharing a single model across threads serializes PyTorch inference —
    # per-worker instances allow true concurrent GPU utilization.

    print(f"  [dedup] Loading {num_dedup_workers} embedding models...")
    dedup_models = [
        SentenceTransformer("all-MiniLM-L6-v2", device=embed_device)
        for _ in range(num_dedup_workers)
    ]
    def dedup_worker(local_model) -> None:
        # Per-worker model — avoids serialization on shared PyTorch state.
        local_deduplicator = GraphDeduplicator(retrieval_model=local_model)
        worker_id = threading.current_thread().name
        worker_id = threading.current_thread().name

        while True:
            item = dedup_queue.get()          # blocks — no busy-polling, no timeout

            if item is _DEDUP_DONE:           # sentinel: nothing more will ever arrive
                break

            stem, full_path, dedup_path = item
            print(f"  [{worker_id}] [{stem}] Deduplicating...")
            try:
                graph       = load_graph(full_path)
                dedup_graph = local_deduplicator.deduplicate(graph, max_workers=4)
                save_graph(dedup_graph, dedup_path)
            except Exception as e:
                print(f"  [{worker_id}] [{stem}] ERROR during dedup: {e}")
            finally:
                # Lock required: multiple dedup threads increment concurrently.
                with completed_lock:
                    completed[0] += 1
                print(f"  Progress: {completed[0]}/{total_pending} complete")

    dedup_threads = [
        threading.Thread(target=dedup_worker, args=(dedup_models[i],), daemon=True, name=f"dedup-{i}")
        for i in range(num_dedup_workers)
    ]
    for t in dedup_threads:
        t.start()

    # ── Extraction workers ────────────────────────────────────────────────────
    # ThreadPoolExecutor over raw threads because:
    #   • Creates exactly num_extraction_workers threads (not one per file).
    #   • executor.__exit__ is a hard synchronisation barrier — all futures are
    #     resolved before execution reaches the line after the `with` block.
    #     This is what makes the sentinel push race-free.

    def extract_one(fp: str, stem: str, full_path: str, dedup_path: str) -> None:
        print(f"\n  [{stem}] Extracting...")
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        graph = process_text(text, chunk_size, overlap, edge_threshold, max_workers_per_file)
        save_graph(graph, full_path)
        # Push ONLY after successful atomic save — dedup workers read from disk,
        # so the file must be complete and renamed before the push.
        dedup_queue.put((stem, full_path, dedup_path))

    with ThreadPoolExecutor(max_workers=num_extraction_workers) as executor:
        future_to_stem = {
            executor.submit(extract_one, *args): args[1]
            for args in needs_extraction
        }
        for future in as_completed(future_to_stem):
            exc = future.exception()
            if exc:
                stem = future_to_stem[future]
                print(f"  [{stem}] ERROR during extraction: {exc}")
    # ← Hard barrier: every extraction future is resolved (success or error).
    #   No worker thread can put anything onto dedup_queue after this line.

    # Signal each dedup worker to stop after draining remaining items.
    for _ in range(num_dedup_workers):
        dedup_queue.put(_DEDUP_DONE)
    for t in dedup_threads:
        t.join()

    print(f"\nPer-file processing complete. ({completed[0]}/{total_pending} processed)")


# ──────────────────────────────────────────────────────────────────────────────
# Merge from checkpoints
# ──────────────────────────────────────────────────────────────────────────────

def merge_from_checkpoints(
        checkpoint_dir: str,
        edge_threshold: float = 0.4,
        max_workers: int = 10,
) -> tuple[Graph | None, Graph | None]:
    """
    Reads per-file checkpoints and produces two merged graphs:
      Path A — merge all full graphs  → deduplicate → merged_full_deduplicated.graphml
      Path B — merge all dedup graphs → deduplicate → merged_dedup_deduplicated.graphml

    Both outputs are saved atomically to checkpoints/merged/.
    Returns (None, None) if checkpoints are missing or empty.
    """
    embed_device    = pick_embedding_device(min_free_mib=4_000)
    retrieval_model = SentenceTransformer("all-MiniLM-L6-v2", device=embed_device)
    deduplicator    = GraphDeduplicator(retrieval_model=retrieval_model)

    full_dir  = os.path.join(checkpoint_dir, "full")
    dedup_dir = os.path.join(checkpoint_dir, "deduplicated")

    if not os.path.exists(full_dir) or not os.path.exists(dedup_dir):
        print("No checkpoints found. Run 'build' first.")
        return None, None

    full_graphml_files  = [fn for fn in sorted(os.listdir(full_dir))  if fn.endswith(".graphml")]
    dedup_graphml_files = [fn for fn in sorted(os.listdir(dedup_dir)) if fn.endswith(".graphml")]

    # Guard: refuse to merge if either set is empty — merging an empty list
    # produces an empty Graph which would be silently saved as a valid checkpoint,
    # poisoning all future merge runs that load it from disk.
    if not full_graphml_files:
        print("No full graph checkpoints found — run 'build' first.")
        return None, None
    if not dedup_graphml_files:
        print("No deduplicated graph checkpoints found — run 'build' first.")
        return None, None

    full_graphs  = [load_graph(os.path.join(full_dir,  fn)) for fn in full_graphml_files]
    dedup_graphs = [load_graph(os.path.join(dedup_dir, fn)) for fn in dedup_graphml_files]

    print(f"\nLoaded {len(full_graphs)} full + {len(dedup_graphs)} deduplicated graphs.\n")

    # ── Path A: merge full graphs → deduplicate ───────────────────────────────

    path_a = os.path.join(os.path.dirname(checkpoint_dir), "merged", "merged_full_deduplicated.graphml")
    os.makedirs(os.path.dirname(path_a), exist_ok=True)

    if os.path.exists(path_a):
        print("[Path A] Checkpoint found — loading...")
        merged_from_full = load_graph(path_a)
    else:
        print(f"[Path A] Merging {len(full_graphs)} full graphs...")
        merged_full      = _merge_graphs(full_graphs, edge_threshold, max_workers)
        print("[Path A] Deduplicating...")
        merged_from_full = deduplicator.deduplicate(merged_full, max_workers=max_workers)
        save_graph(merged_from_full, path_a)

    # ── Path B: merge deduplicated graphs → deduplicate ───────────────────────
    path_b = os.path.join(os.path.dirname(checkpoint_dir), "merged", "merged_dedup_deduplicated.graphml")
    os.makedirs(os.path.dirname(path_b), exist_ok=True)

    if os.path.exists(path_b):
        print("\n[Path B] Checkpoint found — loading...")
        merged_from_dedup = load_graph(path_b)
    else:
        print(f"\n[Path B] Merging {len(dedup_graphs)} deduplicated graphs...")
        merged_dedup      = _merge_graphs(dedup_graphs, edge_threshold, max_workers)
        print("[Path B] Deduplicating...")
        merged_from_dedup = deduplicator.deduplicate(merged_dedup, max_workers=max_workers)
        save_graph(merged_from_dedup, path_b)

    print("\nMerging complete.")
    return merged_from_full, merged_from_dedup


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["build", "merge"],
        help="'build' to process files, 'merge' to merge checkpoints",
    )
    args = parser.parse_args()

    configure_dspy(temperature=0.1, max_tokens=20000)
    litellm.request_timeout = 120

    # input_dir      = "data/10_contracts_dataset/contracts"
    # checkpoint_dir = "data/10_contracts_dataset/graphs/checkpoints"

    input_dir      = "data/abid_et_al/contracts"
    checkpoint_dir = "data/abid_et_al/graphs/checkpoints"

    if args.mode == "build":
        build_index_from_directory(
            directory=input_dir,
            chunk_size=500,
            overlap=50,
            edge_threshold=0.4,
            num_extraction_workers=5,
            max_workers_per_file=10,
            num_dedup_workers=3,
            checkpoint_dir=checkpoint_dir,
        )

    elif args.mode == "merge":
        merged_from_full, merged_from_dedup = merge_from_checkpoints(
            checkpoint_dir=checkpoint_dir,
            edge_threshold=0.4,
            max_workers=60,
        )
        if merged_from_full is not None:
            print(
                f"\nPath A (full → dedup)  | "
                f"nodes={merged_from_full.network.number_of_nodes()} "
                f"edges={merged_from_full.network.number_of_edges()} "
                f"entities={len(merged_from_full.entities)}"
            )
            print(
                f"Path B (dedup → dedup) | "
                f"nodes={merged_from_dedup.network.number_of_nodes()} "
                f"edges={merged_from_dedup.network.number_of_edges()} "
                f"entities={len(merged_from_dedup.entities)}"
            )