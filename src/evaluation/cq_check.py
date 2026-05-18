import json
import re
import sys
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt


import rdflib
import rdflib.plugins.sparql.parser

from openai import OpenAI
from config import cfg


# ---------------------------------------------------------------------------
# LLM client (same pattern as stage3)
# ---------------------------------------------------------------------------
client = OpenAI(api_key="dummy", base_url=cfg.API_BASE)


# ---------------------------------------------------------------------------
# Data model (replaces Pydantic — plain dataclass is sufficient here)
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResult:
    justification: str
    score: float


# ---------------------------------------------------------------------------
# Helper: robust LLM call (mirrors call_llm from stage3)
# ---------------------------------------------------------------------------
def call_llm(prompt: str, retries: int = 3) -> Optional[dict]:
    """
    Send *prompt* to the vLLM server and parse the JSON response.
    Returns a dict on success, None after all retries are exhausted.
    """
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=cfg.MODEL_NAME,
                messages=messages,
                max_tokens=512,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [Try {attempt+1}/{retries}] JSON parse error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  [Try {attempt+1}/{retries}] LLM call error: {type(e).__name__} - {e}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Abox resolution
# ---------------------------------------------------------------------------
def resolve_abox_path(contract_file_name: str) -> Path:
    """
    Map a contract file name such as 'contract_1_term_life.txt'
    to its abox directory, e.g. cfg.ABOX_OUT_DIR / 'abox_C1'.

    Convention: the contract ID is extracted from the filename as the
    first token that matches 'contract_<N>' → 'C<N>'.
    So 'contract_1_term_life.txt' → 'C1' → abox dir 'abox_C1'.
    """
    stem = Path(contract_file_name).stem          # e.g. 'contract_1_term_life'
    match = re.match(r"contract_(\d+)", stem, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Cannot derive contract ID from file name '{contract_file_name}'. "
            "Expected pattern: 'contract_<number>_...'."
        )
    contract_id = f"C{match.group(1)}"            # e.g. 'C1'
    abox_dir = Path(cfg.ABOX_OUT_DIR) / f"abox_{contract_id}"
    return abox_dir


def load_abox(abox_dir: Path) -> str:
    """
    Load the single Turtle file inside *abox_dir*.
    The file is expected to be named after the directory, e.g.
    abox_C1/abox_C1.ttl.
    """
    ttl_file = f"{abox_dir}.ttl"
    if not Path(ttl_file).exists():
        raise FileNotFoundError(
            f"Expected abox file not found: {ttl_file}"
        )
    print(f"    Loading abox file: {Path(ttl_file).name}")
    return Path(ttl_file).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# SPARQL execution (no LLM — query is always pre-supplied)
# ---------------------------------------------------------------------------
def execute_sparql(sparql_query: str, ontology_content: str) -> tuple[Optional[str], Optional[str]]:
    """
    Execute *sparql_query* against *ontology_content* (Turtle string).
    Returns (result_json_str, error_message).  One of them is always None.
    """
    try:
        # Validate syntax first
        rdflib.plugins.sparql.parser.parseQuery(sparql_query)
    except Exception as e:
        return None, f"SPARQL syntax error: {e}"

    try:
        g = rdflib.Graph()
        g.parse(data=ontology_content, format="turtle")
        results = g.query(sparql_query)

        if results.type == "SELECT":
            results_list = [row.asdict() for row in results]
            return json.dumps(results_list, indent=2), None
        elif results.type == "ASK":
            return json.dumps({"boolean": results.askAnswer}, indent=2), None
        elif results.type in ("CONSTRUCT", "DESCRIBE"):
            return results.serialize(format="turtle"), None
        else:
            return "Unknown query result type.", None

    except Exception:
        return None, f"SPARQL execution error:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# LLM-based answer evaluator
# ---------------------------------------------------------------------------
def evaluate_answer(
    question: str,
    expected_answer: str,
    sparql_query: str,
    query_result: Optional[str],
    execution_error: Optional[str],
) -> EvaluationResult:
    """
    Ask the LLM to judge whether *query_result* correctly answers *question*
    given *expected_answer*.  If the query itself failed, score 0.0 immediately.
    """
    if execution_error:
        return EvaluationResult(
            justification=f"Query could not be executed. Error: {execution_error}",
            score=0.0,
        )

    prompt = f"""You are a strict but answer-focused evaluator for ontology competency question evaluation.

Your task is to determine whether the ACTUAL QUERY RESULT answers the ORIGINAL QUESTION with respect to the EXPECTED ANSWER.

Important evaluation principles:
- Grade primarily based on whether the actual query result contains, entails, paraphrases, or semantically supports the expected answer.
- Do NOT require exact string matching.
- Do NOT over-penalise minor wording differences, determiners, capitalisation, singular/plural forms, or synonymous labels.
- Do NOT penalise the query merely because it used a different ontology path, as long as the returned result correctly answers the question.
- Consider the generated SPARQL query only as supporting evidence. The score should mainly reflect the correctness of the returned result.
- An empty result set when an answer is clearly expected should score 0.0.

Scoring rubric:
- 1.0        Correct: The query result clearly contains or semantically entails the expected answer.
- 0.75–0.99  Mostly correct: Strongly supports the expected answer but is indirect or uses a related label.
- 0.40–0.74  Partially correct: Related and gives useful evidence but is incomplete or not the exact type of information requested.
- 0.10–0.39  Weakly related: In the correct general domain but does not directly answer the question.
- 0.0        Incorrect: Unrelated, empty when an answer is expected, or contradicts the expected answer.

Evaluation Task:
- Original Question : "{question}"
- Expected Answer   : "{expected_answer}"
- SPARQL Query      :
```sparql
{sparql_query}
```
- Actual Query Result:
```json
{query_result}
```

Return ONLY valid JSON with exactly two keys — no explanation, no markdown:
{{"justification": "...", "score": 0.0}}"""

    result = call_llm(prompt, retries=3)

    if result is None:
        return EvaluationResult(
            justification="LLM evaluation failed after 3 attempts.",
            score=0.0,
        )

    try:
        return EvaluationResult(
            justification=str(result["justification"]),
            score=float(result["score"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        return EvaluationResult(
            justification=f"Could not parse LLM response: {e}. Raw: {result}",
            score=0.0,
        )


# ---------------------------------------------------------------------------
# Per-CQ evaluation (execute + evaluate)
# ---------------------------------------------------------------------------
def evaluate_cq(cq_item: Dict, ontology_content: str) -> Dict:
    """
    Execute the pre-supplied SPARQL query and evaluate its result.
    Returns a result dict ready for the JSON report.
    """
    question        = cq_item.get("question", cq_item.get("competency_question", ""))
    expected_answer = cq_item.get("answer",   cq_item.get("expected_answer", ""))
    sparql_query    = cq_item.get("sparql",   cq_item.get("sparql_query", ""))
    cq_id           = cq_item.get("id", "")

    print(f"\n  → [{cq_id}] {question}")

    query_result, exec_error = execute_sparql(sparql_query, ontology_content)

    if exec_error:
        print(f"    ✗ Execution failed: {exec_error[:120]}...")
    else:
        print(f"    ✓ Execution OK. Result preview: {str(query_result)[:120]}")

    evaluation = evaluate_answer(
        question=question,
        expected_answer=expected_answer,
        sparql_query=sparql_query,
        query_result=query_result,
        execution_error=exec_error,
    )

    print(f"    Score: {evaluation.score} — {evaluation.justification[:100]}")

    return {
        "id": cq_id,
        "question": question,
        "expected_answer": expected_answer,
        "sparql": sparql_query,
        "query_result": (
            json.loads(query_result)
            if query_result and query_result.strip().startswith("[") or (query_result and query_result.strip().startswith("{"))
            else query_result
        ) if query_result else "N/A",
        "evaluation_summary": evaluation.justification,
        "score": evaluation.score,
    }






COLORS = ["#F23434", "#FF9F18", "#34820D", "#2BC1B0", "#04414E",
          "#0057FF", "#6A0FBD", "#DC0C7F", "#853A05", "#5D5A5A"]


def extract_contract_id(file_name: str) -> str:
    """
    contract_1_xxx.txt -> C1_xxx
    """
    match = re.match(r"contract_(\d+)(.*)", file_name)
    if not match:
        return file_name

    num = match.group(1)        # 1
    suffix = match.group(2)     # _xxx.txt

    suffix = re.sub(r"\.txt$", "", suffix)  # remove extension

    return f"{num}{suffix}"


def plot_from_report(report_path: str, save: bool = True):
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    contracts = report["contracts"]

    # -------------------------------------------------------
    # Sort by contract ID (C1 → C10)
    # -------------------------------------------------------
    # contracts = sorted(
    #     contracts,
    #     key=lambda x: extract_contract_id(x["file_name"])
    # )

    labels = []
    scores = []
    colors = []

    for i, contract in enumerate(contracts):
        cid = f"C{extract_contract_id(contract['file_name'])}"
        score = contract.get("average_score", 0.0)

        labels.append(cid)
        scores.append(score)
        colors.append(COLORS[i % len(COLORS)])


    # -------------------------------------------------------
    # Plot
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    y = range(len(labels))

    ax.barh(
        y,
        scores,
        color=colors,
        edgecolor="black",
        height=0.7,
    )

    # labels
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)

    # -------------------------------------------------------
    # IMPORTANT: zoomed scale (better visibility)
    # -------------------------------------------------------
    ax.set_xlim(0.65, 1.0)

    # annotations
    for i, v in enumerate(scores):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)

    ax.set_xlabel("Average CQ Accuracy Score", fontsize=12)

    ax.grid(axis="x", linestyle=":", alpha=0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    import numpy as np

    overall = np.mean(scores)

    ax.axvline(
        overall,
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
    )

    ax.text(
        overall,
        1.02,
        f"Overall = {overall:.2f}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    plt.tight_layout()

    if save:
        out_path = Path(report_path).parent / "CQ_Coverage.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"📊 Saved plot → {out_path}")

    plt.show()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cq_path = Path(cfg.DATA_DIR) / Path(cfg.DATASET) / "competency_questions.json"
    abox_base = Path(cfg.ABOX_OUT_DIR)

    if not cq_path.exists():
        print(f"❌ CQ file not found: {cq_path}")
        return

    print("🤖  Initialising Multi-Contract Ontology Grader")
    print(f"▶   CQ file  : {cq_path}")
    print(f"▶   Abox base: {abox_base}")

    with open(cq_path, "r", encoding="utf-8") as f:
        cq_data = json.load(f)

    contracts = cq_data.get("contracts", [])
    if not contracts:
        print("❌ No contracts found in CQ file.")
        return

    all_contract_results = []
    grand_total_score = 0.0
    grand_total_cqs   = 0

    for contract in contracts:
        file_name   = contract.get("file_name", "")
        policy_type = contract.get("policy_type", "")
        insurer     = contract.get("insurer", "")
        cq_pairs    = contract.get("cq_pairs", [])

        print(f"\n{'='*65}")
        print(f"📄 Contract : {file_name}")
        print(f"   Type     : {policy_type}")
        print(f"   Insurer  : {insurer}")
        print(f"   # CQs    : {len(cq_pairs)}")
        print(f"{'='*65}")

        # --- Resolve and load the matching abox ---
        try:
            abox_dir = resolve_abox_path(file_name)
            print(f"   Abox dir : {abox_dir}")
            ontology_content = load_abox(abox_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"   ⚠️  Skipping contract — {e}")
            all_contract_results.append({
                "file_name": file_name,
                "policy_type": policy_type,
                "insurer": insurer,
                "error": str(e),
                "average_score": 0.0,
                "cq_results": [],
            })
            continue

        # --- Evaluate each CQ ---
        contract_results = []
        contract_score   = 0.0

        for cq_item in cq_pairs:
            result = evaluate_cq(cq_item, ontology_content)
            contract_results.append(result)
            contract_score   += result["score"]
            grand_total_score += result["score"]
            grand_total_cqs   += 1

        avg = contract_score / len(cq_pairs) if cq_pairs else 0.0
        print(f"\n  Contract average score: {avg:.3f}  ({contract_score:.2f} / {len(cq_pairs)})")

        all_contract_results.append({
            "file_name":     file_name,
            "policy_type":   policy_type,
            "insurer":       insurer,
            "average_score": avg,
            "cq_results":    contract_results,
        })

    # --- Save report ---
    results_dir = Path(__file__).resolve().parent / "eval_results"
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / "evaluation_report_10_contracts_dataset.json"

    grand_avg = grand_total_score / grand_total_cqs if grand_total_cqs else 0.0

    report = {
        "cq_file":        str(cq_path),
        "abox_base_dir":  str(abox_base),
        "total_cqs":      grand_total_cqs,
        "grand_average_score": grand_avg,
        "contracts": all_contract_results,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n\n{'='*65}")
    print("🏁  EVALUATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Contracts evaluated : {len(all_contract_results)}")
    print(f"  Total CQs           : {grand_total_cqs}")
    print(f"  Grand average score : {grand_avg:.3f}")
    print(f"  Report saved to     : {report_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    # main()

    results_dir = Path(__file__).resolve().parent / "eval_results"
    report_path = results_dir / "evaluation_report_10_contracts_dataset.json"
    
    plot_from_report(report_path)