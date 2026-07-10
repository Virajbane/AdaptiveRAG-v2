"""
ragas_eval.py

Ragas-based eval harness, wired to the SAME pipeline as eval_rag.py
(AgentOrchestrator.process + HybridSearchEngine.search), but using Ragas's
standard metric suite and its LLM-as-judge machinery instead of hand-rolled
prompts.

Judge model: local Ollama (free, no API key). Uses the same nomic-embed-text
embeddings you already have running for retrieval, so no extra embedding
service is needed.

WHY A SEPARATE reference-free / reference-based PATH:
    Your golden_set.json currently stores "expected_answer_contains" (a list
    of acceptable keywords), not a full ground-truth answer. Ragas's
    reference-based metrics (context_recall, factual_correctness) need an
    actual reference ANSWER string to compare against -- a keyword list
    isn't enough signal for those specific metrics. So:
      - If every "answer"-type item has a "reference" field -> full suite.
      - Otherwise -> reference-free suite only (faithfulness, response
        relevancy, context precision without reference), and a warning is
        printed telling you which items are missing "reference" and how
        to add it.
    This avoids silently mislabeling keyword-matching as ground-truth
    comparison, which would give you a false sense of precision.

Install (all free / local, no API keys):
    pip install ragas langchain-ollama

Run (defaults sized for constrained hardware: qwen2.5:0.5b judge, max_workers=1):
    ollama pull qwen2.5:0.5b
    ollama pull nomic-embed-text  # you already have this

    python ragas_eval.py --golden golden_set.json --user-id <test_user_id>

answer_relevancy and factual_correctness auto-disable below 2B params since
they gave unreliable/misleading scores at that size in testing (see prior
run logs). If you ever get more capable hardware, pass --judge-model
qwen2.5:7b --num-ctx 4096 to get the full metric suite back.
"""

import argparse
import asyncio
import json
import sys
from typing import List, Optional

from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    FactualCorrectness,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings


def load_golden_set(path: str) -> List[dict]:
    """Same placeholder-skip logic as eval_rag.py, so both harnesses treat
    an unfinished golden_set.json identically."""
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    placeholders = [it["id"] for it in items if "REPLACE" in json.dumps(it) or "SET_AFTER" in json.dumps(it)]
    if placeholders:
        print(f"[WARN] {len(placeholders)} golden set entries still contain "
              f"placeholder values and will be skipped: {placeholders}")
    return [it for it in items if "REPLACE" not in json.dumps(it) and "SET_AFTER" not in json.dumps(it)]


async def build_samples(golden_items: List[dict], orchestrator, hybrid_engine, user_id: str, top_k: int = 6):
    """Runs the real pipeline for each 'answer'-type item and assembles
    Ragas-shaped samples: user_input, response, retrieved_contexts, and
    an optional reference answer if the golden item provides one.

    Deliberately re-fetches contexts fresh (full text) rather than reusing
    any truncated citation preview the orchestrator's state may carry --
    same reasoning as the fix already in eval_rag.py: judges need the full
    chunk text, not a 200-char citation stub, or true-but-late-in-chunk
    facts get wrongly flagged as unsupported.
    """
    samples = []
    missing_reference = []

    for item in golden_items:
        if item.get("type") != "answer":
            continue

        state = await orchestrator.process(item["question"], user_id)
        answer_text = state.get("answer", "") if isinstance(state, dict) else getattr(state, "answer", "")

        # NOTE: deliberately NOT filtering by item.get("document_id") here.
        # The orchestrator itself resolves documents by content/score match
        # and passes document_id_filter=None when confidence is high enough
        # (see "[DOC_RESOLVER] ... no filter applied" in your logs) --
        # that's the path that actually finds the right chunks. Any
        # document_id stored in golden_set.json can go stale (re-ingestion,
        # re-upload, DB reset) without the golden set being updated, and a
        # stale ID silently hard-filters hybrid_engine.search() down to
        # zero results instead of erroring -- which is exactly what
        # happened in your last run (every single item: "0 contexts").
        # Matching the orchestrator's own successful behavior is safer than
        # trusting a filter value that isn't verified against live data.
        retrieved = await hybrid_engine.search(
            query=item["question"], user_id=user_id, top_k=top_k
        )
        contexts = [r["text"] for r in retrieved]
        if not contexts:
            print(f"[WARN] {item['id']}: retrieval returned 0 chunks even WITHOUT a document_id filter. "
                  f"This item's faithfulness/context scores will be meaningless -- check that the document "
                  f"is actually indexed for user_id={user_id}.")

        reference = item.get("reference")
        if not reference:
            missing_reference.append(item["id"])

        sample = {
            "user_input": item["question"],
            "response": answer_text,
            "retrieved_contexts": contexts,
        }
        if reference:
            sample["reference"] = reference

        samples.append(sample)
        print(f"[COLLECTED] {item['id']}: {len(contexts)} contexts, answer len={len(answer_text)}")

    return samples, missing_reference


def build_metrics(evaluator_llm, evaluator_embeddings, have_full_references: bool, judge_model: str):
    param_tag = judge_model.split(":")[-1].lower()
    is_small = any(tag in param_tag for tag in ["0.5b", "1b", "1.5b", "2b"])

    metrics = [Faithfulness(llm=evaluator_llm)]

    # ResponseRelevancy's "noncommittal" classification step needs enough
    # reasoning to reliably tell "declines to answer" apart from "answers
    # but briefly" -- a sub-1B judge conflates these (that's what tanked
    # the 0.5b run's aggregate to 0.10 even on clearly on-topic answers).
    # 7B+ models handle this distinction correctly, so only skip it below 2B.
    if is_small:
        print(f"[SKIP] Leaving out answer_relevancy -- '{judge_model}' unreliably flags normal answers as "
              f"'noncommittal'. Use a 7B+ judge to get this metric.")
    else:
        metrics.append(ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings))

    metrics.append(LLMContextPrecisionWithoutReference(llm=evaluator_llm))

    if have_full_references:
        metrics.append(LLMContextRecall(llm=evaluator_llm))
        # FactualCorrectness does multi-step claim decomposition + entailment
        # checking -- the most reasoning-heavy metric in the suite. A sub-1B
        # judge is unreliable enough at this that it's better left out than
        # reported with false precision. Swap to a 7B+ Ollama model to get it.
        if not is_small:
            metrics.append(FactualCorrectness(llm=evaluator_llm))
        else:
            print(f"[SKIP] Leaving out factual_correctness -- '{judge_model}' is too small to trust for it. "
                  f"context_recall is kept since it's a simpler judgment.")
    return metrics


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="golden_set.json")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--judge-model", default="qwen2.5:0.5b",
                         help="Free local Ollama model used as the Ragas judge. 7B+ is more reliable but too "
                              "slow/heavy for constrained hardware -- 0.5b trades reliability for speed. "
                              "answer_relevancy and factual_correctness auto-disable below 2B params since "
                              "they were unreliable at that size in testing.")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-workers", type=int, default=1,
                         help="Concurrent judge calls. Keep at 1 for a small/constrained GPU.")
    parser.add_argument("--num-ctx", type=int, default=2048,
                         help="Ollama context window for the judge model. Lower this if you hit OOM.")
    parser.add_argument("--out", default="ragas_results.csv")
    args = parser.parse_args()

    # Small (<2B) judge models frequently fail to produce the structured
    # (JSON/pydantic) output Ragas' default prompts expect, and reason less
    # reliably than a 7-8B judge on multi-step grading like faithfulness
    # claim decomposition. This isn't a reason to block you if that's what
    # you have -- it just means: expect a higher parse-failure/retry rate,
    # and treat per-sample scores as directional rather than precise.
    approx_param_count = args.judge_model.split(":")[-1]
    if any(tag in approx_param_count.lower() for tag in ["0.5b", "1b", "1.5b", "2b"]):
        print(f"[CAUTION] '{args.judge_model}' is a small judge model. Faithfulness/relevancy scores from "
              f"it will be noisier than a 7B+ judge -- use them as rough directional signal, and spot-check "
              f"the per-sample CSV rather than trusting the aggregate number alone.\n")

    golden_items = load_golden_set(args.golden)
    if not golden_items:
        print("No usable golden set entries. Fill in golden_set.json first.")
        sys.exit(1)

    # --- same wiring as eval_rag.py -----------------------------------
    from app.services.retrieval.hybrid_search import HybridSearchEngine
    from app.db.mongodb.client import connect_to_mongo
    from app.services.retrieval.bm25_bootstrap import rebuild_bm25_indexes
    from app.agents.orchestrator import AgentOrchestrator

    await connect_to_mongo()
    await rebuild_bm25_indexes()

    hybrid_engine = HybridSearchEngine()
    orchestrator = AgentOrchestrator()
    # -------------------------------------------------------------------

    samples, missing_reference = await build_samples(
        golden_items, orchestrator, hybrid_engine, args.user_id, top_k=args.top_k
    )
    if not samples:
        print("No 'answer'-type items found in golden set -- Ragas needs question/answer/context samples.")
        sys.exit(1)

    have_full_references = len(missing_reference) == 0
    if not have_full_references:
        print(f"\n[INFO] {len(missing_reference)} item(s) have no 'reference' answer field: {missing_reference}")
        print("       Running reference-free metrics only (faithfulness, response_relevancy, "
              "context_precision_without_reference).")
        print("       To also get context_recall + factual_correctness, add a \"reference\": \"<ground truth "
              "answer>\" field to every 'answer'-type golden set item.\n")

    decline_ids = [it["id"] for it in golden_items if it.get("type") == "answer" and it["id"].startswith("decline_")]
    if decline_ids:
        print(f"[INFO] {len(decline_ids)} item(s) are intentional-decline questions: {decline_ids}")
        print("       response_relevancy scores these near 0 by design (it flags 'noncommittal' answers), "
              "regardless of whether the decline was correct. Check these rows individually in the CSV "
              "rather than letting them pull down your aggregate answer_relevancy score.\n")

    dataset = EvaluationDataset.from_list(samples)

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=args.judge_model,
            base_url=args.ollama_url,
            temperature=0,
            num_ctx=args.num_ctx,  # 2048 default keeps memory use low for constrained hardware
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=args.embed_model, base_url=args.ollama_url)
    )

    metrics = build_metrics(evaluator_llm, evaluator_embeddings, have_full_references, args.judge_model)

    # max_workers=1 avoids sending concurrent requests to a constrained GPU
    # (which either OOMs or just queues anyway, so parallelism buys nothing).
    # Higher max_retries + longer timeout compensate for the small judge
    # model occasionally failing to produce parseable structured output.
    run_config = RunConfig(timeout=180, max_retries=5, max_workers=args.max_workers)

    print(f"Running Ragas eval with judge='{args.judge_model}' (Ollama), embeddings='{args.embed_model}', "
          f"max_workers={args.max_workers} ...")
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    print("\n" + "=" * 60)
    print("RAGAS RESULTS (aggregate)")
    print("=" * 60)
    print(result)

    df = result.to_pandas()
    df.to_csv(args.out, index=False)
    print(f"\nPer-sample scores written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())