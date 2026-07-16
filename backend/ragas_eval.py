"""
ragas_eval.py

Ragas-based eval harness, wired to the SAME pipeline as eval_rag.py
(AgentOrchestrator.process + HybridSearchEngine.search), using Ragas's
standard metric suite for judge-based scoring PLUS the same deterministic
checks eval_rag.py runs, imported from rag_eval_common.py so both scripts
can't drift on shared logic.

Full metric suite this script now reports:

  RETRIEVAL LAYER
    - Recall@k, MRR              (NEW -- Ragas has no built-in retrieval
                                   recall@k/MRR metric; added as a plain
                                   loop, same content-match logic as
                                   eval_rag.py so the two Recall@6 numbers
                                   are directly comparable)
    - Context precision           (Ragas, LLM-judged -- unchanged)
    - Context recall               (Ragas, only if references present)

  GENERATION LAYER
    - Faithfulness                 (Ragas, LLM-judged -- unchanged)
    - Answer relevancy              (Ragas, 7B+ judge only -- unchanged)
    - Factual correctness           (Ragas, 7B+ judge + references only)
    - Entity-attribution accuracy   (NEW -- same deterministic check as
                                     eval_rag.py, reported standalone. This
                                     is the metric the report confirmed
                                     Ragas's own faithfulness metric MISSES
                                     -- the LLM judge scored both confirmed
                                     UTMOS fabrications as faithfulness=1.0,
                                     same failure mode as the local judge.)

  SAFETY / ROBUSTNESS LAYER
    - Hallucination trap pass rate  (NEW)
    - False-decline rate            (NEW)

  OPERATIONAL LAYER
    - Ingestion completeness gate   (NEW, runs first)
    - Cache hit accuracy + latency  (NEW)

Install (all free / local, no API keys):
    pip install ragas langchain-ollama

Run:
    ollama pull qwen2.5:0.5b
    ollama pull nomic-embed-text

    python ragas_eval.py --golden golden_set.json --user-id <test_user_id>

answer_relevancy and factual_correctness auto-disable below 2B params.
Pass --judge-model qwen2.5:7b --num-ctx 4096 for the full suite.
"""

import argparse
import asyncio
import json
import sys
import time
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

from rag_eval_common import (
    entity_attribution_pass,
    score_hallucination_trap,
    score_false_decline,
    check_ingestion_completeness,
    CacheMetricTracker,
    load_golden_set_v2,
)


def _content_match(text: str, expected_keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in expected_keywords)


async def run_retrieval_metrics(golden_items: List[dict], hybrid_engine, user_id: str, top_k: int):
    """Recall@k + MRR, added because Ragas has no built-in equivalent --
    its context_precision/context_recall metrics judge RELEVANCE of what
    was retrieved, not whether the retriever found the right thing at
    all. Same content-match logic as eval_rag.py on purpose, so the two
    Recall@6 numbers are apples-to-apples across both harnesses."""
    results = []
    for item in golden_items:
        if item.get("type") != "retrieval":
            continue
        t0 = time.perf_counter()
        retrieved = await hybrid_engine.search(
            query=item["question"], user_id=user_id, top_k=top_k,
            document_id=item.get("document_id"),
        )
        latency = time.perf_counter() - t0
        expected_keywords = item.get("expected_answer_contains", [])
        found, rank = False, None
        for i, r in enumerate(retrieved, start=1):
            if _content_match(r.get("text", ""), expected_keywords):
                found, rank = True, i
                break
        results.append({"id": item["id"], "found": found, "rank": rank, "latency_s": latency})
    return results


def print_retrieval_report(results, top_k: int):
    print("\n" + "=" * 60)
    print("1. RETRIEVAL EVAL (Recall@k / MRR -- not a Ragas built-in metric)")
    print("=" * 60)
    if not results:
        print("(no retrieval-type items in golden set)")
        return
    recall = sum(1 for r in results if r["found"]) / len(results)
    mrr = sum((1 / r["rank"]) if r["found"] else 0 for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    print(f"Recall@{top_k}: {recall:.2%}  ({sum(1 for r in results if r['found'])}/{len(results)})")
    print(f"MRR:        {mrr:.3f}")
    print(f"Avg latency: {avg_latency:.2f}s")
    for r in results:
        status = f"rank {r['rank']}" if r["found"] else "NOT FOUND"
        print(f"  [{r['id']}] {status}")


async def build_samples(golden_items: List[dict], orchestrator, hybrid_engine, user_id: str, top_k: int = 6):
    """Runs the real pipeline for each 'answer'-type item and assembles
    Ragas-shaped samples, PLUS the deterministic checks from
    rag_eval_common (entity-attribution, hallucination trap, false
    decline) attached as extra fields alongside each sample -- Ragas
    doesn't know about these, so they're tracked separately and reported
    in their own section, not passed into `evaluate()`.
    """
    samples = []
    extra_checks = []
    missing_reference = []

    for item in golden_items:
        if item.get("type") != "answer":
            continue

        t0 = time.perf_counter()
        state = await orchestrator.process(item["question"], user_id)
        latency = time.perf_counter() - t0
        answer_text = state.get("answer", "") if isinstance(state, dict) else getattr(state, "answer", "")

        retrieved = await hybrid_engine.search(
            query=item["question"], user_id=user_id, top_k=top_k
        )
        contexts = [r["text"] for r in retrieved]
        if not contexts:
            print(f"[WARN] {item['id']}: retrieval returned 0 chunks. This item's "
                  f"faithfulness/context scores will be meaningless -- check indexing "
                  f"for user_id={user_id}.")

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

        context_text = "\n\n".join(contexts)
        extra_checks.append({
            "id": item["id"],
            "entity_attribution_ok": entity_attribution_pass(answer_text, context_text, item["question"]),
            "hallucination_trap_pass": score_hallucination_trap(item, answer_text),
            "false_decline": score_false_decline(item, answer_text),
            "latency_s": latency,
        })

        print(f"[COLLECTED] {item['id']}: {len(contexts)} contexts, answer len={len(answer_text)}")

    return samples, extra_checks, missing_reference


def print_extra_checks_report(extra_checks):
    print("\n" + "=" * 60)
    print("2b. DETERMINISTIC CHECKS (not part of Ragas's metric suite)")
    print("=" * 60)
    entity_checks = [c["entity_attribution_ok"] for c in extra_checks if c["entity_attribution_ok"] is not None]
    trap_checks = [c["hallucination_trap_pass"] for c in extra_checks if c["hallucination_trap_pass"] is not None]
    false_declines = [c["false_decline"] for c in extra_checks if c["false_decline"] is not None]
    if extra_checks:
        avg_latency = sum(c["latency_s"] for c in extra_checks) / len(extra_checks)
        print(f"Avg latency:                 {avg_latency:.2f}s")
    if entity_checks:
        rate = sum(entity_checks) / len(entity_checks)
        print(f"Entity-attribution accuracy: {rate:.2%}  (n={len(entity_checks)}) -- Ragas's own "
              f"faithfulness metric misses this class of error; see §2.2 of the bug report")
    if trap_checks:
        rate = sum(trap_checks) / len(trap_checks)
        print(f"Hallucination trap pass:     {rate:.2%}  ({sum(trap_checks)}/{len(trap_checks)})")
    if false_declines:
        rate = sum(false_declines) / len(false_declines)
        print(f"False-decline rate:          {rate:.2%}  ({sum(false_declines)}/{len(false_declines)}, lower is better)")
    for c in extra_checks:
        e = "" if c["entity_attribution_ok"] is None else ("entity=OK" if c["entity_attribution_ok"] else "entity=MISMATCH")
        t = "" if c["hallucination_trap_pass"] is None else ("trap=PASS" if c["hallucination_trap_pass"] else "trap=FAIL")
        flags = " ".join(x for x in [e, t] if x)
        if flags:
            print(f"  [{c['id']}] {flags}")


def build_metrics(evaluator_llm, evaluator_embeddings, have_full_references: bool, judge_model: str):
    param_tag = judge_model.split(":")[-1].lower()
    is_small = any(tag in param_tag for tag in ["0.5b", "1b", "1.5b", "2b"])

    metrics = [Faithfulness(llm=evaluator_llm)]

    if is_small:
        print(f"[SKIP] Leaving out answer_relevancy -- '{judge_model}' unreliably flags normal answers as "
              f"'noncommittal'. Use a 7B+ judge to get this metric.")
    else:
        metrics.append(ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings))

    metrics.append(LLMContextPrecisionWithoutReference(llm=evaluator_llm))

    if have_full_references:
        metrics.append(LLMContextRecall(llm=evaluator_llm))
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
    parser.add_argument("--judge-model", default="qwen2.5:0.5b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--cache-hit-threshold-s", type=float, default=5.0)
    parser.add_argument("--out", default="ragas_results.csv")
    args = parser.parse_args()

    approx_param_count = args.judge_model.split(":")[-1]
    if any(tag in approx_param_count.lower() for tag in ["0.5b", "1b", "1.5b", "2b"]):
        print(f"[CAUTION] '{args.judge_model}' is a small judge model. Faithfulness/relevancy scores from "
              f"it will be noisier than a 7B+ judge -- treat as directional, spot-check the CSV, and lean "
              f"on the deterministic checks (entity-attribution, hallucination trap) below, since those "
              f"don't depend on judge size at all.\n")

    golden = load_golden_set_v2(args.golden)
    golden_items = golden["items"]
    ingestion_check_cfg = golden["ingestion_check"]

    if not golden_items:
        print("No usable golden set entries. Fill in golden_set.json first.")
        sys.exit(1)

    from app.services.retrieval.hybrid_search import HybridSearchEngine
    from app.db.mongodb.client import connect_to_mongo
    from app.services.retrieval.bm25_bootstrap import rebuild_bm25_indexes
    from app.agents.orchestrator import AgentOrchestrator

    await connect_to_mongo()
    await rebuild_bm25_indexes()

    hybrid_engine = HybridSearchEngine()
    orchestrator = AgentOrchestrator()

    # --- 0. Ingestion completeness gate ---------------------------------
    ingestion_report = None
    if ingestion_check_cfg:
        expected_pages = ingestion_check_cfg.get("expected_pages", [])
        # chunks_by_page = await your_chunk_store.count_by_page(args.user_id)
        chunks_by_page = {}  # <-- REPLACE with real query before trusting this gate
        ingestion_report = check_ingestion_completeness(chunks_by_page, expected_pages)
        if not chunks_by_page:
            print("[WARN] ingestion_check configured but chunks_by_page query isn't wired "
                  "up yet -- placeholder in main(). This will show FAIL until connected.")

    print("\n" + "=" * 60)
    print("0. INGESTION COMPLETENESS GATE")
    print("=" * 60)
    if ingestion_report is None:
        print("(no top-level 'ingestion_check' in golden set -- skipped)")
    else:
        status = "PASS" if ingestion_report["complete"] else "FAIL"
        print(f"Status: {status}  (coverage: {ingestion_report['coverage']:.1%})")
        if ingestion_report["missing_pages"]:
            print(f"MISSING PAGES: {ingestion_report['missing_pages']}")

    # --- 1. Retrieval metrics (NEW) -------------------------------------
    retrieval_results = await run_retrieval_metrics(golden_items, hybrid_engine, args.user_id, args.top_k)
    print_retrieval_report(retrieval_results, args.top_k)

    # --- 2. Answer samples for Ragas + deterministic checks -------------
    samples, extra_checks, missing_reference = await build_samples(
        golden_items, orchestrator, hybrid_engine, args.user_id, top_k=args.top_k
    )

    # --- 3. Cache metrics (NEW) ------------------------------------------
    cache_tracker = CacheMetricTracker(hit_threshold_s=args.cache_hit_threshold_s)
    for item in golden_items:
        if item.get("type") == "cache":
            await cache_tracker.run(
                item, lambda q, u: orchestrator.process(q, u), args.user_id
            )
    cache_summary = cache_tracker.summary()

    if not samples:
        print("No 'answer'-type items found in golden set -- Ragas needs question/answer/context samples.")
        print_extra_checks_report(extra_checks)
        if cache_summary:
            print("\n" + "=" * 60)
            print("4. CACHE METRICS")
            print("=" * 60)
            print(f"Cache accuracy: {cache_summary['cache_accuracy']:.2%}  (n={cache_summary['n']})")
        sys.exit(0)

    have_full_references = len(missing_reference) == 0
    if not have_full_references:
        print(f"\n[INFO] {len(missing_reference)} item(s) have no 'reference' answer field: {missing_reference}")
        print("       Running reference-free metrics only. Add \"reference\": \"<ground truth answer>\" "
              "to every 'answer'-type item to also get context_recall + factual_correctness.\n")

    decline_ids = [it["id"] for it in golden_items if it.get("type") == "answer" and
                   (it.get("expects_decline") or it["id"].startswith("decline_"))]
    if decline_ids:
        print(f"[INFO] {len(decline_ids)} item(s) are intentional-decline questions: {decline_ids}")
        print("       response_relevancy scores these near 0 by design -- check the hallucination "
              "trap pass rate in section 2b instead of the Ragas relevancy aggregate for these.\n")

    dataset = EvaluationDataset.from_list(samples)

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=args.judge_model,
            base_url=args.ollama_url,
            temperature=0,
            num_ctx=args.num_ctx,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=args.embed_model, base_url=args.ollama_url)
    )

    metrics = build_metrics(evaluator_llm, evaluator_embeddings, have_full_references, args.judge_model)

    run_config = RunConfig(timeout=180, max_retries=5, max_workers=args.max_workers)

    print(f"Running Ragas eval with judge='{args.judge_model}' (Ollama), embeddings='{args.embed_model}', "
          f"max_workers={args.max_workers} ...")
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    print("\n" + "=" * 60)
    print("2a. RAGAS RESULTS (aggregate)")
    print("=" * 60)
    print(result)

    print_extra_checks_report(extra_checks)

    print("\n" + "=" * 60)
    print("3. CACHE METRICS")
    print("=" * 60)
    if cache_summary:
        print(f"Cache accuracy:      {cache_summary['cache_accuracy']:.2%}  (n={cache_summary['n']})")
        print(f"Avg cold latency:    {cache_summary['avg_cold_latency_s']}s")
        print(f"Avg warm latency:    {cache_summary['avg_warm_latency_s']}s")
    else:
        print("(no type:'cache' items in golden set -- skipped)")

    df = result.to_pandas()
    df.to_csv(args.out, index=False)
    print(f"\nPer-sample Ragas scores written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())