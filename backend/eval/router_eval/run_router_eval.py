# eval/router_eval/run_router_eval.py
"""
Calls PlannerAgent directly — no retrieval, no LLM answer generation.
Pure routing-accuracy check. Cheapest, fastest eval in the whole suite —
run this one on EVERY commit, not just PRs.
"""
import asyncio, json
from app.config.settings import settings
from pathlib import Path
from app.agents.planner import PlannerAgent
from app.agents.state import AgentState
from app.services.llm.provider import LLMProvider   # adjust import to your actual factory

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "routing_golden.jsonl"

async def main():
    cases = [json.loads(l) for l in DATASET.read_text().splitlines() if l.strip()]
    planner = PlannerAgent(LLMProvider(model=settings.OLLAMA_FAST_MODEL))  # same model prod uses for planning

    correct = 0
    for case in cases:
        state = AgentState(question=case["question"], user_id=case.get("user_id", "eval"))
        state = await planner._execute(state)
        raw_sources = state.sources_needed or []
        expected = sorted(case["expected_sources"])

        # Guard against malformed planner output — local models occasionally
        # ignore the expected list[str] schema and return something else
        # entirely (e.g. a list of classification dicts). Treat that as a
        # failed case instead of crashing the whole eval run.
        if raw_sources and not all(isinstance(s, str) for s in raw_sources):
            passed = False
        else:
            passed = sorted(raw_sources) == expected
        correct += passed
        print(f"{'✅' if passed else '❌'} {case['id']}: got {state.sources_needed}, expected {case['expected_sources']}")

    print(f"\nRouting accuracy: {correct}/{len(cases)} = {correct/len(cases):.1%}")

if __name__ == "__main__":
    asyncio.run(main())