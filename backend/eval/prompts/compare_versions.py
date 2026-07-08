# eval/prompts/compare_versions.py
"""
Runs the SAME golden dataset through two prompt versions and diffs the
Ragas scores. Use this before promoting a prompt change to app/agents/prompts.py.
"""
import asyncio
from eval.prompts.registry import load_prompt
from eval.harness.pipeline_runner import run_single
from eval.ragas_eval.generation_metrics import score_generation
from eval.harness.state_capture import to_ragas_row
import app.agents.prompts as prompts_module

async def run_with_version(cases, prompt_name: str, version: str):
    original = getattr(prompts_module, f"{prompt_name.upper()}_PROMPT")
    setattr(prompts_module, f"{prompt_name.upper()}_PROMPT", load_prompt(prompt_name, version))
    try:
        rows = []
        for case in cases:
            result = await run_single(case["question"], user_id=case["user_id"])
            rows.append(to_ragas_row(result, ground_truth_answer=case.get("ground_truth_answer", "")))
        return score_generation(rows).to_pandas()
    finally:
        setattr(prompts_module, f"{prompt_name.upper()}_PROMPT", original)  # ALWAYS restore