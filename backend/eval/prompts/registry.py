# eval/prompts/registry.py
"""
Prompt version registry. app/agents/prompts.py stays the single source of
truth for what's ACTIVE in production; this registry lets eval code run
the SAME golden dataset against multiple historical/candidate versions
for comparison, without touching production prompts.
"""
from pathlib import Path

VERSIONS_DIR = Path(__file__).parent / "versions"

PROMPT_VERSIONS = {
    "planner": {
        "v1": VERSIONS_DIR / "planner_v1.txt",
        "v2": VERSIONS_DIR / "planner_v2.txt",   # e.g. the 7B-model prompt rewrite
    },
    "answer": {
        "v1": VERSIONS_DIR / "answer_v1.txt",
    },
}

def load_prompt(name: str, version: str) -> str:
    return PROMPT_VERSIONS[name][version].read_text()