# eval/reports/report_builder.py
"""
Merges Ragas JSON + DeepEval pytest-html output + router/retrieval scores
into ONE Markdown report — the artifact you'd actually attach to a PR or
show in a portfolio demo.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REPORTS_DIR = Path(__file__).resolve().parent

def build_report():
    ragas_gen = json.loads((RESULTS_DIR / "ragas_generation.json").read_text())
    ragas_retr = json.loads((RESULTS_DIR / "ragas_retrieval.json").read_text())

    def avg(rows, key):
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else "N/A"

    lines = [
        f"# RAG Evaluation Report — {datetime.now(timezone.utc).isoformat(timespec='seconds')}Z",
        "",
        "## Generation Quality (Ragas)",
        f"- Faithfulness: **{avg(ragas_gen, 'faithfulness')}**",
        f"- Answer Relevancy: **{avg(ragas_gen, 'answer_relevancy')}**",
        f"- Answer Correctness: **{avg(ragas_gen, 'answer_correctness')}**",
        "",
        "## Retrieval Quality (Ragas)",
        f"- Context Precision: **{avg(ragas_retr, 'llm_context_precision_with_reference')}**",
        f"- Context Recall: **{avg(ragas_retr, 'context_recall')}**",
        "",
        "## Notes",
        "- Full per-case breakdown: see `eval/results/*.json`",
        "- Regression + Bias/Toxicity results: see `eval/reports/*.html` (pytest-html output)",
    ]
    (REPORTS_DIR / "summary.md").write_text("\n".join(lines))
    print("✅ Report written to eval/reports/summary.md")

if __name__ == "__main__":
    build_report()