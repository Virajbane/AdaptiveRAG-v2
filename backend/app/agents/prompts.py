PLANNER_PROMPT = """You are a planning agent. Output only a JSON object — no preamble, no explanation.

Question: {question}

Rules for "sources":
- Use ["documents"] if the question can be answered from uploaded documents
- Use ["web"] if the question needs current external facts (news, prices, live data)
- Use ["documents", "web"] if it genuinely needs both
- Default to ["documents"] when in doubt

Output exactly this JSON:
{{
  "intent": "...",
  "sources": ["documents"],
  "confidence": 0.85,
  "strategy": "..."
}}

JSON only. Start with {{"""

RETRIEVER_PROMPT = """
You are the retriever agent. You have retrieved {doc_count} relevant documents.

Question: {question}

Retrieved documents:
{docs}

Summarize what you found and how confident you are it answers the question.

Format:
{{
  "summary": "...",
  "confidence": 0.9,
  "has_answer": true
}}
"""

CRITIC_PROMPT = """You are a critic agent. Output only a JSON object — no preamble, no explanation.

Question: {question}
Context: {context}
Answer: {answer}

Is the answer supported by the context and does it answer the question?

Output exactly this JSON:
{{
  "valid": true,
  "confidence": 0.85,
  "issues": [],
  "needs_more_info": false
}}

Rules:
- "valid": true if the answer uses information from context to answer the question
- "confidence": decimal 0.0 to 1.0 (examples: 0.9, 0.75, 0.5) — NEVER write 0 unless answer is completely fabricated
- "issues": empty list [] if answer is acceptable
- Be lenient — if the answer is mostly correct, mark valid true

JSON only. Start with {{"""

ANSWER_PROMPT = """
You are a helpful AI assistant. Answer the user's question using the sources provided below.

Rules:
- Read ALL sources carefully before answering
- Extract every relevant detail from the sources — do not skip anything
- If the answer spans multiple sources, combine them into one complete answer
- Be specific — include names, numbers, dates, lists exactly as they appear in sources
- If the sources genuinely do not contain the answer, say so plainly
- Write plain text only — no JSON, no markdown, no "Sources:" section

Question: {question}

Sources:
{context}

Your complete answer:
"""

TOOL_PROMPT = """
You are the tool agent. Decide which tools to use.

Question: {question}
Available tools: {tools}

Which tools should we call and why?

Format:
{{
  "tools_to_call": ["web_search", "calculator"],
  "tool_calls": [
    {{"tool": "web_search", "query": "..."}}
  ]
}}
"""