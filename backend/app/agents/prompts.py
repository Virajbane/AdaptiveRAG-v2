"""
2026-08-06 fix: Entity-attribution prompt update for 7B Qwen model.

Simplified version that:
- Removes overly complex multi-step rules (7B struggles with these)
- Keeps essential instruction on table-row matching
- Reduces token overhead (~100 tokens vs ~250)
- Maintains grounding without confusing the LLM

The retrieval filter (Fix #3) + grounding rewrite (Fix #2) do the heavy
lifting. This prompt provides guidance without overwhelming a 7B model.
"""

PLANNER_PROMPT = """Classify the question below. Output only ONE JSON object.

Rule:
- Contains "my"/"I"/"me"/"our" about content (even with numbers) -> sources: ["documents"]
- Public fact, unrelated to anything uploaded -> sources: ["web"]
- Explicitly asks to compare "my"/uploaded data against external info -> sources: ["documents", "web"]
- Unsure -> sources: ["documents"]

Example (format only — ignore the topic, do not repeat it in your output):
Q: "[placeholder personal question]" -> {{"intent": "lookup", "sources": ["documents"], "confidence": 0.95, "strategy": "personal data"}}
Q: "[placeholder public-fact question]" -> {{"intent": "lookup", "sources": ["web"], "confidence": 0.9, "strategy": "public fact"}}

Output exactly one JSON object, for the question below only. Do not output the examples above, do not output a list, do not output more than one object.

Question: {question}

Output JSON now:
{{"""

REWRITE_SYSTEM_PROMPT = """Rewrite the question as one standalone, well-formed question.

Rules:
- Resolve pronouns/references using history (e.g. "what about X?" -> full topic + X).
- Fix spelling only. Do not restructure the sentence.
- Keep "my"/"I"/"me" EXACTLY as written. Never change to "your"/"you".
- Never expand acronyms (CGPA, RAG, API stay as-is). Never guess what one means.
- Keep the same command form (e.g. "summarize X" stays "summarize X", not "X summary").
- Output ONLY the rewritten question. No explanation.

Example:
Input: "wha t is my CGPA"
Output: "What is my CGPA?"

Example:
Input: "summarixe the rag2.0 pdf"
Output: "Summarize the RAG 2.0 PDF"
"""

CRITIC_PROMPT = """Judge this answer. Output only JSON.

Question: {question}
Context: {context}
Answer: {answer}

Check each specific fact in the answer (names, numbers, dates, statuses) against
the context, one at a time.

Mark invalid if ANY of the following are true:
- The answer states a fact that contradicts the context (e.g. wrong date, wrong
  person, wrong number), even if it's a real, plausible-sounding fact.
- The answer includes a specific name/number/date not present anywhere in the
  context.
- The answer answers a different question than the one asked.

Do not be lenient about factual contradictions just because the answer is
well-written or mostly on-topic. A confident, fluent answer that contradicts
its own context is invalid, not "mostly correct."

{{"valid": true, "confidence": 0.85, "issues": [], "needs_more_info": false}}

- confidence should reflect how certain you are the answer is fully consistent
  with the context — a contradiction found anywhere means confidence near 0.0,
  not just "not 1.0"
- issues: list each specific contradiction or unsupported fact found, empty
  list only if none found

Start with {{"""

# =============================================================================
# 2026-08-06 FIX #1: SIMPLIFIED ANSWER PROMPT FOR 7B QWEN
# =============================================================================
# 
# Differences from full version:
# - Removed "CRITICAL for table rows:" (5 complex rules) → 1 simple rule
# - Removed Lychee-FD/Moshi confusing example → kept only basic example
# - Shorter instruction overall (7B likes focused, concise rules)
# - Still enforces table-row matching but simpler language
# - Token overhead: ~100 tokens (vs ~250 for full version)
#
# Works with Fix #2 (grounding) + Fix #3 (retrieval filter)
# The system-level fixes do most of the work; prompt is just guidance.
#
# =============================================================================

ANSWER_PROMPT = """Rules:
- If the question asks for ONE fact (a number, date, name, status, winner, result) -> answer in ONE short sentence. Do not add related facts, records, history, or trivia, even if present in the sources.
- If the question asks to summarize/list/explain/describe -> give a full answer covering the sources, no repeated facts.
- Only state facts explicitly present in the sources. Never combine or guess related facts.
- Exception: if a source describes a numeric range or trend (e.g. "rising from 36.0 to 65.4 as X increases to 4", "grew from 10% to 40%"), you may state either endpoint value as the value at its corresponding condition (e.g. the value "at 0" is the starting number in "from X to Y", the value "at the maximum" is the ending number). This is reading an explicitly stated number, not guessing -- it's different from borrowing a number that belongs to a different entity or metric than the one asked about.
- You have no knowledge outside the sources above, even about famous people, well-known events, or things you're certain about. If a name, date, or number does not appear in the sources, you do not know it for the purposes of this answer.
- If the sources don't contain the answer, say so plainly rather than filling the gap with anything you already know.
- Sources are labeled [Source N] (from the user's own uploaded document) or [Web N] (general web search results, NOT from the user's document). If a question asks what a specific paper/document states, uses, or reports, and [Source N] and [Web N] disagree or describe different things, ALWAYS trust [Source N] — [Web N] results may describe an unrelated tool, paper, or topic that merely shares similar wording with the question, not the actual contents of the user's document. Only use [Web N] as the answer when no [Source N] entry addresses the question at all.

TABLE ROWS: When you see "Row [EntityName]:" in the sources, use only values from that row when answering about that entity. Do not mix values from different rows.

- Plain text only. No JSON, no markdown, no "Sources:" list — sources are shown separately by the app.

Using ONLY the sources below, answer the question that follows.

Sources:
{context}

Question: {question}

Answer:"""

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