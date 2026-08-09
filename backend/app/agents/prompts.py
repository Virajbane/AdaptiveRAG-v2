"""
2026-08-09 UPDATE: Consolidated all agent prompts in this file.
This is the single source of truth for all system prompts used by:
- PlannerAgent (PLANNER_PROMPT)
- RewriterAgent (REWRITE_SYSTEM_PROMPT)
- CriticAgent (CRITIC_PROMPT)
- AnswerAgent (ANSWER_PROMPT)
- ToolAgent (TOOL_PROMPT)

Import these in your agent files instead of hardcoding prompts.

2026-08-09 FIX (routing bug): PLANNER_PROMPT now includes explicit rules
for "direct_llm" and "database" sources that were previously defined in
SOURCE_REGISTRY but unreachable by the classifier. Added document-intent
rules for Figure/Table/Section queries. Added multi-source routing examples
(documents + web, documents + tool). planner.py now asserts at import time
that every SOURCE_REGISTRY key appears somewhere in this string, so this
kind of drift fails loudly instead of silently.

2026-08-09 ENHANCEMENT: Added high-confidence document-intent rules:
- "According to the paper/document/PDF"
- "In Figure X / Table Y / Section Z"
- "What does the paper say / report / mention"
These route to documents without requiring LLM inference.
"""

# =============================================================================
# PLANNER_PROMPT: Single-turn classification of question's information needs.
# Outputs JSON object with "sources" array constrained to SOURCE_REGISTRY.
# =============================================================================
#
# Rules prioritize in order:
# 1. Explicit document-intent phrases
# 2. Personal data (my/I/we/our)
# 3. Calculation/math
# 4. External APIs (weather/email/Slack/etc)
# 5. Current events/news
# 6. General knowledge (no data needed)
# 7. Database queries
# 8. Comparison across sources (documents + web, documents + tool)
#
# Fallback: unsure -> documents (assume they're asking about their data)

PLANNER_PROMPT = """Classify the question below. Output only ONE JSON object with "sources" array.

Rules (in priority order):

1. Contains explicit document-intent phrases ("according to the paper", "in Figure X", "in Section Y", "what does the document say") -> sources: ["documents"]

2. Contains "my"/"I"/"me"/"our" about personal content or uploads -> sources: ["documents"]

3. Contains math/calculation/solve/percentage/unit conversion -> sources: ["calculator"]

4. Contains weather/temperature/forecast/climate OR "post"/"send"/"alert"/"message"/"email"/"Slack" with action intent -> sources: ["tool"]

5. Public fact, current events, breaking news, latest benchmarks, recent developments -> sources: ["web"]

6. GitHub repository search, code repository queries, public source code -> sources: ["web"]

7. General knowledge, definitions, explanations, "what is X", no personal data or documents needed -> sources: ["direct_llm"]

8. Counts/stats/records from OUR OWN app database (not the user's documents) -> sources: ["database"]

9. Explicitly asks to compare "my" uploaded content against external/current info -> sources: ["documents", "web"]

10. Asks to search uploaded docs AND check external info (tools/weather/email) -> sources: ["documents", "tool"]

11. Unsure or ambiguous -> sources: ["documents"]

Examples (format only):
Q: "What is 235 * 18?" → {"sources": ["calculator"], "confidence": 0.95}
Q: "What is the weather in Mumbai?" → {"sources": ["tool"], "confidence": 0.9}
Q: "Post a message to #eng-alerts" → {"sources": ["tool"], "confidence": 0.95}
Q: "Latest AI benchmarks 2026" → {"sources": ["web"], "confidence": 0.9}
Q: "What is my CGPA?" → {"sources": ["documents"], "confidence": 0.95}
Q: "What is the capital of France?" → {"sources": ["direct_llm"], "confidence": 0.9}
Q: "How many users signed up last week?" → {"sources": ["database"], "confidence": 0.85}
Q: "According to Figure 4, what is the UTMOS score?" → {"sources": ["documents"], "confidence": 0.95}
Q: "In the paper, what does Table 3 show?" → {"sources": ["documents"], "confidence": 0.95}
Q: "What does my report say about GPT-4?" → {"sources": ["documents"], "confidence": 0.9}
Q: "Compare my uploaded research with latest papers on transformers" → {"sources": ["documents", "web"], "confidence": 0.85}
Q: "Check my files and also get current weather" → {"sources": ["documents", "tool"], "confidence": 0.8}

IMPORTANT:
- Only output valid source names from: documents, web, calculator, tool, database, direct_llm
- If a name is not in that list, omit it (closed output space)
- Do NOT invent new source names
- Output exactly one JSON object, nothing else
- If unsure, prefer ["documents"] over web

Output your JSON now:
{"""


# =============================================================================
# REWRITE_SYSTEM_PROMPT: Fix spelling/pronouns, keep meaning intact
# =============================================================================

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


# =============================================================================
# CRITIC_PROMPT: Fact-check the answer against available evidence
# =============================================================================
#
# 2026-08-09 FIX: Now handles multi-source evidence validation.
# Evidence may include documents, web results, calculator results, weather,
# tool outputs, database results, or metadata. Critic validates against
# whatever evidence was actually available to AnswerAgent.

CRITIC_PROMPT = """Judge this answer. Output only JSON.

Question: {question}
Available Evidence:
{context}
Answer: {answer}

Validation Rules:
1. Check each specific fact (names, numbers, dates, statuses) against the
   available evidence, one at a time.

2. Mark invalid if ANY of the following are true:
   - The answer contradicts the evidence (wrong date, wrong person, wrong number)
   - The answer includes a fact not present anywhere in the evidence
   - The answer answers a different question than asked
   - For calculator/weather/tool results: the answer doesn't match the result

3. For evidence types:
   - [Source N]: document chunks from uploaded files
   - [Web N]: web search results
   - [Calculator Result]: exact calculation output
   - [Weather Result]: API weather data
   - [Slack Result] / [Email Result]: action confirmation
   - [Metadata]: document metadata (title, author, etc.)
   - [Database Result]: database query results

4. Special case - Tool Results:
   If the question asks "What is 235 * 18?" and evidence shows
   [Calculator Result] 235 * 18 = 4230, then answer "4230" is valid.
   If answer is "4200" instead, it's invalid (contradicts evidence).

5. Do not accept plausible external facts. If an answer claims a "fact"
   that is not in the available evidence, mark it as unsupported,
   even if it's a real fact in the world.

Do not be lenient about contradictions just because the answer is well-written.
A confident answer that contradicts its own evidence is invalid.

Output JSON format:
{{"valid": true, "confidence": 0.85, "issues": [], "failure_type": "generation", "needs_more_info": false}}

Fields:
- valid: true if all facts check out, false if any contradiction or unsupported claim
- confidence: 0.0-1.0, how certain you are the answer is consistent with evidence
  (contradiction found → near 0.0, not just "not 1.0")
- issues: list each specific contradiction or unsupported fact, empty if none found
- failure_type: if valid=false, classify the failure:
  * "generation": answer hallucinated or made a synthesis error despite good evidence
  * "retrieval": missing/wrong evidence (docs not found when expected)
  * "tool": tool execution failed or returned invalid result
  * "planning": correct source exists but wrong source was selected
  * "unknown": can't determine
- needs_more_info: true if evidence is too limited to make a confident judgment

Start with {{"""


# =============================================================================
# ANSWER_PROMPT: Generate answer using only source facts
# =============================================================================
#
# 2026-08-06 FIX #1: SIMPLIFIED FOR 7B QWEN
# - Removed overly complex multi-step rules (7B struggles with these)
# - Keeps essential instruction on table-row matching
# - Reduces token overhead (~100 tokens vs ~250)
# - Maintains grounding without confusing the LLM
#
# Works with Fix #2 (grounding) + Fix #3 (retrieval filter)
# The system-level fixes do most of the work; prompt is just guidance.

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


# =============================================================================
# TOOL_PROMPT: (Reserved for future tool-selection logic)
# =============================================================================

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