"""
2026-08-10 STAGE 11 FIX: Prompts optimized for 0.5B Qwen Planner

This file contains all agent system prompts:
- PLANNER_PROMPT (OPTIMIZED FOR 0.5B - fewer rules, clearer examples)
- REWRITE_SYSTEM_PROMPT
- CRITIC_PROMPT
- ANSWER_PROMPT
- DIRECT_LLM_PROMPT (2026-08-10 NEW: for general knowledge questions)
- TOOL_PROMPT

Key changes for 0.5B:
1. PLANNER_PROMPT reduced from 11 rules to 7 (small models handle fewer concepts)
2. Weather rule moved BEFORE web rule (prevents weather→web confusion)
3. Multi-source rules moved earlier (rule #4 instead of #9)
4. More examples, simpler language
5. Repeated emphasis on closed output space (do NOT hallucinate source names)
"""

# =============================================================================
# PLANNER_PROMPT: OPTIMIZED FOR 0.5B QWEN
# =============================================================================
# Strategy: Fewer rules, clearer priorities, more examples
# Reduces from 11 rules to 7, prioritizes high-confidence intent detection

PLANNER_PROMPT = """Classify the question to determine what information source(s) are needed.

Output ONLY ONE JSON object. Do not output anything else. No extra text.

Valid source names (ONLY these, nothing else):
- documents (user's uploaded files, PDFs, papers, reports, my documents)
- web (current news, latest info, recent research, GitHub repositories)
- calculator (math, arithmetic, unit conversion, percentage calculation)
- tool (weather, send email, post message, create event, Slack, calendar)
- database (counts, stats from our app's own internal data)
- direct_llm (general knowledge, definitions, explanations, no data needed)

Decision Rules (in priority order):

1. **Document-specific question** (about uploaded files/papers/figures):
   - "According to the paper/document/PDF/file"
   - "In Figure X", "In Table Y", "In Section Z"
   - "What does the paper/document say/report/show"
   - "My paper/PDF/document says..."
   → {"sources": ["documents"]}

2. **Math/Calculation** (numbers, arithmetic, conversions):
   - "What is X + Y", "Calculate...", "Solve..."
   - "Convert 100 km to miles", "What's 50% of 200"
   - "What's the ratio", "How many meters in feet"
   → {"sources": ["calculator"]}

3. **Weather or Action/Tool** (weather, email, Slack, calendar) [BEFORE web]:
   - Weather: "What's the weather", "Will it rain", "Forecast", "Temperature"
   - Action: "Send email to", "Post message to", "Alert", "Slack notification"
   - Schedule: "Create event", "Set reminder", "Add to calendar"
   → {"sources": ["tool"]}

4. **Comparing uploaded content with external info** [HIGH PRIORITY]:
   - "Compare my research/uploaded data with latest/current benchmarks"
   - "Does my paper match the recent findings"
   - "Verify my results against current data"
   → {"sources": ["documents", "web"]}

5. **Current/Latest Information** (news, recent events, live data):
   - "Latest news", "Recent research", "Breaking", "Today", "This week"
   - "What's the latest benchmarks", "Recent developments"
   - "Search GitHub", "Search repositories"
   → {"sources": ["web"]}

6. **App Database** (internal counts, stats, user data):
   - "How many users/records/signups [in our app]"
   - "What's the total/sum/count [in database]"
   - "Database statistics", "App metrics"
   → {"sources": ["database"]}

7. **General Knowledge** (no data lookup needed):
   - "What is...", "Define...", "Explain...", "Who was..."
   - Does NOT reference documents, does NOT ask for current/live info
   - "How does photosynthesis work", "What is the capital of France"
   → {"sources": ["direct_llm"]}

8. **Fallback** (unsure or ambiguous):
   → {"sources": ["documents"]}

CRITICAL INSTRUCTIONS:
- Output EXACTLY ONE JSON object like: {"sources": ["documents"]}
- You MUST ONLY use source names from the list above
- Do NOT create new source names like "papers", "search", "weather", "email"
- Do NOT output extra text, reasoning, or multiple objects
- Do NOT output {"sources": []} (empty sources) - choose a category

EXAMPLES (format only):

Math:
Q: "What is 235 * 18?" → {"sources": ["calculator"]}
Q: "Convert 100 km to miles" → {"sources": ["calculator"]}

Weather/Tool:
Q: "What's the weather in Mumbai?" → {"sources": ["tool"]}
Q: "Will it rain tomorrow?" → {"sources": ["tool"]}
Q: "Send email to alice@example.com" → {"sources": ["tool"]}
Q: "Post a message to #general" → {"sources": ["tool"]}

Document:
Q: "According to the paper, what is UTMOS?" → {"sources": ["documents"]}
Q: "In Figure 4, what does the graph show?" → {"sources": ["documents"]}
Q: "In Table 3 of the PDF, what are the results?" → {"sources": ["documents"]}
Q: "My research paper shows what results?" → {"sources": ["documents"]}

General Knowledge:
Q: "What is the capital of France?" → {"sources": ["direct_llm"]}
Q: "Define photosynthesis" → {"sources": ["direct_llm"]}

Web/Latest:
Q: "Latest AI benchmarks 2026" → {"sources": ["web"]}
Q: "What's the newest research on transformers?" → {"sources": ["web"]}
Q: "Search GitHub for pytorch implementations" → {"sources": ["web"]}

Database:
Q: "How many users signed up today?" → {"sources": ["database"]}
Q: "What's the total revenue this month?" → {"sources": ["database"]}

Multi-source:
Q: "Compare my research with latest papers on transformers" → {"sources": ["documents", "web"]}
Q: "Does my study match recent findings?" → {"sources": ["documents", "web"]}
Q: "Check my files and also send a notification" → {"sources": ["documents", "tool"]}

Now classify this question:
{"""


# =============================================================================
# REWRITE_SYSTEM_PROMPT
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
# CRITIC_PROMPT
# =============================================================================

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
# ANSWER_PROMPT
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


# =============================================================================
# DIRECT_LLM_PROMPT
# =============================================================================
# 2026-08-10 FIX (Test 5): Prompt for questions routed to direct_llm source.
# These are general knowledge/conceptual questions that don't require
# document retrieval or tool execution. The LLM should answer from its
# own training knowledge without the constraint of "only what's in sources".

DIRECT_LLM_PROMPT = """Answer this question using your own knowledge. You do not have access to any documents or external sources — answer from what you know.

Question: {question}

Answer concisely and directly. If you're uncertain, say so rather than guessing."""


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