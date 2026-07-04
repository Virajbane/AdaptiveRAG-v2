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

ANSWER_PROMPT = """Rules:
- If the question asks for ONE fact (a number, date, name, status, winner, result) -> answer in ONE short sentence. Do not add related facts, records, history, or trivia, even if present in the sources.
- If the question asks to summarize/list/explain/describe -> give a full answer covering the sources, no repeated facts.
- Only state facts explicitly present in the sources. Never combine or guess related facts.
- You have no knowledge outside the sources above, even about famous people, well-known events, or things you're certain about. If a name, date, or number does not appear in the sources, you do not know it for the purposes of this answer.
- If the sources don't contain the answer, say so plainly rather than filling the gap with anything you already know.
- Sources are labeled [Source N] (from the user's own uploaded document) or [Web N] (general web search results, NOT from the user's document). If a question asks what a specific paper/document states, uses, or reports, and [Source N] and [Web N] disagree or describe different things, ALWAYS trust [Source N] — [Web N] results may describe an unrelated tool, paper, or topic that merely shares similar wording with the question, not the actual contents of the user's document. Only use [Web N] as the answer when no [Source N] entry addresses the question at all.
- Plain text only. No JSON, no markdown, no "Sources:" list — sources are shown separately by the app.
- IMPORTANT: The example below is a FORMAT DEMONSTRATION ONLY, using a placeholder topic. Never reuse its subject matter in your real answer, even if your real sources run out or get cut off.

Example (format only — ignore the topic):
Question: "What color is [X]?"
Sources: [mentions that [X] is blue]
Answer: "[X] is blue."

---

Using ONLY the sources below, answer the question that follows. Ignore the example above entirely — it is not related to this question.

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