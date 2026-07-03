PLANNER_PROMPT = """Classify this question. Output only JSON.

Question: {question}

Rule:
- Contains "my"/"I"/"me" about content (even with numbers) -> sources: ["documents"]
- Public fact, unrelated to anything uploaded -> sources: ["web"]
- Explicitly asks to compare "my"/uploaded data against external info -> sources: ["documents", "web"]
- Unsure -> sources: ["documents"]

Q: "What is my total balance?" -> {{"intent": "lookup", "sources": ["documents"], "confidence": 0.95, "strategy": "personal data"}}
Q: "Who is the PM of Japan?" -> {{"intent": "lookup", "sources": ["web"], "confidence": 0.9, "strategy": "public fact"}}

Output JSON now:
{{"intent": "...", "sources": [...], "confidence": 0.85, "strategy": "..."}}

Start with {{"""

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

ANSWER_PROMPT = """Answer the question using only the sources below.

Question: {question}
Sources:
{context}

Rules:
- If the question asks for ONE fact (a number, date, name, status) -> answer in ONE short sentence. Nothing else.
- If the question asks to summarize/list/explain/describe -> give a full answer covering the sources, no repeated facts.
- Only state facts explicitly present in the sources. Never combine or guess related facts.
- You have no knowledge outside the sources above, even about famous people, well-known events, or things you're certain about. If a name, date, or number does not appear in the sources, you do not know it for the purposes of this answer.
- If the sources don't contain the answer, say so plainly rather than filling the gap with anything you already know.
- Plain text only. No JSON, no markdown, no "Sources:" list — sources are shown separately by the app.

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