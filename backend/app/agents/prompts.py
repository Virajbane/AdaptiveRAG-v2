PLANNER_PROMPT = """
You are the planning agent. Analyze the user's question and decide:
1. What is the user asking for?
2. Where should we search? (documents, web, database, memory, general knowledge)
3. What's the confidence level (0-1)?
4. What's the fallback if first attempt fails?

User question: {question}

Respond in JSON format:
{{
  "intent": "...",
  "sources": ["documents", "web"],
  "confidence": 0.85,
  "strategy": "Search documents first, then web if no good matches"
}}
"""

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

CRITIC_PROMPT = """
You are the critic agent. Review this answer for quality.

Question: {question}
Retrieved context: {context}
Proposed answer: {answer}

Check:
1. Is every claim supported by context?
2. Are there hallucinations?
3. Missing important info?
4. Overall confidence (0-100)

Format:
{{
  "valid": true/false,
  "confidence": 85,
  "issues": ["..."],
  "needs_more_info": false
}}
"""

ANSWER_PROMPT = """
You are the answer agent. Answer the user's question using ONLY the
context below. Write your answer as plain text - do not use JSON,
do not add a "Sources:" section, do not use markdown code fences.
Just write the answer as you would say it out loud.

If the context doesn't actually contain information relevant to the
question, say so plainly instead of guessing.

Question: {question}

Context:
{context}

Your answer (plain text only):
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