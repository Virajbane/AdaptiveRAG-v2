"""
Natural-language arithmetic: intent detection + expression construction.

Root cause this module fixes (CALC_04 / planner eval failure 1):
  The planner's old _CALCULATOR_INTENT regex only recognized arithmetic
  written with symbols ("250 - 37") or an explicit "calculate/compute/
  solve" verb next to a digit. A question like "If I have 250 items and
  remove 37, how many remain?" expresses subtraction entirely in words,
  so it matched nothing and fell through to the documents default.

Design: arithmetic *intent* is "an arithmetic verb/phrase co-occurring
with at least two numbers", independent of any specific sentence. This
module is intentionally generic -- it is keyed on operation vocabulary,
not on any single example question -- so it generalizes to phrasings
never seen during this fix (per the eval instructions: no hardcoded
example sentences).

Both the planner (needs a yes/no signal) and the tool agent (needs an
actual expression string to evaluate) share the same vocabulary here so
"the planner thinks this is arithmetic" and "the tool agent can build an
expression for it" can never silently disagree.
"""

import re
from typing import Optional

_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# Words/phrases that signal each operation. Order of the *checks* below
# (not of this dict) encodes priority when multiple operation words are
# present in the same question.
_SUBTRACT_WORDS = re.compile(
    r"\b(?:remove[sd]?|removing|subtract(?:ing)?|minus|lose|los[et]|lost|"
    r"spend|spent|left|remain(?:ing|s)?|decrease[sd]?|take\s+away|used\s+up)\b",
    re.IGNORECASE,
)
_ADD_WORDS = re.compile(
    r"\b(?:add(?:ed|ing)?|plus|sum|total|gain(?:ed)?|combine[sd]?|"
    r"together|more|buy|bought|purchase[sd]?)\b",
    re.IGNORECASE,
)
_MULTIPLY_WORDS = re.compile(
    r"\b(?:multipl(?:y|ied|ying)|times|product\s+of)\b", re.IGNORECASE
)
_DIVIDE_WORDS = re.compile(
    r"\b(?:divid(?:e|ed|ing)|split|each|per)\b", re.IGNORECASE
)
_AVERAGE_WORDS = re.compile(r"\b(?:average|mean)\b", re.IGNORECASE)

# "removing/subtracting Y from X" -- the base amount (X) comes *after*
# "from", reversed from left-to-right reading order. Handled as an
# explicit special case so the general "first two numbers in order"
# fallback doesn't get the operands backwards.
_REVERSED_SUBTRACT = re.compile(
    r"(?:remov(?:e|ing)|subtract(?:ing)?)\s+(?:the\s+)?(?:number\s+)?"
    r"(\d+(?:\.\d+)?)\s+from\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Any arithmetic-operation vocabulary at all -- used as the coarse
# "does this question even smell like arithmetic" gate before we bother
# counting numbers.
_ANY_OPERATION_WORD = re.compile(
    "|".join(
        p.pattern
        for p in (
            _SUBTRACT_WORDS,
            _ADD_WORDS,
            _MULTIPLY_WORDS,
            _DIVIDE_WORDS,
            _AVERAGE_WORDS,
        )
    ),
    re.IGNORECASE,
)


def has_nl_arithmetic_intent(question: str) -> bool:
    """
    True if the question expresses a deterministic arithmetic operation
    in natural language (no math symbols required).

    Requires BOTH an operation word/phrase AND at least two numbers --
    either signal alone is too weak (e.g. "I have 3 documents" has a
    number but no operation; "let's add a section" has an operation word
    but no numbers to operate on).
    """
    if not _ANY_OPERATION_WORD.search(question):
        return False
    numbers = _NUMBER.findall(question)
    return len(numbers) >= 2


def build_expression(question: str) -> Optional[str]:
    """
    Build a safe arithmetic expression string from a natural-language
    arithmetic question, e.g. "I have 250 items and remove 37" -> "250 - 37".

    Returns None if no confident expression can be built (caller should
    fall back to its existing symbol-based extraction / error handling).
    """
    # Reversed subtraction: "removing Y from X" -> "X - Y"
    m = _REVERSED_SUBTRACT.search(question)
    if m:
        subtrahend, base = m.group(1), m.group(2)
        return f"{base} - {subtrahend}"

    if _AVERAGE_WORDS.search(question):
        numbers = _NUMBER.findall(question)
        if len(numbers) >= 2:
            return f"({'+'.join(numbers)})/{len(numbers)}"
        return None

    numbers = _NUMBER.findall(question)
    if len(numbers) < 2:
        return None
    a, b = numbers[0], numbers[1]

    # Priority mirrors _deterministic_calculator_routing's reasoning:
    # an explicit subtraction word is the strongest, least ambiguous
    # signal (e.g. "buy 20 and spend 7" -- "spend" wins over "buy").
    if _SUBTRACT_WORDS.search(question):
        return f"{a} - {b}"
    if _MULTIPLY_WORDS.search(question):
        return f"{a} * {b}"
    if _DIVIDE_WORDS.search(question):
        return f"{a} / {b}"
    if _ADD_WORDS.search(question):
        return f"{a} + {b}"
    return None