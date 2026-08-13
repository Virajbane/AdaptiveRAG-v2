"""
Natural-language arithmetic: intent detection + expression construction.

STAGE 14 FIX (2026-08-14):
  Complete rewrite to support:
  - N-operand expressions (not just 2)
  - Percentage syntax ("18% of 3500" → "3500 * 18 / 100")
  - Chained operations ("divide 1440 by 24 and add 35" → "1440 / 24 + 35")
  - Averaging all operands ("average of 18, 24, 30, 36" → "(18+24+30+36)/4")
  
  Design principle: intent detection remains generic (no hardcoded sentences),
  but expression building now handles the full range of NL arithmetic patterns.

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

# Percentage syntax: "18% of 3500" or "18 percent of 3500"
_PERCENTAGE_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)|"
    r"(\d+(?:\.\d+)?)\s+percent\s+of\s+(\d+(?:\.\d+)?)",
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
    arithmetic question.
    
    Examples:
    - "I have 250 items and remove 37" → "250 - 37"
    - "average of 18, 24, 30, 36" → "(18+24+30+36)/4"
    - "18% of 3500" → "3500 * 18 / 100"
    - "divide 1440 by 24 and add 35" → "1440 / 24 + 35"

    Returns None if no confident expression can be built (caller should
    fall back to its existing symbol-based extraction / error handling).
    """
    
    # ── PERCENTAGE: "18% of 3500" or "18 percent of 3500" ─────────────
    pct_match = _PERCENTAGE_OF.search(question)
    if pct_match:
        # Groups: (percent, base) or (None, None, percent, base)
        percent = pct_match.group(1) or pct_match.group(3)
        base = pct_match.group(2) or pct_match.group(4)
        if percent and base:
            return f"{base} * {percent} / 100"
    
    # ── REVERSED SUBTRACTION: "remove Y from X" → "X - Y" ─────────────
    m = _REVERSED_SUBTRACT.search(question)
    if m:
        subtrahend, base = m.group(1), m.group(2)
        return f"{base} - {subtrahend}"
    
    # ── AVERAGE: "average of 1, 2, 3, 4" → "(1+2+3+4)/4" ──────────────
    if _AVERAGE_WORDS.search(question):
        numbers = _NUMBER.findall(question)
        if len(numbers) >= 2:
            return f"({'+'.join(numbers)})/{len(numbers)}"
        return None
    
    # ── CHAINED OPERATIONS: "divide X by Y and add Z" ──────────────────
    # Strategy: if multiple operation words are present, chain them
    chained = _try_build_chained_expression(question)
    if chained:
        return chained
    
    # ── SINGLE OPERATION (fallback) ──────────────────────────────────────
    # Use all available numbers with the detected operation
    numbers = _NUMBER.findall(question)
    if len(numbers) < 2:
        return None
    
    # Priority: explicit operation verb determines which to use
    if _SUBTRACT_WORDS.search(question):
        # For subtraction with multiple numbers, use: a - b - c - ...
        # But for 2 numbers, just "a - b"
        if len(numbers) == 2:
            return f"{numbers[0]} - {numbers[1]}"
        else:
            # For multiple subtractions, chain them left-to-right
            return _build_expression_with_operator(numbers, "-")
    
    if _MULTIPLY_WORDS.search(question):
        # Multiplication: a * b * c
        return _build_expression_with_operator(numbers, "*")
    
    if _DIVIDE_WORDS.search(question):
        # Division: a / b / c (evaluated left-to-right)
        return _build_expression_with_operator(numbers, "/")
    
    if _ADD_WORDS.search(question):
        # Addition: a + b + c
        return _build_expression_with_operator(numbers, "+")
    
    return None


def _build_expression_with_operator(numbers: list[str], op: str) -> str:
    """
    Build expression with N operands and the same operator.
    E.g., [10, 20, 30] + "+" → "10 + 20 + 30"
    """
    if len(numbers) == 1:
        return numbers[0]
    return f" {op} ".join(numbers)


def _try_build_chained_expression(question: str) -> Optional[str]:
    """
    Handle chained operations like "divide 1440 by 24 and add 35".
    
    Strategy:
    1. Find all numbers in order: [1440, 24, 35]
    2. Find all operation words in order: [divide, add]
    3. Chain them: "1440 / 24 + 35"
    
    Returns None if pattern doesn't match (caller tries single operation).
    """
    numbers = _NUMBER.findall(question)
    if len(numbers) < 3:
        # Chained ops need at least 3 operands
        return None
    
    # Find all operations mentioned, in order
    # (this is heuristic: assumes operations appear roughly in numeric order)
    operations = []
    
    q_lower = question.lower()
    # Mark positions of each operation in the question string
    op_positions = []
    
    for match in _DIVIDE_WORDS.finditer(q_lower):
        op_positions.append((match.start(), "/"))
    for match in _MULTIPLY_WORDS.finditer(q_lower):
        op_positions.append((match.start(), "*"))
    for match in _SUBTRACT_WORDS.finditer(q_lower):
        op_positions.append((match.start(), "-"))
    for match in _ADD_WORDS.finditer(q_lower):
        op_positions.append((match.start(), "+"))
    
    # Sort by position in the question (appears left-to-right)
    op_positions.sort()
    operations = [op for _, op in op_positions]
    
    # If we have fewer operations than needed (n-1 operations for n numbers),
    # fall back to single operation detection
    if len(operations) < len(numbers) - 1:
        return None
    
    # Build the expression: number1 op1 number2 op2 number3 ...
    if len(operations) >= len(numbers) - 1:
        # Take only the first n-1 operations (for n numbers)
        expr = str(numbers[0])
        for i, op in enumerate(operations[:len(numbers)-1]):
            expr += f" {op} {numbers[i+1]}"
        return expr
    
    return None