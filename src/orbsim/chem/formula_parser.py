from __future__ import annotations

import re
from collections import defaultdict

from orbsim.chem.elements import SYMBOL_TO_ELEMENT


_TOKEN_PATTERN = re.compile(r"[A-Z][a-z]?|\d+|[()]")


def parse_formula(formula: str) -> dict[str, int]:
    if not formula or not isinstance(formula, str):
        raise ValueError("Formula must be a non-empty string.")
    tokens = _TOKEN_PATTERN.findall(formula)
    if not tokens:
        raise ValueError("Formula contains no valid tokens.")

    stack: list[defaultdict[str, int]] = [defaultdict(int)]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "(":
            stack.append(defaultdict(int))
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("Unmatched closing parenthesis.")
            group = stack.pop()
            multiplier = 1
            if index + 1 < len(tokens) and tokens[index + 1].isdigit():
                multiplier = int(tokens[index + 1])
                index += 1
            for symbol, count in group.items():
                stack[-1][symbol] += count * multiplier
        elif token.isdigit():
            raise ValueError(f"Unexpected number token '{token}'.")
        else:
            if token not in SYMBOL_TO_ELEMENT:
                raise ValueError(f"Unknown element symbol '{token}'.")
            count = 1
            if index + 1 < len(tokens) and tokens[index + 1].isdigit():
                count = int(tokens[index + 1])
                index += 1
            stack[-1][token] += count
        index += 1

    if len(stack) != 1:
        raise ValueError("Unmatched opening parenthesis.")

    return dict(stack[0])


def expand_formula_to_atomic_numbers(formula: str) -> list[int]:
    counts = parse_formula(formula)
    elements: list[int] = []
    for symbol, count in counts.items():
        element = SYMBOL_TO_ELEMENT[symbol]
        elements.extend([element.atomic_number] * count)
    return elements
