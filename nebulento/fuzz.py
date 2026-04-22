"""Fuzzy string similarity strategies wrapping rapidfuzz and difflib."""

import logging
from difflib import SequenceMatcher
from enum import IntEnum, auto
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import rapidfuzz

LOG = logging.getLogger("nebulento")

_Choices = Union[List[str], Dict[str, str]]


class MatchStrategy(IntEnum):
    """Fuzzy similarity algorithm to use when comparing utterances to templates.

    Choose based on the trade-off between precision and recall that suits your
    deployment — see the benchmark table in the README for measured F1 / FP rates.
    """

    SIMPLE_RATIO = auto()
    """Character-level Levenshtein ratio (rapidfuzz ``ratio``).  Balanced baseline."""

    RATIO = auto()
    """Identical to ``SIMPLE_RATIO`` via the rapidfuzz ``ratio`` scorer."""

    PARTIAL_RATIO = auto()
    """Best substring match score.  Very permissive — high false-positive risk."""

    TOKEN_SORT_RATIO = auto()
    """Sort tokens before comparing — handles word-order variation well."""

    TOKEN_SET_RATIO = auto()
    """Set-intersection approach — highest recall, most false positives."""

    PARTIAL_TOKEN_RATIO = auto()
    """Partial ratio applied after token splitting.  Not recommended for intent gating."""

    PARTIAL_TOKEN_SORT_RATIO = auto()
    """Token-sort then partial ratio.  Not recommended for intent gating."""

    PARTIAL_TOKEN_SET_RATIO = auto()
    """Token-set then partial ratio.  Not recommended for intent gating."""

    DAMERAU_LEVENSHTEIN_SIMILARITY = auto()
    """Edit-distance similarity including transpositions.  Zero false positives on the
    benchmark dataset — recommended default for production deployments."""


def fuzzy_match(x: str, against: str,
                strategy: MatchStrategy = MatchStrategy.SIMPLE_RATIO) -> float:
    """Compute a fuzzy similarity score between two strings.

    Args:
        x: First string.
        against: Second string to compare against.
        strategy: Algorithm to use.  Defaults to ``SIMPLE_RATIO``.

    Returns:
        Similarity score in ``[0.0, 1.0]`` — ``1.0`` is a perfect match.
    """
    if strategy == MatchStrategy.RATIO:
        return rapidfuzz.fuzz.ratio(x, against) / 100
    if strategy == MatchStrategy.PARTIAL_RATIO:
        return rapidfuzz.fuzz.partial_ratio(x, against) / 100
    if strategy == MatchStrategy.TOKEN_SORT_RATIO:
        return rapidfuzz.fuzz.token_sort_ratio(x, against) / 100
    if strategy == MatchStrategy.TOKEN_SET_RATIO:
        return rapidfuzz.fuzz.token_set_ratio(x, against) / 100
    if strategy == MatchStrategy.PARTIAL_TOKEN_SORT_RATIO:
        return rapidfuzz.fuzz.partial_token_sort_ratio(x, against) / 100
    if strategy == MatchStrategy.PARTIAL_TOKEN_SET_RATIO:
        return rapidfuzz.fuzz.partial_token_set_ratio(x, against) / 100
    if strategy == MatchStrategy.PARTIAL_TOKEN_RATIO:
        return rapidfuzz.fuzz.partial_token_ratio(x, against) / 100
    if strategy == MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY:
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(x, against)
    return SequenceMatcher(None, x, against).ratio()


def match_one(query: str, choices: _Choices,
              match_func: Optional[Callable] = None,
              strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
              ) -> Tuple[str, float]:
    """Return the single best match from *choices* for *query*.

    Args:
        query: Utterance to score against each choice.
        choices: List of candidate strings, or dict mapping display-value → key.
        match_func: Custom scoring callable ``(str, str, MatchStrategy) -> float``.
            Defaults to :func:`fuzzy_match`.
        strategy: Algorithm passed to *match_func*.

    Returns:
        ``(best_match, score)`` tuple where *score* is in ``[0.0, 1.0]``.
    """
    return match_all(query, choices, match_func, strategy)[0]


def match_all(query: str, choices: _Choices,
              match_func: Optional[Callable] = None,
              strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
              ) -> List[Tuple[str, float]]:
    """Score *query* against every choice and return all results sorted best-first.

    Args:
        query: Utterance to score.
        choices: List of candidate strings, or dict mapping display-value → key.
        match_func: Custom scoring callable ``(str, str, MatchStrategy) -> float``.
            Defaults to :func:`fuzzy_match`.
        strategy: Algorithm passed to *match_func*.

    Returns:
        List of ``(match, score)`` tuples, sorted by score descending.

    Raises:
        ValueError: If *choices* is neither a list nor a dict.
    """
    match_func = match_func or fuzzy_match
    if isinstance(choices, dict):
        _choices: Sequence[str] = list(choices.keys())
    elif isinstance(choices, list):
        _choices = choices
    else:
        raise ValueError("choices must be a list or dict")

    matches: List[Tuple[str, float]] = []
    for c in _choices:
        if isinstance(choices, dict):
            matches.append((choices[c], match_func(query, c, strategy)))
        else:
            matches.append((c, match_func(query, c, strategy)))
    return sorted(matches, key=lambda k: k[1], reverse=True)
