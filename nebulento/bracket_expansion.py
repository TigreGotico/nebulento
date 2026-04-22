"""Template expansion and text normalisation utilities."""

import itertools
import re
from typing import Dict, List

# Apostrophe variants replaced with a space to preserve word boundaries.
# "it's" → "it s" so both sides of a match normalise identically.
_APOSTROPHES = (
    "'",   # U+0027 ASCII apostrophe
    "’",  # RIGHT SINGLE QUOTATION MARK
    "‘",  # LEFT SINGLE QUOTATION MARK
    "ʼ",  # MODIFIER LETTER APOSTROPHE
    "ʹ",  # MODIFIER LETTER PRIME
    "`",        # U+0060 GRAVE ACCENT
    "´",  # ACUTE ACCENT
    "＇",  # FULLWIDTH APOSTROPHE
)
_APOS_RE = re.compile("|".join(re.escape(a) for a in _APOSTROPHES))
_WS_RE = re.compile(r"\s+")


def _drop_apostrophes(text: str) -> str:
    """Replace all apostrophe variants with a single space."""
    return _APOS_RE.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    return _WS_RE.sub(" ", text).strip()


def clean_braces(example: str) -> str:
    """Normalise accidental double-braces: ``{{entity}}`` → ``{entity}``.

    Args:
        example: Raw training template string.

    Returns:
        Template with ``{{``/``}}`` collapsed to single braces.
    """
    return example.replace("{{", "{").replace("}}", "}")


def translate_padatious(example: str) -> str:
    """Translate Padatious ``:0`` word-slot tokens to ``{word0}`` entity syntax.

    Allows intent files written for Padatious to be used with Nebulento without
    modification.

    Args:
        example: Template string that may contain ``:0`` tokens.

    Returns:
        Template with each ``:0`` replaced by a sequentially numbered entity
        placeholder ``{word0}``, ``{word1}``, etc.
    """
    if ":0" not in example:
        return example
    tokens = example.split()
    i = 0
    for idx, token in enumerate(tokens):
        if token == ":0":
            tokens[idx] = "{" + f"word{i}" + "}"
            i += 1
    return " ".join(tokens)


def normalize_example(example: str) -> str:
    """Normalise a training template for storage.

    Applies ``clean_braces``, ``translate_padatious``, apostrophe-dropping, and
    whitespace collapsing.  Entity placeholders (``{name}``) are preserved.

    Args:
        example: Raw training template string.

    Returns:
        Normalised template ready for ``expand_template``.
    """
    text = clean_braces(translate_padatious(example))
    text = _drop_apostrophes(text)
    return _normalize_whitespace(text)


def normalize_utterance(text: str) -> str:
    """Normalise a plain query utterance for matching.

    Applies apostrophe-dropping and whitespace collapsing.  Does **not** touch
    entity placeholder syntax.

    Args:
        text: Raw utterance string (e.g. from STT output).

    Returns:
        Normalised utterance string.
    """
    return _normalize_whitespace(_drop_apostrophes(text))


def expand_template(template: str) -> List[str]:
    """Expand a template into all concrete string variants.

    Handles:
    - ``(one|of|these)`` alternation
    - ``[optional]`` syntax (equivalent to ``(optional|)``)
    - Nested combinations of the above

    Args:
        template: Template string, e.g. ``"(play|start) [some] {song}"``.

    Returns:
        Sorted list of all expanded variants with internal whitespace collapsed.

    Example::

        expand_template("(hi|hello) [there]")
        # ["hello", "hello there", "hi", "hi there"]
    """
    def expand_optional(text: str) -> str:
        return re.sub(r"\[([^\[\]]+)\]", lambda m: f"({m.group(1)}|)", text)

    def expand_alternatives(text: str):
        parts = []
        for segment in re.split(r"(\([^\(\)]+\))", text):
            if segment.startswith("(") and segment.endswith(")"):
                parts.append(segment[1:-1].split("|"))
            else:
                parts.append([segment])
        return itertools.product(*parts)

    def fully_expand(texts):
        result = set(texts)
        while True:
            expanded = {
                re.sub(r" +", " ", "".join(option)).strip()
                for text in result
                for option in expand_alternatives(text)
            }
            if expanded == result:
                break
            result = expanded
        return sorted(result)

    return fully_expand([expand_optional(template)])


def expand_slots(template: str, slots: Dict[str, List[str]]) -> List[str]:
    """Expand a template by substituting slot placeholders with sample values.

    First expands alternation/optional syntax via :func:`expand_template`, then
    fills each ``{slot}`` placeholder with every value in *slots*, producing the
    Cartesian product of all combinations.

    Args:
        template: Template string containing ``{slot}`` placeholders.
        slots: Mapping of slot name → list of possible replacement strings.
            Slots absent from *slots* are left as-is (``{name}`` unchanged).

    Returns:
        List of all fully-expanded concrete strings.

    Example::

        expand_slots("buy {item} from {shop}", {
            "item": ["milk", "eggs"],
            "shop": ["Tesco"],
        })
        # ["buy eggs from Tesco", "buy milk from Tesco"]
    """
    all_sentences: List[str] = []
    for sentence in expand_template(template):
        matches = re.findall(r"\{([^\{\}]+)\}", sentence)
        if matches:
            slot_options = [slots.get(m, [f"{{{m}}}"]) for m in matches]
            for combination in itertools.product(*slot_options):
                filled = sentence
                for slot, replacement in zip(matches, combination):
                    filled = filled.replace(f"{{{slot}}}", replacement)
                all_sentences.append(filled)
        else:
            all_sentences.append(sentence)
    return all_sentences
