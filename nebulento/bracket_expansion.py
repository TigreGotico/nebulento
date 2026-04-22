import itertools
import re
from typing import List, Dict

# Apostrophe variants — replaced with a space to preserve word boundaries
# "it's" -> "it s", not "its", so both sides of a match reduce identically
_APOSTROPHES = (
    "'",   # U+0027 ASCII apostrophe
    "’",  # RIGHT SINGLE QUOTATION MARK
    "‘",  # LEFT SINGLE QUOTATION MARK
    "ʼ",  # MODIFIER LETTER APOSTROPHE
    "ʹ",  # MODIFIER LETTER PRIME
    "`",   # U+0060 GRAVE ACCENT
    "´",  # ACUTE ACCENT
    "＇",  # FULLWIDTH APOSTROPHE
)
_APOS_RE = re.compile("|".join(re.escape(a) for a in _APOSTROPHES))
_WS_RE = re.compile(r"\s+")


def _drop_apostrophes(text: str) -> str:
    return _APOS_RE.sub(" ", text)


def _normalize_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def clean_braces(example: str) -> str:
    """Normalise {{entity}} → {entity}."""
    return example.replace("{{", "{").replace("}}", "}")


def translate_padatious(example: str) -> str:
    """Translate Padatious :0 word-slot syntax to {word0:word} format."""
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
    """Normalise a training template (preserves {entity} placeholders)."""
    text = clean_braces(translate_padatious(example))
    text = _drop_apostrophes(text)
    text = _normalize_whitespace(text)
    return text


def normalize_utterance(text: str) -> str:
    """Normalise a plain query utterance for consistent matching."""
    text = _drop_apostrophes(text)
    text = _normalize_whitespace(text)
    return text


def expand_template(template: str) -> List[str]:
    def expand_optional(text):
        return re.sub(r"\[([^\[\]]+)\]", lambda m: f"({m.group(1)}|)", text)

    def expand_alternatives(text):
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
            expanded = set()
            for text in result:
                options = list(expand_alternatives(text))
                expanded.update([
                    re.sub(r" +", " ", "".join(option)).strip()
                    for option in options
                ])
            if expanded == result:
                break
            result = expanded
        return sorted(result)

    return fully_expand([expand_optional(template)])


def expand_slots(template: str, slots: Dict[str, List[str]]) -> List[str]:
    """Expand a template substituting slot placeholders with their sample values."""
    all_sentences = []
    for sentence in expand_template(template):
        matches = re.findall(r"\{([^\{\}]+)\}", sentence)
        if matches:
            slot_options = [slots.get(match, [f"{{{match}}}"]) for match in matches]
            for combination in itertools.product(*slot_options):
                filled = sentence
                for slot, replacement in zip(matches, combination):
                    filled = filled.replace(f"{{{slot}}}", replacement)
                all_sentences.append(filled)
        else:
            all_sentences.append(sentence)
    return all_sentences
