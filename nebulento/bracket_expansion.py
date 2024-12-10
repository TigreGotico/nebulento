import itertools
import re
from typing import List, Dict


def expand_template(template: str) -> List[str]:
    def expand_optional(text):
        """Replace [optional] with two options: one with and one without."""
        return re.sub(r"\[([^\[\]]+)\]", lambda m: f"({m.group(1)}|)", text)

    def expand_alternatives(text):
        """Expand (alternative|choices) into a list of choices."""
        parts = []
        for segment in re.split(r"(\([^\(\)]+\))", text):
            if segment.startswith("(") and segment.endswith(")"):
                options = segment[1:-1].split("|")
                parts.append(options)
            else:
                parts.append([segment])
        return itertools.product(*parts)

    def fully_expand(texts):
        """Iteratively expand alternatives until all possibilities are covered."""
        result = set(texts)
        while True:
            expanded = set()
            for text in result:
                options = list(expand_alternatives(text))
                expanded.update(["".join(option).strip() for option in options])
            if expanded == result:  # No new expansions found
                break
            result = expanded
        return sorted(result)  # Return a sorted list for consistency

    # Expand optional items first
    template = expand_optional(template)

    # Fully expand all combinations of alternatives
    return fully_expand([template])


def expand_slots(template: str, slots: Dict[str, List[str]]) -> List[str]:
    """Expand a template by first expanding alternatives and optional components,
    then substituting slot placeholders with their corresponding options.

    Args:
        template (str): The input string template to expand.
        slots (dict): A dictionary where keys are slot names and values are lists of possible replacements.

    Returns:
        list[str]: A list of all expanded combinations.
    """
    # Expand alternatives and optional components
    base_expansions = expand_template(template)

    # Process slots
    all_sentences = []
    for sentence in base_expansions:
        matches = re.findall(r"\{([^\{\}]+)\}", sentence)
        if matches:
            # Create all combinations for slots in the sentence
            slot_options = [slots.get(match, [f"{{{match}}}"]) for match in matches]
            for combination in itertools.product(*slot_options):
                filled_sentence = sentence
                for slot, replacement in zip(matches, combination):
                    filled_sentence = filled_sentence.replace(f"{{{slot}}}", replacement)
                all_sentences.append(filled_sentence)
        else:
            # No slots to expand
            all_sentences.append(sentence)

    return all_sentences
