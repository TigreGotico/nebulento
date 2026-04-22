import re
import logging
from nebulento.fuzz import MatchStrategy, match_one
from nebulento.bracket_expansion import expand_template
import quebra_frases

LOG = logging.getLogger('nebulento')

_APOS = re.compile(r"[''ʼ]")
_WS = re.compile(r"\s+")


def _normalize(text: str, ignore_case: bool = True) -> str:
    text = _APOS.sub("", text)
    text = _WS.sub(" ", text).strip()
    if ignore_case:
        text = text.lower()
    return text


class IntentContainer:
    def __init__(self, fuzzy_strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                 ignore_case=True):
        self.fuzzy_strategy = fuzzy_strategy
        self.ignore_case = ignore_case
        self.registered_intents = {}
        self.registered_entities = {}

    @property
    def intent_names(self):
        return list(self.registered_intents)

    def match_entities(self, sentence):
        sentence = _normalize(sentence, self.ignore_case)
        matches = {}
        for entity, samples in self.registered_entities.items():
            chunked = quebra_frases.chunk(sentence, samples)
            matches[entity] = [s for s in samples if s in chunked]
        return matches

    def match_fuzzy(self, sentence):
        sentence = _normalize(sentence, self.ignore_case)
        entities = self.match_entities(sentence)
        for intent, samples in self.registered_intents.items():

            sent, score = match_one(sentence, samples,
                                    strategy=self.fuzzy_strategy)
            sentence_tokens = quebra_frases.word_tokenize(sentence)
            sent_tokens = quebra_frases.word_tokenize(sent)
            remainder = [w for w in sentence_tokens if w not in sent_tokens]
            consumed = [w for w in sentence_tokens if w in sent_tokens]

            tagged_entities = {}
            for ent, v in entities.items():
                if v and any("{" + ent + "}" in s for s in samples):
                    score = 0.25 + score * 0.75
                    tagged_entities[ent] = v
                    consumed += [_ for _ in v if _ not in consumed]
                    remainder = [_ for _ in remainder if _ not in v]
            remainder = " ".join(remainder)
            consumed = " ".join(consumed)
            yield {"best_match": sent,
                   "conf": min(score, 1),
                   "entities": tagged_entities,
                   "match_strategy": self.fuzzy_strategy.name,
                   "utterance": sentence,
                   "utterance_remainder": remainder,
                   "utterance_consumed": consumed,
                   "name": intent}

    def add_intent(self, name, lines):
        expanded = []
        for line in lines:
            expanded += expand_template(line)
        expanded = [_normalize(line, self.ignore_case) for line in expanded]
        self.registered_intents[name] = expanded

    def remove_intent(self, name):
        if name in self.registered_intents:
            del self.registered_intents[name]

    def add_entity(self, name, lines):
        expanded = []
        for line in lines:
            expanded += expand_template(line)
        expanded = [_normalize(line, self.ignore_case) for line in expanded]
        self.registered_entities[name] = expanded

    def remove_entity(self, name):
        if name in self.registered_entities:
            del self.registered_entities[name]

    def calc_intents(self, query):
        for intent in self.match_fuzzy(query):
            yield intent

    def calc_intent(self, query):
        return max(
            self.calc_intents(query),
            key=lambda x: x["conf"],
            default={"best_match": None,
                     "conf": 0,
                     "match_strategy": self.fuzzy_strategy,
                     "utterance": query,
                     "name": None}
        )
