import logging
from nebulento.fuzz import MatchStrategy, match_one
from nebulento.bracket_expansion import expand_template, expand_slots
import quebra_frases

LOG = logging.getLogger('nebulento')


class IntentContainer:
    def __init__(self, fuzzy_strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                 ignore_case=True):
        self.fuzzy_strategy = fuzzy_strategy
        self.ignore_case = ignore_case
        self.registered_intents = {}
        self.registered_entities = {}

    def match_entities(self, sentence):
        if self.ignore_case:
            sentence = sentence.lower()
        matches = {}
        for entity, samples in self.registered_entities.items():
            chunked = quebra_frases.chunk(sentence, samples)
            matches[entity] = [s for s in samples if s in chunked]
        return matches

    def match_fuzzy(self, sentence):
        if self.ignore_case:
            sentence = sentence.lower()
        entities = self.match_entities(sentence)
        for intent, samples in self.registered_intents.items():
            samples = self.registered_intents[intent]

            sent, score = match_one(sentence, samples,
                                    strategy=self.fuzzy_strategy)
            remainder = [
                w for w in quebra_frases.word_tokenize(sentence)
                if w not in quebra_frases.word_tokenize(sent)]
            consumed = [
                w for w in quebra_frases.word_tokenize(sentence)
                if w in quebra_frases.word_tokenize(sent)]

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
        for l in lines:
            expanded += expand_template(l)
        if self.ignore_case:
            expanded = [l.lower() for l in expanded]
        self.registered_intents[name] = expanded

    def remove_intent(self, name):
        if name in self.registered_intents:
            del self.registered_intents[name]

    def add_entity(self, name, lines):
        expanded = []
        for l in lines:
            expanded += expand_template(l)
        if self.ignore_case:
            expanded = [l.lower() for l in expanded]
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
