import re
import logging
from typing import List, Iterator, Optional

from nebulento.fuzz import MatchStrategy, match_one
from nebulento.bracket_expansion import expand_template, normalize_example, normalize_utterance
import quebra_frases

LOG = logging.getLogger("nebulento")


class IntentContainer:
    def __init__(self, fuzzy_strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                 ignore_case: bool = True):
        self.fuzzy_strategy = fuzzy_strategy
        self.ignore_case = ignore_case
        self.registered_intents: dict = {}
        self.registered_entities: dict = {}
        self.available_contexts: dict = {}
        self.required_contexts: dict = {}
        self.excluded_contexts: dict = {}
        self.excluded_keywords: dict = {}

    # ── public properties ──────────────────────────────────────────────────

    @property
    def intent_names(self) -> List[str]:
        return list(self.registered_intents)

    # ── normalisation ──────────────────────────────────────────────────────

    def _norm(self, text: str) -> str:
        text = normalize_utterance(text)
        if self.ignore_case:
            text = text.lower()
        return text

    # ── registration ───────────────────────────────────────────────────────

    def add_intent(self, name: str, lines: List[str]):
        if name in self.registered_intents:
            raise RuntimeError(f"Intent already registered: {name!r}. Call remove_intent first.")
        expanded = set()
        for line in lines:
            for e in expand_template(normalize_example(line)):
                expanded.add(self._norm(e))
        self.registered_intents[name] = list(expanded)

    def remove_intent(self, name: str):
        self.registered_intents.pop(name, None)

    def add_entity(self, name: str, lines: List[str]):
        name = name.lower()
        if name in self.registered_entities:
            raise RuntimeError(f"Entity already registered: {name!r}. Call remove_entity first.")
        expanded = set()
        for line in lines:
            for e in expand_template(line):
                expanded.add(self._norm(e))
        self.registered_entities[name] = list(expanded)

    def remove_entity(self, name: str):
        self.registered_entities.pop(name.lower(), None)

    # ── context / keyword gating ───────────────────────────────────────────

    def set_context(self, intent_name: str, context_name: str, context_val=None):
        self.available_contexts.setdefault(intent_name, {})[context_name] = context_val

    def unset_context(self, intent_name: str, context_name: str):
        if intent_name in self.available_contexts:
            self.available_contexts[intent_name].pop(context_name, None)

    def require_context(self, intent_name: str, context_name: str):
        self.required_contexts.setdefault(intent_name, []).append(context_name)

    def unrequire_context(self, intent_name: str, context_name: str):
        if intent_name in self.required_contexts:
            self.required_contexts[intent_name] = [
                c for c in self.required_contexts[intent_name] if c != context_name
            ]

    def exclude_context(self, intent_name: str, context_name: str):
        self.excluded_contexts.setdefault(intent_name, []).append(context_name)

    def unexclude_context(self, intent_name: str, context_name: str):
        if intent_name in self.excluded_contexts:
            self.excluded_contexts[intent_name] = [
                c for c in self.excluded_contexts[intent_name] if c != context_name
            ]

    def exclude_keywords(self, intent_name: str, samples: List[str]):
        self.excluded_keywords.setdefault(intent_name, [])
        self.excluded_keywords[intent_name] += samples

    # ── filtering ──────────────────────────────────────────────────────────

    def _filter(self, query: str) -> List[str]:
        excluded = []
        q_lower = query.lower()
        query_words = set(q_lower.split())

        for intent_name, keywords in self.excluded_keywords.items():
            def _hit(kw, _qw=query_words, _ql=q_lower):
                if " " not in kw:
                    return kw.lower() in _qw
                return bool(re.search(r"\b" + re.escape(kw.lower()) + r"\b", _ql))
            if any(_hit(kw) for kw in keywords):
                excluded.append(intent_name)

        for intent_name, contexts in self.required_contexts.items():
            avail = self.available_contexts.get(intent_name, {})
            if any(c not in avail for c in contexts):
                excluded.append(intent_name)

        for intent_name, contexts in self.excluded_contexts.items():
            avail = self.available_contexts.get(intent_name, {})
            if any(c in avail for c in contexts):
                excluded.append(intent_name)

        return excluded

    # ── matching ───────────────────────────────────────────────────────────

    def match_entities(self, sentence: str) -> dict:
        sentence = self._norm(sentence)
        matches = {}
        for entity, samples in self.registered_entities.items():
            chunked = quebra_frases.chunk(sentence, samples)
            matches[entity] = [s for s in samples if s in chunked]
        return matches

    def match_fuzzy(self, sentence: str) -> Iterator[dict]:
        sentence = self._norm(sentence)
        entities = self.match_entities(sentence)
        excluded = self._filter(sentence)

        for intent, samples in self.registered_intents.items():
            if intent in excluded:
                continue
            if not samples:
                continue

            sent, score = match_one(sentence, samples, strategy=self.fuzzy_strategy)
            sentence_tokens = quebra_frases.word_tokenize(sentence)
            sent_tokens = quebra_frases.word_tokenize(sent)
            remainder = [w for w in sentence_tokens if w not in sent_tokens]
            consumed = [w for w in sentence_tokens if w in sent_tokens]

            tagged_entities = {}
            for ent, v in entities.items():
                if v and any("{" + ent + "}" in s for s in samples):
                    score = 0.25 + score * 0.75
                    tagged_entities[ent] = v
                    consumed += [w for w in v if w not in consumed]
                    remainder = [w for w in remainder if w not in v]

            # lowercase entity keys for consistent output
            tagged_entities = {k.lower(): v for k, v in tagged_entities.items()}

            yield {
                "best_match": sent,
                "conf": min(score, 1.0),
                "entities": tagged_entities,
                "match_strategy": self.fuzzy_strategy.name,
                "utterance": sentence,
                "utterance_remainder": " ".join(remainder),
                "utterance_consumed": " ".join(consumed),
                "name": intent,
            }

    # ── public query API ───────────────────────────────────────────────────

    def calc_intents(self, query: str) -> Iterator[dict]:
        yield from self.match_fuzzy(query)

    def calc_intent(self, query: str) -> Optional[dict]:
        _EMPTY = {
            "best_match": None,
            "conf": 0.0,
            "entities": {},
            "match_strategy": self.fuzzy_strategy.name,
            "utterance": self._norm(query),
            "utterance_consumed": "",
            "utterance_remainder": "",
            "name": None,
        }
        candidates = list(self.calc_intents(query))
        if not candidates:
            return _EMPTY

        best_conf = max(r["conf"] for r in candidates)
        ties = [r for r in candidates if r["conf"] == best_conf]

        if len(ties) > 1:
            LOG.debug("tied intents: %s", [t["name"] for t in ties])
            # more consumed words = more specific match; name breaks remaining ties
            ties.sort(key=lambda t: (-len(t["utterance_consumed"].split()), t["name"]))

        return ties[0]
