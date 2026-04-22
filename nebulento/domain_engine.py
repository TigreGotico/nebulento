"""Domain-aware intent container for hierarchical intent organisation."""

from collections import defaultdict
from typing import Dict, List, Optional

from nebulento.container import IntentContainer, MatchResult
from nebulento.fuzz import MatchStrategy


class DomainIntentContainer:
    """Two-level intent engine: domain classification followed by intent matching.

    Intents are grouped into *domains*.  At query time the engine first selects
    the most likely domain, then runs the domain-specific intent container to
    find the best intent within that domain.

    Domains can also be selected explicitly, bypassing the top-level classifier.

    Example::

        from nebulento import DomainIntentContainer

        d = DomainIntentContainer()
        d.register_domain_intent("media", "play", ["play {song}", "put on {song}"])
        d.register_domain_intent("home",  "lights_on", ["lights on", "turn on the lights"])

        # teach the domain classifier what each domain sounds like
        d.domain_engine.add_intent("media", ["play music", "next track"])
        d.domain_engine.add_intent("home",  ["lights on", "thermostat"])

        result = d.calc_intent("play some jazz")
        # result["name"] == "play"

    Args:
        fuzzy_strategy: Similarity algorithm forwarded to every
            :class:`~nebulento.container.IntentContainer` created internally.
        ignore_case: Case-folding flag forwarded to child containers.
    """

    def __init__(self, fuzzy_strategy: MatchStrategy = MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                 ignore_case: bool = True) -> None:
        self.fuzzy_strategy = fuzzy_strategy
        self.ignore_case = ignore_case
        #: Top-level classifier that maps free-text queries to a domain name.
        self.domain_engine: IntentContainer = IntentContainer(
            fuzzy_strategy=fuzzy_strategy, ignore_case=ignore_case
        )
        #: Per-domain intent containers, keyed by domain name.
        self.domains: Dict[str, IntentContainer] = {}
        #: Raw training samples accumulated per domain (for inspection / re-training).
        self.training_data: Dict[str, List[str]] = defaultdict(list)

    # ── domain management ──────────────────────────────────────────────────

    def remove_domain(self, domain_name: str) -> None:
        """Remove a domain and all its intents, entities, and training data.

        Args:
            domain_name: Domain to remove.
        """
        self.training_data.pop(domain_name, None)
        self.domains.pop(domain_name, None)
        if domain_name in self.domain_engine.intent_names:
            self.domain_engine.remove_intent(domain_name)

    # ── intent management ──────────────────────────────────────────────────

    def register_domain_intent(self, domain_name: str, intent_name: str,
                                intent_samples: List[str]) -> None:
        """Register an intent inside a domain.

        Creates the domain's :class:`~nebulento.container.IntentContainer`
        on first use.

        Args:
            domain_name: Target domain (created if it does not exist).
            intent_name: Unique intent name within the domain.
            intent_samples: Training templates for the intent.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(
                fuzzy_strategy=self.fuzzy_strategy, ignore_case=self.ignore_case
            )
        self.domains[domain_name].add_intent(intent_name, intent_samples)
        self.training_data[domain_name] += intent_samples

    def remove_domain_intent(self, domain_name: str, intent_name: str) -> None:
        """Remove a specific intent from a domain.

        Args:
            domain_name: Domain that owns the intent.
            intent_name: Intent to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)

    # ── entity management ──────────────────────────────────────────────────

    def register_domain_entity(self, domain_name: str, entity_name: str,
                                entity_samples: List[str]) -> None:
        """Register an entity inside a domain.

        Creates the domain's container on first use.

        Args:
            domain_name: Target domain.
            entity_name: Entity name.
            entity_samples: Sample values for the entity.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(
                fuzzy_strategy=self.fuzzy_strategy, ignore_case=self.ignore_case
            )
        self.domains[domain_name].add_entity(entity_name, entity_samples)

    def remove_domain_entity(self, domain_name: str, entity_name: str) -> None:
        """Remove a specific entity from a domain.

        Args:
            domain_name: Domain that owns the entity.
            entity_name: Entity to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_entity(entity_name)

    # ── query API ──────────────────────────────────────────────────────────

    def calc_domain(self, query: str) -> MatchResult:
        """Classify *query* into the best-matching domain.

        Uses the top-level :attr:`domain_engine` which should be trained with
        representative utterances per domain via ``domain_engine.add_intent``.

        Args:
            query: Raw utterance to classify.

        Returns:
            :class:`~nebulento.container.MatchResult` dict whose ``name`` key
            is the predicted domain name (or ``None`` if no domain matched).
        """
        return self.domain_engine.calc_intent(query)

    def calc_intent(self, query: str,
                    domain: Optional[str] = None) -> MatchResult:
        """Return the best-matching intent for *query*, optionally within *domain*.

        If *domain* is ``None``, the domain is inferred by :meth:`calc_domain`.
        If the inferred or supplied domain has no registered intents, a no-match
        result is returned.

        Args:
            query: Raw utterance to evaluate.
            domain: Domain to restrict matching to.  ``None`` triggers automatic
                domain classification.

        Returns:
            :class:`~nebulento.container.MatchResult` dict.  ``name`` is
            ``None`` when no domain or intent could be matched.
        """
        resolved_domain: Optional[str] = domain or self.domain_engine.calc_intent(query).get("name")  # type: ignore[assignment]
        if resolved_domain in self.domains:
            return self.domains[resolved_domain].calc_intent(query)
        return {
            "best_match": None,
            "conf": 0.0,
            "entities": {},
            "match_strategy": self.fuzzy_strategy.name,
            "utterance": query,
            "utterance_consumed": "",
            "utterance_remainder": "",
            "name": None,
        }
