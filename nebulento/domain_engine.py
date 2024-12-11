from collections import defaultdict
from typing import Dict, List, Optional

from nebulento.container import IntentContainer
from nebulento.fuzz import MatchStrategy


class DomainIntentContainer:
    """
    A domain-aware intent recognition engine that organizes intents and entities
    into specific domains, providing flexible and hierarchical intent matching.
    """

    def __init__(self, fuzzy_strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY,
                 ignore_case=True):
        """
        Initialize the DomainIntentContainer.

        Attributes:
            domain_engine (IntentContainer): A top-level intent container for cross-domain calculations.
            domains (Dict[str, IntentContainer]): A mapping of domain names to their respective intent containers.
            training_data (Dict[str, List[str]]): A mapping of domain names to their associated training samples.
        """
        self.fuzzy_strategy = fuzzy_strategy
        self.ignore_case = ignore_case
        self.domain_engine = IntentContainer(fuzzy_strategy=fuzzy_strategy, ignore_case=ignore_case)
        self.domains: Dict[str, IntentContainer] = {}
        self.training_data: Dict[str, List[str]] = defaultdict(list)
        self.must_train = True

    def remove_domain(self, domain_name: str):
        """
        Remove a domain and its associated intents and training data.

        Args:
            domain_name (str): The name of the domain to remove.
        """
        if domain_name in self.training_data:
            self.training_data.pop(domain_name)
        if domain_name in self.domains:
            self.domains.pop(domain_name)
        if domain_name in self.domain_engine.intent_names:
            self.domain_engine.remove_intent(domain_name)

    def register_domain_intent(self, domain_name: str, intent_name: str, intent_samples: List[str]):
        """
        Register an intent within a specific domain.

        Args:
            domain_name (str): The name of the domain.
            intent_name (str): The name of the intent to register.
            intent_samples (List[str]): A list of sample sentences for the intent.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(fuzzy_strategy=self.fuzzy_strategy,
                                                        ignore_case=self.ignore_case)
        self.domains[domain_name].add_intent(intent_name, intent_samples)
        self.training_data[domain_name] += intent_samples
        self.must_train = True

    def remove_domain_intent(self, domain_name: str, intent_name: str):
        """
        Remove a specific intent from a domain.

        Args:
            domain_name (str): The name of the domain.
            intent_name (str): The name of the intent to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_intent(intent_name)

    def register_domain_entity(self, domain_name: str, entity_name: str, entity_samples: List[str]):
        """
        Register an entity within a specific domain.

        Args:
            domain_name (str): The name of the domain.
            entity_name (str): The name of the entity to register.
            entity_samples (List[str]): A list of sample phrases for the entity.
        """
        if domain_name not in self.domains:
            self.domains[domain_name] = IntentContainer(fuzzy_strategy=self.fuzzy_strategy,
                                                        ignore_case=self.ignore_case)
        self.domains[domain_name].add_entity(entity_name, entity_samples)

    def remove_domain_entity(self, domain_name: str, entity_name: str):
        """
        Remove a specific entity from a domain.

        Args:
            domain_name (str): The name of the domain.
            entity_name (str): The name of the entity to remove.
        """
        if domain_name in self.domains:
            self.domains[domain_name].remove_entity(entity_name)

    def calc_domain(self, query: str):
        """
        Calculate the best matching domain for a query.

        Args:
            query (str): The input query.

        Returns:
            MatchData: The best matching domain.
        """
        return self.domain_engine.calc_intent(query)

    def calc_intent(self, query: str, domain: Optional[str] = None):
        """
        Calculate the best matching intent for a query within a specific domain.

        Args:
            query (str): The input query.
            domain (Optional[str]): The domain to limit the search to. Defaults to None.

        Returns:
            MatchData: The best matching intent.
        """
        domain: str = domain or self.domain_engine.calc_intent(query).name
        if domain in self.domains:
            return self.domains[domain].calc_intent(query)
        return {"best_match": None,
                "conf": 0,
                "match_strategy": self.fuzzy_strategy,
                "utterance": query,
                "name": None}
