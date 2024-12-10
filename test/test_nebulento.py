import unittest

from nebulento import IntentContainer, MatchStrategy


class TestIntentContainer(unittest.TestCase):
    def test_syntax(self):
        container = IntentContainer()
        container.add_intent('hello', ["(hello|hi|hey) world"])
        self.assertEqual(set(container.registered_intents["hello"]),
                         set(['hello world', 'hi world', 'hey world']))

        container = IntentContainer()
        container.add_intent('hello', ["hello (world|)"])
        self.assertEqual(set(container.registered_intents["hello"]),
                         set(['hello world', 'hello']))

        container.add_intent('hey', ["hey [world]"])
        self.assertEqual(set(container.registered_intents["hey"]),
                         set(['hey world', 'hey']))

        container.add_intent('hi', ["hi [{person}|people]"])
        self.assertEqual(set(container.registered_intents["hi"]),
                         set(['hi {person}', 'hi people', 'hi']))

    # test intent parsing
    def test_intents(self):
        container = IntentContainer()
        container.add_intent('hello', [
            'hello', 'hi', 'how are you', "what's up"
        ])
        container.add_intent('buy', [
            'buy {item}', 'purchase {item}', 'get {item}', 'get {item} for me'
        ])
        container.add_entity('item', [
            'milk', 'cheese'
        ])
        self.assertEqual(container.calc_intent('hello'),
                         {'best_match': 'hello',
                          'conf': 1.0,
                          'entities': {},
                          'match_strategy': 'DAMERAU_LEVENSHTEIN_SIMILARITY',
                          'name': 'hello',
                          'utterance': 'hello',
                          'utterance_consumed': 'hello',
                          'utterance_remainder': ''})

        self.assertEqual(container.calc_intent('buy milk'),
                         {'best_match': 'buy {item}',
                          'conf': 0.625,
                          'entities': {'item': ['milk']},
                          'match_strategy': 'DAMERAU_LEVENSHTEIN_SIMILARITY',
                          'name': 'buy',
                          'utterance': 'buy milk',
                          'utterance_consumed': 'buy milk',
                          'utterance_remainder': ''})
        self.assertEqual(container.calc_intent('buy beer'),
                         {'best_match': 'buy {item}',
                          'conf': 0.5,
                          'entities': {},
                          'match_strategy': 'DAMERAU_LEVENSHTEIN_SIMILARITY',
                          'name': 'buy',
                          'utterance': 'buy beer',
                          'utterance_consumed': 'buy',
                          'utterance_remainder': 'beer'}
                         )

    def test_case(self):
        container = IntentContainer()
        container.add_intent('test', ['Testing cAPitalizAtion'])
        self.assertEqual(
            container.calc_intent('Testing cAPitalizAtion')['conf'], 1.0)
        self.assertEqual(
            container.calc_intent('teStiNg CapitalIzation')['conf'], 1.0)

        container = IntentContainer(ignore_case=False)
        container.add_intent('test', ['Testing cAPitalizAtion'])
        self.assertEqual(
            container.calc_intent('teStiNg CapitalIzation')['conf'],
            0.6363636363636364)

    def test_entities(self):
        container = IntentContainer()
        container.add_intent('test3', ['I see {thing} (in|on) {place}'])
        self.assertEqual(
            container.calc_intent('I see a bin in there'),
            {'best_match': 'i see {thing} in {place}',
             'conf': 0.5416666666666667,
             'entities': {},
             'match_strategy': 'DAMERAU_LEVENSHTEIN_SIMILARITY',
             'name': 'test3',
             'utterance': 'i see a bin in there',
             'utterance_consumed': 'i see in',
             'utterance_remainder': 'a bin there'}
        )
        container.add_entity("place", ["floor", "table"])
        self.assertEqual(
            container.calc_intent('I see trash in the floor'),
            {'best_match': 'i see {thing} in {place}',
             'conf': 0.53125,
             'entities': {'place': ['floor']},
             'match_strategy': 'DAMERAU_LEVENSHTEIN_SIMILARITY',
             'name': 'test3',
             'utterance': 'i see trash in the floor',
             'utterance_consumed': 'i see in floor',
             'utterance_remainder': 'trash the'}
        )
        container.add_entity("thing", ["food"])
        self.assertEqual(
            container.calc_intent('I see food in the table'),
            {'best_match': 'i see {thing} in {place}',
             'conf': 0.6484375,
             'entities': {'place': ['table'], 'thing': ['food']},
             'match_strategy': 'DAMERAU_LEVENSHTEIN_SIMILARITY',
             'name': 'test3',
             'utterance': 'i see food in the table',
             'utterance_consumed': 'i see in table food',
             'utterance_remainder': 'the'}
        )

    # test match strategies
    def test_token_set(self):
        container = IntentContainer(
            fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
        container.add_intent('buy', [
            'buy {item}', 'purchase {item}', 'get {item}', 'get {item} for me'
        ])
        container.add_entity('item', [
            'milk', 'cheese'
        ])
        self.assertEqual(container.calc_intent('buy milk'),
                         {'best_match': 'buy {item}',
                          'conf': 0.6666666666666667,
                          'entities': {'item': ['milk']},
                          'match_strategy': 'TOKEN_SET_RATIO',
                          'name': 'buy',
                          'utterance': 'buy milk',
                          'utterance_consumed': 'buy milk',
                          'utterance_remainder': ''})
        self.assertEqual(container.calc_intent('buy beer'),
                         {'best_match': 'buy {item}',
                          'conf': 0.5555555555555556,
                          'entities': {},
                          'match_strategy': 'TOKEN_SET_RATIO',
                          'name': 'buy',
                          'utterance': 'buy beer',
                          'utterance_consumed': 'buy',
                          'utterance_remainder': 'beer'})

    def test_token_sort(self):
        container = IntentContainer(
            fuzzy_strategy=MatchStrategy.TOKEN_SORT_RATIO)
        container.add_intent('buy', [
            'buy {item}', 'purchase {item}', 'get {item}', 'get {item} for me'
        ])
        container.add_entity('item', [
            'milk', 'cheese'
        ])
        self.assertEqual(container.calc_intent('buy milk'),
                         {'best_match': 'buy {item}',
                          'conf': 0.6666666666666667,
                          'entities': {'item': ['milk']},
                          'match_strategy': 'TOKEN_SORT_RATIO',
                          'name': 'buy',
                          'utterance': 'buy milk',
                          'utterance_consumed': 'buy milk',
                          'utterance_remainder': ''})
        self.assertEqual(container.calc_intent('buy beer'),
                         {'best_match': 'buy {item}',
                          'conf': 0.33333333333333337,
                          'entities': {},
                          'match_strategy': 'TOKEN_SORT_RATIO',
                          'name': 'buy',
                          'utterance': 'buy beer',
                          'utterance_consumed': 'buy',
                          'utterance_remainder': 'beer'})

    def test_partial_token_set(self):
        container = IntentContainer(
            fuzzy_strategy=MatchStrategy.PARTIAL_TOKEN_SET_RATIO)
        container.add_intent('buy', [
            'buy {item}', 'purchase {item}', 'get {item}', 'get {item} for me'
        ])
        container.add_entity('item', [
            'milk', 'cheese'
        ])
        self.assertEqual(container.calc_intent('buy milk'),
                         {'best_match': 'buy {item}',
                          'conf': 1,
                          'entities': {'item': ['milk']},
                          'match_strategy': 'PARTIAL_TOKEN_SET_RATIO',
                          'name': 'buy',
                          'utterance': 'buy milk',
                          'utterance_consumed': 'buy milk',
                          'utterance_remainder': ''})
        self.assertEqual(container.calc_intent('buy beer'),
                         {'best_match': 'buy {item}',
                          'conf': 1.0,
                          'entities': {},
                          'match_strategy': 'PARTIAL_TOKEN_SET_RATIO',
                          'name': 'buy',
                          'utterance': 'buy beer',
                          'utterance_consumed': 'buy',
                          'utterance_remainder': 'beer'})

    def test_partial_token_sort(self):
        container = IntentContainer(
            fuzzy_strategy=MatchStrategy.PARTIAL_TOKEN_SORT_RATIO)
        container.add_intent('buy', [
            'buy {item}', 'purchase {item}', 'get {item}', 'get {item} for me'
        ])
        container.add_entity('item', [
            'milk', 'cheese'
        ])
        self.assertEqual(container.calc_intent('buy milk'),
                         {'best_match': 'buy {item}',
                          'conf': 0.7857142857142857,
                          'entities': {'item': ['milk']},
                          'match_strategy': 'PARTIAL_TOKEN_SORT_RATIO',
                          'name': 'buy',
                          'utterance': 'buy milk',
                          'utterance_consumed': 'buy milk',
                          'utterance_remainder': ''})
        self.assertEqual(container.calc_intent('buy beer'),
                         {'best_match': 'buy {item}',
                          'conf': 0.5454545454545454,
                          'entities': {},
                          'match_strategy': 'PARTIAL_TOKEN_SORT_RATIO',
                          'name': 'buy',
                          'utterance': 'buy beer',
                          'utterance_consumed': 'buy',
                          'utterance_remainder': 'beer'})
