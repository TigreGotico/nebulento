# Nebulento

*A lightweight, dead simple fuzzy matching intent parser*

Built on top of [rapidfuzz](https://github.com/maxbachmann/rapidfuzz)

Finds the closest matching intent via fuzzy match between the text and all of the training sentences you provided. Works best when you have a small number of sentences (dozens to hundreds) and need some resiliency to spelling errors (i.e., from text chat).

## Usage

```python
from nebulento import IntentContainer, MatchStrategy

container = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)

container.add_intent('hello', [
    'hello', 'hi', 'how are you', "what's up"
])
container.add_intent('buy', [
    'buy {item}', 'purchase {item}', 'get {item}', 'get {item} for me'
])
container.add_entity('item', [
    'milk', 'cheese'
])

container.calc_intent('hello')
# {'best_match': 'hello',
#  'conf': 1.0,
#  'entities': {},
#  'match_strategy': 'TOKEN_SET_RATIO',
#  'name': 'hello',
#  'utterance': 'hello',
#  'utterance_consumed': 'hello',
#  'utterance_remainder': ''}
                         
container.calc_intent('buy milk')
# {'best_match': 'buy {item}',
#  'conf': 0.71875,
#  'entities': {'item': ['milk']},
#  'match_strategy': 'TOKEN_SET_RATIO',
#  'name': 'buy',
#  'utterance': 'buy milk',
#  'utterance_consumed': 'buy milk',
#  'utterance_remainder': ''}


container.add_intent('look_at_thing', ['I see {thing} (in|on) {place}'])
container.add_entity("place", ["floor", "table"])

container.calc_intent('I see trash in the floor')
#{'best_match': 'i see {thing} in {place}',
# 'conf': 0.65625,
# 'entities': {'place': ['floor']},
# 'match_strategy': 'TOKEN_SET_RATIO',
# 'name': 'look_at_thing',
# 'utterance': 'i see trash in the floor',
# 'utterance_consumed': 'i see in floor',
# 'utterance_remainder': 'trash the'}
            
container.add_entity("thing", ["food"])
container.calc_intent('I see food in the table')
#{'best_match': 'i see {thing} in {place}',
# 'conf': 0.7007978723404256,
# 'entities': {'place': ['table'], 'thing': ['food']},
# 'match_strategy': 'TOKEN_SET_RATIO',
# 'name': 'look_at_thing',
# 'utterance': 'i see food in the table',
# 'utterance_consumed': 'i see in table food',
# 'utterance_remainder': 'the'}

```

## Match Strategies

- **Ratio**: Use `MatchStrategy.RATIO` when you need to compare strings and determine their overall similarity. It is effective for handling cases where strings have minor differences due to typos or spelling variations.
- **Partial Ratio**: `MatchStrategy.PARTIAL_RATIO` is useful when you want to focus on the best matching substring between two strings. It handles cases where one string is a subset or prefix of the other, providing a more targeted similarity measure.
- **Token Set Ratio**: `MatchStrategy.TOKEN_SET_RATIO` is ideal when you want to compare strings without considering their word order. It captures the essence of the strings’ content, making it suitable for scenarios where word arrangement might vary but the overall content remains similar.
- **Token Sort Ratio**: Use `MatchStrategy.TOKEN_SORT_RATIO` when you want to compare strings and consider word order variations. It is particularly effective when the words are expected to be similar, but their arrangement may differ.
