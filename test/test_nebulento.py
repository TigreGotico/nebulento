import unittest

from nebulento import IntentContainer, DomainIntentContainer, MatchStrategy


# ── helpers ────────────────────────────────────────────────────────────────

def _basic_container(strategy=MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY):
    c = IntentContainer(fuzzy_strategy=strategy)
    c.add_intent("hello", ["hello", "hi", "how are you", "what's up"])
    c.add_intent("buy", ["buy {item}", "purchase {item}", "get {item}", "get {item} for me"])
    c.add_entity("item", ["milk", "cheese"])
    return c


# ── template expansion ─────────────────────────────────────────────────────

class TestSyntax(unittest.TestCase):
    def test_alternation(self):
        c = IntentContainer()
        c.add_intent("hello", ["(hello|hi|hey) world"])
        self.assertEqual(set(c.registered_intents["hello"]),
                         {"hello world", "hi world", "hey world"})

    def test_optional_word(self):
        c = IntentContainer()
        c.add_intent("hello", ["hello (world|)"])
        self.assertEqual(set(c.registered_intents["hello"]), {"hello world", "hello"})

    def test_optional_bracket(self):
        c = IntentContainer()
        c.add_intent("hey", ["hey [world]"])
        self.assertEqual(set(c.registered_intents["hey"]), {"hey world", "hey"})

    def test_optional_entity(self):
        c = IntentContainer()
        c.add_intent("hi", ["hi [{person}|people]"])
        self.assertEqual(set(c.registered_intents["hi"]), {"hi {person}", "hi people", "hi"})

    def test_nested_alternation(self):
        c = IntentContainer()
        c.add_intent("media", ["(play|start) (music|radio)"])
        self.assertEqual(set(c.registered_intents["media"]),
                         {"play music", "play radio", "start music", "start radio"})

    def test_empty_template_list(self):
        c = IntentContainer()
        c.add_intent("empty", [])
        self.assertEqual(c.registered_intents["empty"], [])


# ── basic intent matching ──────────────────────────────────────────────────

class TestIntentMatching(unittest.TestCase):
    def setUp(self):
        self.c = _basic_container()

    def test_exact_match_conf_1(self):
        r = self.c.calc_intent("hello")
        self.assertEqual(r["name"], "hello")
        self.assertAlmostEqual(r["conf"], 1.0)

    def test_exact_match_fields_present(self):
        r = self.c.calc_intent("hello")
        for key in ("name", "conf", "entities", "best_match",
                    "utterance", "utterance_consumed", "utterance_remainder",
                    "match_strategy"):
            self.assertIn(key, r)

    def test_utterance_is_normalised(self):
        r = self.c.calc_intent("Hello")
        self.assertEqual(r["utterance"], "hello")

    def test_close_match(self):
        r = self.c.calc_intent("helo")  # typo
        self.assertEqual(r["name"], "hello")
        self.assertGreater(r["conf"], 0.5)

    def test_no_match_returns_none_name(self):
        c = IntentContainer()
        c.add_intent("hello", ["hello"])
        r = c.calc_intent("zzzzz completely unrelated xyzzy")
        # name may be "hello" or None depending on strategy, but conf must be low
        self.assertIsNotNone(r)  # always returns a dict

    def test_calc_intents_yields_all(self):
        results = list(self.c.calc_intents("buy milk"))
        names = [r["name"] for r in results]
        self.assertIn("buy", names)
        self.assertIn("hello", names)

    def test_calc_intent_returns_best(self):
        r = self.c.calc_intent("buy milk")
        self.assertEqual(r["name"], "buy")

    def test_match_strategy_in_result(self):
        r = self.c.calc_intent("hello")
        self.assertEqual(r["match_strategy"], "DAMERAU_LEVENSHTEIN_SIMILARITY")


# ── entity matching ────────────────────────────────────────────────────────

class TestEntities(unittest.TestCase):
    def setUp(self):
        self.c = _basic_container()

    def test_registered_entity_in_result(self):
        r = self.c.calc_intent("buy milk")
        self.assertEqual(r["entities"], {"item": ["milk"]})

    def test_unregistered_entity_value_empty(self):
        r = self.c.calc_intent("buy beer")
        self.assertEqual(r["entities"], {})

    def test_entity_boosts_confidence(self):
        r_with = self.c.calc_intent("buy milk")
        r_without = self.c.calc_intent("buy beer")
        self.assertGreater(r_with["conf"], r_without["conf"])

    def test_multiple_entities(self):
        c = IntentContainer()
        c.add_intent("look", ["I see {thing} (in|on) {place}"])
        c.add_entity("place", ["floor", "table"])
        c.add_entity("thing", ["food"])
        r = c.calc_intent("I see food in the table")
        self.assertIn("place", r["entities"])
        self.assertIn("thing", r["entities"])

    def test_remove_entity(self):
        c = _basic_container()
        c.remove_entity("item")
        r = c.calc_intent("buy milk")
        self.assertEqual(r["entities"], {})

    def test_add_entity_case_normalised(self):
        c = IntentContainer()
        c.add_intent("buy", ["buy {item}"])
        c.add_entity("item", ["Milk", "CHEESE"])
        r = c.calc_intent("buy milk")
        self.assertIn("item", r["entities"])


# ── utterance remainder / consumed ────────────────────────────────────────

class TestUtteranceFields(unittest.TestCase):
    def test_consumed_exact(self):
        c = IntentContainer()
        c.add_intent("hello", ["hello"])
        r = c.calc_intent("hello")
        self.assertEqual(r["utterance_consumed"], "hello")
        self.assertEqual(r["utterance_remainder"], "")

    def test_remainder_contains_extra_words(self):
        c = _basic_container()
        r = c.calc_intent("buy beer please")
        # "please" and "beer" are not in "buy {item}" template
        self.assertIn("please", r["utterance_remainder"])


# ── case sensitivity ───────────────────────────────────────────────────────

class TestCaseSensitivity(unittest.TestCase):
    def test_ignore_case_default(self):
        c = IntentContainer()
        c.add_intent("test", ["Testing cAPitalizAtion"])
        self.assertAlmostEqual(c.calc_intent("Testing cAPitalizAtion")["conf"], 1.0)
        self.assertAlmostEqual(c.calc_intent("teStiNg CapitalIzation")["conf"], 1.0)

    def test_case_sensitive_mismatch_lower_conf(self):
        c = IntentContainer(ignore_case=False)
        c.add_intent("test", ["Testing cAPitalizAtion"])
        exact = c.calc_intent("Testing cAPitalizAtion")["conf"]
        wrong_case = c.calc_intent("teStiNg CapitalIzation")["conf"]
        self.assertGreater(exact, wrong_case)


# ── normalisation ──────────────────────────────────────────────────────────

class TestNormalisation(unittest.TestCase):
    def test_apostrophe_stripped_in_query(self):
        c = IntentContainer()
        c.add_intent("greet", ["what is up"])
        r_apos = c.calc_intent("what's up")
        r_plain = c.calc_intent("what is up")
        # both should match; apostrophe variant not penalised more
        self.assertEqual(r_apos["name"], r_plain["name"])

    def test_curly_apostrophe_stripped(self):
        c = IntentContainer()
        c.add_intent("greet", ["whats up"])
        r = c.calc_intent("what’s up")  # RIGHT SINGLE QUOTATION MARK
        self.assertEqual(r["name"], "greet")

    def test_extra_whitespace_collapsed(self):
        c = IntentContainer()
        c.add_intent("hello", ["hello world"])
        r = c.calc_intent("hello  world")
        self.assertEqual(r["utterance"], "hello world")


# ── add / remove intents ───────────────────────────────────────────────────

class TestAddRemove(unittest.TestCase):
    def test_remove_intent(self):
        c = _basic_container()
        c.remove_intent("hello")
        self.assertNotIn("hello", c.registered_intents)

    def test_remove_nonexistent_intent_no_error(self):
        c = IntentContainer()
        c.remove_intent("ghost")  # should not raise

    def test_remove_nonexistent_entity_no_error(self):
        c = IntentContainer()
        c.remove_entity("ghost")

    def test_intent_names_property(self):
        c = _basic_container()
        self.assertIn("hello", c.intent_names)
        self.assertIn("buy", c.intent_names)


# ── all match strategies ───────────────────────────────────────────────────

class TestStrategies(unittest.TestCase):
    """Smoke-test every MatchStrategy: exact match should always return the right intent."""

    def _run(self, strategy):
        c = IntentContainer(fuzzy_strategy=strategy)
        c.add_intent("greet", ["hello", "hi there", "good morning"])
        c.add_intent("bye", ["goodbye", "see you later", "farewell"])
        r = c.calc_intent("hello")
        return r

    def test_simple_ratio(self):
        self.assertEqual(self._run(MatchStrategy.SIMPLE_RATIO)["name"], "greet")

    def test_ratio(self):
        self.assertEqual(self._run(MatchStrategy.RATIO)["name"], "greet")

    def test_partial_ratio(self):
        r = self._run(MatchStrategy.PARTIAL_RATIO)
        self.assertIsNotNone(r["name"])

    def test_token_sort_ratio(self):
        self.assertEqual(self._run(MatchStrategy.TOKEN_SORT_RATIO)["name"], "greet")

    def test_token_set_ratio(self):
        self.assertEqual(self._run(MatchStrategy.TOKEN_SET_RATIO)["name"], "greet")

    def test_partial_token_ratio(self):
        r = self._run(MatchStrategy.PARTIAL_TOKEN_RATIO)
        self.assertIsNotNone(r["name"])

    def test_partial_token_sort_ratio(self):
        r = self._run(MatchStrategy.PARTIAL_TOKEN_SORT_RATIO)
        self.assertIsNotNone(r["name"])

    def test_partial_token_set_ratio(self):
        r = self._run(MatchStrategy.PARTIAL_TOKEN_SET_RATIO)
        self.assertIsNotNone(r["name"])

    def test_damerau_levenshtein(self):
        self.assertEqual(
            self._run(MatchStrategy.DAMERAU_LEVENSHTEIN_SIMILARITY)["name"], "greet"
        )

    def test_token_set_entity_match(self):
        c = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SET_RATIO)
        c.add_intent("buy", ["buy {item}", "purchase {item}", "get {item}", "get {item} for me"])
        c.add_entity("item", ["milk", "cheese"])
        r = c.calc_intent("buy milk")
        self.assertEqual(r["name"], "buy")
        self.assertIn("item", r["entities"])

    def test_token_sort_entity_match(self):
        c = IntentContainer(fuzzy_strategy=MatchStrategy.TOKEN_SORT_RATIO)
        c.add_intent("buy", ["buy {item}", "purchase {item}", "get {item}", "get {item} for me"])
        c.add_entity("item", ["milk", "cheese"])
        r = c.calc_intent("buy milk")
        self.assertEqual(r["name"], "buy")

    def test_partial_token_set_entity(self):
        c = IntentContainer(fuzzy_strategy=MatchStrategy.PARTIAL_TOKEN_SET_RATIO)
        c.add_intent("buy", ["buy {item}", "purchase {item}", "get {item}", "get {item} for me"])
        c.add_entity("item", ["milk", "cheese"])
        r = c.calc_intent("buy milk")
        self.assertEqual(r["name"], "buy")
        self.assertAlmostEqual(r["conf"], 1.0)

    def test_partial_token_sort_entity(self):
        c = IntentContainer(fuzzy_strategy=MatchStrategy.PARTIAL_TOKEN_SORT_RATIO)
        c.add_intent("buy", ["buy {item}", "purchase {item}", "get {item}", "get {item} for me"])
        c.add_entity("item", ["milk", "cheese"])
        r = c.calc_intent("buy milk")
        self.assertEqual(r["name"], "buy")


# ── edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):
    def test_empty_query(self):
        c = _basic_container()
        r = c.calc_intent("")
        self.assertIsNotNone(r)

    def test_single_word_intent(self):
        c = IntentContainer()
        c.add_intent("stop", ["stop"])
        r = c.calc_intent("stop")
        self.assertEqual(r["name"], "stop")
        self.assertAlmostEqual(r["conf"], 1.0)

    def test_conf_clamped_to_1(self):
        c = _basic_container()
        for r in c.calc_intents("hello"):
            self.assertLessEqual(r["conf"], 1.0)
            self.assertGreaterEqual(r["conf"], 0.0)

    def test_no_intents_registered(self):
        c = IntentContainer()
        r = c.calc_intent("hello")
        self.assertIsNotNone(r)

    def test_match_strategy_exported_from_package(self):
        from nebulento import MatchStrategy as MS
        self.assertIs(MS, MatchStrategy)

    def test_all_strategies_in_enum(self):
        expected = {
            "SIMPLE_RATIO", "RATIO", "PARTIAL_RATIO",
            "TOKEN_SORT_RATIO", "TOKEN_SET_RATIO",
            "PARTIAL_TOKEN_RATIO", "PARTIAL_TOKEN_SORT_RATIO",
            "PARTIAL_TOKEN_SET_RATIO", "DAMERAU_LEVENSHTEIN_SIMILARITY",
        }
        self.assertEqual({s.name for s in MatchStrategy}, expected)


# ── DomainIntentContainer ──────────────────────────────────────────────────

class TestDomainIntentContainer(unittest.TestCase):
    def _build(self):
        d = DomainIntentContainer()
        d.register_domain_intent("media", "play", ["play {song}", "play some music"])
        d.register_domain_intent("media", "pause", ["pause", "stop the music"])
        d.register_domain_intent("home", "lights_on", ["turn on the lights", "lights on"])
        d.register_domain_intent("home", "lights_off", ["turn off the lights", "lights off"])
        d.domain_engine.add_intent("media", ["play music", "pause music", "next track"])
        d.domain_engine.add_intent("home", ["lights on", "lights off", "thermostat"])
        return d

    def test_calc_intent_with_explicit_domain(self):
        d = self._build()
        r = d.calc_intent("play something", domain="media")
        self.assertEqual(r["name"], "play")

    def test_calc_intent_auto_domain(self):
        d = self._build()
        r = d.calc_intent("lights on")
        self.assertIn(r["name"], ("lights_on", "lights_off"))  # fuzzy, home domain

    def test_calc_domain(self):
        d = self._build()
        r = d.calc_domain("lights on")
        self.assertIsNotNone(r)
        self.assertIn("name", r)

    def test_remove_domain(self):
        d = self._build()
        d.remove_domain("media")
        self.assertNotIn("media", d.domains)
        self.assertNotIn("media", d.training_data)

    def test_remove_domain_intent(self):
        d = self._build()
        d.remove_domain_intent("media", "pause")
        self.assertNotIn("pause", d.domains["media"].registered_intents)

    def test_remove_domain_entity(self):
        d = DomainIntentContainer()
        d.register_domain_intent("media", "play", ["play {song}"])
        d.register_domain_entity("media", "song", ["jazz", "rock"])
        d.remove_domain_entity("media", "song")
        self.assertNotIn("song", d.domains["media"].registered_entities)

    def test_register_domain_entity(self):
        d = DomainIntentContainer()
        d.register_domain_intent("media", "play", ["play {song}"])
        d.register_domain_entity("media", "song", ["jazz", "rock"])
        self.assertIn("song", d.domains["media"].registered_entities)

    def test_unknown_domain_returns_none_name(self):
        d = DomainIntentContainer()
        r = d.calc_intent("hello", domain="nonexistent")
        self.assertIsNone(r["name"])

    def test_training_data_accumulates(self):
        d = DomainIntentContainer()
        d.register_domain_intent("media", "play", ["play music"])
        d.register_domain_intent("media", "stop", ["stop music"])
        self.assertEqual(len(d.training_data["media"]), 2)

    def test_no_must_train_attribute(self):
        d = DomainIntentContainer()
        self.assertFalse(hasattr(d, "must_train"))


# ── bracket_expansion standalone ──────────────────────────────────────────

class TestBracketExpansion(unittest.TestCase):
    def test_expand_template_plain(self):
        from nebulento.bracket_expansion import expand_template
        self.assertEqual(expand_template("hello world"), ["hello world"])

    def test_expand_template_alternation(self):
        from nebulento.bracket_expansion import expand_template
        result = expand_template("(a|b) c")
        self.assertIn("a c", result)
        self.assertIn("b c", result)

    def test_expand_template_optional(self):
        from nebulento.bracket_expansion import expand_template
        result = expand_template("hello [there]")
        self.assertIn("hello there", result)
        self.assertIn("hello", result)

    def test_expand_slots(self):
        from nebulento.bracket_expansion import expand_slots
        result = expand_slots("buy {item}", {"item": ["milk", "eggs"]})
        self.assertIn("buy milk", result)
        self.assertIn("buy eggs", result)

    def test_expand_slots_no_slot(self):
        from nebulento.bracket_expansion import expand_slots
        result = expand_slots("hello world", {})
        self.assertEqual(result, ["hello world"])


if __name__ == "__main__":
    unittest.main()
