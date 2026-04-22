"""
Benchmark dataset: training templates + labelled inference utterances.

test_match entries are natural human utterances — not filled-in templates.
They include contractions, filler words, politeness markers, regional phrasing,
word-order variation, and colloquialisms that real STT output produces.
"""

INTENTS = {
    # ── media ──────────────────────────────────────────────────────────────
    "play_music": {
        "train": [
            "play {song}",
            "play some {song}",
            "put on {song}",
            "start playing {song}",
            "i want to hear {song}",
            "can you play {song}",
        ],
        "test_match": [
            # exact template fills
            "play bohemian rhapsody",
            "put on some jazz",
            # natural human phrasing
            "i'd like to listen to something calm",
            "chuck on some music",
            "could you stick some background music on",
            "fancy some beatles",
            "play me something upbeat",
            "i feel like listening to pink floyd",
            "put some tunes on",
            "can you get some music going",
            "stick on a playlist",
            "play us something",
            "let's have some music",
            "i want something relaxing on in the background",
            "get some lo-fi on please",
            "play that song i like",
            "put on something chill",
            "can we have some music",
            "i could do with some music",
            "start some music",
        ],
    },

    "pause_music": {
        "train": [
            "pause",
            "pause the music",
            "pause playback",
            "stop playing",
            "hold on",
            "pause [the] (music|song|track|playback)",
            "can you pause",
            "pause [for a moment]",
        ],
        "test_match": [
            "pause",
            "hang on a sec",
            "wait wait wait",
            "hold up",
            "stop the music for a moment",
            "can you just pause that",
            "shh",
            "quiet the music",
            "turn it off for a bit",
            "pause please",
        ],
    },

    "next_track": {
        "train": [
            "next [song|track]",
            "skip [this] [song|track]",
            "play the next [song|track]",
            "next please",
            "skip to the next [one|song|track]",
            "go to the next [song|track]",
        ],
        "test_match": [
            "next",
            "not this one",
            "i don't like this song skip it",
            "can we have a different song",
            "ooh skip this one",
            "next track please",
            "move on",
            "pass",
            "change the song",
            "something else please",
        ],
    },

    "set_volume": {
        "train": [
            "set volume to {level}",
            "volume {level}",
            "set the volume to {level}",
            "turn the volume (up|down)",
            "make it (louder|quieter)",
            "volume (up|down)",
        ],
        "test_match": [
            "turn it up a bit",
            "bit louder please",
            "i can't hear it",
            "it's too loud",
            "could you turn it down",
            "quieter",
            "volume up",
            "crank it up",
            "ease it off a bit",
            "bit quieter",
        ],
    },

    # ── timers & alarms ────────────────────────────────────────────────────
    "set_timer": {
        "train": [
            "set a timer for {duration}",
            "set [a] {duration} timer",
            "timer for {duration}",
            "remind me in {duration}",
            "start a {duration} timer",
            "countdown {duration}",
        ],
        "test_match": [
            "set a timer for five minutes",
            "five minute timer please",
            "remind me in half an hour",
            "can you set a ten minute timer",
            "i need a timer for twenty minutes",
            "put a timer on for fifteen minutes",
            "countdown from three minutes",
            "give me a two minute warning",
            "set thirty second timer",
            "i need reminding in an hour",
            "timer on for forty five minutes",
            "set one hour timer",
            "quick two minute timer",
            "ninety second timer",
            "put a timer on",
        ],
    },

    "set_alarm": {
        "train": [
            "set an alarm for {time}",
            "set [an] alarm [at] {time}",
            "wake me up at {time}",
            "wake me up [in the morning]",
            "alarm at {time}",
            "alarm for {time}",
        ],
        "test_match": [
            "wake me up at seven",
            "i need to be up by six thirty",
            "set an alarm for half past seven",
            "alarm for eight o'clock",
            "make sure i'm up at six",
            "get me up at seven fifteen",
            "i've got to be up early set an alarm for six am",
            "wake me at noon",
            "can you wake me up tomorrow morning",
            "set my alarm for quarter to eight",
        ],
    },

    "cancel_timer": {
        "train": [
            "cancel [the] timer",
            "stop [the] timer",
            "cancel [the] alarm",
            "stop [the] alarm",
            "delete [the] timer",
        ],
        "test_match": [
            "cancel the timer",
            "forget the timer",
            "turn off the alarm",
            "i don't need the timer anymore",
            "kill the alarm",
            "stop that timer",
            "get rid of the alarm",
            "cancel it",
        ],
    },

    # ── weather ────────────────────────────────────────────────────────────
    "weather_query": {
        "train": [
            "what is the weather [today|tomorrow]",
            "what is the weather like [today|tomorrow]",
            "how is the weather [today|tomorrow]",
            "weather [forecast] [today|tomorrow]",
            "will it rain [today|tomorrow]",
            "will it be (rainy|sunny|cold|hot) [today|tomorrow]",
            "is it going to be (hot|cold|sunny|rainy) [today|tomorrow]",
            "is it going to rain [today|tomorrow]",
            "what is the temperature [today|tomorrow]",
            "temperature [today|tomorrow]",
            "do i need an umbrella [today|tomorrow]",
            "is it (cold|hot|warm) [outside|today|tomorrow]",
            "how (cold|hot|warm) is it [today|tomorrow]",
        ],
        "test_match": [
            "what's it like out there",
            "should i bring a coat",
            "do i need a jacket today",
            "is it going to be nice later",
            "how's the weather looking",
            "will i need an brolly",
            "is it chucking it down out there",
            "what's the forecast",
            "going to be hot today",
            "pretty cold out today is it",
            "any chance of rain",
            "is it worth going outside",
            "what are the conditions like",
            "tell me the weather",
            "is it nice out",
        ],
    },

    # ── smart home ─────────────────────────────────────────────────────────
    "lights_on": {
        "train": [
            "turn on [the] lights",
            "lights on",
            "lights [please]",
            "switch on [the] lights",
            "turn [the] lights on",
            "turn on [the] {room} lights",
            "brighten [the] lights",
            "can you turn on [the] lights",
        ],
        "test_match": [
            "it's dark in here",
            "can you brighten it up a bit",
            "little bit of light please",
            "lights on please",
            "turn the lights on would you",
            "flick the lights on",
            "could you switch the lights on",
            "it's too dark",
            "lights",
            "bit of light in here",
        ],
    },

    "lights_off": {
        "train": [
            "turn off [the] lights",
            "lights off",
            "switch off [the] lights",
            "switch [the] lights off",
            "turn [the] lights off",
            "turn off [the] {room} lights",
            "dim [the] lights",
            "can you turn off [the] lights",
        ],
        "test_match": [
            "lights off please",
            "kill the lights",
            "turn the lights out",
            "i'm going to sleep turn the lights off",
            "could you switch the lights off",
            "dim it down a bit",
            "it's too bright",
            "flick the lights off",
            "go dark",
            "lights out",
        ],
    },

    "thermostat_set": {
        "train": [
            "set the thermostat to {temperature}",
            "set thermostat to {temperature}",
            "set the temperature to {temperature}",
            "set temperature to {temperature}",
            "make it {temperature}",
            "heat [the house] to {temperature}",
            "cool [the house] to {temperature}",
            "change the temperature to {temperature}",
        ],
        "test_match": [
            "it's freezing in here can you turn the heating up",
            "bump the thermostat up a few degrees",
            "set it to twenty two",
            "crank the heating on",
            "make it a bit warmer",
            "it's boiling turn the temperature down",
            "drop it to eighteen degrees",
            "set the heat to twenty",
            "could you warm the place up a bit",
            "cool it down please",
        ],
    },

    # ── communication ──────────────────────────────────────────────────────
    "call_contact": {
        "train": [
            "call {contact}",
            "phone {contact}",
            "ring {contact}",
            "dial {contact}",
            "call {contact} [please]",
            "can you call {contact}",
        ],
        "test_match": [
            "give mum a ring",
            "can you ring alice for me",
            "i need to call john",
            "get bob on the phone",
            "call my wife",
            "phone the office",
            "ring dad",
            "give sarah a call",
            "call my brother please",
            "can you get alice on the line",
            "dial charlie",
            "ring the doctor",
        ],
    },

    "send_message": {
        "train": [
            "send [a] message to {contact}",
            "text {contact}",
            "send {contact} a message",
            "message {contact}",
            "send a text to {contact}",
        ],
        "test_match": [
            "drop alice a text",
            "send a message to my mum",
            "text john and tell him i'm running late",
            "can you message sarah",
            "send bob a quick text",
            "shoot charlie a message",
            "text my brother",
            "let dad know i'm on my way",
            "send a message to the team",
            "ping alice",
        ],
    },

    # ── information ────────────────────────────────────────────────────────
    "time_query": {
        "train": [
            "what time is it",
            "what is the time",
            "what's the time [right now]",
            "current time",
            "tell me the time",
            "time [please]",
            "what time",
            "do you know what time it is",
            "can you tell me the time",
        ],
        "test_match": [
            "what's the time",
            "got the time",
            "any idea what time it is",
            "quick what time is it",
            "i've lost track of the time",
            "what time is it right now",
            "time check",
            "how late is it",
            "is it late",
            "what hour is it",
        ],
    },

    "date_query": {
        "train": [
            "what day is it [today]",
            "what is today['s] date",
            "what is the date [today]",
            "what date is it [today]",
            "today's date",
            "date [please]",
            "tell me today's date",
            "what day of the week is it",
        ],
        "test_match": [
            "what day is it",
            "what's today",
            "i've lost track of the date",
            "what's the date today",
            "is today monday",
            "which day are we on",
            "what's today's date",
            "do you know what day it is",
            "what day of the week is it today",
            "what month are we in",
        ],
    },

    "search_query": {
        "train": [
            "search for {query}",
            "look up {query}",
            "find {query}",
            "search {query}",
            "google {query}",
            "what is {query}",
        ],
        "test_match": [
            "can you look up the nearest pharmacy",
            "google how to make sourdough",
            "what is photosynthesis",
            "search for cheap flights to berlin",
            "find a good pizza recipe",
            "look up the opening hours for the library",
            "what's the capital of australia",
            "search how to fix a leaky tap",
            "find out when the next train to london is",
            "what does procrastinate mean",
            "look up symptoms of a cold",
            "google padacioso python library",
        ],
    },

    # ── navigation ─────────────────────────────────────────────────────────
    "navigate_to": {
        "train": [
            "navigate to {place}",
            "take me to {place}",
            "drive to {place}",
            "directions to {place}",
            "how do i get to {place}",
            "get directions to {place}",
        ],
        "test_match": [
            "how do i get to the airport from here",
            "take me home",
            "i need directions to the nearest hospital",
            "get me to the train station",
            "navigate me to work",
            "find a route to the city centre",
            "can you direct me to the supermarket",
            "i'm lost where's the nearest petrol station",
            "take me to tesco",
            "route to london please",
            "how far is the airport",
            "get me there",
        ],
    },

    # ── reminders & notes ──────────────────────────────────────────────────
    "add_reminder": {
        "train": [
            "remind me to {task}",
            "set a reminder [to|for] {task}",
            "reminder to {task}",
            "don't let me forget to {task}",
            "add a reminder to {task}",
        ],
        "test_match": [
            "don't let me forget to call the dentist",
            "i need a reminder to take my medication at noon",
            "remind me about the meeting tomorrow",
            "can you remind me to pick up milk on the way home",
            "set a reminder for my doctor's appointment",
            "i'll forget to water the plants remind me",
            "nudge me to send that email later",
            "remind me to ring mum tonight",
            "don't forget the school run reminder",
            "set a reminder so i don't miss the deadline",
        ],
    },

    "add_note": {
        "train": [
            "take a note {text}",
            "note [that] {text}",
            "write [this] down {text}",
            "add [a] note {text}",
            "make a note [that] {text}",
        ],
        "test_match": [
            "note down that the meeting's been moved to thursday",
            "write this down milk eggs and bread",
            "add a note i need to renew my passport before june",
            "make a note the client wants revisions by friday",
            "take a note call the plumber about the leak",
            "jot this down dentist appointment tuesday three pm",
            "note to self bring charger tomorrow",
            "write down the wifi password for the guest",
            "add a note project deadline end of month",
            "make a note to follow up with sarah",
        ],
    },

    # ── shopping ───────────────────────────────────────────────────────────
    "add_shopping": {
        "train": [
            "add {item} to [my] [shopping] list",
            "put {item} on [my] [shopping] list",
            "buy {item}",
            "i need [to buy] {item}",
            "add {item} to the shopping list",
            "i need {item}",
        ],
        "test_match": [
            "we're out of milk",
            "we need more coffee",
            "can you add bread to the shopping list",
            "don't forget eggs when you're at the shops",
            "put butter on the list",
            "i've run out of shampoo add it to the list",
            "we need to get some pasta",
            "add a few apples to the shopping",
            "running low on washing up liquid",
            "pick up some cheese as well",
        ],
    },

    # ── system / device ────────────────────────────────────────────────────
    "stop": {
        "train": [
            "stop",
            "cancel",
            "cancel [that]",
            "abort",
            "abort [mission]",
            "never mind",
            "forget it",
            "stop [that|it|please]",
        ],
        "test_match": [
            "actually don't worry about it",
            "forget what i said",
            "never mind that",
            "stop stop stop",
            "no no no cancel",
            "leave it",
            "don't bother",
            "that's fine forget it",
            "abort",
            "scrap that",
        ],
    },

    "help": {
        "train": [
            "help",
            "help me",
            "i need help",
            "what can you do",
            "what do you know",
            "show me what you can do",
            "list your skills",
            "how can you help [me]",
            "what are your capabilities",
            "tell me what you can do",
        ],
        "test_match": [
            "what are you capable of",
            "i'm not sure what to ask you",
            "what commands do you know",
            "can you give me some examples",
            "i'm new here what can you do",
            "show me what you've got",
            "give me some ideas of what to say",
            "what sorts of things can i ask you",
            "what's your feature set",
            "run me through what you can do",
        ],
    },
}

# Utterances that should NOT match any intent — plausible but genuinely off-topic
NO_MATCH_UTTERANCES = [
    # conversational filler
    "um yeah so anyway",
    "right okay then",
    "hmm not sure about that",
    "oh interesting",
    "fair enough",
    # random statements
    "the dog ate my homework",
    "i've been thinking about getting a new sofa",
    "did you see the match last night",
    "my knee's been giving me trouble",
    "i went to the dentist yesterday",
    # things that share words with intents but aren't commands
    "my friend alice rang me earlier",          # 'alice' overlaps call_contact
    "the music at the restaurant was terrible", # 'music' overlaps play_music
    "i watched a documentary about weather",    # 'weather' overlaps weather_query
    "the lights were beautiful at the concert", # 'lights' overlaps lights
    "the alarm went off in the night",          # 'alarm' overlaps set_alarm
    "my shopping bag broke",                    # 'shopping' overlaps add_shopping
    "the timer on the oven broke",              # 'timer' overlaps set_timer
    "i always stop for coffee on the way",      # 'stop' overlaps stop
    "she left me a note on the table",          # 'note' overlaps add_note
    # nonsense / gibberish
    "blarg wump fizz",
    "one fish two fish red fish blue fish",
    "supercalifragilistic",
    "lorem ipsum dolor sit amet",
    "the mitochondria is the powerhouse of the cell",
]
