"""Guide-driven post captions for TikTok clips.

POST captions — the text under the video on TikTok — NOT on-screen
text. Nothing in this module draws on the video: the on-screen quote
line stays off by default (v1.4.1) and subtitles stay opt-in.

Works the way Sara works with Claude: she hands over a video and her
caption guide, and the captions follow the guide's rules. The guide
lives in ``caption_guide.md`` next to this module (git-ignored, local
only) and is re-read on every run, so editing the guide changes the
captions without a code change. When the file is missing, the built-in
default rules below apply instead (they match
``caption_guide.example.md``).

Rule-based generation is the main path — no API key needed. When a
Gemini key is configured, the guide + clip context are sent to Gemini
instead; any failure falls back to the rule-based captions. NOTE: the
Gemini path is untested without a real key (none has been provided).

The clip context carries two text sources: the Whisper transcript
(what the streamer says) and the on-screen text (game dialogue and
objective text read off the frames by game_ocr.read_screen_text).
Entities come from both — the streamer may barely speak while the
game says everything — but the reaction is detected from the
transcript only: game dialogue is not the streamer's reaction.

The guide's moment-type templates (apology, song reaction, emotional,
career moment, meeting someone famous) are parsed out of the guide
file at runtime and picked by the moment kind detected from the
transcript + headline — never from the on-screen text. Viral moments
need no template: the full-context engine already names the specific
thing and the number. Candidates are filtered against the clipping
guide's risky words (fat, mean, bully, fight, gun, dead, hate, f*ck):
a caption must stay PG, so any candidate carrying one is dropped.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "GUIDE_FILENAME",
    "RISKY_WORDS",
    "detect_moment_kind",
    "detect_reaction",
    "extract_entities",
    "generate_captions",
    "generate_captions_rule_based",
    "generate_captions_with_llm",
    "load_caption_guide",
    "parse_guide_rules",
    "parse_guide_templates",
]

GUIDE_FILENAME = "caption_guide.md"
MAX_CAPTION_CHARS = 150

# Risky words from the clipping guide: captions carrying one get 0
# views, so candidates containing one (whole word, case-insensitive)
# are dropped. Generic moderation words, not guide text — safe to
# hardcode. Masked forms included: TikTok reads through them.
RISKY_WORDS = {
    "fat", "mean", "bully", "fight", "gun", "dead", "hate", "fuck",
    "f*ck", "f**k", "fck",
}
_RISKY_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(RISKY_WORDS)) + r")\b",
    re.IGNORECASE)

# Trigger words the guide lists for captions: capitalized words that
# make a caption hit harder. Parsed from the guide when present;
# these are the fallback when the guide has no trigger-words section.
DEFAULT_TRIGGER_WORDS = ["PROVED", "SPEECHLESS", "REGRETTED", "REVEALED"]

# Built-in moment-type templates, used when the guide file is missing
# or has no templates section. GENERIC REWRITES in our own words —
# they match caption_guide.example.md, NOT the wording of any
# user-provided guide (that content stays in the local-only guide
# file and is parsed at runtime). Placeholders: (streamer) is the
# display name; (quick context) is the specific thing in the clip.
# Templates carry their own emojis — the topic emoji + heart
# decoration is never appended to them.
DEFAULT_TEMPLATES = {
    "meets_famous": [
        "the moment (streamer) had no clue who that was ❤️‍🩹",
    ],
    "song_reaction": [
        "the moment (streamer) heard that new song and lost it 😭❤️‍🩹",
    ],
    "emotional": [
        "the moment (streamer) got emotional on stream ❤️‍🩹",
        "(streamer) was crushed after (quick context) ❤️‍🩹",
    ],
    "apology": [
        "the moment (streamer) owned up and said sorry ❤️‍🩹",
    ],
    "career_moment": [
        "(streamer) just had the biggest moment of his career ❤️‍🩹",
    ],
}

# Built-in default rules, used when caption_guide.md is missing. Written
# in our own words; they match caption_guide.example.md.
DEFAULT_GUIDE_RULES = {
    # Viral moments get FULL context: streamer + game + the specific
    # thing + the number, when there is one.
    "full_context": True,
    # One variant drops the key detail to pull comments.
    "mystery": True,
    # Capitalize the reaction word (SPEECHLESS, REVEALED, PROVED).
    "caps_words": True,
    # Trigger words, capitalized where they fit.
    "trigger_words": list(DEFAULT_TRIGGER_WORDS),
}


# ---------------------------------------------------------------------------
# Guide file
# ---------------------------------------------------------------------------

def _default_guide_path():
    return Path(__file__).resolve().parent / GUIDE_FILENAME


def load_caption_guide(guide_path=None):
    """Read the caption guide file, re-read on every call so edits take
    effect without a restart or code change. Returns "" when the file
    is missing or unreadable — the caller falls back to the built-in
    default rules."""
    path = Path(guide_path) if guide_path else _default_guide_path()
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def parse_guide_rules(guide_text):
    """Turn guide text into the flags the template picker uses.

    An empty guide (file missing) means the built-in defaults. A guide
    that is present but silent on a point keeps the default for that
    point — except full context: a present guide that never mentions
    full context is taken to prefer shorter / mystery captions, so the
    full-context template stops being the best pick.

    Also returns "trigger_words" (the guide's capitalized trigger
    words, or the defaults) and "templates" (the guide's moment-type
    templates overlaid on the generic built-ins — see
    _templates_for_guide).
    """
    if not (guide_text or "").strip():
        rules = dict(DEFAULT_GUIDE_RULES)
        rules["trigger_words"] = list(DEFAULT_TRIGGER_WORDS)
        rules["templates"] = _templates_for_guide("")
        return rules
    lower = guide_text.lower()
    return {
        "full_context": "full context" in lower,
        "mystery": ("mystery" in lower or "comment" in lower
                    or "drop" in lower),
        "caps_words": ("no caps" not in lower
                       and "don't capitalize" not in lower
                       and "do not capitalize" not in lower),
        "trigger_words": _parse_trigger_words(guide_text),
        "templates": _templates_for_guide(guide_text),
    }


# Heading keyword -> moment kind, for the guide's template sections.
# The viral-moment section has no quoted template (it is an example
# sentence), so it maps to nothing and is skipped.
_TEMPLATE_HEADING_KINDS = [
    ("famous", "meets_famous"),
    ("song", "song_reaction"),
    ("emotional", "emotional"),
    ("apologiz", "apology"),
    ("apologis", "apology"),
    ("career", "career_moment"),
]


def parse_guide_templates(guide_text):
    """Parse the guide's moment-type templates out of the guide text.

    Format: a heading line like ``If the Streamer Meets Someone
    Famous:`` followed by one or more quoted template lines ``"..."``
    and a ``Tip:`` line. Returns {kind: [template, ...]} for the
    kinds the guide covers — {} when the guide has no templates
    section. Only the OUTERMOST pair of quotes is stripped, so nested
    quotes inside a template (the career template's "I MADE IT
    MOMENT") survive. ``Tip:`` and ``Example:`` lines are not
    templates and are ignored.
    """
    templates = {}
    current_kind = None
    for raw_line in (guide_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("If ") and line.endswith(":"):
            lower = line.lower()
            current_kind = None
            for keyword, kind in _TEMPLATE_HEADING_KINDS:
                if keyword in lower:
                    current_kind = kind
                    break
            continue
        if current_kind is None:
            continue
        if line.startswith('"') and line.endswith('"') and len(line) >= 2:
            templates.setdefault(current_kind, []).append(line[1:-1])
    return templates


def _templates_for_guide(guide_text):
    """Parsed templates overlaid on the generic built-ins: a guide
    that only covers some moment kinds still gets a template for the
    rest. Returns a fresh dict every call."""
    merged = {kind: list(lines)
              for kind, lines in DEFAULT_TEMPLATES.items()}
    for kind, lines in parse_guide_templates(guide_text).items():
        merged[kind] = list(lines)
    return merged


def _parse_trigger_words(guide_text):
    """The guide's trigger words: the comma-separated ALL-CAPS words
    on the first non-empty line after a "Trigger Words" heading.
    Defaults when the guide has no such section or the line carries
    no ALL-CAPS words."""
    lines = (guide_text or "").splitlines()
    for index, line in enumerate(lines):
        if "trigger word" not in line.lower():
            continue
        for following in lines[index + 1:]:
            stripped = following.strip()
            if not stripped:
                continue
            words = [word.strip(" .,;:") for word in
                     re.split(r"[,\s]+", stripped)]
            found = [word for word in words
                     if len(word) >= 3 and word.isalpha()
                     and word.isupper()]
            return found or list(DEFAULT_TRIGGER_WORDS)
    return list(DEFAULT_TRIGGER_WORDS)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

_MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,}\b")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Words that are capitalized only because they start a sentence (or are
# filler / pronouns / platform words) — never treat them as names.
_NAME_STOPWORDS = {
    "i", "do", "does", "did", "look", "like", "what", "who", "when",
    "where", "why", "how", "you", "he", "she", "we", "they", "it",
    "this", "that", "these", "those", "the", "a", "an", "and", "but",
    "so", "well", "now", "then", "just", "hey", "hi", "hello", "yep",
    "yeah", "yes", "no", "oh", "wow", "damn", "bro", "bruh", "man",
    "dude", "guys", "lol", "lmao", "haha", "okay", "ok", "right",
    "really", "actually", "literally", "mean", "think", "know", "see",
    "get", "got", "let", "come", "go", "say", "said", "tell", "can",
    "could", "will", "would", "should", "is", "are", "was", "were",
    "be", "been", "have", "has", "had", "my", "your", "his", "her",
    "our", "their", "me", "him", "us", "them", "am", "in", "on", "at",
    "to", "of", "for", "with", "from", "by", "as", "if", "or", "not",
    "all", "new", "one", "two", "first", "last", "big", "little",
    "good", "bad", "nice", "great", "crazy", "insane", "holy", "fuck",
    "shit", "god", "jesus", "christ", "lord", "omg", "wtf", "btw",
    "pls", "please", "sorry", "thanks", "thank", "live", "stream",
    "clip", "chat", "kick", "twitch", "tiktok", "youtube",
    "sure", "maybe", "woohoo", "alright", "listen", "wait", "hold",
    "rest",  # "Rest of you, get to the spot!" — game dialogue, not a name
}

# Game terms the extractor recognizes as the "specific thing" in a
# clip: places, characters and distinctive nouns. GTA 6 first (the
# main game clipped), plus nouns that carry a moment in any game.
# Places are kept apart from things: the money sentence names the
# THING being sold ("$500 strawberry farmer"), not the place.
_GAME_PLACES = {"leonida", "vice city"}
_GAME_TERMS = _GAME_PLACES | {
    "lucia", "jason", "strawberry", "farmer",
    "mission", "heist", "police", "cop", "gym", "casino", "blackjack",
    "poker", "alligator", "gator", "boat", "plane", "helicopter",
    "car", "chase", "crash", "crash site", "fight",
}

# Person/role nouns: in a money thing ("$500 strawberry farmer") the
# product comes first and the seller last.
_PERSON_TERMS = {"farmer", "lucia", "jason"}


def _term_pattern(term):
    """Regex matching a game term plus its plural: "strawberry" also
    matches "strawberries", "farmer" also matches "farmers"."""
    if term.endswith("y"):
        return re.compile(r"\b(?:" + re.escape(term) + r"|"
                          + re.escape(term[:-1]) + r"ies)\b")
    return re.compile(r"\b" + re.escape(term) + r"s?\b")


def _clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _split_sentences(text):
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def extract_entities(transcript, game=None, on_screen_text=None):
    """Pull the caption ingredients out of the transcript and the
    on-screen text (game dialogue + objectives read off the frames).

    Returns {"money", "names", "game_terms", "sentences"}: money is the
    first dollar amount mentioned ("" when none); names are capitalized
    words that are not sentence filler; game terms are recognized
    places / characters / objects, in first-seen order. The on-screen
    text feeds the same extraction — the streamer may barely speak
    while the game says everything — but it never feeds the reaction
    (see detect_reaction: game dialogue is not the streamer's
    reaction).
    """
    text = _clean_text(transcript)
    screen = _clean_text(on_screen_text)
    combined = (text + " " + screen).strip() if screen else text
    sentences = _split_sentences(combined)
    money_match = _MONEY_RE.search(combined)
    money = money_match.group(0) if money_match else ""

    names = []
    for match in _NAME_RE.finditer(combined):
        word = match.group(0)
        if word.lower() in _NAME_STOPWORDS:
            continue
        if word not in names:
            names.append(word)

    lower = combined.lower()
    found = []
    for term in _GAME_TERMS:
        match = _term_pattern(term).search(lower)
        if match:
            found.append((match.start(), term))
    # Same start position: the longer phrase wins ("crash site" over
    # "crash") so the thing names the place, not the bare word.
    found.sort(key=lambda item: (item[0], -len(item[1])))
    game_terms = [term for _, term in found]
    return {
        "money": money,
        "names": names,
        "game_terms": game_terms,
        "sentences": sentences,
    }


# ---------------------------------------------------------------------------
# Reaction detection — never invent an emotion the streamer did not show
# ---------------------------------------------------------------------------

# Transcript signal -> capitalized reaction word, checked in order.
_REACTION_SIGNALS = [
    ("speechless", "SPEECHLESS"),
    ("can't believe", "SHOCKED"),
    ("cannot believe", "SHOCKED"),
    ("no way", "SHOCKED"),
    ("oh my god", "SHOCKED"),
    ("what the", "SHOCKED"),
    ("insane", "SHOCKED"),
    ("scream", "SCREAMING"),
    ("laugh", "LAUGHING"),
    ("haha", "LAUGHING"),
    ("lmao", "LAUGHING"),
    ("funny", "LAUGHING"),
]


def detect_reaction(transcript):
    """The streamer's reaction as a capitalized word, or None.

    Only reactions actually present in the transcript count — a
    caption must never invent an emotion the streamer did not show.
    """
    lower = _clean_text(transcript).lower()
    if not lower:
        return None
    for signal, word in _REACTION_SIGNALS:
        if signal in lower:
            return word
    return None


# ---------------------------------------------------------------------------
# Moment-kind detection — which guide template a clip gets
# ---------------------------------------------------------------------------

# Transcript + headline signals only — never the on-screen text: game
# dialogue is not the streamer's moment. Checked in priority order:
# apology > song_reaction > emotional > career_moment > meets_famous.
# Viral moments need no kind: the full-context engine already covers
# them (money + the specific thing).
_APOLOGY_RE = re.compile(
    r"\b(sorry|apologiz\w*|apologis\w*|my bad|i apologize)\b")
_MUSIC_RE = re.compile(
    r"\b(grammy|song|rapper|album|track|music|beat)\b")
_EMOTIONAL_RE = re.compile(
    r"\b(cry|crying|tears|emotional|heartbroken|sad|broke down|break down)\b")
_CAREER_RE = re.compile(
    r"\b(i made it|tier 1|tier one|dream come true)\b")
_MEETING_RE = re.compile(
    r"\b(just met|met|meeting|ran into|bumped into)\b")


def detect_moment_kind(transcript, headline, entities=None):
    """The clip's moment kind from the transcript + headline, or None.

    One of "meets_famous", "song_reaction", "emotional", "apology",
    "career_moment". The kind decides the guide template the best
    pick uses; None means the existing full-context / generic
    behaviour. On-screen text is never consulted: what the game says
    is not the streamer's moment. ``entities`` (from extract_entities)
    supplies the capitalized names the meets-famous check needs; when
    it is None the names are extracted from the transcript + headline
    only.
    """
    text = _clean_text(transcript)
    head = _clean_text(headline)
    combined = (text + " " + head).strip().lower()
    if not combined:
        return None
    if _APOLOGY_RE.search(combined):
        return "apology"
    if _MUSIC_RE.search(combined):
        return "song_reaction"
    if _EMOTIONAL_RE.search(combined):
        return "emotional"
    if _CAREER_RE.search(combined):
        return "career_moment"
    if _MEETING_RE.search(combined):
        names = entities.get("names") if entities else None
        if names is None:
            names = extract_entities(
                (text + " " + head).strip()).get("names", [])
        if names:
            return "meets_famous"
    return None


# Transcript signals for the REGRETTED candidate: the streamer wishes
# he hadn't done it. Checked on the transcript only.
_REGRET_RE = re.compile(
    r"\b(regret\w*|shouldn'?t have|wish i hadn'?t|wish i didn'?t)\b")


def _contains_risky_word(text):
    """True when a caption carries a risky word from the clipping
    guide (whole word, case-insensitive) — such a caption gets 0
    views, so it is dropped, not posted."""
    return bool(_RISKY_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Moment-type lines (from visual moment detection). "{game_suffix}" is
# " in GTA 6" or "" when the game is unknown.
_MOMENT_LINES = {
    "chase": "{name} got chased down{game_suffix} and barely got away",
    "crash": "{name} just CRASHED in the worst way possible{game_suffix}",
    "fight": "{name} got into a FIGHT{game_suffix} and it went wild",
    "stunt": "{name} just pulled off the craziest stunt{game_suffix}",
    "reaction": "{name} could NOT believe what just happened{game_suffix}",
    "funny": "{name} could NOT stop laughing at what just happened{game_suffix}",
}

# Game-event OCR lines. Only labels with a known reading are used —
# an unknown banner never becomes a caption.
_OCR_LINES = {
    "wasted": "{name} just got WASTED{game_suffix}",
    "busted": "{name} just got BUSTED{game_suffix}",
    "mission passed": "{name} just PASSED the mission{game_suffix}",
}

# Emoji per topic, Claude's pattern: a topic emoji + a heart
# (🍓💖). Each entry is a regex checked against the lowercased
# caption, first match wins ("strawberry" before "farmer": the
# strawberry farmer gets 🍓, not 🌾). The patterns allow the usual
# word forms ("crashed", "chased", "laughing") but stay word-bound
# so "car" never matches "card".
_EMOJI_TOPICS = [
    (r"\bstrawberr(?:y|ies)\b", "🍓"),
    (r"\bcrash(?:ed|ing|es|s)?\b", "💥"),
    (r"\$", "💰"),
    (r"\bchase[ds]?\b", "🚗"),
    (r"\bcars?\b", "🚗"),
    (r"\bpolice\b", "🚔"),
    (r"\bcops?\b", "🚔"),
    (r"\bbusted\b", "🚔"),
    (r"\bfight(?:ing|s)?\b", "🥊"),
    (r"\bstunts?\b", "🤯"),
    (r"\bmissions?\b", "🎯"),
    (r"\bheists?\b", "🎯"),
    (r"\bcasino\b", "🎰"),
    (r"\bblackjack\b", "🎰"),
    (r"\bpoker\b", "🎰"),
    (r"\bgym\b", "💪"),
    (r"\bboats?\b", "🚤"),
    (r"\bplanes?\b", "🚁"),
    (r"\bhelicopters?\b", "🚁"),
    (r"\balligators?\b", "🐊"),
    (r"\bgators?\b", "🐊"),
    (r"\bwasted\b", "💀"),
    (r"\bfarmers?\b", "🌾"),
    (r"\bleonida\b", "🌴"),
    (r"\bvice city\b", "🌴"),
    (r"\blaugh(?:ed|ing|s)?\b", "😂"),
    (r"\bfunny\b", "😂"),
    (r"\bspeechless\b", "😱"),
    (r"\bshocked\b", "😱"),
]
_EMOJI_DEFAULT = "🔥"
_EMOJI_SECOND = "💖"


def _caption_emojis(text):
    """1-2 emojis for a caption: the topic emoji + a heart. The emoji
    always relates to the caption's content; with no topic match the
    default is 🔥."""
    lower = (text or "").lower()
    for pattern, emoji in _EMOJI_TOPICS:
        if re.search(pattern, lower):
            return f"{emoji}{_EMOJI_SECOND}"
    return f"{_EMOJI_DEFAULT}{_EMOJI_SECOND}"


# Capitalized tokens, lowercased when the guide asks for no caps.
_CAPS_TOKENS = {
    "SPEECHLESS": "speechless",
    "SHOCKED": "shocked",
    "SCREAMING": "screaming",
    "LAUGHING": "laughing",
    "NO IDEA": "no idea",
    "PROVED": "proved",
    "REGRETTED": "regretted",
    "REVEALED": "revealed",
    "CRASHED": "crashed",
    "FIGHT": "fight",
    "PASSED": "passed",
    "WASTED": "wasted",
    "BUSTED": "busted",
    "NOT": "not",
}


def _display_name(streamer):
    name = _clean_text(streamer)
    return name.title() if name else "The streamer"


def _game_suffix(game):
    game = _clean_text(game)
    return f" in {game}" if game else ""


def _truncate(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0]
    return (cut or text[:limit - 1]).rstrip(" .,!?") + "…"


def _caps_thing(thing, entities):
    """Uppercase the distinctive noun in a thing phrase, e.g.
    "$500 strawberry farmer" -> "$500 STRAWBERRY farmer". The first
    recognized game term in the phrase, in phrase order, is the
    distinctive one."""
    best = None
    for term in entities.get("game_terms", []):
        match = re.search(r"\b" + re.escape(term) + r"\b", thing,
                          re.IGNORECASE)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), match.end())
    if best is None:
        return thing
    start, end = best
    return thing[:start] + thing[start:end].upper() + thing[end:]


# Words a quoted clause must not END with: Whisper fragments like
# "Sure not leaving the" end mid-thought, and a caption built on one
# reads broken ("had NO IDEA Sure not leaving the").
_CLAUSE_END_STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "with", "in", "on", "at",
    "and", "but", "or", "so", "his", "her", "their", "my", "your",
    "is", "was", "had", "has", "have", "be", "been", "do", "did",
    "does", "not", "no",
}


def _is_complete_clause(clause):
    """True when a quoted clause reads as a complete thought: at least
    3 words and ending on a content word, not a function word."""
    words = (clause or "").split()
    if len(words) < 3:
        return False
    last = re.sub(r"[^a-z0-9]", "", words[-1].lower())
    return last not in _CLAUSE_END_STOPWORDS


def _specific_thing(entities, headline, streamer):
    """The concrete thing the caption names, as (thing, style, strong).

    Style "noun" reads after "revealed" ("a $500 STRAWBERRY farmer",
    "Leonida", "the mission"); style "clause" reads after "hearing"
    ("Chaz is not clean"). Strong things (money + nouns, a complete
    quoted clause) can headline the best pick on their own; weak
    things (a place, a common noun) only do when a reaction carries
    them — "had NO IDEA GTA 6 had the mission" is grammatical but
    weak, so it stays an alternative, not the best pick.

    A bare person's name is never a thing: "had NO IDEA GTA 6 had
    Ethan" is nonsense, and "revealed Ethan" claims the person is in
    the game. Names feed the mystery topic instead.

    Prefers money + the distinctive nouns in the money sentence, then
    a complete quoted clause in the story headline, then a place, then
    a common noun. Returns ("", "", False) when nothing specific
    exists — the templates fall back to the moment itself.
    """
    money = entities.get("money", "")
    sentences = entities.get("sentences", [])
    if money:
        money_sentence = next(
            (s for s in sentences if money in s), "")
        lower_sentence = money_sentence.lower()
        # The THING being sold, not the place: "strawberry farmer",
        # not "Leonida". Places only fill in when no thing noun is
        # in the money sentence.
        nouns = [t for t in entities.get("game_terms", [])
                 if t not in _GAME_PLACES
                 and _term_pattern(t).search(lower_sentence)]
        if not nouns:
            nouns = [t for t in entities.get("game_terms", [])
                     if _term_pattern(t).search(lower_sentence)]
        if nouns:
            # Product before seller: "strawberry farmer", not "farmer
            # strawberry" — the money sentence names the thing being
            # sold ("<seller> selling <product> for $X"), and the
            # product is the distinctive noun.
            nouns.sort(key=lambda t: 1 if t in _PERSON_TERMS else 0)
            thing = f"{money} {' '.join(nouns[:2])}"
            return (_caps_thing(thing, entities), "noun", True)
    headline = _clean_text(headline)
    quoted = re.search(r'"([^"]{3,60})"', headline)
    if quoted and _is_complete_clause(quoted.group(1)):
        return (quoted.group(1).strip(), "clause", True)
    terms = entities.get("game_terms", [])
    for term in terms:
        if term in _GAME_PLACES:
            return (term.title(), "noun", False)
    if terms:
        return (f"the {terms[0]}", "noun", False)
    return ("", "", False)


def _mystery_topic(entities, hide_names=False):
    """What the mystery variant points at, without naming the detail:
    a place ("Leonida"), a common noun ("the mission"), or a name
    ("Chaz"). "" when nothing exists. With hide_names (meeting a
    famous person, reacting to a song) a person's name is never the
    topic: the guide's mystery rule says the name stays out so the
    comments argue about it."""
    terms = entities.get("game_terms", [])
    for term in terms:
        if term in _GAME_PLACES:
            return term.title()
    if terms:
        return f"the {terms[0]}"
    if hide_names:
        return ""
    names = entities.get("names", [])
    if names:
        return names[0]
    return ""


def _render_template(template, name, thing, thing_style, moment_word):
    """Fill a guide template's placeholders: (streamer) is the display
    name; (quick context) is the specific thing when it is a noun
    ("the crash", "$500 STRAWBERRY farmer"), else "the <moment word>".
    A clause-style thing ("Chaz is not clean") reads after "hearing",
    not after "after", so it never fills (quick context). Returns None
    when the template cannot be filled."""
    rendered = (template or "").replace("(streamer)", name)
    if "(quick context)" in rendered:
        if thing and thing_style == "noun":
            fill = thing
        elif moment_word:
            fill = f"the {moment_word}"
        else:
            return None
        rendered = rendered.replace("(quick context)", fill)
    return _clean_text(rendered)


def _uncap(text, caps_words):
    """Lowercase the capitalized reaction words when the guide asks
    for no caps. A no-op when caps are on (the default)."""
    if caps_words:
        return text
    for caps, lower in _CAPS_TOKENS.items():
        text = re.sub(r"\b" + re.escape(caps) + r"\b", lower, text)
    return text


def _normalize(text):
    """Dedup key: lowercase alphanumeric only."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _build_candidates(context, entities, reaction, thing, thing_style,
                      guide_rules, moment_kind=None, templates=None):
    """All caption candidates in preference order: each is
    (key, text, reason). Keys: template, full_context, mystery,
    proved, regret, ocr, moment, headline, generic. Any candidate
    carrying a risky word is dropped (see RISKY_WORDS) — the best
    pick then falls to the next clean candidate."""
    name = _display_name(context.get("streamer"))
    game = _clean_text(context.get("game"))
    game_suffix = _game_suffix(game)
    game_word = game or "this game"
    headline = _clean_text(context.get("headline"))
    moment_type = (context.get("moment_type") or "").strip().lower()
    ocr_hits = context.get("ocr_hits") or []
    caps_words = guide_rules.get("caps_words", True)
    candidates = []

    def add(key, text, reason):
        text = _uncap(_truncate(_clean_text(text), MAX_CAPTION_CHARS),
                      caps_words)
        if text and not _contains_risky_word(text):
            candidates.append((key, text, reason))

    # The moment word, for the PROVED line and for filling a
    # template's (quick context): "reaction" / "funny" are not things
    # to point at — they read as "after this moment", while chase /
    # crash / fight / stunt keep their word.
    moment_word = {"chase": "chase", "crash": "crash", "fight": "fight",
                   "stunt": "stunt"}.get(moment_type)
    if not moment_word:
        moment_word = ("mission" if "mission" in entities.get("game_terms", [])
                       else "moment")
    # A risky moment word ("fight") never reaches a caption: it reads
    # as "moment" instead, the same fallback reaction / funny use.
    if _contains_risky_word(moment_word):
        moment_word = "moment"

    # 0. Guide template for the moment kind: when the clip matches one
    # of the guide's moment types, that template is the best pick and
    # its other variants are alternatives. A template's own emojis
    # stay — the decoration step never appends to a template caption.
    if moment_kind:
        kind_reason = {
            "meets_famous": "Guide template: hides the name so the "
                            "comments argue about it.",
            "song_reaction": "Guide template: hides the artist so the "
                             "comments argue about it.",
            "emotional": "Guide template for an emotional moment.",
            "apology": "Guide template: frames the apology as rare "
                       "maturity.",
            "career_moment": "Guide template for a big career moment.",
        }.get(moment_kind, "Guide template for this moment.")
        for template in (templates or {}).get(moment_kind, []):
            rendered = _render_template(template, name, thing,
                                        thing_style, moment_word)
            if rendered:
                add("template", rendered, kind_reason)

    # 1. Full context (the viral moment template): streamer + game +
    # the specific thing + the number, when there is one.
    if thing:
        if thing_style == "clause":
            if reaction:
                add("full_context",
                    f"{name} was {reaction} after hearing {thing}",
                    "Full context: names the streamer and the exact moment.")
            else:
                add("full_context",
                    f"{name} had NO IDEA {thing}",
                    "Full context: names the streamer and the exact moment.")
        else:
            # No article: money things ("$500 STRAWBERRY farmer"),
            # places ("Leonida") and "the"-things ("the mission")
            # all read without one.
            if reaction:
                add("full_context",
                    f"{name} was {reaction} after {game_word} revealed "
                    f"{thing}",
                    "Full context: streamer, game, the specific thing and "
                    "the number.")
            else:
                add("full_context",
                    f"{name} had NO IDEA {game_word} had {thing}",
                    "Full context: streamer, game, the specific thing and "
                    "the number.")

    # 2. Mystery: drops the key detail to pull comments. For a famous
    # meeting or a song reaction the name stays out of the mystery
    # too — the guide's rule is that the comments argue about it.
    topic = _mystery_topic(
        entities,
        hide_names=moment_kind in ("meets_famous", "song_reaction"))
    # A risky topic ("the fight") never names the mystery: the
    # no-topic variant ("what the game was hiding") carries it instead.
    if topic and _contains_risky_word(topic):
        topic = ""
    if topic:
        add("mystery",
            f"the moment {name} realized what {game_word} just REVEALED "
            f"about {topic}",
            "Drops the key detail to pull comments.")
    else:
        add("mystery",
            f"the moment {name} realized what {game_word} was hiding",
            "Drops the key detail to pull comments.")

    # 3. PROVED: a bold claim that baits shares and arguments.
    add("proved",
        f"{name} PROVED {game_word} is the most unhinged game ever "
        f"after this {moment_word}",
        "A bold claim that baits shares and arguments.")

    # 3b. REGRETTED: the streamer wishes he hadn't done it — only
    # when the transcript itself signals regret.
    transcript_lower = _clean_text(context.get("transcript")).lower()
    if _REGRET_RE.search(transcript_lower):
        add("regret",
            f"{name} REGRETTED what he just did{game_suffix}",
            "Names the regret the streamer showed.")

    # 4. Game-event OCR: what the game screen itself announced.
    for hit in ocr_hits:
        label = (hit.get("label") if isinstance(hit, dict)
                 else str(hit)).strip().lower()
        line = _OCR_LINES.get(label)
        if line:
            add("ocr",
                line.format(name=name, game_suffix=game_suffix),
                "Names what the game screen announced.")
            break

    # 5. Moment type: what kind of moment the clip is.
    line = _MOMENT_LINES.get(moment_type)
    if line:
        add("moment",
            line.format(name=name, game_suffix=game_suffix),
            "Names the moment itself, no fake reaction.")

    # 6. The story headline already names the moment — unless its
    # quoted clause is a Whisper fragment ("Sure not leaving the"),
    # which would carry the broken wording into the caption.
    if headline:
        quoted = re.search(r'"([^"]{3,60})"', headline)
        if quoted is None or _is_complete_clause(quoted.group(1)):
            add("headline", headline,
                "The headline already names the moment.")

    # 7. Generic fallbacks — always available, never the best pick
    # when anything specific exists.
    add("generic",
        f"{name} did NOT see that coming{game_suffix} live",
        "Short hook for a clip with no specific detail.")
    add("generic",
        f"{name} could NOT believe what just happened{game_suffix} live",
        "Short hook for a clip with no specific detail.")
    return candidates


def generate_captions_rule_based(clip_context, guide_rules=None):
    """Rule-based captions: best pick + 4 alternatives, each with a
    one-line reason. No API key needed. ``guide_rules`` comes from
    parse_guide_rules; None means the built-in defaults."""
    context = clip_context or {}
    rules = guide_rules or dict(DEFAULT_GUIDE_RULES)
    transcript = context.get("transcript") or ""
    on_screen = (context.get("on_screen_text")
                 or context.get("game_dialogue") or "")
    entities = extract_entities(transcript, context.get("game"), on_screen)
    # The reaction comes from the transcript only: game dialogue is
    # not the streamer's reaction.
    reaction = detect_reaction(transcript)
    # The moment kind comes from the transcript + headline only, for
    # the same reason: what the game says is not the streamer's moment.
    moment_kind = detect_moment_kind(transcript, context.get("headline"))
    templates = rules.get("templates") or _templates_for_guide("")
    thing, thing_style, _strong = _specific_thing(
        entities, context.get("headline"), context.get("streamer"))
    candidates = _build_candidates(context, entities, reaction, thing,
                                   thing_style, rules, moment_kind,
                                   templates)

    def first_of(keys):
        for key, text, reason in candidates:
            if key in keys:
                return (key, text, reason)
        return None

    # Best pick: when the clip matches one of the guide's moment
    # types, that type's template wins — the guide's style for the
    # moment beats the generic rules. Otherwise the guide's
    # viral-moment rule — FULL context whenever a specific thing
    # exists (from the transcript or the on-screen text), strong or
    # weak; a weak thing ("the crash site") still names the moment,
    # and a generic hook only wins when there is genuinely no
    # specific thing anywhere. The mystery variant is an alternative,
    # never the best pick: it drops the detail the best pick exists
    # to name.
    best = None
    if moment_kind:
        best = first_of({"template"})
    if best is None and thing:
        best = first_of({"full_context"})
    if best is None:
        best = first_of({"regret", "ocr", "moment", "headline",
                         "generic"})
    if best is None:  # pragma: no cover — generic fallbacks always exist
        best = ("generic", "Watch this clip", "Fallback caption.")

    seen = {_normalize(best[1])}
    alternatives = []
    alt_reasons = []
    for key, text, reason in candidates:
        if len(alternatives) >= 4:
            break
        norm = _normalize(text)
        if norm in seen or not norm:
            continue
        seen.add(norm)
        alternatives.append((key, text))
        alt_reasons.append(reason)
    # Fillers, only reached when the transcript is nearly empty.
    name = _display_name(context.get("streamer"))
    for filler in (f"{name} just had a wild moment live",
                   f"{name} with the clip of the day"):
        if len(alternatives) >= 4:
            break
        if _normalize(filler) not in seen:
            seen.add(_normalize(filler))
            alternatives.append(("generic", filler))
            alt_reasons.append("Short hook for a clip with no specific detail.")
    # Emojis last, after dedupe (they are stripped by _normalize, so
    # they never affect it) and after truncation (they are never cut).
    # Template captions carry their own emojis from the guide, so
    # they skip the topic emoji + heart decoration.
    def decorate(key, text):
        if key == "template":
            return text
        return f"{text} {_caption_emojis(text)}"

    decorated_best = decorate(best[0], best[1])
    decorated_alts = [decorate(key, text)
                      for key, text in alternatives[:4]]
    return {
        "best": decorated_best,
        "alternatives": decorated_alts,
        "reasons": {"best": best[2], "alternatives": alt_reasons[:4]},
        "provider": "rule_based",
    }


# ---------------------------------------------------------------------------
# LLM path (Gemini) — untested without a real key
# ---------------------------------------------------------------------------

def _build_caption_prompt(clip_context, guide_text):
    """The caption prompt: the guide's rules + everything known about
    the clip, asking for one best pick and four alternatives as JSON."""
    context = clip_context or {}
    transcript = _clean_text(context.get("transcript"))[:2000]
    on_screen = _clean_text(
        context.get("on_screen_text")
        or context.get("game_dialogue"))[:2000]
    ocr_hits = context.get("ocr_hits") or []
    ocr_labels = ", ".join(
        (h.get("label") if isinstance(h, dict) else str(h))
        for h in ocr_hits) or "none"
    trigger_words = ", ".join(parse_guide_rules(guide_text or "")
                              .get("trigger_words")
                              or DEFAULT_TRIGGER_WORDS)
    risky_words = ", ".join(sorted(RISKY_WORDS - {"f*ck", "f**k", "fck"}))
    return (
        "You write TikTok post captions for streamer clips. Follow the "
        "caption guide below exactly — it is the knowledge for this job.\n\n"
        "CAPTION GUIDE:\n"
        f"{guide_text or '(no guide file — use the default rules: name the specific moment, give full context on viral moments, one variant drops the key detail to pull comments, capitalize the reaction word, one line, never invent a reaction the streamer did not have.)'}\n\n"
        "CLIP:\n"
        f"Streamer: {context.get('streamer') or 'unknown'}\n"
        f"Game: {context.get('game') or 'unknown'}\n"
        f"Story headline: {context.get('headline') or 'unknown'}\n"
        f"Moment type: {context.get('moment_type') or 'unknown'}\n"
        f"Game screen events: {ocr_labels}\n"
        f"Transcript: {transcript or '(no dialogue)'}\n"
        f"On-screen text (game dialogue and objectives read off the "
        f"video frames): {on_screen or '(none)'}\n\n"
        "Return JSON only, exactly this shape:\n"
        '{"best": "<one best caption>", '
        '"alternatives": ["<caption>", "<caption>", "<caption>", "<caption>"], '
        '"reasons": {"best": "<one line>", "alternatives": ["<one line>", "<one line>", "<one line>", "<one line>"]}}\n'
        "No hashtags in the captions — they are added separately at post time. "
        "End each caption with 1-2 relevant emojis (a topic emoji plus "
        "a heart, like 🍓💖). "
        f"Use the trigger words where they fit: {trigger_words}. "
        f"Keep every caption PG: never use these words or masked forms "
        f"of them: {risky_words}, f*ck. "
        "Never invent a reaction the streamer did not have."
    )


def _default_gemini_fn(prompt):
    """Call the shared Gemini client (ai_story_engine). Raises when no
    key is configured or the call fails — the caller falls back to the
    rule-based captions."""
    from ai_story_engine import gemini_client
    from config import GEMINI_MODEL
    if gemini_client is None:
        raise RuntimeError("Gemini client not configured")
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt)
    from llm_providers import parse_json_response
    return parse_json_response(response.text or "")


def _strip_hashtags(text):
    """Remove #tags from an LLM caption — hashtags are added separately
    at post time, so a caption must not carry its own."""
    return _clean_text(re.sub(r"#\w+", "", text or ""))


def generate_captions_with_llm(clip_context, guide_text, llm_fn=None):
    """Ask Gemini for captions following the guide. Raises on any
    failure — the caller falls back to the rule-based captions.

    ``llm_fn(prompt) -> dict`` is injectable for tests; the default
    calls the shared Gemini client. UNTESTED without a real Gemini key
    (none has been provided).
    """
    fn = llm_fn or _default_gemini_fn
    data = fn(_build_caption_prompt(clip_context, guide_text))
    if not isinstance(data, dict):
        raise ValueError("LLM captions are not a JSON object")
    best = _strip_hashtags(data.get("best", ""))
    alts = [_strip_hashtags(a) for a in data.get("alternatives", []) or []]
    alts = [a for a in alts if a]
    reasons = data.get("reasons", {}) or {}
    if not best or not alts:
        raise ValueError("LLM captions missing best pick or alternatives")
    alt_reasons = [str(r).strip() for r in
                   (reasons.get("alternatives", []) or [])]
    # Pad a short alternatives list with rule-based ones (deduped) —
    # a short LLM list must never fail the clip.
    if len(alts) < 4:
        rule_based = generate_captions_rule_based(clip_context)
        seen = {_normalize(best)} | {_normalize(a) for a in alts}
        for alt, reason in zip(rule_based["alternatives"],
                               rule_based["reasons"]["alternatives"]):
            if len(alts) >= 4:
                break
            if _normalize(alt) not in seen:
                seen.add(_normalize(alt))
                alts.append(alt)
                alt_reasons.append(reason)
    while len(alt_reasons) < len(alts):
        alt_reasons.append("Alternative take on the same moment.")
    return {
        "best": _truncate(best, MAX_CAPTION_CHARS),
        "alternatives": [_truncate(a, MAX_CAPTION_CHARS) for a in alts[:4]],
        "reasons": {
            "best": str(reasons.get("best", "")).strip()
            or "Best pick from the guide.",
            "alternatives": alt_reasons[:4],
        },
        "provider": "gemini",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_captions(clip_context, guide_text=None, guide_path=None,
                      use_llm=None, llm_fn=None):
    """Generate post captions for a clip: best pick + 4 alternatives,
    each with a one-line reason.

    clip_context: streamer, game, transcript, headline, moment_type,
    ocr_hits, category, on_screen_text (or game_dialogue: the game
    dialogue + objectives read off the frames). guide_text overrides
    the guide file when given (tests); otherwise caption_guide.md is
    re-read every run. use_llm: None means "use Gemini when a key is
    configured"; an injected llm_fn also enables the LLM path. Any LLM
    failure falls back to the rule-based captions — a clip never fails
    over captions.
    """
    if guide_text is None:
        guide_text = load_caption_guide(guide_path)
    rules = parse_guide_rules(guide_text)
    if use_llm is None:
        if llm_fn is not None:
            use_llm = True
        else:
            try:
                from config import GEMINI_API_KEY
                use_llm = bool(GEMINI_API_KEY)
            except Exception:
                use_llm = False
    if use_llm:
        try:
            result = generate_captions_with_llm(clip_context, guide_text,
                                                llm_fn=llm_fn)
        except Exception as exc:
            logger.warning(
                "Gemini captions failed (%s); using rule-based captions",
                exc)
            result = generate_captions_rule_based(clip_context, rules)
    else:
        result = generate_captions_rule_based(clip_context, rules)
    # The on-screen text that fed the captions, stored with the clip
    # so the dashboard can show what the captions were built from.
    on_screen = _clean_text(
        (clip_context or {}).get("on_screen_text")
        or (clip_context or {}).get("game_dialogue"))[:500]
    result["on_screen_text"] = on_screen
    return result
