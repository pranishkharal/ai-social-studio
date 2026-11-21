"""Story headlines, quote picker and post captions for TikTok clips.

Built from the study of @zysno, @godsclipper and @kicksclipping
(see tiktok-clipping-accounts-analysis.md): their headlines name the
SPECIFIC moment in the clip ("Zavala breaks the guitar Amanda got him"),
while our old heuristic titles were generic ("COMPLETELY SPEECHLESS").

This module is stdlib only, makes no network calls and holds no secrets.
The transcript passed in is already prestige-sanitized by the caller
(see AIStoryEngine.prestige_sanitize); this module deliberately does NOT
import ai_story_engine (that would be a circular import).
"""

import re

__all__ = [
    "GAME_HASHTAGS",
    "build_story_headline",
    "compose_hashtags",
    "compose_post_caption",
    "game_hashtag",
    "pick_quote_line",
]

MAX_HEADLINE_CHARS = 90
MAX_QUOTE_CHARS = 120

# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NAME_RE = re.compile(r"\b[A-Z][a-z]{1,}\b")

# Words that are capitalized only because they start a sentence (or are
# filler / pronouns / platform words) -- never treat them as names.
_NAME_STOPWORDS = {
    "i", "do", "does", "did", "look", "like", "what", "who", "when", "where",
    "why", "how", "you", "he", "she", "we", "they", "it", "this", "that",
    "these", "those", "the", "a", "an", "and", "but", "so", "well", "now",
    "then", "just", "hey", "hi", "hello", "yep", "yeah", "yes", "no", "oh",
    "wow", "damn", "bro", "bruh", "man", "dude", "guys", "lol", "lmao",
    "haha", "okay", "ok", "right", "really", "actually", "literally",
    "mean", "think", "know", "see", "get", "got", "let", "come", "go",
    "say", "said", "tell", "can", "could", "will", "would", "should", "is",
    "are", "was", "were", "be", "been", "have", "has", "had", "my", "your",
    "his", "her", "our", "their", "me", "him", "us", "them", "am", "in",
    "on", "at", "to", "of", "for", "with", "from", "by", "as", "if", "or",
    "not", "all", "new", "one", "two", "first", "last", "big", "little",
    "good", "bad", "nice", "great", "crazy", "insane", "holy", "fuck",
    "shit", "god", "jesus", "christ", "lord", "omg", "wtf", "btw", "pls",
    "please", "sorry", "thanks", "thank", "congrats", "happy", "sad",
    "live", "stream", "clip", "chat", "kick", "twitch", "tiktok",
    "youtube", "gta", "fortnite", "valorant", "minecraft",
}

_FILLER_WORDS = {
    "um", "uh", "yeah", "yep", "like", "okay", "ok", "hey", "well", "so",
    "hmm", "uh-huh", "mm", "nah", "bro", "man", "dude", "you", "know",
}


def _clean_text(text):
    """Collapse whitespace; return "" for None/blank input."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def _words_to_text(words):
    """Rebuild a transcript from Whisper-style word dicts (or plain str)."""
    if not words:
        return ""
    pieces = []
    for w in words:
        if isinstance(w, dict):
            pieces.append(str(w.get("word", "")))
        else:
            pieces.append(str(w))
    return _clean_text(" ".join(pieces))


def _resolve_transcript(transcript, words):
    """Use the transcript, falling back to the word list when it is empty."""
    text = _clean_text(transcript)
    if not text:
        text = _words_to_text(words)
    return text


def _split_sentences(text):
    """Split a transcript into sentences, keeping end punctuation."""
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _word_count(sentence):
    return len(sentence.split())


def _display_name(streamer):
    name = _clean_text(streamer)
    return name.title() if name else "The streamer"


def _extract_names(text, streamer=""):
    """Capitalized names mentioned in the transcript, in first-seen order.

    Excludes sentence starters, filler and the streamer's own name.
    """
    exclude = {s.lower() for s in [streamer] if s}
    names = []
    for match in _NAME_RE.finditer(text or ""):
        word = match.group(0)
        lower = word.lower()
        if lower in _NAME_STOPWORDS or lower in exclude:
            continue
        if word not in names:
            names.append(word)
    return names


# ---------------------------------------------------------------------------
# Quote picker (Feature 2)
# ---------------------------------------------------------------------------

_EMOTION_WORDS = {
    "crazy", "insane", "wow", "damn", "unbelievable", "unreal", "wild",
    "holy", "badass", "literally", "actually", "bro", "dude", "omg",
    "fuck", "f*ck", "shit", "hell", "god", "hilarious", "funniest",
    "impossible", "ridiculous", "legendary", "epic", "nuts", "mental",
}

_EMOTION_PHRASES = {
    "no way", "oh my god", "oh my", "what the", "can't believe",
    "cannot believe", "just did", "look at", "did you see",
    "are you kidding", "shut up", "no fucking way", "holy shit",
    "holy crap", "what just happened", "i swear",
}


def _score_sentence(sentence):
    """Punchiness score for a spoken sentence; None means unusable.

    Rewards exclamation/question marks, emotion words and the 4-15 word
    sweet spot. Fragments, very long sentences and filler-only lines
    are rejected.
    """
    words = sentence.split()
    n = len(words)
    if n < 3 or n > 20 or len(sentence) > MAX_QUOTE_CHARS:
        return None
    cleaned = [re.sub(r"[^a-z0-9*']", "", w.lower()) for w in words]
    if all(w in _FILLER_WORDS or not w for w in cleaned):
        return None

    score = 0
    if 4 <= n <= 15:
        score += 3
    elif n == 3 or 16 <= n <= 20:
        score += 1
    if "!" in sentence:
        score += 3
    if "?" in sentence:
        score += 2

    lower = sentence.lower()
    emotion = 0
    for phrase in _EMOTION_PHRASES:
        if phrase in lower:
            emotion += 2
    for w in cleaned:
        if w in _EMOTION_WORDS:
            emotion += 2
    score += min(emotion, 6)
    return score


def pick_quote_line(transcript, words=None):
    """Pick the single punchiest spoken sentence from the clip.

    This is ONE static line (a lower-third quote), not captions.
    Returns "" when nothing suitable exists (empty transcript, only
    fragments).
    """
    text = _resolve_transcript(transcript, words)
    best = ""
    best_score = -1
    for sentence in _split_sentences(text):
        score = _score_sentence(sentence)
        if score is not None and score > best_score:
            best_score = score
            best = sentence
    return best


# ---------------------------------------------------------------------------
# Story headline builder (Feature 1)
# ---------------------------------------------------------------------------

# Irregular past tense -> third-person singular present.
_IRREGULAR_PAST_TO_PRESENT = {
    "said": "says", "told": "tells", "thought": "thinks", "knew": "knows",
    "saw": "sees", "went": "goes", "came": "comes", "got": "gets",
    "took": "takes", "gave": "gives", "made": "makes", "found": "finds",
    "lost": "loses", "won": "wins", "broke": "breaks", "bought": "buys",
    "sold": "sells", "stole": "steals", "ate": "eats", "drank": "drinks",
    "drove": "drives", "fell": "falls", "ran": "runs", "paid": "pays",
    "beat": "beats", "read": "reads", "wrote": "writes", "drew": "draws",
    "built": "builds", "slept": "sleeps", "woke": "wakes", "spoke": "speaks",
    "chose": "chooses", "wore": "wears", "tore": "tears", "flew": "flies",
    "grew": "grows", "brought": "brings", "taught": "teaches",
    "kept": "keeps", "sent": "sends", "spent": "spends", "met": "meets",
    "sat": "sits", "stood": "stands", "understood": "understands",
    "shot": "shoots", "cut": "cuts", "put": "puts", "set": "sets",
    "shut": "shuts", "hurt": "hurts", "cost": "costs", "hit": "hits",
    "let": "lets", "quit": "quits", "felt": "feels", "left": "leaves",
    "held": "holds", "led": "leads", "fed": "feeds", "bled": "bleeds",
    "sped": "speeds", "lent": "lends", "bent": "bends", "dealt": "deals",
    "dreamt": "dreams", "meant": "means", "lit": "lights",
    "smelt": "smells", "swept": "sweeps", "wept": "weeps",
    "crept": "creeps", "began": "begins", "swam": "swims", "sang": "sings",
    "rang": "rings", "sank": "sinks", "stank": "stinks", "shrank": "shrinks",
    "sprang": "springs", "rode": "rides", "rose": "rises", "froze": "freezes",
    "threw": "throws", "blew": "blows", "dug": "digs", "hung": "hangs",
    "stuck": "sticks", "struck": "strikes", "swore": "swears",
    "bore": "bears", "died": "dies", "lied": "lies", "tied": "ties",
}

# Third-person singular present forms we accept as verbs in the transcript.
_PRESENT_VERBS = set(_IRREGULAR_PAST_TO_PRESENT.values()) | {
    "smashes", "destroys", "exposes", "reveals", "announces", "admits",
    "confesses", "apologizes", "thanks", "celebrates", "cheers",
    "watches", "listens", "hears", "dances", "plays", "laughs", "cries",
    "screams", "yells", "whispers", "jumps", "catches", "drops", "picks",
    "opens", "closes", "starts", "stops", "tries", "fails", "misses",
    "saves", "blocks", "knocks", "punches", "slaps", "kisses", "hugs",
    "texts", "calls", "posts", "streams", "bans", "kicks", "mutes",
    "gifts", "donates", "follows", "reports", "leaks", "cheats",
    "hacks", "scams", "robs", "fires", "hires", "marries", "dumps",
    "proposes", "graduates", "fixes", "spills", "cooks", "orders",
    "delivers", "arrives", "returns", "stays", "waits", "walks",
    "wants", "needs", "loves", "hates", "believes", "remembers",
    "forgets", "learns", "helps", "kills", "crashes", "bets", "pulls",
    "asks", "answers", "shows", "does", "slips", "falls", "earns",
    "finds", "says", "goes", "gets",
}

# Verbs too weak or auxiliary to headline a clip.
_WEAK_VERBS = {
    "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "do", "does", "did", "can", "could", "will", "would", "should",
    "may", "might", "must", "thinks", "knows", "feels", "seems",
    "looks", "sounds", "means", "likes",
}

# Words that end the object phrase (new clause / filler).
_OBJECT_STOPS = {
    "and", "but", "so", "because", "then", "though", "while", "when",
    "if", "or", "like", "um", "uh",
}

# Trailing words trimmed off the end of an object phrase.
_OBJECT_TRIM_TAIL = {
    "for", "with", "to", "on", "in", "at", "of", "and", "but", "so",
    "the", "a", "an", "his", "her", "their", "my", "your",
}

# Object words too vague on their own to count as specific.
_WEAK_OBJECTS = {
    "it", "that", "this", "there", "here", "him", "her", "them", "us",
    "me", "you", "one", "something", "anything", "everything",
    "nothing", "up", "down", "out", "in", "on", "off", "away",
}

# First/second person -> third person for the headline subject.
_PRONOUN_MAP = {
    "your": "his", "my": "his", "our": "his",
    "you": "him", "me": "him",
    "myself": "himself", "yourself": "himself",
    "we": "they", "us": "them",
}

# Filler words that end a short "Name ..." clause.
_CLAUSE_STOPS = {
    "like", "what", "um", "uh", "yeah", "yep", "okay", "ok", "so",
    "and", "but", "then", "bro", "man", "dude", "hey", "well", "hmm",
    "oh", "wow", "lol", "you", "know",
}

# Category-flavoured fallbacks for transcripts with no usable content.
# Deliberately never the generic "COMPLETELY SPEECHLESS" line.
_FALLBACK_LINES = {
    "Big Wins": "{name} just pulled off the biggest win live",
    "Funny Moments": "{name} could NOT stop laughing at what just happened",
    "Angry or Emotional Reactions": "{name} broke down live after what just happened",
    "Exciting Gameplay": "{name} just pulled off an insane play live",
    "Unexpected Moments": "{name} did NOT see that coming live",
    "Streamer Reactions": "{name} could NOT believe what just happened live",
    "Interesting Conversations": "{name} just said something nobody expected live",
    "Viral Potential": "{name} could NOT believe what just happened live",
}
_DEFAULT_FALLBACK = "{name} could NOT believe what just happened live"

# Moment-type fallbacks, used when visual moment detection
# (visual_moments.py) says what KIND of moment the clip is. A crash
# reads differently from a chase -- the headline names the moment type
# instead of falling back to a generic line. Only used when the
# transcript yields no specific verb + object headline.
_MOMENT_FALLBACK_LINES = {
    "chase": "{name} got chased down live and barely got away",
    "crash": "{name} just crashed in the worst way possible live",
    "fight": "{name} got into a fight live and it went wild",
    "stunt": "{name} just pulled off the craziest stunt live",
    "reaction": "{name} could NOT believe what just happened live",
    "funny": "{name} could NOT stop laughing at what just happened",
}


# Common verbs whose stem ends in -en: "happened" -> "happen", never
# "happene". Kept small and explicit — a purely orthographic rule
# cannot tell "happened" (stem "happen") from "deleted" (stem
# "delete"), so the frequent -en verbs are listed instead.
_EN_VERBS = frozenset({
    "happen", "open", "listen", "sharpen", "widen", "deepen",
    "fasten", "soften", "loosen", "stiffen", "frighten", "threaten",
    "reckon", "lengthen", "strengthen", "shorten", "hasten",
    "moisten", "sweeten", "whiten", "blacken", "redden", "ripen",
    "sicken", "weaken", "waken", "awaken", "lighten", "brighten",
    "tighten", "quieten",
})


def _ed_to_present(word):
    """Convert a regular -ed past tense to third-person present.

    Handles: tried -> tries, agreed -> agrees, danced -> dances,
    stopped -> stops, liked -> likes, kissed -> kisses,
    happened -> happens (not "happenes").
    """
    # Verbs ending in -en (happen, open, listen, ...): the -ed form
    # drops "ed", not just "d". Without this the vowel+consonant+"ed"
    # branch below misfires on them ("happened" -> "happenes").
    if word[:-2] in _EN_VERBS:
        stem = word[:-2]
    elif word.endswith("ied"):
        stem = word[:-3] + "y"
    elif word.endswith("eed"):
        stem = word[:-1]
    elif word.endswith(("ced", "ged")):
        stem = word[:-1]
    elif word.endswith("sed") and not word.endswith("ssed"):
        stem = word[:-1]
    elif (
        len(word) >= 5
        and word[-3] not in "aeiou"
        and word[-4] in "aeiou"
        and word[-5] not in "aeiou"
    ):
        # Single vowel + single consonant + "ed": liked -> like.
        stem = word[:-1]
    else:
        stem = word[:-2]
    # Doubled consonant: stopped -> stopp -> stop. Only for consonants
    # that double before -ed (p/b/g/m/n/r/t): stems that genuinely end
    # in a double consonant (kiss, pass, call, kill, add) must NOT be
    # stripped, or "kissed" becomes "kises" and "killed" "kils".
    if (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and stem[-1] in "pbgmnrt"
    ):
        stem = stem[:-1]
    if stem.endswith(("s", "x", "z", "ch", "sh")):
        return stem + "es"
    if stem.endswith("y") and len(stem) > 1 and stem[-2] not in "aeiou":
        return stem[:-1] + "ies"
    return stem + "s"


def _tokenize(sentence):
    """Return [(original, cleaned-lowercase), ...] for a sentence."""
    tokens = []
    for raw in sentence.split():
        cleaned = re.sub(r"^[^A-Za-z0-9*'’]+|[^A-Za-z0-9*'’]+$", "", raw)
        tokens.append((raw.strip(".,!?\"'()[]"), cleaned.lower()))
    return tokens


def _find_verb(tokens):
    """Find the first headline-worthy verb: (index, present-tense form)."""
    for i, (_, lower) in enumerate(tokens):
        if not lower or lower in _WEAK_VERBS:
            continue
        if lower in _IRREGULAR_PAST_TO_PRESENT:
            return i, _IRREGULAR_PAST_TO_PRESENT[lower]
        if lower in _PRESENT_VERBS:
            return i, lower
        if lower.endswith("ed") and len(lower) > 3:
            return i, _ed_to_present(lower)
    return None, None


def _extract_object(tokens, verb_idx):
    """Words after the verb, cleaned up as a headline object phrase."""
    words = []
    for original, lower in tokens[verb_idx + 1:]:
        if not lower:
            continue
        if lower in _OBJECT_STOPS:
            break
        if lower == "i":
            continue
        words.append(_PRONOUN_MAP.get(lower, original))
        if len(words) >= 6:
            break
    while words and words[-1].lower().strip(".,!?") in _OBJECT_TRIM_TAIL:
        words.pop()
    if not words:
        return ""
    has_specific = any(
        re.sub(r"[^a-z0-9]", "", w.lower()) not in _WEAK_OBJECTS
        and len(re.sub(r"[^a-z0-9]", "", w.lower())) > 1
        for w in words
    )
    if not has_specific:
        return ""
    return " ".join(words).strip(" .,!?") or ""


def _name_clause(text, name):
    """Short "Name ..." clause from the sentence that mentions the name.

    E.g. "Chaz is not clean like what the F*CK..." -> "Chaz is not clean".
    Returns "" when the name has no usable clause.
    """
    for sentence in _split_sentences(text):
        tokens = _tokenize(sentence)
        for i, (original, lower) in enumerate(tokens):
            if original != name:
                continue
            clause = [name]
            for _, nxt in tokens[i + 1:]:
                if not nxt or nxt in _CLAUSE_STOPS:
                    break
                clause.append(nxt)
                if len(clause) >= 4:
                    break
            if len(clause) >= 2:
                return " ".join(clause)
            return ""
    return ""


def _truncate(text, limit):
    """Truncate at a word boundary, adding an ellipsis when cut."""
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0]
    return (cut or text[:limit - 1]).rstrip(" .,!?") + "…"


def build_story_headline(streamer, transcript, words=None, category=None,
                         game=None, moment_type=None):
    """Write a SPECIFIC headline naming the moment in the transcript.

    Prefers a "{Streamer} {verb} {specific object/context}" sentence built
    from what was actually said (names, quoted phrases, distinctive
    nouns). Never returns the generic "COMPLETELY SPEECHLESS" line when
    the transcript has real content. Capped at ~90 characters.

    ``moment_type`` is the visual moment kind from
    visual_moments.py (chase / crash / fight / stunt / reaction /
    funny): when the transcript yields no specific headline, the
    fallback names the moment type instead of the generic line.
    """
    name = _display_name(streamer)
    text = _resolve_transcript(transcript, words)
    moment_line = _MOMENT_FALLBACK_LINES.get(
        (moment_type or "").strip().lower())

    def fallback():
        if moment_line:
            return moment_line.format(name=name)
        if game and len(str(game).strip()) <= 20:
            return f"{name} just had a wild {str(game).strip()} moment live"
        template = _FALLBACK_LINES.get(category or "", _DEFAULT_FALLBACK)
        return template.format(name=name)

    if not text:
        return fallback()

    # Usable content = at least 3 real words (not just filler).
    real_words = [
        w for w in re.findall(r"[A-Za-z0-9*']+", text.lower())
        if w not in _FILLER_WORDS
    ]
    if len(real_words) < 3:
        return fallback()

    sentences = _split_sentences(text)
    scored = []
    for idx, sentence in enumerate(sentences):
        score = _score_sentence(sentence)
        scored.append((score if score is not None else -1, idx, sentence))
    # Punchiest sentences first, transcript order breaks ties.
    scored.sort(key=lambda item: (-item[0], item[1]))

    headline = ""
    quote_headline = False
    for _, _, sentence in scored:
        tokens = _tokenize(sentence)
        verb_idx, verb = _find_verb(tokens)
        if verb_idx is None:
            continue
        obj = _extract_object(tokens, verb_idx)
        if not obj:
            continue
        headline = f"{name} {verb} {obj}"
        break

    if not headline:
        # No verb + object: use the best spoken line as the headline
        # (kicksclipping sometimes headlines with a quote instead).
        quote = pick_quote_line(text)
        if quote:
            headline = f'{name}: "{quote}"'
            quote_headline = True
        else:
            names = _extract_names(text, streamer)
            if names:
                return f"{name} just had a wild moment with {names[0]} live"
            return fallback()

    # Add a named person's clause when it fits and adds new information.
    # Verb headlines only — a quote headline is already a full quote,
    # and appending a second quote reads as two headlines at once.
    if not quote_headline:
        for other in _extract_names(text, streamer):
            if other.lower() in headline.lower():
                continue
            clause = _name_clause(text, other)
            if not clause:
                continue
            with_clause = f'{headline} — "{clause}"'
            if len(with_clause) <= MAX_HEADLINE_CHARS:
                headline = with_clause
            break

    return _truncate(headline, MAX_HEADLINE_CHARS)


# ---------------------------------------------------------------------------
# Post captions and hashtags (Feature 4)
# ---------------------------------------------------------------------------

GAME_HASHTAGS = {
    "gta 6": "gta6",
    "grand theft auto 6": "gta6",
    "grand theft auto vi": "gta6",
    "gta v": "gtav",
    "grand theft auto v": "gtav",
    "fortnite": "fortnite",
    "valorant": "valorant",
    "minecraft": "minecraft",
    "blackjack": "blackjack",
    "call of duty": "callofduty",
    "league of legends": "leagueoflegends",
    "rocket league": "rocketleague",
    "apex legends": "apexlegends",
    "counter-strike 2": "cs2",
    "cs2": "cs2",
    "ea fc 26": "eafc26",
    "fifa": "fifa",
    "poker": "poker",
    "slots": "slots",
    "just chatting": "justchatting",
    "irl": "irl",
}


def _slugify_tag(text):
    """Lowercase alphanumeric tag slug: "GTA 6" -> "gta6"."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def game_hashtag(game):
    """Hashtag slug for a game, e.g. "GTA 6" -> "gta6" ("" when unknown)."""
    if not game or not str(game).strip():
        return ""
    key = re.sub(r"\s+", " ", str(game).strip().lower())
    return GAME_HASHTAGS.get(key) or _slugify_tag(game)


def compose_hashtags(streamer, game=None):
    """The tag block for a post: "#<streamer> #fyp #kick" (+ game tag)."""
    tags = []
    slug = _slugify_tag(streamer)
    if slug:
        tags.append(f"#{slug}")
    tags.append("#fyp")
    tags.append("#kick")
    game_tag = game_hashtag(game)
    if game_tag:
        tags.append(f"#{game_tag}")
    return " ".join(tags)


def compose_post_caption(title, streamer, game=None):
    """Post caption: title + "#<streamer> #fyp #kick" (+ game tag)."""
    title = _clean_text(title)
    tags = compose_hashtags(streamer, game)
    return f"{title} {tags}" if title else tags
