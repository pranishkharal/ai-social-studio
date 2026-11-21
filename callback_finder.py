"""
callback_finder.py — Find BEFORE/AFTER callback candidates in a transcript.

A callback is a setup early in a stream (a claim, a bet, a question, a rare
word, an action) and a later payoff where the same thing comes back, usually
with a loud / chat / heat spike. This module is a *finder*, not a judge: it
proposes ranked setup/payoff pairs with reasons, and the caller (the
pipeline, or Sara in the dashboard) decides which ones become clips.

The finder works on sentence-level transcript segments — the shape
``AIStoryEngine.transcribe_clip()`` produces can be turned into segments
with :func:`segments_from_words` (words are grouped into sentences on
``. ? !`` punctuation, falling back to long pauses when Whisper did not
punctuate). Each segment may carry ``loud`` (dB above the clip's normal
level), ``chat`` (chat-rate multiple) and ``heat`` (0-100) signals; the
finder also reads ``[LOUD +12dB]`` / ``[CHAT x4]`` / ``[HEAT 80]`` tags
embedded in the segment text, so tagged transcripts work as-is.

How a pair scores (all rules need a gap between ``min_gap`` and
``max_gap`` seconds, default 5 min to 2 h):
  * the same strong claim said again later ("I will never die" ... "I will
    never die") — even with no spike on the payoff line;
  * a claim + a shared rare word (or a "told you / called it" cue) + a
    spike on the payoff line;
  * a question answered later (2+ shared rare words + a spike);
  * the same fail action twice ("died" ... "died") with a spike;
  * a rare word returning at a peak (IDF-weighted, so "platypus" counts
    far more than "game");
  * a topic of the setup segment returning at a peak near the payoff.

Rare words are weighted by IDF (inverse document frequency) over the whole
transcript, and words are stemmed by truncation (``w[:7]``), so
"indonesian" and "indonesia" count as one topic. At most 2 pairs per setup
or payoff region are kept, so one big peak does not flood the list.

A candidate becomes a multi-part format spec with
:func:`candidate_to_callback_spec` (claim 3-6 s, payoff 8-15 s), which
``clip_formats.render_format`` renders.

Technique adapted from gta6-clipper (https://gitlab.com/nocodehack.io/
gta6-clipper, MIT licence), callbacks.py — reimplemented here for this
repo's transcript shapes; no code copied.
"""

import math
import re
from collections import Counter

# --- claim / payoff / fail patterns ---------------------------------------
# A "claim" is a promise, prediction or bet: something the stream can
# later prove right or wrong. STRONG claims are distinctive enough that
# hearing the same phrase again later means something on its own; the
# weaker ones ("easy", "i promise", "i bet") are said all the time, so
# they need a shared word or a spike too.
CLAIM = re.compile(
    r"\b(i will never|i'll never|never again|never gonna|i('m| am) never|"
    r"i guarantee|i promise|trust me|no way i|no chance|i bet|bet you|"
    r"watch (this|me|how)|easy|too easy|this time|"
    r"i('m| am) (gonna|going to) win|i can't lose|can't die|won't die|"
    r"you('ll| will) see|i('ll| will) prove|i called it|prediction|"
    r"mark my words)\b",
    re.I,
)
STRONG = re.compile(
    r"i will never|i'll never|never again|i guarantee|trust me|no way i|"
    r"this time|watch (this|me|how)|you('ll| will) see|mark my words|"
    r"i called it|prediction",
    re.I,
)
PAYOFF = re.compile(
    r"\b(told you|i said|remember (when|that)|called it|knew it|again|"
    r"here we go again|i lied|lied|wasted|busted|dead|died|die|fail|lost|"
    r"lose|never mind|nevermind)\b",
    re.I,
)
CALLBACK_CUE = re.compile(
    r"told you|i said|called it|knew it|i lied", re.I)
FAIL = re.compile(
    r"\b(died|dead|die|wasted|busted|arrested|failed|fail|crashed|crash|"
    r"fell|missed|lost|killed|exploded)\b",
    re.I,
)

# Words too common to link two moments (chat slang, fillers, pronouns).
STOP = set(
    """that this with have from they them were been what when your about
    just like yeah okay know going gonna there here then than into over
    some more very really want chat bro man dude shit fuck fucking oh
    don't didn't can't it's i'm that's you're what's let's gotta wanna
    thing things right well come look need think because would could
    should still back only even also much many make made good dont cant
    its youre thats whats lets people""".split()
)

_TAG_LOUD = re.compile(r"\[LOUD \+(\d+(?:\.\d+)?)dB\]", re.I)
_TAG_CHAT = re.compile(r"\[CHAT x(\d+(?:\.\d+)?)", re.I)
_TAG_HEAT = re.compile(r"\[HEAT (\d+(?:\.\d+)?)\]", re.I)
_TAG_ANY = re.compile(r"\s*\[[A-Z]+[^\]]*\]")

DEFAULT_MIN_GAP = 300.0    # 5 min: a callback needs real distance
DEFAULT_MAX_GAP = 7200.0   # 2 h
DEFAULT_TOP = 25


def _tag_value(pattern, text):
    """First float captured by ``pattern`` in ``text``, or 0.0."""
    m = pattern.search(text)
    if not m:
        return 0.0
    try:
        return float(m.group(1))
    except ValueError:
        return 0.0


def parse_segments(segments):
    """
    Normalise transcript segments into finder lines.

    Args:
        segments: list of dicts with "start", "end" and "text" (or
            "full_text"). Optional "loud" / "chat" / "heat" keys, or
            the same signals as ``[LOUD +12dB]``-style tags in the text.

    Returns:
        List of dicts: {"i", "s", "e", "txt", "plain", "loud", "chat",
        "heat"}. "plain" is the text with tags stripped.
    """
    lines = []
    for seg in segments or []:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
        except (TypeError, ValueError):
            continue
        text = str(seg.get("text") or seg.get("full_text") or "").strip()
        if not text:
            continue
        loud = seg.get("loud")
        chat = seg.get("chat")
        heat = seg.get("heat")
        lines.append({
            "i": len(lines),
            "s": start,
            "e": end,
            "txt": text,
            "loud": float(loud) if loud is not None else _tag_value(_TAG_LOUD, text),
            "chat": float(chat) if chat is not None else _tag_value(_TAG_CHAT, text),
            "heat": float(heat) if heat is not None else _tag_value(_TAG_HEAT, text),
            "plain": _TAG_ANY.sub("", text).strip(),
        })
    return lines


def segments_from_words(words, max_pause=0.8):
    """
    Group Whisper word dicts ({"word", "start", "end"}) into sentence
    segments for the callback finder.

    A sentence ends on ``. ? !`` punctuation, or on a pause longer than
    ``max_pause`` seconds when Whisper did not punctuate (Whisper tiny
    often does not). Words without timings are skipped.
    """
    segments = []
    current = []
    prev_end = None
    for w in words or []:
        try:
            start = float(w["start"])
            end = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        word = str(w.get("word", "")).strip()
        if not word:
            continue
        if current and prev_end is not None and start - prev_end > max_pause:
            segments.append(_words_to_segment(current))
            current = []
        current.append({"word": word, "start": start, "end": end})
        prev_end = end
        if re.search(r"[.?!]$", word):
            segments.append(_words_to_segment(current))
            current = []
    if current:
        segments.append(_words_to_segment(current))
    return segments


def _words_to_segment(words):
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": " ".join(w["word"] for w in words),
    }


def _topic_words(text):
    """Lowercased 4+ letter words, stop-words removed, stemmed to 7 chars."""
    out = []
    for w in re.findall(r"[a-z']{4,}", text.lower()):
        if w in STOP or w.strip("'") in STOP:
            continue
        out.append(w[:7])
    return out


def _spike(line):
    """Combined spike strength: 0 = nothing tagged. A +21 dB line with
    chat x5 scores ~3.8."""
    return line["loud"] / 10.0 + line["chat"] / 3.0 + line["heat"] / 40.0


def find_callbacks(lines, min_gap=DEFAULT_MIN_GAP, max_gap=DEFAULT_MAX_GAP,
                   top=DEFAULT_TOP):
    """
    Rank setup/payoff callback candidates in parsed transcript lines.

    Args:
        lines: output of :func:`parse_segments` (or
            ``parse_segments(segments_from_words(words))``).
        min_gap / max_gap: payoff must start this many seconds after
            the setup ends (default 300 s / 7200 s).
        top: maximum candidates returned.

    Returns:
        List of dicts, best first: {"score", "setup": (start, end,
        text), "payoff": (start, end, text), "gap_min", "why", "tags"}.
    """
    lines = list(lines or [])
    if not lines:
        return []

    # Document frequency per topic word, for IDF weighting.
    df = Counter(w for line in lines for w in set(_topic_words(line["plain"])))
    n_lines = len(lines)

    def idf(word):
        return math.log(n_lines / df[word]) if df[word] else 0.0

    # Biggest spike within 30 s of each line: a topic can come back just
    # before or after the peak.
    near = []
    for line in lines:
        nearby = [x for x in lines if abs(x["s"] - line["s"]) <= 30]
        near.append(max((_spike(x) for x in nearby), default=0.0))

    def local_count(line, word):
        """Times ``word`` is said within 2 min of ``line`` — 2+ means it
        is a topic of that segment, not a passing word."""
        return sum(
            word in _topic_words(x["plain"])
            for x in lines if abs(x["s"] - line["s"]) <= 120
        )

    candidates = {}

    def add(setup, payoff, score, why):
        key = (setup["i"], payoff["i"])
        if key not in candidates or candidates[key][0] < score:
            candidates[key] = (score, why)
        elif why not in candidates[key][1]:
            candidates[key] = (candidates[key][0] + 0.5,
                               candidates[key][1] + " + " + why)

    for setup in lines:
        setup_words = {w for w in _topic_words(setup["plain"]) if df[w] <= 8}
        claim_match = CLAIM.search(setup["plain"])
        is_claim = bool(claim_match)
        is_question = setup["plain"].rstrip().endswith("?")
        fail_setup = FAIL.search(setup["plain"])
        for payoff in lines:
            gap = payoff["s"] - setup["e"]
            if gap < min_gap or gap > max_gap:
                continue
            spike = _spike(payoff)
            cue_match = PAYOFF.search(payoff["plain"])
            has_cue = bool(cue_match)
            shared = sorted(
                setup_words & set(_topic_words(payoff["plain"])),
                key=lambda w: -idf(w),
            )
            shared_weight = sum(idf(w) for w in shared[:3])

            strong = STRONG.search(setup["plain"]) if is_claim else None
            same_claim_again = bool(
                strong and re.search(re.escape(strong.group(0)),
                                     payoff["plain"], re.I)
            )
            if same_claim_again and spike == 0:
                add(setup, payoff, 2.5 + shared_weight + has_cue,
                    f"same claim again ('{strong.group(0)}'), "
                    "no tag on the payoff line")
            # A bare "again" after any claim is noise: a claim needs a
            # shared word or a real call-back cue.
            elif is_claim and (shared or (has_cue and CALLBACK_CUE.search(payoff["plain"]))) and spike > 0:
                cue_text = (f"cue {cue_match.group(0)} "
                            if has_cue else "")
                add(setup, payoff,
                    3 + shared_weight + spike + has_cue * 1.5,
                    f"claim ('{claim_match.group(0)}') then "
                    f"{cue_text}{shared[:3]}")
            elif is_question and len(shared) >= 2 and spike > 0:
                add(setup, payoff, 1.5 + shared_weight + spike,
                    f"question answered later, shared {shared[:3]}")
            elif (fail_setup and FAIL.search(payoff["plain"]) and spike >= 2
                    and (has_cue or shared)):
                add(setup, payoff,
                    1.5 + spike + shared_weight * 0.5 + has_cue,
                    f"same action again ({fail_setup.group(0)} / "
                    f"{FAIL.search(payoff['plain']).group(0)})")
            elif shared and shared_weight >= 6 and spike >= 2 and df[shared[0]] <= 4:
                add(setup, payoff, shared_weight * 0.4 + spike,
                    f"rare word returns {shared[:3]}")
            elif (shared and df[shared[0]] <= 6 and len(shared[0]) >= 7
                    and near[payoff["i"]] >= 2
                    and local_count(setup, shared[0]) >= 2):
                add(setup, payoff,
                    idf(shared[0]) * 0.6 + near[payoff["i"]],
                    f"topic returns at a peak {shared[:2]}")

    ranked = sorted(candidates.items(), key=lambda kv: -kv[1][0])
    out = []
    seen_setup_starts = []
    seen_payoff_starts = []
    for (i_setup, i_payoff), (score, why) in ranked:
        setup, payoff = lines[i_setup], lines[i_payoff]
        # At most 2 pairs per setup or payoff region (within 3 min):
        # no wall of the same pair. One big peak may still pay off
        # several different setups.
        if (sum(abs(setup["s"] - x) < 180 for x in seen_setup_starts) >= 2
                or sum(abs(payoff["s"] - y) < 180 for y in seen_payoff_starts) >= 2):
            continue
        seen_setup_starts.append(setup["s"])
        seen_payoff_starts.append(payoff["s"])
        tags = ("payoff "
                + (f"LOUD+{payoff['loud']:.0f} " if payoff["loud"] else "")
                + (f"CHAT x{payoff['chat']:g}" if payoff["chat"] else "")).strip()
        out.append({
            "score": round(score, 1),
            "setup": (setup["s"], setup["e"], setup["plain"]),
            "payoff": (payoff["s"], payoff["e"], payoff["plain"]),
            "gap_min": round((payoff["s"] - setup["e"]) / 60.0, 1),
            "why": why,
            "tags": tags,
        })
        if len(out) >= top:
            break
    return out


def candidate_to_callback_spec(candidate, hook=""):
    """
    Turn a :func:`find_callbacks` candidate into a ``clip_formats``
    callback spec: the claim as a 3-6 s part and the payoff as an
    8-15 s part. The claim window starts at the setup line (plus a
    0.5 s lead-in, never before 0); the payoff window starts 2 s
    before the payoff line so the reaction is not cut off.
    """
    setup_start, setup_end, _ = candidate["setup"]
    payoff_start, payoff_end, _ = candidate["payoff"]
    claim_start = max(0.0, setup_start - 0.5)
    claim_end = min(setup_end + 1.0, claim_start + 6.0)
    if claim_end - claim_start < 3.0:
        claim_end = min(claim_start + 3.0, claim_end)
    payoff_start = max(0.0, payoff_start - 2.0)
    payoff_end = min(payoff_end + 2.0, payoff_start + 15.0)
    if payoff_end - payoff_start < 8.0:
        payoff_end = payoff_start + 8.0
    return {
        "format": "callback",
        "hook": hook,
        "parts": [
            {"start": round(claim_start, 2), "end": round(claim_end, 2),
             "label": "THE CLAIM"},
            {"start": round(payoff_start, 2), "end": round(payoff_end, 2),
             "label": "THE PAYOFF"},
        ],
        "callback": {
            "score": candidate["score"],
            "gap_min": candidate["gap_min"],
            "why": candidate["why"],
        },
    }
