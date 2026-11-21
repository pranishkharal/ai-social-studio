"""Tests for story_headlines.py -- story headlines (Feature 1), the quote
picker (Feature 2) and auto post captions with hashtags (Feature 4).

Uses the REAL transcript from checkpoints/clip_11.json (adinross, live
run 2026-11-19) plus synthetic win / funny / drama / empty transcripts.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from story_headlines import (  # noqa: E402
    build_story_headline,
    compose_hashtags,
    compose_post_caption,
    pick_quote_line,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real transcript: adinross clip 11 -- "Do it the fuck. Chaz is not clean
# like what the fuck am I already? Look at the good Jake. I think you
# found your calling..." (loaded from the checkpoint, not copied here).
CLIP_11_TRANSCRIPT = json.loads(
    (REPO_ROOT / "checkpoints" / "clip_11.json").read_text()
)["steps"]["transcribed"]["data"]["transcript"]

WIN_TRANSCRIPT = (
    "I just won the jackpot! Oh my god, we actually did it, "
    "fifty thousand dollars!"
)
FUNNY_TRANSCRIPT = (
    "Bro, he just slipped on a banana and fell straight into the pool. "
    "I cannot breathe, that was the funniest thing ever."
)
DRAMA_TRANSCRIPT = (
    "Zavala just smashed the guitar Amanda got him for his birthday. "
    "I cannot believe he actually did that on stream."
)


# ---------------------------------------------------------------------------
# Story headlines (Feature 1)
# ---------------------------------------------------------------------------

def test_headline_real_clip_11_names_the_moment():
    headline = build_story_headline("adinross", CLIP_11_TRANSCRIPT)
    assert "SPEECHLESS" not in headline
    # At least one transcript-specific word: a name or a distinctive noun.
    assert any(w in headline for w in ("Chaz", "Jake", "calling", "beer"))
    assert "Adinross" in headline
    assert len(headline) <= 90


def test_headline_win_moment():
    headline = build_story_headline("adinross", WIN_TRANSCRIPT)
    assert "SPEECHLESS" not in headline
    assert "jackpot" in headline.lower()
    assert "Adinross" in headline
    assert len(headline) <= 90


def test_headline_funny_moment():
    headline = build_story_headline("adinross", FUNNY_TRANSCRIPT)
    assert "SPEECHLESS" not in headline
    assert "banana" in headline.lower()
    assert len(headline) <= 90


def test_headline_drama_moment_names_people():
    headline = build_story_headline("zavala", DRAMA_TRANSCRIPT)
    assert "SPEECHLESS" not in headline
    assert "guitar" in headline.lower()
    assert "Amanda" in headline
    assert len(headline) <= 90


def test_headline_empty_transcript_avoids_generic_speechless():
    headline = build_story_headline("adinross", "")
    assert headline
    assert "SPEECHLESS" not in headline
    assert "Adinross" in headline
    assert build_story_headline("adinross", "   ") == headline


def test_headline_filler_only_transcript_falls_back():
    headline = build_story_headline("adinross", "Um, uh, like, you know, yeah.")
    assert "SPEECHLESS" not in headline
    assert "Adinross" in headline


def test_headline_category_flavoured_fallback():
    headline = build_story_headline("adinross", "", category="Big Wins")
    assert "SPEECHLESS" not in headline
    assert "win" in headline.lower()


def test_headline_uses_words_when_transcript_empty():
    words = [{"word": "I"}, {"word": "won"}, {"word": "the"},
             {"word": "jackpot!"}]
    headline = build_story_headline("adinross", "", words=words)
    assert "jackpot" in headline.lower()
    assert "SPEECHLESS" not in headline


def test_headline_long_transcript_stays_capped():
    long_transcript = (
        "He just smashed the enormous limited edition guitar that Amanda "
        "bought him for his birthday last year in front of everybody."
    ) * 3
    headline = build_story_headline("zavala", long_transcript)
    assert len(headline) <= 90


# ---------------------------------------------------------------------------
# Quote picker (Feature 2)
# ---------------------------------------------------------------------------

def test_quote_real_clip_11_is_a_real_sentence():
    quote = pick_quote_line(CLIP_11_TRANSCRIPT)
    assert quote
    assert quote in CLIP_11_TRANSCRIPT
    assert 3 <= len(quote.split()) <= 20
    assert len(quote) <= 120


def test_quote_win_moment_picks_punchiest_line():
    quote = pick_quote_line(WIN_TRANSCRIPT)
    assert "fifty thousand" in quote


def test_quote_empty_transcript_returns_empty():
    assert pick_quote_line("") == ""
    assert pick_quote_line("   ") == ""
    assert pick_quote_line(None) == ""


def test_quote_fragments_only_returns_empty():
    assert pick_quote_line("Yep. Um. Yeah.") == ""


def test_quote_uses_words_when_transcript_empty():
    words = [{"word": "Oh"}, {"word": "my"}, {"word": "god,"},
             {"word": "that"}, {"word": "was"}, {"word": "insane!"}]
    assert pick_quote_line("", words=words)


# ---------------------------------------------------------------------------
# Post captions and hashtags (Feature 4)
# ---------------------------------------------------------------------------

def test_caption_adds_streamer_fyp_kick_tags():
    caption = compose_post_caption("Adinross wins the jackpot", "adinross")
    assert caption == "Adinross wins the jackpot #adinross #fyp #kick"


def test_caption_adds_game_tag():
    caption = compose_post_caption("Adinross wins the jackpot", "adinross",
                                   game="GTA 6")
    assert caption.endswith("#gta6")
    assert "#adinross" in caption
    assert "#fyp" in caption
    assert "#kick" in caption


def test_caption_known_games():
    assert compose_post_caption("T", "xqc", game="Fortnite").endswith("#fortnite")
    assert compose_post_caption("T", "xqc", game="Valorant").endswith("#valorant")
    assert compose_post_caption("T", "xqc", game="Minecraft").endswith("#minecraft")
    assert compose_post_caption("T", "xqc", game="Blackjack").endswith("#blackjack")


def test_caption_unknown_game_slugified():
    caption = compose_post_caption("T", "xqc", game="Elden Ring")
    assert caption.endswith("#eldenring")


def test_caption_sanitizes_streamer_tag():
    caption = compose_post_caption("T", "AdinRoss")
    assert "#adinross" in caption


def test_compose_hashtags_block():
    assert compose_hashtags("adinross") == "#adinross #fyp #kick"
    assert compose_hashtags("adinross", game="GTA 6") == (
        "#adinross #fyp #kick #gta6")


# ---------------------------------------------------------------------------
# Heuristic integration (ai_story_engine wiring)
# ---------------------------------------------------------------------------

def test_heuristic_metadata_has_story_headline_and_quote():
    from ai_story_engine import AIStoryEngine

    engine = AIStoryEngine()
    sanitized = engine.prestige_sanitize(CLIP_11_TRANSCRIPT)
    meta = engine._heuristic_metadata("adinross", sanitized, 48.8)

    # New key, and the headline is specific, not generic SPEECHLESS.
    assert "quote_line" in meta
    assert meta["quote_line"]
    assert meta["quote_line"] in sanitized
    assert "SPEECHLESS" not in meta["viral_title"]
    assert any(w in meta["viral_title"]
               for w in ("Chaz", "Jake", "calling", "beer"))
    # Banner is the headline plus emoji.
    assert meta["hook_banner"].startswith(meta["viral_title"])
    assert meta["screen_caption"] == meta["hook_banner"]
    # Caption + hashtags come from the composer.
    assert meta["tiktok_caption"].startswith(meta["viral_title"])
    assert "#adinross" in meta["tiktok_caption"]
    assert "#fyp" in meta["tiktok_caption"]
    assert "#kick" in meta["tiktok_caption"]
    assert meta["hashtags"] == "#adinross #fyp #kick"
    # Existing category / score / reason logic is unchanged.
    assert meta["category"] == "Viral Potential"
    assert meta["ai_score"] == 65
    assert meta["llm_provider"] == "heuristic"


def test_heuristic_metadata_win_category_unchanged():
    from ai_story_engine import AIStoryEngine

    engine = AIStoryEngine()
    meta = engine._heuristic_metadata(
        "adinross", engine.prestige_sanitize(WIN_TRANSCRIPT), 80.0)
    assert meta["category"] == "Big Wins"
    assert meta["ai_score"] == 92
    assert "jackpot" in meta["viral_title"].lower()
    assert meta["quote_line"]


def test_empty_transcript_analysis_has_quote_line_key():
    from ai_story_engine import AIStoryEngine

    engine = AIStoryEngine()
    meta = engine.analyze_story_and_generate_metadata("adinross", "", 30)
    assert meta["quote_line"] == ""
    assert "SPEECHLESS" not in meta["viral_title"]
    assert meta["llm_provider"] == "heuristic"


def test_sanitize_metadata_keeps_short_quote_drops_long_quote():
    from ai_story_engine import AIStoryEngine

    engine = AIStoryEngine()
    base = {"viral_title": "T", "hook_banner": "H", "tiktok_caption": "C",
            "blue_comment": "B", "category": "Big Wins"}
    out = engine._sanitize_metadata({**base, "quote_line": "short quote here"})
    assert out["quote_line"] == "short quote here"
    out = engine._sanitize_metadata({**base, "quote_line": "x" * 121})
    assert out["quote_line"] == ""
    # Missing quote_line still yields the key (empty).
    out = engine._sanitize_metadata(dict(base))
    assert out["quote_line"] == ""


def test_analysis_prompt_requires_specific_headline_and_quote():
    from ai_story_engine import AIStoryEngine

    engine = AIStoryEngine()
    prompt = engine._build_analysis_prompt(
        "adinross", "Gaming", 30, 80.0, "some transcript")
    assert "quote_line" in prompt
    assert "specific moment" in prompt.lower()
    assert "SPEECHLESS" in prompt  # the rule names the banned generic line


def test_quote_headline_never_gets_second_quoted_clause():
    """A quote-style headline is already a full quote — appending a
    second quoted clause reads as two headlines at once (seen in the
    first demo run: 'Adinross: "..." — "Here you\'re not leaving"')."""
    from story_headlines import build_story_headline

    transcript = (
        "Hey, freezing cold. That's so we already know. What's here? "
        "Going for a Louis Vuitton. How many great songs is there? "
        "I try to show. Yeah. Yeah. I like that. Did you get my wrong? "
        "Here. You're. You're. Now leave me the fuck alone. Woo. Woo. "
        "Thank you. God damn. I've been so annoying, bro."
    )
    headline = build_story_headline("adinross", transcript)
    assert headline.count('"') <= 2
    assert " — \"" not in headline
