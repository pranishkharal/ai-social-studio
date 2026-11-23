"""Tests for the guide-driven post captions (captions.py).

Post captions are the text under the video on TikTok — NOT on-screen
text. These tests cover the rule-based templates, entity extraction,
the guide file (read + fallback), the LLM fallback chain, and the
pipeline storing the captions on the clip record.
"""

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import captions as captions_module  # noqa: E402
import pipeline as pipeline_module  # noqa: E402
from captions import (  # noqa: E402
    detect_moment_kind,
    detect_reaction,
    extract_entities,
    generate_captions,
    generate_captions_rule_based,
    generate_captions_with_llm,
    load_caption_guide,
    parse_guide_rules,
    parse_guide_templates,
)
from pipeline import ClipPipeline  # noqa: E402
from story_headlines import compose_post_caption  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# The strawberry example from Sara's Claude screenshot: the guide's
# viral moment template gives FULL context (streamer + game + the
# specific thing + the number).
STRAWBERRY_TRANSCRIPT = (
    "A character rants about Leonida strawberries and a Japanese "
    "farmer selling them for $500 a pop. No way, what the fuck, "
    "that is insane."
)
STRAWBERRY_CONTEXT = {
    "streamer": "adinross",
    "game": "GTA 6",
    "transcript": STRAWBERRY_TRANSCRIPT,
    "headline": "Adinross reacts to the strawberry farmer",
    "moment_type": "reaction",
}

# The real adinross transcript from the 2026-11-19 live run (clip #11).
CLIP_11_TRANSCRIPT = json.loads(
    (REPO_ROOT / "checkpoints" / "clip_11.json").read_text()
)["steps"]["transcribed"]["data"]["transcript"]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def test_extract_money_amounts():
    entities = extract_entities("He lost $50,000 on one hand, then won $500 back.")
    assert entities["money"] == "$50,000"
    assert extract_entities("No money here at all.")["money"] == ""


def test_extract_names_and_game_terms():
    entities = extract_entities(STRAWBERRY_TRANSCRIPT)
    assert entities["game_terms"] == ["leonida", "strawberry", "farmer"]
    # "strawberries" (plural) still matches the "strawberry" term.
    assert "strawberry" in entities["game_terms"]

    entities_11 = extract_entities(CLIP_11_TRANSCRIPT)
    assert "Chaz" in entities_11["names"]
    assert "Jake" in entities_11["names"]
    assert entities_11["money"] == ""


def test_detect_reaction_only_from_transcript():
    assert detect_reaction("He was speechless, no words at all.") == "SPEECHLESS"
    assert detect_reaction("No way, I can't believe that just happened.") == "SHOCKED"
    assert detect_reaction("He started screaming at the screen.") == "SCREAMING"
    assert detect_reaction("Haha that was so funny, I am laughing.") == "LAUGHING"
    # No reaction words in the transcript: no reaction is invented.
    assert detect_reaction("Yeah so we drive over there and then we wait.") is None
    assert detect_reaction("") is None


# ---------------------------------------------------------------------------
# Rule-based templates
# ---------------------------------------------------------------------------

def test_full_context_best_pick_matches_claude_shape():
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="",
                               use_llm=False)
    assert result["provider"] == "rule_based"
    # Same shape as Claude's best pick: streamer + reaction + game +
    # the specific thing + the number, key noun capitalized — now
    # with Claude's emojis (topic + heart).
    assert result["best"] == (
        "Adinross was SHOCKED after GTA 6 revealed $500 STRAWBERRY farmer 🍓💖")
    assert result["reasons"]["best"]
    assert len(result["alternatives"]) == 4
    assert len(result["reasons"]["alternatives"]) == 4


def test_mystery_variant_drops_the_key_detail():
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="",
                               use_llm=False)
    mystery = result["alternatives"][0]
    assert mystery.startswith("the moment Adinross realized")
    assert "$500" not in mystery  # the number is the dropped detail
    assert "Leonida" in mystery  # the topic stays, so it still hooks


def test_proved_template_uses_moment_word():
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="",
                               use_llm=False)
    proved = [a for a in result["alternatives"] if "PROVED" in a]
    assert len(proved) == 1
    # "reaction" is not a thing to point at — it reads as "moment".
    assert "after this moment" in proved[0]


def test_no_fake_reaction_without_reaction_signals():
    flat = {
        "streamer": "xqc", "game": "GTA 6",
        "transcript": "Yeah so we drive over there and then we wait.",
        "headline": "",
    }
    result = generate_captions(flat, guide_text="", use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"])
    for fake in ("SPEECHLESS", "SHOCKED", "SCREAMING", "LAUGHING"):
        assert fake not in all_text


def test_moment_type_template():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Yeah so we drive over there and then we wait.",
        "headline": "", "moment_type": "crash",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert any("CRASHED" in a for a in
               [result["best"]] + result["alternatives"])


def test_ocr_template_names_the_game_event():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Yeah so we drive over there and then we wait.",
        "headline": "", "ocr_hits": [{"label": "WASTED"}],
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert any("WASTED" in a for a in
               [result["best"]] + result["alternatives"])
    # An unknown OCR label never becomes a caption.
    context["ocr_hits"] = [{"label": "SOMETHING WEIRD"}]
    result = generate_captions(context, guide_text="", use_llm=False)
    assert "SOMETHING WEIRD" not in " ".join(
        [result["best"]] + result["alternatives"])


def test_captions_never_empty():
    result = generate_captions({"streamer": "adinross"}, guide_text="",
                               use_llm=False)
    assert result["best"]
    assert len(result["alternatives"]) == 4
    assert len(set([result["best"]] + result["alternatives"])) == 5


def test_fragment_clause_is_rejected():
    # Demo regression (clip #30): the headline's quoted clause was a
    # Whisper fragment ("Sure not leaving the") and the best pick came
    # out as the broken "Adinross had NO IDEA Sure not leaving the".
    # A fragment clause must never become a caption — and the headline
    # itself is dropped as an alternative when it carries one.
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Sure, not leaving the fuck alone. "
                      "So when can I get missions?",
        "headline": 'Adinross knows him had a soft spot — '
                    '"Sure not leaving the"',
        "moment_type": "stunt",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"])
    assert "Sure not leaving the" not in all_text
    assert result["best"] == (
        "Adinross had NO IDEA GTA 6 had the mission 🎯💖")


def test_mystery_topic_uses_the_for_common_nouns():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "So when can I get missions?",
        "headline": "",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    mystery = [a for a in [result["best"]] + result["alternatives"]
               if a.startswith("the moment")]
    assert mystery
    assert "about the mission" in mystery[0]
    assert "about Mission" not in mystery[0]


def test_weak_thing_is_the_best_pick():
    # The guide's viral-moment rule: FULL context whenever a specific
    # thing exists, even a weak one — "the mission" still names the
    # moment, and a generic hook only wins when nothing specific
    # exists at all. The mystery variant is an alternative now.
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "So when can I get missions?",
        "headline": "",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert result["best"] == (
        "Adinross had NO IDEA GTA 6 had the mission 🎯💖")
    assert any(a.startswith("the moment Adinross realized")
               for a in result["alternatives"])


def test_sentence_starters_are_not_names():
    entities = extract_entities(
        "Sure, not leaving the fuck alone. Maybe I'll make a picture. "
        "Ethan wants that to be weird.")
    assert entities["names"] == ["Ethan"]


def test_bare_name_is_not_a_full_context_thing():
    # "had NO IDEA GTA 6 had Ethan" is nonsense, and "revealed Ethan"
    # claims the person is in the game — a bare name never becomes
    # the full-context thing.
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Ethan wants that to be weird.",
        "headline": "",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"])
    assert "had Ethan" not in all_text
    assert "revealed Ethan" not in all_text


def test_caps_off_when_guide_asks():
    guide = "Write plain captions. No caps anywhere, keep it calm."
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text=guide,
                               use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"])
    assert "SPEECHLESS" not in all_text
    assert "SHOCKED" not in all_text
    assert "PROVED" not in all_text


# ---------------------------------------------------------------------------
# On-screen text (game dialogue + objectives read off the frames)
# ---------------------------------------------------------------------------

# The real on-screen text from the 2026-11-21 live demo 2 window
# (65-80 s of the Adin Ross GTA 6 Day 4 capture), as RapidOCR reads it.
LIVE2_ON_SCREEN = (
    "Brittney and the new guys, you're with me. Rest of you, get to "
    "the spot! One of you better have something I can smoke. "
    "Go to the crash site.")


def test_on_screen_text_feeds_entities():
    entities = extract_entities("Oh shit.", on_screen_text=LIVE2_ON_SCREEN)
    assert "crash site" in entities["game_terms"]
    assert "Brittney" in entities["names"]
    # The bare word loses to the longer phrase at the same position.
    assert entities["game_terms"][0] == "crash site"


def test_on_screen_text_makes_specific_best_pick():
    # The live demo 2 case: the transcript is just "Oh shit." but the
    # frames say the moment — the best pick names the crash site,
    # and the mystery variant (which drops it) is only an alternative.
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Oh shit.",
        "headline": "Adinross could NOT believe what just happened live",
        "moment_type": "chase",
        "on_screen_text": LIVE2_ON_SCREEN,
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert result["best"] == (
        "Adinross had NO IDEA GTA 6 had the crash site 💥💖")
    mystery = [a for a in result["alternatives"]
               if a.startswith("the moment Adinross realized")]
    assert mystery and "about the crash site" in mystery[0]
    assert result["on_screen_text"] == LIVE2_ON_SCREEN


def test_on_screen_money_sentence_is_the_thing():
    # A dollar amount on screen counts like one in the transcript:
    # the money sentence names the thing being sold.
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Oh shit.",
        "headline": "",
        "on_screen_text": ("Japanese farmer selling strawberries "
                           "for $500 a pop."),
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert result["best"] == (
        "Adinross had NO IDEA GTA 6 had $500 STRAWBERRY farmer 🍓💖")


def test_on_screen_text_does_not_invent_a_reaction():
    # Game dialogue shouting "oh my god" is not the streamer
    # reacting — the reaction comes from the transcript only.
    context = {
        "streamer": "xqc", "game": "GTA 6",
        "transcript": "Yeah so we drive over there and then we wait.",
        "headline": "",
        "on_screen_text": "Oh my god, what the hell is happening!",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"])
    assert "SHOCKED" not in all_text
    assert "SPEECHLESS" not in all_text


def test_empty_on_screen_text_falls_back_to_transcript_only():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Oh shit.", "headline": "",
        "on_screen_text": "",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    # No specific thing anywhere: a generic hook is the best pick.
    assert result["best"].startswith("Adinross did NOT see that coming")
    assert result["on_screen_text"] == ""


def test_mystery_is_alternative_not_best():
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="",
                               use_llm=False)
    assert result["best"].startswith("Adinross was SHOCKED after GTA 6")
    assert any(a.startswith("the moment Adinross realized")
               for a in result["alternatives"])


# ---------------------------------------------------------------------------
# Emojis
# ---------------------------------------------------------------------------

def test_emojis_are_present_and_topic_relevant():
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="",
                               use_llm=False)
    # Every caption ends with a topic emoji + a heart, Claude's pattern.
    for caption in [result["best"]] + result["alternatives"]:
        assert caption.endswith("💖")
    assert result["best"].endswith("🍓💖")  # strawberry moment
    crash = generate_captions({
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Oh shit.", "headline": "",
        "moment_type": "crash",
    }, guide_text="", use_llm=False)
    assert any(a.endswith("💥💖") for a in
               [crash["best"]] + crash["alternatives"])
    # No topic at all: the default flame, never a random emoji.
    flat = generate_captions({
        "streamer": "xqc", "game": "GTA 6",
        "transcript": "Yeah so we drive over there and then we wait.",
        "headline": "",
    }, guide_text="", use_llm=False)
    assert flat["best"].endswith("🔥💖")


def test_emoji_word_boundaries():
    # "car" must not match "card", and word forms ("crashed",
    # "chased", "laughing") still match their topic.
    chased = generate_captions({
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Oh shit.", "headline": "",
        "moment_type": "chase",
    }, guide_text="", use_llm=False)
    assert any(a.endswith("🚗💖") for a in
               [chased["best"]] + chased["alternatives"])


# ---------------------------------------------------------------------------
# Guide file
# ---------------------------------------------------------------------------

def test_load_guide_missing_file_falls_back_to_defaults(tmp_path):
    assert load_caption_guide(tmp_path / "no-such-guide.md") == ""
    rules = parse_guide_rules("")
    assert rules["full_context"] is True
    assert rules["mystery"] is True
    assert rules["caps_words"] is True


def test_load_guide_reads_file(tmp_path):
    guide = tmp_path / "caption_guide.md"
    guide.write_text("Viral moments get FULL context. Mystery pulls comments.\n")
    assert "FULL context" in load_caption_guide(guide)


def test_full_context_best_pick_regardless_of_guide_wording():
    # Even a guide that never mentions full context still gets the
    # full-context best pick when a specific thing exists: the
    # viral-moment rule is the default, and the mystery variant is an
    # alternative, not the best pick.
    guide = "Keep captions short and mysterious. Drop the key detail."
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text=guide,
                               use_llm=False)
    assert result["best"].startswith(
        "Adinross was SHOCKED after GTA 6 revealed $500 STRAWBERRY farmer")
    assert any(a.startswith("the moment Adinross realized")
               for a in result["alternatives"])


def test_real_guide_file_is_read():
    # Sara's actual guide file is present in the repo (git-ignored,
    # local only) and the generator reads it by default.
    text = load_caption_guide()
    assert "Watch Time = Views" in text
    rules = parse_guide_rules(text)
    assert rules["full_context"] is True
    assert rules["mystery"] is True


# ---------------------------------------------------------------------------
# Moment-kind detection
# ---------------------------------------------------------------------------

def test_detect_moment_kind_apology():
    assert detect_moment_kind(
        "I'm so sorry chat, I apologize, that was my fault.", "") == "apology"
    assert detect_moment_kind("My bad, that was on me.", "") == "apology"


def test_detect_moment_kind_song_reaction():
    assert detect_moment_kind(
        "Yo that new song is insane, the rapper dropped a grammy "
        "winning track.", "") == "song_reaction"


def test_detect_moment_kind_emotional():
    assert detect_moment_kind(
        "I just broke down crying, I'm so emotional right now.",
        "") == "emotional"


def test_detect_moment_kind_career_moment():
    assert detect_moment_kind(
        "We did it chat, I made it, dream come true.",
        "") == "career_moment"
    assert detect_moment_kind(
        "I'm officially a tier 1 streamer now.", "") == "career_moment"


def test_detect_moment_kind_meets_famous():
    transcript = "Bro I just met Drake at the mall, I had no idea who he was."
    assert detect_moment_kind(transcript, "") == "meets_famous"
    # With pre-extracted entities the same check applies.
    entities = extract_entities(transcript)
    assert detect_moment_kind(transcript, "", entities) == "meets_famous"
    # A meeting phrase with no name is not a famous meeting.
    assert detect_moment_kind("I just met someone at the mall.",
                              "") is None


def test_detect_moment_kind_priority():
    # Apology outranks song and emotional when several fire at once.
    assert detect_moment_kind(
        "I'm sorry, that song made me cry.", "") == "apology"
    # Song outranks emotional.
    assert detect_moment_kind(
        "That song made me cry, I'm so emotional.",
        "") == "song_reaction"


@pytest.mark.parametrize("transcript,headline", [
    (STRAWBERRY_TRANSCRIPT, "Adinross reacts to the strawberry farmer"),
    (CLIP_11_TRANSCRIPT, ""),
    ("Yeah so we drive over there and then we wait.", ""),
    ("Oh shit.", "Adinross could NOT believe what just happened live"),
    ("So when can I get missions?", ""),
    ("Ethan wants that to be weird.", ""),
])
def test_detect_moment_kind_none_for_existing_contexts(transcript, headline):
    assert detect_moment_kind(transcript, headline) is None


def test_moment_kind_ignores_on_screen_text():
    # The game apologizing on screen is not the streamer apologizing:
    # signals in the on-screen text must never pick a template.
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Oh shit.", "headline": "",
        "on_screen_text": "Sorry about the crash, I was sad about the song.",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert "sorry" not in result["best"].lower()
    assert result["best"].startswith("Adinross had NO IDEA GTA 6 had the crash")


# ---------------------------------------------------------------------------
# Guide template parsing
# ---------------------------------------------------------------------------

def test_parse_real_guide_templates():
    templates = parse_guide_templates(load_caption_guide())
    assert sorted(templates) == [
        "apology", "career_moment", "emotional", "meets_famous",
        "song_reaction"]
    # The viral-moment section is an example sentence, not a template.
    assert "viral_moment" not in templates
    assert len(templates["emotional"]) == 3
    assert len(templates["career_moment"]) == 2
    # Nested quotes survive: only the outermost pair is stripped, so
    # the first career template still carries an inner quoted phrase.
    assert re.search(r'"[^"]+"', templates["career_moment"][0])
    assert "(streamer)" in templates["apology"][0]


def test_parse_example_guide_templates_match_builtins():
    example = (REPO_ROOT / "caption_guide.example.md").read_text()
    parsed = parse_guide_templates(example)
    assert parsed == captions_module.DEFAULT_TEMPLATES


def test_parse_templates_empty_guide():
    assert parse_guide_templates("") == {}
    assert parse_guide_templates(None) == {}


def test_missing_guide_uses_generic_templates():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "I'm so sorry chat, I apologize.",
        "headline": "",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert result["best"] == (
        "the moment Adinross owned up and said sorry ❤️‍🩹")


# ---------------------------------------------------------------------------
# Template rendering (real guide)
# ---------------------------------------------------------------------------

def _real_guide_context(transcript, headline="", moment_type="reaction"):
    return {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": transcript, "headline": headline,
        "moment_type": moment_type,
    }


def _guide_template(kind, index=0, streamer="Adinross", quick_context=""):
    """Render one template from the real guide file, the way the
    generator does — the test never hardcodes the guide's wording
    (the guide is local-only and must not leak into the repo)."""
    templates = parse_guide_templates(load_caption_guide())
    text = templates[kind][index]
    text = text.replace("(streamer)", streamer)
    if "(quick context)" in text:
        text = text.replace("(quick context)", quick_context)
    return text


def test_apology_template_is_best_pick():
    result = generate_captions(
        _real_guide_context(
            "I'm so sorry chat, I apologize, that was my fault."),
        use_llm=False)
    assert result["best"] == _guide_template("apology")
    assert result["best"].endswith("❤️‍🩹")
    assert not result["best"].endswith("💖")


def test_song_template_is_best_pick():
    result = generate_captions(
        _real_guide_context(
            "Yo that new song is insane, the rapper dropped a grammy "
            "winning track."),
        use_llm=False)
    assert result["best"] == _guide_template("song_reaction")


def test_emotional_template_is_best_pick():
    result = generate_captions(
        _real_guide_context(
            "I just broke down crying after the crash.",
            moment_type="crash"),
        use_llm=False)
    assert result["best"] == _guide_template("emotional", 0)
    # The (quick context) variant fills in the specific thing.
    assert any(a == _guide_template("emotional", 1,
                                   quick_context="the crash")
               for a in result["alternatives"])


def test_career_template_keeps_nested_quotes():
    result = generate_captions(
        _real_guide_context("We did it chat, I made it, dream come true."),
        use_llm=False)
    assert result["best"] == _guide_template("career_moment", 0)
    # The inner quoted phrase survives rendering.
    assert re.search(r'"[^"]+"', result["best"])
    assert result["best"].endswith("❤️‍🩹")
    assert any(a == _guide_template("career_moment", 1)
               for a in result["alternatives"])


def test_meets_famous_hides_the_name():
    result = generate_captions(
        _real_guide_context(
            "Bro I just met Drake at the mall, I had no idea who he was.",
            headline="Adinross meets a stranger on stream"),
        use_llm=False)
    assert result["best"] == _guide_template("meets_famous")
    # The name stays out of every caption, not just the best pick:
    # the comments are supposed to argue about it.
    all_text = " ".join([result["best"]] + result["alternatives"])
    assert "Drake" not in all_text


# ---------------------------------------------------------------------------
# Risky words (clipping guide)
# ---------------------------------------------------------------------------

def test_fight_moment_produces_no_fight_caption():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "He got into a fight with the cops.",
        "headline": "", "moment_type": "fight",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"]).lower()
    assert "fight" not in all_text
    assert result["best"]
    assert len(result["alternatives"]) == 4


def test_profanity_never_reaches_a_caption():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "What the fuck, that was insane.",
        "headline": 'Adinross reacts — "what the fuck just happened"',
        "moment_type": "reaction",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"]).lower()
    assert "fuck" not in all_text
    assert "f*ck" not in all_text
    assert result["best"]
    assert len(result["alternatives"]) == 4


def test_risky_word_match_is_whole_word():
    # "mean" is risky, but "meaningful" is not the word "mean" — a
    # headline carrying it survives as a caption.
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "Yeah so we drive over there and then we wait.",
        "headline": "Adinross had a meaningful moment live",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert any("meaningful" in a for a in
               [result["best"]] + result["alternatives"])
    context["headline"] = "Adinross was mean to the chat"
    result = generate_captions(context, guide_text="", use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"]).lower()
    assert "was mean to the chat" not in all_text


# ---------------------------------------------------------------------------
# Trigger words
# ---------------------------------------------------------------------------

def test_trigger_words_parsed_from_real_guide():
    rules = parse_guide_rules(load_caption_guide())
    assert rules["trigger_words"] == [
        "PROVED", "SPEECHLESS", "REGRETTED", "REVEALED"]
    assert parse_guide_rules("")["trigger_words"] == [
        "PROVED", "SPEECHLESS", "REGRETTED", "REVEALED"]


def test_regretted_candidate_and_no_caps():
    context = {
        "streamer": "adinross", "game": "GTA 6",
        "transcript": "I regret doing that, I shouldn't have said it.",
        "headline": "",
    }
    result = generate_captions(context, guide_text="", use_llm=False)
    assert "REGRETTED" in result["best"]
    # With a no-caps guide the same word is lowercased.
    result = generate_captions(context, guide_text="No caps anywhere.",
                               use_llm=False)
    all_text = " ".join([result["best"]] + result["alternatives"])
    assert "REGRETTED" not in all_text
    assert "regretted" in result["best"]


# ---------------------------------------------------------------------------
# LLM path — fallback behaviour (no real key in tests)
# ---------------------------------------------------------------------------

def test_llm_fn_returning_valid_json_is_used():
    def fake_llm(prompt):
        assert "CAPTION GUIDE" in prompt
        assert "adinross" in prompt.lower()
        return {
            "best": "LLM best pick",
            "alternatives": ["LLM alt 1", "LLM alt 2", "LLM alt 3",
                             "LLM alt 4"],
            "reasons": {"best": "LLM reason",
                        "alternatives": ["r1", "r2", "r3", "r4"]},
        }

    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="guide",
                               llm_fn=fake_llm)
    assert result["provider"] == "gemini"
    assert result["best"] == "LLM best pick"
    assert len(result["alternatives"]) == 4


def test_llm_failure_falls_back_to_rule_based():
    def failing_llm(prompt):
        raise RuntimeError("no network")

    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="guide",
                               llm_fn=failing_llm)
    assert result["provider"] == "rule_based"
    assert result["best"]
    assert len(result["alternatives"]) == 4


def test_llm_bad_json_falls_back_to_rule_based():
    def bad_llm(prompt):
        return {"best": "", "alternatives": []}

    with pytest.raises(ValueError):
        generate_captions_with_llm(STRAWBERRY_CONTEXT, "guide",
                                   llm_fn=bad_llm)
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="guide",
                               llm_fn=bad_llm)
    assert result["provider"] == "rule_based"


def test_llm_short_list_is_padded_with_rule_based():
    def short_llm(prompt):
        return {
            "best": "LLM best pick",
            "alternatives": ["LLM alt 1"],
            "reasons": {"best": "LLM reason", "alternatives": ["r1"]},
        }

    result = generate_captions_with_llm(STRAWBERRY_CONTEXT, "guide",
                                        llm_fn=short_llm)
    assert result["best"] == "LLM best pick"
    assert len(result["alternatives"]) == 4
    assert len(result["reasons"]["alternatives"]) == 4


def test_llm_hashtags_are_stripped():
    def taggy_llm(prompt):
        return {
            "best": "LLM best pick #fyp #kick",
            "alternatives": ["alt #gta6", "alt 2", "alt 3", "alt 4"],
            "reasons": {"best": "r", "alternatives": ["r"] * 4},
        }

    result = generate_captions_with_llm(STRAWBERRY_CONTEXT, "guide",
                                        llm_fn=taggy_llm)
    assert "#" not in result["best"]
    assert "#" not in result["alternatives"][0]


def test_no_key_means_rule_based(monkeypatch):
    import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="")
    assert result["provider"] == "rule_based"


# ---------------------------------------------------------------------------
# Post-caption composition (best caption + hashtags)
# ---------------------------------------------------------------------------

def test_post_caption_is_best_plus_hashtags():
    result = generate_captions(STRAWBERRY_CONTEXT, guide_text="",
                               use_llm=False)
    post = compose_post_caption(result["best"], "adinross", "GTA 6")
    assert post.startswith(result["best"])
    assert "#adinross" in post
    assert "#fyp" in post and "#kick" in post
    assert "#gta6" in post


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_setup(tmp_path, monkeypatch, fresh_db):
    """ClipPipeline with the editor captured and side-effect dirs isolated
    (same shape as tests/test_tiktok_pipeline.py::pipeline_setup)."""
    monkeypatch.setattr(pipeline_module, "SUBTITLES_DIR", tmp_path / "subtitles")
    monkeypatch.setattr(pipeline_module, "EDITED_DIR", tmp_path / "edited")
    monkeypatch.setattr(pipeline_module, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "SUBTITLE_STYLE", "classic")
    monkeypatch.setattr(pipeline_module, "BGM_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "BROLL_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "QUOTE_LINE_ENABLED", False)
    (tmp_path / "subtitles").mkdir()
    (tmp_path / "edited").mkdir()
    (tmp_path / "clips").mkdir()

    import config
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")

    pipe = ClipPipeline()
    monkeypatch.setattr(pipe, "_mark", lambda *a, **k: None)
    monkeypatch.setattr(
        pipe.ai, "transcribe_clip",
        lambda path, **kwargs: {"full_text": CLIP_11_TRANSCRIPT, "words": []})

    captured = {}

    def fake_edit_clip(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "edited" / "out.mp4")

    monkeypatch.setattr(pipe.editor, "edit_clip", fake_edit_clip)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    return pipe, raw, tmp_path, captured


def _clip_row(clip_id):
    import database
    conn = database.get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?",
                       (clip_id,)).fetchone()
    conn.close()
    return dict(row)


def test_process_moment_stores_post_captions(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw), game="GTA 6")
    assert result["status"] == "ready_to_upload"
    post = result["post_captions"]
    assert post["best"] == ("Adinross was SHOCKED after hearing "
                            "Chaz is not clean 😱💖")
    assert len(post["alternatives"]) == 4
    assert post["provider"] == "rule_based"
    # The TikTok post description is the best caption + hashtags.
    assert result["caption"].startswith(post["best"])
    assert "#gta6" in result["caption"]
    assert "#adinross" in result["caption"]
    # ... and the full set is stored on the clip record for the dashboard.
    stored = json.loads(_clip_row(result["clip_id"])["post_captions"])
    assert stored["best"] == post["best"]
    assert stored["alternatives"] == post["alternatives"]


def test_process_moment_caption_fallback_when_generation_fails(
        pipeline_setup, monkeypatch):
    pipe, raw, tmp_path, captured = pipeline_setup

    def boom(*args, **kwargs):
        raise RuntimeError("caption generator exploded")

    monkeypatch.setattr(captions_module, "generate_captions", boom)
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw), game="GTA 6")
    # The clip still succeeds; the post caption falls back to the old
    # title-based caption.
    assert result["status"] == "ready_to_upload"
    assert result["post_captions"] is None
    assert result["caption"].startswith(result["title"])
    assert "#gta6" in result["caption"]


def test_process_moment_captions_without_game(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    assert result["status"] == "ready_to_upload"
    assert result["post_captions"]["best"]
    assert result["caption"].startswith(result["post_captions"]["best"])
    assert "#adinross" in result["caption"]


def test_process_moment_uses_on_screen_text(pipeline_setup, monkeypatch):
    # The pipeline feeds the frames' text into the caption generator:
    # a silent moment ("Oh shit.") still gets a specific best pick
    # from the on-screen objective.
    pipe, raw, tmp_path, captured = pipeline_setup
    monkeypatch.setattr(
        pipe.ai, "transcribe_clip",
        lambda path, **kwargs: {"full_text": "Oh shit.", "words": []})
    import game_ocr
    monkeypatch.setattr(
        game_ocr, "read_screen_text",
        lambda path, **kwargs: {"text": "Go to the crash site.",
                                "lines": ["Go to the crash site."],
                                "frames_ocr": 1, "sampled": 5,
                                "scan_s": 0.1})
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw), game="GTA 6")
    assert result["status"] == "ready_to_upload"
    post = result["post_captions"]
    assert post["best"] == (
        "Adinross had NO IDEA GTA 6 had the crash site 💥💖")
    assert post["on_screen_text"] == "Go to the crash site."
    stored = json.loads(_clip_row(result["clip_id"])["post_captions"])
    assert stored["best"] == post["best"]


def test_process_moment_stores_apology_template(pipeline_setup, monkeypatch):
    # An apology clip gets the guide's apology template as its best
    # pick, stored on the clip record for the dashboard.
    pipe, raw, tmp_path, captured = pipeline_setup
    monkeypatch.setattr(
        pipe.ai, "transcribe_clip",
        lambda path, **kwargs: {
            "full_text": "I'm so sorry chat, I apologize, that was my fault.",
            "words": []})
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw), game="GTA 6")
    assert result["status"] == "ready_to_upload"
    post = result["post_captions"]
    assert post["best"] == _guide_template("apology")
    assert len(post["alternatives"]) == 4
    stored = json.loads(_clip_row(result["clip_id"])["post_captions"])
    assert stored["best"] == post["best"]
