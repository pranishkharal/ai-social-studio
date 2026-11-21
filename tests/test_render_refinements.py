"""Tests for the rendering refinements (gta6-clipper feature 21):

- sentence-boundary snapping (silence_cutter.snap_clip_bounds)
- profanity masking (karaoke_subtitles.mask_profanity_word)
- themes (themes.py + themes.json, incl. the clip_editor wiring)
- the theme progress bar (render_modes.build_progress_bar_filter)
- timestamp remapping for speaker segments (silence_cutter.remap_*)

Pure tests except the clip_editor card check (Pillow only). No network.
"""

import pytest

from karaoke_subtitles import (generate_karaoke_ass, mask_profanity_enabled,
                               mask_profanity_word)
from silence_cutter import (compute_keep_segments, remap_segments,
                            remap_time, snap_clip_bounds)
from themes import caption_style, get_theme, hex_to_rgb, hook_colors, list_themes, progress_bar


def _words(spec):
    """spec: [(word, start, end), ...] -> Whisper word dicts."""
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


# ---------------------------------------------------------------------------
# sentence-boundary snapping
# ---------------------------------------------------------------------------


def _sentence_words():
    # Three sentences over ~11.5 s, 0.4 s words, 0.1 s gaps, 1 s pauses
    # between sentences. Sentence ends: 3.4 s, 7.4 s, 11.4 s.
    words = []
    t = 0.0
    for sentence in ("I will never die to the cops.",
                     "Watch me win this round now.",
                     "That was way too easy man."):
        for i, word in enumerate(sentence.split()):
            last = i == len(sentence.split()) - 1
            words.append((word, round(t, 2), round(t + 0.4, 2)))
            t += 0.5
        t += 1.0  # pause between sentences
    return _words(words)


def test_snap_extends_to_sentence_end_within_4s():
    words = _sentence_words()
    # Pick ends at 6.0 s, mid-sentence-2 ("this" 6.0-6.4). Rule 1:
    # sentence 2 ends at 7.4 s, within 4 s — the end extends to it
    # plus a 0.25 s breath.
    start, end = snap_clip_bounds(words, 0.0, 6.0)
    assert end == pytest.approx(7.65, abs=0.05)
    assert start == pytest.approx(0.0, abs=0.05)


def test_snap_trims_dead_pause_after_sentence_end():
    words = _sentence_words()
    # Pick ends at 4.4 s, in the pause after sentence 1 (ends 3.4 s).
    # Rule 1 finds the sentence end right there: the clip ends at
    # 3.4 s + a breath instead of padding with dead air.
    start, end = snap_clip_bounds(words, 0.0, 4.4)
    assert end == pytest.approx(3.65, abs=0.05)


def test_snap_backs_off_to_last_stop_that_keeps_12s():
    # Four sentences, then a long unpunctuated run with no pauses.
    words = []
    t = 0.0
    for sentence in ("I will never die to the cops.",
                     "Watch me win this round now.",
                     "That was way too easy man.",
                     "And now the final test ends here."):
        for word in sentence.split():
            words.append((word, round(t, 2), round(t + 0.4, 2)))
            t += 0.5
        t += 1.0
    for i in range(16):  # unpunctuated run, 17.0-24.9 s
        words.append((f"r{i}", round(t, 2), round(t + 0.4, 2)))
        t += 0.5
    words = _words(words)
    # Pick ends at 21 s, mid-run, with no sentence end within 4 s.
    # Rule 3 backs off to the last stop that keeps >= 12 s: sentence
    # 4 ends at 15.9 s — the end becomes 15.9 s + a breath.
    start, end = snap_clip_bounds(words, 0.0, 21.0)
    assert end == pytest.approx(16.15, abs=0.05)
    assert end - start >= 12.0


def test_snap_exact_mode_never_goes_mid_word():
    words = _sentence_words()
    # Same pick as the rule-1 test, but exact: the end is the last
    # word that finishes by 6.0 s ("win", ends 5.9 s) + a breath —
    # it does NOT extend to the sentence end at 7.4 s.
    start, end = snap_clip_bounds(words, 0.0, 6.0, exact=True)
    assert end == pytest.approx(6.0, abs=0.05)
    _, rule_end = snap_clip_bounds(words, 0.0, 6.0)
    assert rule_end == pytest.approx(7.65, abs=0.05)


def test_snap_start_moves_to_first_word():
    words = _words([("hello", 2.0, 2.4), ("world.", 2.5, 2.9),
                    ("again.", 5.0, 5.4)])
    start, end = snap_clip_bounds(words, 0.0, 6.0)
    assert start == pytest.approx(1.8)  # first word - 0.2 s lead-in


def test_snap_no_words_unchanged():
    assert snap_clip_bounds([], 1.0, 9.0) == (1.0, 9.0)
    assert snap_clip_bounds(None, 1.0, 9.0) == (1.0, 9.0)


def test_snap_unpunctuated_transcript_snaps_on_breaths():
    # Whisper tiny often drops punctuation: pauses still count as stops.
    words = _words([("hello", 0.0, 0.4), ("chat", 0.5, 0.9),
                    ("what", 3.0, 3.4), ("is", 3.5, 3.7),
                    ("up", 3.8, 4.2)])
    start, end = snap_clip_bounds(words, 0.0, 1.5)
    # Pick ends at 1.5 s; "chat" ends at 0.9 s with a 2.1 s pause
    # after it — a breath, so rule 2 keeps the pick end region.
    assert end <= 1.6


def test_compute_keep_segments_snap_extends_end():
    words = _sentence_words()
    plain = compute_keep_segments(words, snap=False)
    snapped = compute_keep_segments(words, snap=True)
    # Without snap the segment ends at the last word + 0.15 s pad;
    # with snap it ends at the sentence end + a 0.25 s breath.
    assert plain == [(0.0, 11.55)]
    assert snapped == [(0.0, 11.65)]


def test_compute_keep_segments_snap_off_by_default():
    words = _sentence_words()
    assert compute_keep_segments(words) == [(0.0, 11.55)]


# ---------------------------------------------------------------------------
# timestamp remapping (speaker segments into trimmed time)
# ---------------------------------------------------------------------------


def test_remap_time_inside_and_outside_segments():
    keep = [(0.0, 5.0), (10.0, 15.0)]  # 5 s cut out of the middle
    assert remap_time(2.0, keep) == 2.0
    assert remap_time(12.0, keep) == 7.0  # 5 s kept + 2 s in
    assert remap_time(7.0, keep) == 5.0   # inside the dropped gap
    assert remap_time(20.0, keep) == 10.0  # past the end: trimmed length


def test_remap_segments_keeps_keys_and_drops_collapsed():
    keep = [(0.0, 5.0), (10.0, 15.0)]
    segments = [
        {"speaker": "alice", "start": 1.0, "end": 3.0},
        {"speaker": "alice+bob", "start": 11.0, "end": 13.0,
         "split": ["alice", "bob"]},
        {"speaker": "bob", "start": 6.0, "end": 7.0},  # inside the gap
    ]
    out = remap_segments(segments, keep)
    assert len(out) == 2  # the gap segment collapses away
    assert out[0] == {"speaker": "alice", "start": 1.0, "end": 3.0}
    assert out[1]["start"] == 6.0
    assert out[1]["end"] == 8.0
    assert out[1]["split"] == ["alice", "bob"]


def test_remap_segments_no_keep_segments_is_identity():
    segments = [{"speaker": "a", "start": 1.0, "end": 2.0}]
    assert remap_segments(segments, []) == segments


# ---------------------------------------------------------------------------
# profanity masking
# ---------------------------------------------------------------------------


def test_mask_profanity_word():
    assert mask_profanity_word("fuck") == "f***"
    assert mask_profanity_word("fucking") == "f******"
    assert mask_profanity_word("shit") == "s***"
    assert mask_profanity_word("FUCK") == "F***"
    assert mask_profanity_word("fuck!") == "f***!"
    assert mask_profanity_word("hello") == "hello"
    assert mask_profanity_word("shitake") == "shitake"  # not a swear word
    assert mask_profanity_word("class") == "class"


def test_mask_profanity_enabled_env(monkeypatch):
    monkeypatch.delenv("KARAOKE_MASK_PROFANITY", raising=False)
    assert mask_profanity_enabled() is True
    monkeypatch.setenv("KARAOKE_MASK_PROFANITY", "0")
    assert mask_profanity_enabled() is False


def test_generate_karaoke_ass_masks_profanity(tmp_path):
    words = _words([("what", 0.0, 0.3), ("the", 0.3, 0.5),
                    ("fuck", 0.5, 0.9), ("man", 1.0, 1.3)])
    out = tmp_path / "clip.ass"
    generate_karaoke_ass(words, str(out), style="minimal")
    text = out.read_text(encoding="utf-8")
    assert "f***" in text
    assert "fuck" not in text.lower()


def test_generate_karaoke_ass_mask_can_be_disabled(tmp_path):
    words = _words([("what", 0.0, 0.3), ("the", 0.3, 0.5),
                    ("fuck", 0.5, 0.9)])
    out = tmp_path / "clip.ass"
    generate_karaoke_ass(words, str(out), style="minimal",
                         mask_profanity=False)
    text = out.read_text(encoding="utf-8")
    assert "fuck" in text


# ---------------------------------------------------------------------------
# themes
# ---------------------------------------------------------------------------


def test_list_themes_has_five():
    themes = list_themes()
    assert len(themes) == 5
    names = [name for name, _ in themes]
    assert "clean" in names
    assert "bold-yellow" in names
    assert "neon" in names


def test_get_theme_default_is_clean(monkeypatch):
    monkeypatch.delenv("CLIP_THEME", raising=False)
    assert get_theme()["name"] == "clean"
    assert get_theme(None)["name"] == "clean"


def test_get_theme_env_var(monkeypatch):
    monkeypatch.setenv("CLIP_THEME", "neon")
    assert get_theme()["name"] == "neon"
    # An explicit name wins over the env var.
    assert get_theme("bold-yellow")["name"] == "bold-yellow"


def test_get_theme_unknown_falls_back_to_clean():
    assert get_theme("no-such-theme")["name"] == "clean"


def test_hex_to_rgb():
    assert hex_to_rgb("FFE500") == (255, 229, 0)
    assert hex_to_rgb("#FF3B30") == (255, 59, 48)
    assert hex_to_rgb("F00") == (255, 0, 0)
    assert hex_to_rgb("garbage") == (255, 255, 255)
    assert hex_to_rgb("garbage", (1, 2, 3)) == (1, 2, 3)


def test_hook_colors_default_is_white_box_black_text():
    box, text = hook_colors("clean")
    assert box == (255, 255, 255, 255)
    assert text == (10, 10, 10, 255)
    box, text = hook_colors("bold-yellow")
    assert box == (255, 229, 0, 255)
    assert text == (0, 0, 0, 255)


def test_progress_bar_themes():
    assert progress_bar("clean") == (False, "FFE500")
    assert progress_bar("neon") == (True, "22D3EE")
    assert progress_bar("bold-yellow") == (True, "FFE500")


def test_caption_style_themes():
    assert caption_style("clean") == "hormozi"
    assert caption_style("minimal-white") == "minimal"
    assert caption_style("nope") == "hormozi"  # unknown -> clean -> hormozi


# ---------------------------------------------------------------------------
# progress bar filter
# ---------------------------------------------------------------------------


def test_build_progress_bar_filter():
    from render_modes import build_progress_bar_filter
    filt = build_progress_bar_filter(20.0, "22D3EE")
    assert "drawbox" in filt
    assert "0x22D3EE" in filt
    assert "min(t/20.00,1)" in filt
    assert "ih-12" in filt


def test_build_progress_bar_filter_bad_color_falls_back():
    from render_modes import build_progress_bar_filter
    filt = build_progress_bar_filter(10.0, "zzz")
    assert "0xFFE500" in filt


# ---------------------------------------------------------------------------
# clip_editor theme wiring (Pillow card only — no ffmpeg)
# ---------------------------------------------------------------------------


def test_caption_card_theme_colours(tmp_path):
    from PIL import Image

    from clip_editor import ClipEditor
    editor = ClipEditor()

    default_png = tmp_path / "default.png"
    editor.generate_screen_caption_card("HELLO WORLD", str(default_png))
    yellow_png = tmp_path / "yellow.png"
    editor.generate_screen_caption_card("HELLO WORLD", str(yellow_png),
                                        theme="bold-yellow")

    def _has_colour(path, target, tolerance=12):
        with Image.open(path) as img:
            pixels = img.convert("RGBA").getdata()
        count = 0
        for r, g, b, a in pixels:
            if a < 200:
                continue
            if all(abs(c - t) <= tolerance for c, t in zip((r, g, b), target)):
                count += 1
        return count

    # Default theme: a white box (thousands of white pixels).
    assert _has_colour(default_png, (255, 255, 255)) > 10000
    assert _has_colour(default_png, (255, 229, 0)) == 0
    # Bold-yellow theme: the box is yellow, not white.
    assert _has_colour(yellow_png, (255, 229, 0)) > 10000
    assert _has_colour(yellow_png, (255, 255, 255)) < 100
