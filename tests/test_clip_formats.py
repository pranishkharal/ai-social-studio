"""Tests for clip_formats.py — multi-part Short formats + render cache.

Planning, validation and the cache are pure. The render tests use a
fake part renderer (ffmpeg window copy) plus real ffmpeg assembly, on
small synthetic fixtures. No network.
"""

import os
import subprocess
import time

import pytest

import clip_formats as cf
from clip_formats import (PartCache, cache_key, chat_row_timings,
                          format_of, plan_parts, render_card_png,
                          render_chat_row_png, render_format,
                          validate_spec)


def _moment(start, end, title="", label="", score=0):
    return {"start": start, "end": end, "title": title,
            "label": label, "score": score}


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------


def test_format_of_defaults():
    assert format_of({"format": "chat"}) == "chat"
    assert format_of({"parts": [1, 2]}) == "callback"
    assert format_of({}) == "moment"
    assert format_of({"start": 0, "end": 5}) == "moment"


def test_validate_spec_unknown_format():
    with pytest.raises(ValueError, match="Unknown format"):
        validate_spec({"format": "nope"})


def test_validate_spec_needs_parts():
    with pytest.raises(ValueError, match="compilation needs at least 3"):
        validate_spec({"format": "compilation",
                       "parts": [_moment(0, 5), _moment(6, 11)]})
    with pytest.raises(ValueError, match="callback needs at least 2"):
        validate_spec({"format": "callback", "parts": [_moment(0, 5)]})


def test_validate_spec_rejects_short_parts():
    with pytest.raises(ValueError, match="too short"):
        validate_spec({"format": "callback",
                       "parts": [_moment(0, 0.2), _moment(5, 15)]})


def test_validate_spec_caps_compilation():
    with pytest.raises(ValueError, match="at most 5"):
        validate_spec({"format": "compilation",
                       "parts": [_moment(i * 10, i * 10 + 5)
                                 for i in range(6)]})


def test_validate_spec_single_part_needs_span():
    with pytest.raises(ValueError, match="needs a start/end"):
        validate_spec({"format": "waitforit"})
    assert validate_spec({"format": "waitforit", "start": 0, "end": 10}) == "waitforit"


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def test_plan_callback_trims_claim_and_payoff():
    spec = plan_parts("callback",
                      [_moment(10, 30, label="CLAIM"), _moment(400, 430)],
                      hook="HE CALLED IT")
    assert spec["format"] == "callback"
    claim, payoff = spec["parts"]
    assert claim["end"] - claim["start"] == 6.0  # claim capped at 6 s
    assert payoff["end"] - payoff["start"] == 15.0  # payoff capped at 15 s
    assert claim["label"] == "CLAIM"
    assert spec["hook"] == "HE CALLED IT"


def test_plan_compilation_best_last():
    moments = [_moment(0, 5, score=90), _moment(10, 15, score=50),
               _moment(20, 25, score=70)]
    spec = plan_parts("compilation", moments)
    starts = [p["start"] for p in spec["parts"]]
    # Ascending score: worst first (#3), best last (#1).
    assert starts == [10, 20, 0]


def test_plan_compilation_caps_at_five():
    moments = [_moment(i * 10, i * 10 + 5, score=i) for i in range(8)]
    spec = plan_parts("compilation", moments)
    assert len(spec["parts"]) == 5


def test_plan_escalation_keeps_stream_order_and_word():
    moments = [_moment(0, 5), _moment(10, 15), _moment(20, 25)]
    spec = plan_parts("escalation", moments, word="ATTEMPT")
    assert spec["word"] == "ATTEMPT"
    assert [p["start"] for p in spec["parts"]] == [0, 10, 20]


def test_plan_thenvsnow_labels():
    spec = plan_parts("thenvsnow", [_moment(0, 5), _moment(100, 105)],
                      labels=["BEFORE", "AFTER"])
    assert spec["labels"] == ["BEFORE", "AFTER"]
    assert len(spec["parts"]) == 2


def test_plan_single_moment_formats():
    for fmt in ("reveal", "chat", "waitforit", "moment"):
        spec = plan_parts(fmt, [_moment(5, 20)])
        assert spec["format"] == fmt
        assert len(spec["parts"]) == 1
        assert spec["parts"][0]["start"] == 5


def test_plan_parts_unknown_format():
    with pytest.raises(ValueError, match="Unknown format"):
        plan_parts("nope", [_moment(0, 5)])


def test_plan_parts_too_few_moments():
    with pytest.raises(ValueError, match="at least 3"):
        plan_parts("compilation", [_moment(0, 5)])


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_key_deterministic_and_sensitive():
    assert cache_key("a", 1, 2.5) == cache_key("a", 1, 2.5)
    assert cache_key("a", 1, 2.5) != cache_key("a", 1, 2.6)
    assert len(cache_key("x")) == 12


def test_part_cache_renders_once(tmp_path):
    cache = PartCache(tmp_path / "cache")
    calls = []

    def render(tmp):
        calls.append(1)
        tmp.write_bytes(b"fake mp4")

    first, cached = cache.get_or_render("abc123", render)
    assert cached is False
    assert first.read_bytes() == b"fake mp4"
    second, cached = cache.get_or_render("abc123", render)
    assert cached is True
    assert second == first
    assert len(calls) == 1
    # No half-written tmp file is left behind.
    assert not list((tmp_path / "cache").glob("*_tmp.mp4"))


def test_part_cache_failed_render_leaves_nothing(tmp_path):
    cache = PartCache(tmp_path / "cache")

    def render(tmp):
        pass  # writes nothing

    with pytest.raises(RuntimeError, match="produced no file"):
        cache.get_or_render("badkey", render)
    assert not (tmp_path / "cache" / "badkey.mp4").exists()


def test_part_cache_sweeps_old_files(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old = cache_dir / "old.mp4"
    old.write_bytes(b"x")
    os.utime(old, (1, 1))  # 1970: far older than 7 days
    new = cache_dir / "new.mp4"
    new.write_bytes(b"x")
    cache = PartCache(cache_dir)
    cache.get_or_render("other", lambda t: t.write_bytes(b"y"))
    assert not old.exists()
    assert new.exists()


# ---------------------------------------------------------------------------
# cards + chat rows (Pillow)
# ---------------------------------------------------------------------------


def _make_frame_png(path, size=(1080, 1920)):
    from PIL import Image
    Image.new("RGB", size, (20, 60, 120)).save(path, "PNG")
    return path


def test_render_card_png(tmp_path):
    frame = _make_frame_png(tmp_path / "frame.png")
    out = tmp_path / "card.png"
    render_card_png(frame, [("HE CALLED IT", 78, "FFD400"),
                            ("#3", 330, "FFD400"),
                            ("KAI LOST IT", 80, "FFFFFF")], out)
    assert out.exists()
    from PIL import Image
    with Image.open(out) as img:
        assert img.size == (1080, 1920)


def test_render_chat_row_png(tmp_path):
    out = tmp_path / "row.png"
    w, h = render_chat_row_png("speedfan", "#9146FF", "KEKW", out)
    assert out.exists()
    assert 0 < w <= 980
    assert h > 56


def test_chat_row_timings():
    rows = chat_row_timings(["KEKW", "W", "LOL"], payoff_t=5.0, duration=20.0)
    assert len(rows) == 3
    assert rows[0][0] == 5.2
    assert rows[1][0] == 5.75
    assert rows[0][3] == "KEKW"
    # Rows that would start in the last 1.2 s are dropped.
    rows = chat_row_timings(["a", "b", "c", "d", "e", "f"],
                            payoff_t=18.5, duration=20.0)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# render_modes format builders
# ---------------------------------------------------------------------------


def test_badge_filter_contains_drawtext():
    from render_modes import build_badge_filter
    filt = build_badge_filter("EARLIER")
    assert "drawtext=" in filt
    assert "EARLIER" in filt
    assert "boxcolor=black@0.55" in filt


def test_waitforit_filter_contains_tag_bar_flash():
    from render_modes import build_waitforit_filter
    filt = build_waitforit_filter(7.5)
    assert "WAIT FOR IT" in filt
    assert "lt(t,7.50)" in filt
    assert "min(t/7.50,1)" in filt  # the filling bar
    assert "between(t,7.50,7.58)" in filt  # the flash


def test_reveal_filter_contains_blur_and_tag():
    from render_modes import build_reveal_filter
    filt = build_reveal_filter(1.6, width=720, blur_height=413)
    assert "boxblur=45:3" in filt
    assert "crop=720:413" in filt
    assert "WATCH HIS FACE" in filt
    assert "lt(t,1.60)" in filt


def test_speaker_layout_filter_single_and_split():
    from render_modes import build_speaker_layout_filter
    traj = {"alice": {"x": 0, "crop_w": 607, "raw_h": 1080},
            "bob": {"x": 1313, "crop_w": 607, "raw_h": 1080}}
    segments = [
        {"speaker": "alice", "start": 0.0, "end": 5.0},
        {"speaker": "alice+bob", "start": 5.0, "end": 8.0,
         "split": ["alice", "bob"]},
        {"speaker": "bob", "start": 8.0, "end": 12.0},
        {"speaker": "alice", "start": 12.0, "end": 16.0},
    ]
    filt, out = build_speaker_layout_filter(segments, traj)
    assert out == "[speaker_out]"
    assert "vstack=inputs=3" in filt  # the split span
    assert "concat=n=4:v=1:a=0" in filt
    # Alice is used in two full-screen turns: her crop is split in two.
    assert "split=2" in filt


def test_speaker_layout_filter_missing_trajectory():
    from render_modes import build_speaker_layout_filter
    with pytest.raises(ValueError, match="No trajectory"):
        build_speaker_layout_filter(
            [{"speaker": "alice", "start": 0, "end": 5}], {})


# ---------------------------------------------------------------------------
# render_format with a fake part renderer (real ffmpeg assembly)
# ---------------------------------------------------------------------------


def _make_source(path, duration=60.0, size="720x1280"):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={duration}",
         "-f", "lavfi", "-i", f"sine=f=440:d={duration}",
         "-shortest", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=180)
    return path


def _fake_renderer(source_path, part, hook, out_path, work_dir,
                   layout=None, theme=None, streamer=""):
    """Part renderer for tests: window copy, no ClipEditor."""
    return cf.extract_window(source_path, part["start"], part["end"], out_path)


def _duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30)
    return float(out.stdout.strip())


@pytest.fixture(scope="module")
def source60(tmp_path_factory):
    path = tmp_path_factory.mktemp("fmt") / "source.mp4"
    return _make_source(path, duration=60.0)


def _make_rated_clip(path, rate=48000, duration=3.0, with_audio=True):
    """A small clip with audio at a given sample rate (or no audio)."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i",
           f"testsrc2=s=640x360:r=30:d={duration}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=440:d={duration}",
                "-ar", str(rate)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def _audio_params(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30)
    fields = out.stdout.strip().split(",")
    if len(fields) >= 2 and fields[0].isdigit():
        return int(fields[0]), int(fields[1])
    return None


def test_join_clips_mixed_sample_rates(tmp_path):
    # Regression: the format joiner (concat demuxer) failed with exit
    # 69 when a 96 kHz part met a 48 kHz card — the AAC decoder got
    # packets for the wrong sample rate. The join now normalises
    # every input's audio to 48 kHz stereo first.
    a = _make_rated_clip(tmp_path / "a.mp4", rate=48000)
    b = _make_rated_clip(tmp_path / "b.mp4", rate=44100)
    out = tmp_path / "joined.mp4"
    cf.join_clips([a, b], out)
    assert out.exists()
    assert abs(_duration(out) - 6.0) < 0.6
    assert _audio_params(out) == (48000, 2)


def test_join_clips_silent_clip_gets_silence(tmp_path):
    # A video-only input must not break the join either: it gets a
    # silent track so the concat demuxer sees uniform streams.
    a = _make_rated_clip(tmp_path / "a.mp4", rate=48000)
    b = _make_rated_clip(tmp_path / "silent.mp4", with_audio=False)
    out = tmp_path / "joined.mp4"
    cf.join_clips([a, b], out)
    assert out.exists()
    assert abs(_duration(out) - 6.0) < 0.6
    assert _audio_params(out) == (48000, 2)


def test_render_format_callback_joins_two_parts(tmp_path, source60):
    spec = plan_parts("callback",
                      [_moment(2, 8), _moment(40, 55)],
                      hook="HE CALLED IT")
    result = render_format(source60, spec, tmp_path / "cb.mp4",
                           part_renderer=_fake_renderer,
                           work_dir=tmp_path / "work",
                           cache_dir=tmp_path / "cache")
    assert result["format"] == "callback"
    out = tmp_path / "cb.mp4"
    assert out.exists()
    # 6 s claim + 15 s payoff = 21 s (gap 32 s < 10 min: no EARLIER tag).
    assert abs(_duration(out) - 21.0) < 0.6
    assert len(result["parts"]) == 2


def test_render_format_callback_long_gap_tags_earlier(tmp_path, source60):
    # Gap of 700 s (> 10 min) needs a longer source; fake it by using a
    # 60 s source with parts at 2 s and 50 s and a forced tag.
    spec = plan_parts("callback", [_moment(2, 8), _moment(40, 55)])
    spec["tag"] = "EARLIER"
    result = render_format(source60, spec, tmp_path / "cb2.mp4",
                           part_renderer=_fake_renderer,
                           work_dir=tmp_path / "work",
                           cache_dir=tmp_path / "cache")
    assert any("EARLIER tag" in line for line in result["log"])


def test_render_format_compilation_cards_and_badges(tmp_path, source60):
    moments = [_moment(0, 5, score=10), _moment(10, 15, score=30),
               _moment(20, 25, score=20)]
    spec = plan_parts("compilation", moments, hook="3 TIMES HE LOST IT")
    result = render_format(source60, spec, tmp_path / "comp.mp4",
                           part_renderer=_fake_renderer,
                           work_dir=tmp_path / "work",
                           cache_dir=tmp_path / "cache")
    out = tmp_path / "comp.mp4"
    assert out.exists()
    # 3 parts x 5 s + cards 0.9 + 0.6 + 0.6 = 17.1 s.
    assert abs(_duration(out) - 17.1) < 0.8
    assert any("#3" in line for line in result["log"])
    assert any("#1" in line for line in result["log"])


def test_render_format_escalation_badges(tmp_path, source60):
    moments = [_moment(0, 5), _moment(10, 15), _moment(20, 25)]
    spec = plan_parts("escalation", moments, word="STRIKE")
    result = render_format(source60, spec, tmp_path / "esc.mp4",
                           part_renderer=_fake_renderer,
                           work_dir=tmp_path / "work",
                           cache_dir=tmp_path / "cache")
    assert (tmp_path / "esc.mp4").exists()
    assert any("STRIKE 1/3" in line for line in result["log"])
    assert any("STRIKE 3/3" in line for line in result["log"])


def test_render_format_part_cache_reused_across_formats(tmp_path, source60):
    cache_dir = tmp_path / "cache"
    calls = []

    def counting_renderer(source_path, part, hook, out_path, work_dir,
                          layout=None, theme=None, streamer=""):
        calls.append((part["start"], part["end"]))
        return _fake_renderer(source_path, part, hook, out_path, work_dir)

    spec = plan_parts("callback", [_moment(2, 8), _moment(40, 55)])
    render_format(source60, spec, tmp_path / "a.mp4",
                  part_renderer=counting_renderer,
                  work_dir=tmp_path / "w1", cache_dir=cache_dir)
    assert len(calls) == 2
    # Same parts again (a second format on the same moments): cached.
    render_format(source60, spec, tmp_path / "b.mp4",
                  part_renderer=counting_renderer,
                  work_dir=tmp_path / "w2", cache_dir=cache_dir)
    assert len(calls) == 2


def test_render_format_moment_is_a_copy(tmp_path, source60):
    spec = plan_parts("moment", [_moment(5, 15)])
    result = render_format(source60, spec, tmp_path / "m.mp4",
                           part_renderer=_fake_renderer,
                           work_dir=tmp_path / "work",
                           cache_dir=tmp_path / "cache")
    assert result["format"] == "moment"
    assert abs(_duration(tmp_path / "m.mp4") - 10.0) < 0.4


def test_render_format_missing_source(tmp_path):
    spec = plan_parts("moment", [_moment(0, 5)])
    with pytest.raises(FileNotFoundError):
        render_format(tmp_path / "nope.mp4", spec, tmp_path / "o.mp4",
                      part_renderer=_fake_renderer)


# ---------------------------------------------------------------------------
# single-clip formats on a finished clip
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip1080(tmp_path_factory):
    path = tmp_path_factory.mktemp("fmt1") / "clip.mp4"
    return _make_source(path, duration=12.0, size="1080x1920")


def test_apply_single_clip_format_waitforit(tmp_path, clip1080):
    out = tmp_path / "wfi.mp4"
    cf.apply_single_clip_format(clip1080, "waitforit", out,
                                spec={"payoff": 8.0})
    assert out.exists()
    assert abs(_duration(out) - 12.0) < 0.3


def test_apply_single_clip_format_chat(tmp_path, clip1080):
    out = tmp_path / "chat.mp4"
    cf.apply_single_clip_format(
        clip1080, "chat", out,
        spec={"payoff": 6.0, "chat": ["KEKW", "W", "LOL"]})
    assert out.exists()
    assert abs(_duration(out) - 12.0) < 0.3


def test_apply_single_clip_format_chat_needs_lines(tmp_path, clip1080):
    with pytest.raises(ValueError, match="needs chat lines"):
        cf.apply_single_clip_format(clip1080, "chat", tmp_path / "c.mp4",
                                    spec={"payoff": 6.0})


def test_apply_single_clip_format_reveal_no_face_falls_back(tmp_path, clip1080, monkeypatch):
    # No facecam (monkeypatched: the real check needs OpenCV): the
    # reveal degrades to a plain moment copy.
    monkeypatch.setattr(cf, "_has_facecam", lambda path: False)
    out = tmp_path / "rev.mp4"
    log = []
    cf.apply_single_clip_format(clip1080, "reveal", out,
                                spec={"reveal": 1.6}, log=log)
    assert out.exists()
    assert abs(_duration(out) - 12.0) < 0.3
    assert any("no facecam found" in line for line in log)


def test_apply_single_clip_format_reveal_blurs(tmp_path, clip1080, monkeypatch):
    monkeypatch.setattr(cf, "_has_facecam", lambda path: True)
    out = tmp_path / "rev2.mp4"
    log = []
    cf.apply_single_clip_format(clip1080, "reveal", out,
                                spec={"reveal": 1.6}, log=log)
    assert out.exists()
    assert abs(_duration(out) - 12.0) < 0.3
    assert any("face blurred 1.6s" in line for line in log)


def test_apply_single_clip_format_unknown():
    with pytest.raises(ValueError, match="not a single-clip format"):
        cf.apply_single_clip_format("x.mp4", "callback", "y.mp4")
