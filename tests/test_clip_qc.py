"""Tests for automatic clip QC (gta6-clipper feature 2).

Real ffmpeg synthetic clips for the checks themselves (a good clip
passes; short / quiet / silent / black / frozen clips fail or warn
for the right reason), plus pipeline integration: a clip that passes
QC is stored with its report, a clip that fails gets exactly one
re-cut and then fails with the reason stored.

Frame-based checks (cards, frozen picture, letterbox) need OpenCV and
are skipped when cv2 is unavailable — run with the venv-whisper
site-packages on PYTHONPATH to exercise them.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clip_qc import _is_card, check_clip  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _needs_cv2():
    # Frame-based checks need OpenCV; the ffmpeg-only tests must run
    # without it, so the skip is applied per-test, not per-module.
    # The shared fake cv2 (installed by test_face_tracker_mp) cannot
    # sample frames, so it counts as unavailable here too.
    cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")
    if "fake" in getattr(cv2, "__version__", ""):
        pytest.skip("fake cv2 in this session — real OpenCV needed")


def _make_clip(path, duration=15, video_filter="format=yuv420p",
               audio_source="sine=f=300", audio_filters=None, loudnorm=True):
    """A synthetic 9:16 clip: moving test picture + a steady tone,
    normalised to -14 LUFS unless told otherwise."""
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc2=s=540x960:r=25:d={duration}",
           "-f", "lavfi", "-i", f"{audio_source}:d={duration}",
           "-vf", video_filter]
    af = list(audio_filters or [])
    if loudnorm:
        af.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    if af:
        cmd += ["-af", ",".join(af)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True, timeout=180)
    return path


def _make_frozen_clip(path, duration=15):
    """A synthetic 9:16 clip whose picture freezes for ~2 s mid-clip
    (7-9 s): frames 0-7 s, one frame looped, then 9-15 s. (ffmpeg 8
    changed freezeframes to a two-input filter, so trim+loop+concat
    builds the frozen stretch instead.)"""
    fc = ("[0:v]trim=0:7,setpts=PTS-STARTPTS[a];"
          "[0:v]trim=start=6.96:end=7.0,setpts=PTS-STARTPTS,"
          "loop=50:1:0,setpts=N/25/TB[b];"
          "[0:v]trim=9:15,setpts=PTS-STARTPTS[c];"
          "[a][b][c]concat=n=3:v=1:a=0[v]")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc2=s=540x960:r=25:d={duration}",
           "-f", "lavfi", "-i", f"sine=f=300:d={duration}",
           "-filter_complex", fc, "-map", "[v]", "-map", "1:a",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True, timeout=180)
    return path


def _card_frames(panel_mean, panel_std, motion, n=7):
    return [{"panel_mean": panel_mean, "panel_std": panel_std,
             "motion": motion} for _ in range(n)]


def test_is_card_dark_textured_scene_is_not_a_card():
    # Regression: a real dusk GTA scene (panel mean ~24, std ~30,
    # slow motion) ended a clip and QC failed it as "ends on a card".
    # Dark but textured is gameplay, not a card.
    is_card, info = _is_card(
        _card_frames(panel_mean=24.0, panel_std=30.0, motion=3.0), 0.4)
    assert is_card is False, info


def test_is_card_dark_flat_panel_is_a_card():
    # A black/logo card: dark AND flat AND still.
    is_card, info = _is_card(
        _card_frames(panel_mean=5.0, panel_std=2.0, motion=0.0), 0.4)
    assert is_card is True, info


def test_is_card_solid_colour_frame_is_a_card():
    # A solid-colour frame is a card at any brightness (std ~ 0).
    is_card, info = _is_card(
        _card_frames(panel_mean=76.0, panel_std=0.5, motion=0.0), 0.4)
    assert is_card is True, info


def test_is_card_moving_dark_scene_is_not_a_card():
    # Even a dark flat panel does not count when the picture moves.
    is_card, info = _is_card(
        _card_frames(panel_mean=10.0, panel_std=5.0, motion=8.0), 0.4)
    assert is_card is False, info


# ------------------------------------------------------------------ checks

def test_good_clip_passes(tmp_path):
    report = check_clip(_make_clip(tmp_path / "good.mp4"))
    assert report["status"] == "PASS", report
    assert report["fail"] == [] and report["warn"] == []
    assert report["duration"] == pytest.approx(15, abs=0.5)
    assert abs(report["lufs"] - (-14)) <= 2


def test_short_clip_fails_duration(tmp_path):
    report = check_clip(_make_clip(tmp_path / "short.mp4", duration=8))
    assert report["status"] == "FAIL"
    assert any("duration" in f for f in report["fail"])


def test_quiet_clip_fails_loudness(tmp_path):
    clip = _make_clip(tmp_path / "quiet.mp4", audio_filters=["volume=0.05"],
                      loudnorm=False)
    report = check_clip(clip)
    assert report["status"] == "FAIL"
    assert any("loudness" in f for f in report["fail"])


def test_silent_clip_fails_silence(tmp_path):
    clip = _make_clip(tmp_path / "silent.mp4",
                      audio_source="anullsrc=r=48000:cl=stereo",
                      loudnorm=False)
    report = check_clip(clip)
    assert report["status"] == "FAIL"
    assert any("silence" in f for f in report["fail"]), report


def test_black_frames_fail(tmp_path):
    # A full black flash mid-clip (0.5 s) must be caught.
    vf = ("format=yuv420p,"
          "drawbox=w=iw:h=ih:color=black:t=fill:enable='between(t,7,7.5)'")
    report = check_clip(_make_clip(tmp_path / "flash.mp4", video_filter=vf))
    assert report["status"] == "FAIL"
    assert any("black frames" in f for f in report["fail"]), report


def test_ends_on_black_card_fails(tmp_path):
    _needs_cv2()
    # The game panel goes black for the last 3 s: ends on a card.
    vf = ("format=yuv420p,"
          "drawbox=y=ih*0.45:w=iw:h=ih:color=black:t=fill:enable='gt(t,12)'")
    report = check_clip(_make_clip(tmp_path / "blackend.mp4", video_filter=vf))
    assert report["status"] == "FAIL"
    assert any("ends on" in f for f in report["fail"]), report


def test_frozen_middle_warns(tmp_path):
    # Picture frozen 7-9 s: a WARN, not a FAIL.
    report = check_clip(_make_frozen_clip(tmp_path / "frozen.mp4"))
    assert report["status"] == "WARN", report
    assert any("frozen" in w for w in report["warn"]), report


def test_static_clip_warns_no_face_or_motion(tmp_path):
    _needs_cv2()
    # A still colour frame: no motion anywhere, so the first-second
    # check warns (it also opens on a flat card, so it FAILs overall —
    # the warn must still be present).
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", "color=c=red:s=540x960:r=25:d=15",
           "-f", "lavfi", "-i", "sine=f=300:d=15",
           "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(tmp_path / "static.mp4")]
    subprocess.run(cmd, check=True, timeout=180)
    report = check_clip(tmp_path / "static.mp4")
    assert any("no face or motion" in w for w in report["warn"]), report


def test_missing_file_fails(tmp_path):
    report = check_clip(tmp_path / "nope.mp4")
    assert report["status"] == "FAIL"
    assert report["fail"] == ["file not found"]


def test_garbage_file_fails(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    report = check_clip(bad)
    assert report["status"] == "FAIL"
    assert any("unreadable" in f for f in report["fail"])


# --------------------------------------------------- pipeline integration

@pytest.fixture
def pipeline_qc_setup(tmp_path, monkeypatch, fresh_db):
    """ClipPipeline with a fake editor that copies a fixture clip to
    the render path, so QC runs for real on a known file."""
    import pipeline as pipeline_module
    from pipeline import ClipPipeline

    monkeypatch.setattr(pipeline_module, "SUBTITLES_DIR", tmp_path / "subtitles")
    monkeypatch.setattr(pipeline_module, "EDITED_DIR", tmp_path / "edited")
    monkeypatch.setattr(pipeline_module, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "SUBTITLE_STYLE", "classic")
    monkeypatch.setattr(pipeline_module, "BGM_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "BROLL_ENABLED", False)
    (tmp_path / "subtitles").mkdir()
    (tmp_path / "edited").mkdir()
    (tmp_path / "clips").mkdir()

    pipe = ClipPipeline()
    monkeypatch.setattr(pipe, "_mark", lambda *a, **k: None)
    state = {"fixture": None, "render_calls": 0}

    def fake_edit_clip(**kwargs):
        state["render_calls"] += 1
        shutil.copyfile(state["fixture"], kwargs["output_path"])
        return kwargs["output_path"]

    monkeypatch.setattr(pipe.editor, "edit_clip", fake_edit_clip)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    clip_id = fresh_db["seed_clip"](streamer="xqc")
    return {"pipe": pipe, "state": state, "clip_id": clip_id,
            "raw": str(raw), "tmp": tmp_path}


def _finish(pipe, clip_id, raw, **overrides):
    kwargs = dict(
        clip_id=clip_id, streamer="xqc", raw_clip_path=raw, duration=15,
        transcript_text="hello world", words=[], metadata={},
        hype_score=80.0, trigger_reason="test",
    )
    kwargs.update(overrides)
    return pipe._finish_clip(**kwargs)


def test_pipeline_stores_qc_report_on_pass(pipeline_qc_setup):
    import database

    setup = pipeline_qc_setup
    setup["state"]["fixture"] = str(
        _make_clip(setup["tmp"] / "good.mp4"))
    result = _finish(setup["pipe"], setup["clip_id"], setup["raw"])
    assert result["status"] == "ready_to_upload"
    assert result["qc"]["status"] == "PASS"
    assert setup["state"]["render_calls"] == 1  # no re-cut needed

    conn = database.get_connection()
    row = conn.execute("SELECT status, qc_report FROM clips WHERE id = ?",
                       (setup["clip_id"],)).fetchone()
    conn.close()
    assert row["status"] == "ready_to_upload"
    stored = json.loads(row["qc_report"])
    assert stored["status"] == "PASS"


def test_pipeline_warn_still_uploads(pipeline_qc_setup):
    setup = pipeline_qc_setup
    setup["state"]["fixture"] = str(
        _make_frozen_clip(setup["tmp"] / "frozen.mp4"))
    result = _finish(setup["pipe"], setup["clip_id"], setup["raw"])
    assert result["status"] == "ready_to_upload"
    assert result["qc"]["status"] == "WARN"


def test_pipeline_fail_gets_one_recut_then_fails(pipeline_qc_setup):
    import database

    setup = pipeline_qc_setup
    # An 8 s clip fails QC on duration — and the re-cut fails too,
    # because it is cut from the same footage.
    setup["state"]["fixture"] = str(
        _make_clip(setup["tmp"] / "short.mp4", duration=8))
    result = _finish(setup["pipe"], setup["clip_id"], setup["raw"])
    assert setup["state"]["render_calls"] == 2  # first render + one re-cut
    assert result["status"] == "failed"
    assert "qc_failed" in result["error"]
    assert "duration" in result["error"]
    assert result["qc"]["status"] == "FAIL"

    conn = database.get_connection()
    row = conn.execute("SELECT status, qc_report FROM clips WHERE id = ?",
                       (setup["clip_id"],)).fetchone()
    conn.close()
    assert row["status"] == "failed"
    stored = json.loads(row["qc_report"])
    assert stored["status"] == "FAIL"
    assert any("duration" in f for f in stored["fail"])


def test_pipeline_recut_can_rescue_a_clip(pipeline_qc_setup):
    """A transient first-render failure (bad file) followed by a good
    re-cut: the clip uploads instead of failing."""
    setup = pipeline_qc_setup
    good = str(_make_clip(setup["tmp"] / "good.mp4"))
    bad = setup["tmp"] / "bad.mp4"
    bad.write_bytes(b"not a video")
    calls = {"n": 0}

    def flaky_edit_clip(**kwargs):
        calls["n"] += 1
        src = str(bad) if calls["n"] == 1 else good
        shutil.copyfile(src, kwargs["output_path"])
        return kwargs["output_path"]

    setup["pipe"].editor.edit_clip = flaky_edit_clip
    result = _finish(setup["pipe"], setup["clip_id"], setup["raw"])
    assert calls["n"] == 2
    assert result["status"] == "ready_to_upload"
    assert result["qc"]["status"] == "PASS"
