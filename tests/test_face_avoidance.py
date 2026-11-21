"""Tests for FEATURE 1: face-aware banner placement.

- face_avoidance.py is pure (no cv2): mapping, face-region union,
  banner/quote placement, mirror helpers.
- The card generators honour banner_y / quote_y overrides.
- edit_clip wires the tracker output into placement: a monkeypatched
  SpeakerTracker returns a known trajectory + face boxes, and a real
  ffmpeg render proves the banner is placed where the face is not.
- The mirror interaction: the tracker analyses the unmirrored source
  but the crop runs after hflip, so both the trajectory and the face
  boxes must be mirrored for the placement to dodge the face as
  rendered.

The analyze_face_boxes tests reuse the fake cv2 from
tests/test_face_tracker_mp.py (imported first so its stub is the one
installed in sys.modules); cv2 is not installed under system python3.
"""

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_face_tracker_mp as _ftm  # noqa: E402  (installs fake cv2)
import clip_editor  # noqa: E402
import face_avoidance as fa  # noqa: E402
from speaker_tracker import SpeakerTracker  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


# --------------------------------------------------------------------------
# map_box_to_output
# --------------------------------------------------------------------------

def test_map_box_to_output_known_box():
    # 1920x1080 source, crop_w 607 (9:16), crop at x 0.
    # scale = 1080/607 = 1.7794...
    box = fa.map_box_to_output((100, 100, 200, 100), 0, 607, 1080)
    assert box == (178, 178, 534, 356)


def test_map_box_to_output_with_crop_offset():
    # Same box, crop window starting at x 500: the box shifts left.
    box = fa.map_box_to_output((600, 100, 200, 100), 500, 607, 1080)
    assert box == (178, 178, 534, 356)


def test_map_box_to_output_clamps_at_left_edge():
    # Box straddles the crop's left edge: clipped to x 0.
    box = fa.map_box_to_output((400, 100, 200, 100), 500, 607, 1080)
    assert box == (0, 178, 178, 356)


def test_map_box_to_output_clamps_at_bottom_edge():
    # Box runs past the bottom of the 1920-tall frame.
    box = fa.map_box_to_output((100, 1000, 100, 100), 0, 607, 1080)
    assert box is not None
    assert box[3] == 1920
    assert box[1] == 1779


def test_map_box_to_output_none_when_fully_outside_crop():
    # Entirely left of the crop window: not in the render at all.
    assert fa.map_box_to_output((0, 100, 100, 100), 500, 607, 1080) is None
    # Entirely right of the crop window.
    assert fa.map_box_to_output((1500, 100, 100, 100), 0, 607, 1080) is None


# --------------------------------------------------------------------------
# face_region_for_clip
# --------------------------------------------------------------------------

def test_face_region_union_over_samples():
    # Two samples, crop panning right between them: the region is the
    # union of the mapped boxes (face fixed in the source, crop moves).
    face_boxes = [(0, [(100, 100, 100, 100)]),
                  (30, [(100, 100, 100, 100)])]
    smoothed_x = [0] * 30 + [200] * 30
    region = fa.face_region_for_clip(face_boxes, smoothed_x, 607, 1080,
                                     margin=0)
    # Sample 0 (crop 0): x 178-356. Sample 30 (crop 200): the box is
    # at (100-200)*scale < 0 on the left, clipped to 0; right edge
    # (200-200)*scale = 0 -> the box maps to None (fully outside)...
    # so the union is just the first sample's box.
    assert region == (178, 178, 356, 356)


def test_face_region_union_spans_panning_crop():
    # Face stays visible while the crop pans: the union spans both
    # mapped positions.
    face_boxes = [(0, [(400, 100, 100, 100)]),
                  (30, [(400, 100, 100, 100)])]
    smoothed_x = [0] * 30 + [200] * 30
    region = fa.face_region_for_clip(face_boxes, smoothed_x, 607, 1080,
                                     margin=0)
    # Sample 0: x (400..500)*scale = 712..890. Sample 30 (crop 200):
    # x (200..300)*scale = 356..534. Union x 356..890, y 178..356.
    assert region == (356, 178, 890, 356)


def test_face_region_margin_expands_and_clamps():
    face_boxes = [(0, [(0, 0, 100, 100)])]
    region = fa.face_region_for_clip(face_boxes, [0] * 10, 607, 1080,
                                     margin=40)
    # Mapped box (0, 0, 178, 178) expanded by 40, clamped to the frame.
    assert region == (0, 0, 218, 218)


def test_face_region_none_when_no_faces():
    assert fa.face_region_for_clip([], [0] * 10, 607, 1080) is None
    assert fa.face_region_for_clip([(0, []), (5, [])], [0] * 10, 607, 1080) is None
    # Faces exist but are outside the crop window in every sample.
    assert fa.face_region_for_clip(
        [(0, [(1500, 100, 100, 100)])], [0] * 10, 607, 1080) is None


def test_face_region_clamps_out_of_range_frame_index():
    # A sample past the end of the trajectory uses the last crop_x.
    face_boxes = [(999, [(100, 100, 100, 100)])]
    region = fa.face_region_for_clip(face_boxes, [0] * 10, 607, 1080,
                                     margin=0)
    assert region == (178, 178, 356, 356)


# --------------------------------------------------------------------------
# boxes_overlap
# --------------------------------------------------------------------------

def test_boxes_overlap_with_gap():
    assert fa.boxes_overlap((0, 0, 100, 100), (50, 50, 150, 150))
    assert not fa.boxes_overlap((0, 0, 100, 100), (200, 200, 300, 300))
    # 10 px apart: no overlap without a gap, overlap with a 20 px gap.
    assert not fa.boxes_overlap((0, 0, 100, 100), (110, 0, 200, 100))
    assert fa.boxes_overlap((0, 0, 100, 100), (110, 0, 200, 100), gap=20)
    assert fa.overlap_area((0, 0, 100, 100), (50, 50, 150, 150)) == 2500
    assert fa.overlap_area((0, 0, 100, 100), (200, 200, 300, 300)) == 0


# --------------------------------------------------------------------------
# mirror helpers
# --------------------------------------------------------------------------

def test_mirror_box_and_trajectory():
    assert fa.mirror_box((100, 50, 80, 60), 1920) == (1740, 50, 80, 60)
    # Mirroring twice returns the original box.
    assert fa.mirror_box(fa.mirror_box((100, 50, 80, 60), 1920), 1920) == (100, 50, 80, 60)
    assert fa.mirror_trajectory([0, 300, 1313], 1313) == [1313, 1013, 0]
    assert fa.mirror_face_boxes([(0, [(100, 50, 80, 60)])], 1920) == [
        (0, [(1740, 50, 80, 60)])]


def test_mirrored_face_and_trajectory_map_to_same_output():
    # Mirroring BOTH the face and the crop preserves the face's output
    # position: the render is a mirror image, but the face sits in the
    # same place — so placement must mirror both or neither.
    box = (400, 130, 80, 80)
    plain = fa.map_box_to_output(box, 339, 202, 360)
    mirrored = fa.map_box_to_output(fa.mirror_box(box, 640),
                                    fa.mirror_trajectory([339], 438)[0],
                                    202, 360)
    assert plain == mirrored
    assert plain is not None


# --------------------------------------------------------------------------
# choose_banner_y
# --------------------------------------------------------------------------

def test_choose_banner_y_no_face_returns_800():
    assert fa.choose_banner_y(None, 100, "speaker_tracking") == 800


def test_choose_banner_y_face_at_top_drops_to_800():
    # Facecam face across the top: the 260 candidate overlaps it, so
    # the banner drops to the historical 800 position.
    face = (0, 0, 1080, 500)
    assert fa.choose_banner_y(face, 100, "speaker_tracking") == 800


def test_choose_banner_y_face_at_centre_goes_to_top():
    # Face in the middle (covers 800): the top candidate is free.
    face = (0, 700, 1080, 1000)
    assert fa.choose_banner_y(face, 100, "speaker_tracking") == 260


def test_choose_banner_y_face_at_bottom_goes_to_top():
    face = (0, 1200, 1080, 1420)
    assert fa.choose_banner_y(face, 100, "speaker_tracking") == 260


def test_choose_banner_y_top_and_centre_covered_goes_to_bottom():
    # Faces cover both 260 and 800: the banner goes just above the
    # Kick bar (1410 - box_h).
    face = (0, 0, 1080, 950)
    assert fa.choose_banner_y(face, 100, "speaker_tracking") == 1310


def test_choose_banner_y_face_off_to_the_side_keeps_top():
    # A narrow face at the far left does not overlap the centred
    # banner box, so the top candidate is free.
    face = (0, 200, 100, 500)
    assert fa.choose_banner_y(face, 100, "speaker_tracking", box_w=800) == 260


def test_choose_banner_y_all_overlap_returns_least_overlap():
    # The face covers the top and middle and ends at y 1350, so
    # every candidate overlaps — but the bottom slot (1310-1410)
    # only overlaps 40 px of it, less than the 100 px of the top
    # and middle slots, so it wins.
    face = (0, 0, 1080, 1350)
    assert fa.choose_banner_y(face, 100, "speaker_tracking") == 1310


# --------------------------------------------------------------------------
# choose_quote_y
# --------------------------------------------------------------------------

def test_choose_quote_y_no_face_returns_1240():
    assert fa.choose_quote_y(None, None, 56) == 1240


def test_choose_quote_y_face_in_lower_third_moves_up():
    # Face covers the 1240 band (1150-1350): the quote steps down
    # toward the Kick bar, finds no room, then steps up clear.
    face = (0, 1150, 1080, 1350)
    assert fa.choose_quote_y(face, None, 56) == 1060


def test_choose_quote_y_avoids_banner_box():
    # No face conflict at 1240, but the banner sits at the bottom
    # (1214-1410): the quote moves up clear of it.
    face = (0, 0, 1080, 500)  # face at the top, far from the quote
    banner_box = (75, 1214, 875, 1410)
    assert fa.choose_quote_y(face, banner_box, 56) == 1120


def test_choose_quote_y_stays_when_nothing_overlaps():
    face = (0, 0, 1080, 500)
    banner_box = (140, 260, 940, 360)
    assert fa.choose_quote_y(face, banner_box, 56) == 1240


# --------------------------------------------------------------------------
# Card generators honour the overrides (pixel checks)
# --------------------------------------------------------------------------

def _alpha_bbox(path):
    from PIL import Image
    img = Image.open(path)
    assert img.mode == "RGBA"
    assert img.size == (1080, 1920)
    return img.split()[3].getbbox()


def test_banner_card_honours_banner_y(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "card.png"
    editor.generate_screen_caption_card("TEST BANNER", str(out),
                                        position="top",
                                        layout="speaker_tracking",
                                        banner_y=260)
    bbox = _alpha_bbox(out)
    assert bbox is not None
    assert bbox[1] == 260, f"banner top edge at {bbox[1]}, expected 260"


def test_banner_card_default_still_800_for_face_tracking(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "card.png"
    editor.generate_screen_caption_card("TEST BANNER", str(out),
                                        position="top",
                                        layout="speaker_tracking")
    bbox = _alpha_bbox(out)
    assert bbox is not None
    assert bbox[1] == 800, f"banner top edge at {bbox[1]}, expected 800"


def test_banner_card_banner_y_ignored_for_bottom_position(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "card.png"
    editor.generate_screen_caption_card("TEST BANNER", str(out),
                                        position="bottom",
                                        layout="speaker_tracking",
                                        banner_y=260)
    _lines, _w, box_h = editor._measure_caption_box("TEST BANNER")
    bbox = _alpha_bbox(out)
    assert bbox is not None
    assert bbox[1] == 1410 - box_h


def test_quote_card_honours_quote_y(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "quote.png"
    editor.generate_quote_card("Bro, I cannot believe this", str(out),
                               quote_y=1060)
    bbox = _alpha_bbox(out)
    assert bbox is not None
    centre = (bbox[1] + bbox[3]) // 2
    assert abs(centre - 1060) <= 35, f"quote centred at {centre}, expected 1060"
    # And it is nowhere near the default 1240 band.
    assert bbox[3] < 1150


def test_quote_card_default_still_1240(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "quote.png"
    editor.generate_quote_card("Bro, I cannot believe this", str(out))
    bbox = _alpha_bbox(out)
    assert bbox is not None
    centre = (bbox[1] + bbox[3]) // 2
    assert abs(centre - 1240) <= 35, f"quote centred at {centre}, expected 1240"


# --------------------------------------------------------------------------
# analyze_face_boxes (fake cv2: 60 frames, 1920x1080, 30 fps)
# --------------------------------------------------------------------------

def test_analyze_face_boxes_samples_and_frame_indices(monkeypatch):
    # HAAR_PLAN boxes are in the detector's half-resolution space;
    # _detect_faces_haar doubles them back to full resolution.
    monkeypatch.setattr(_ftm, "HAAR_PLAN", [[100, 100, 50, 50]])
    tracker = SpeakerTracker(models_dir=str(REPO_ROOT / "models"),
                             backend="haar")
    samples = tracker.analyze_face_boxes("fake.mp4", sample_fps=6)
    # 30 fps / 6 = every 5th frame: 0, 5, ..., 55.
    assert [idx for idx, _ in samples] == list(range(0, 60, 5))
    assert len(samples) == 12
    for _idx, boxes in samples:
        assert boxes == [(200, 200, 100, 100)]


def test_analyze_face_boxes_empty_when_no_faces(monkeypatch):
    monkeypatch.setattr(_ftm, "HAAR_PLAN", None)
    tracker = SpeakerTracker(models_dir=str(REPO_ROOT / "models"),
                             backend="haar")
    samples = tracker.analyze_face_boxes("fake.mp4", sample_fps=6)
    assert len(samples) == 12
    assert all(boxes == [] for _idx, boxes in samples)


def test_analyze_face_boxes_sample_fps_controls_cadence(monkeypatch):
    monkeypatch.setattr(_ftm, "HAAR_PLAN", [[10, 10, 20, 20]])
    tracker = SpeakerTracker(models_dir=str(REPO_ROOT / "models"),
                             backend="haar")
    samples = tracker.analyze_face_boxes("fake.mp4", sample_fps=30)
    assert [idx for idx, _ in samples] == list(range(60))


# --------------------------------------------------------------------------
# edit_clip integration (fake tracker, real + fake ffmpeg)
# --------------------------------------------------------------------------

# Fake tracker plan, set per test: a 640x360 source, crop_w 202.
FAKE_PLAN = {
    "trajectory": [339] * 60,
    "boxes": [(0, [(400, 200, 80, 60)]), (30, [(400, 200, 80, 60)])],
    "dims": (640, 360, 202),
}


class FakeTracker:
    def __init__(self, *a, **k):
        pass

    def analyze_video_trajectory(self, path, *a, **k):
        rw, rh, cw = FAKE_PLAN["dims"]
        return (list(FAKE_PLAN["trajectory"]), rw, rh, cw)

    def analyze_face_boxes(self, path, sample_fps=6):
        return [(idx, [tuple(b) for b in boxes])
                for idx, boxes in FAKE_PLAN["boxes"]]

    def get_ffmpeg_crop_filter(self, smoothed_x, orig_fps=30.0,
                               crop_w=202, raw_h=360):
        avg = int(sum(smoothed_x) / len(smoothed_x))
        return f"crop={crop_w}:{raw_h}:{avg}:0,scale=1080:1920"

    def close(self):
        pass  # mirrors SpeakerTracker.close(); edit_clip calls it.


@pytest.fixture
def fake_tracker(monkeypatch):
    mod = types.ModuleType("speaker_tracker")
    mod.SpeakerTracker = FakeTracker
    monkeypatch.setitem(sys.modules, "speaker_tracker", mod)
    return mod


@pytest.fixture
def editor_setup(tmp_path, monkeypatch):
    """ClipEditor with logging + watermark stubbed (same shape as
    tests/test_banner_timing.py::editor_setup)."""
    monkeypatch.setattr(clip_editor, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)
    watermark = tmp_path / "wm.png"
    from PIL import Image
    Image.new("RGBA", (1082, 108), (0, 0, 0, 255)).save(watermark)
    monkeypatch.setattr(clip_editor, "generate_kick_watermark",
                        lambda streamer: str(watermark))
    return clip_editor.ClipEditor(), tmp_path


def _record_card_calls(editor, monkeypatch):
    """Wrap both card generators to record the y overrides they get,
    while still drawing the real PNGs."""
    recorded = {}
    real_banner = clip_editor.ClipEditor.generate_screen_caption_card
    real_quote = clip_editor.ClipEditor.generate_quote_card

    def banner_spy(text, output_path, position="top",
                   layout="blurred_backdrop", banner_y=None, theme=None):
        recorded["banner_y"] = banner_y
        return real_banner(editor, text, output_path, position=position,
                           layout=layout, banner_y=banner_y, theme=theme)

    def quote_spy(text, output_path, quote_y=None):
        recorded["quote_y"] = quote_y
        return real_quote(editor, text, output_path, quote_y=quote_y)

    monkeypatch.setattr(editor, "generate_screen_caption_card", banner_spy)
    monkeypatch.setattr(editor, "generate_quote_card", quote_spy)
    return recorded


def _make_synthetic_clip(path, duration=2):
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i",
         f"testsrc2=duration={duration}:size=640x360:rate=30",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-300:]


def _fake_ffmpeg(monkeypatch):
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(clip_editor.subprocess, "run", fake_run)
    return calls


def _render_filter(calls):
    renders = [c for c in calls if "-filter_complex" in c]
    assert len(renders) == 1
    return " ".join(renders[0])


# -- mirror: the crop filter must use the mirrored trajectory ---------

def test_edit_clip_mirror_uses_mirrored_trajectory(editor_setup,
                                                   fake_tracker,
                                                   monkeypatch):
    editor, tmp_path = editor_setup
    calls = _fake_ffmpeg(monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    # mirror=True (the default): trajectory 339 mirrors to 438-339=99.
    editor.edit_clip(str(raw), str(tmp_path / "out.mp4"), streamer="xqc",
                     layout="speaker_tracking", mirror=True,
                     generate_thumbnail=False)
    filt = _render_filter(calls)
    assert "crop=202:360:99:0" in filt


def test_edit_clip_no_mirror_uses_trajectory_as_is(editor_setup,
                                                   fake_tracker,
                                                   monkeypatch):
    editor, tmp_path = editor_setup
    calls = _fake_ffmpeg(monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    editor.edit_clip(str(raw), str(tmp_path / "out.mp4"), streamer="xqc",
                     layout="speaker_tracking", mirror=False,
                     generate_thumbnail=False)
    filt = _render_filter(calls)
    assert "crop=202:360:339:0" in filt


# -- placement: banner + quote dodge the face --------------------------

def test_edit_clip_places_banner_and_quote_away_from_face(
        editor_setup, fake_tracker, monkeypatch):
    """Face box (400,200,80,60) maps to output region ~(286,1029)-
    (794,1430): it covers the 800 banner slot and the 1240 quote
    band, so the banner goes to the top (260) and the quote moves
    up to 940. Fake ffmpeg: asserts the recorded placement."""
    editor, tmp_path = editor_setup
    calls = _fake_ffmpeg(monkeypatch)
    recorded = _record_card_calls(editor, monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    editor.edit_clip(str(raw), str(tmp_path / "out.mp4"), streamer="xqc",
                     layout="speaker_tracking", mirror=False,
                     hook_banner="TEST BANNER",
                     quote_line="Bro, I cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    assert recorded.get("banner_y") == 260
    assert recorded.get("quote_y") == 940
    assert "[with_card]" in _render_filter(calls)
    assert "[with_quote]" in _render_filter(calls)


def test_edit_clip_mirrored_placement_avoids_mirrored_face(
        editor_setup, fake_tracker, monkeypatch):
    """Same face, mirror=True: the face box and the trajectory are
    both mirrored, so the face lands in the same output region and
    the placement is identical — the banner dodges the face AS
    RENDERED, not as tracked. (Without the mirror fix the face
    would map outside the crop and the banner would stay at 800.)"""
    editor, tmp_path = editor_setup
    _fake_ffmpeg(monkeypatch)
    recorded = _record_card_calls(editor, monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    editor.edit_clip(str(raw), str(tmp_path / "out.mp4"), streamer="xqc",
                     layout="speaker_tracking", mirror=True,
                     hook_banner="TEST BANNER",
                     quote_line="Bro, I cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    assert recorded.get("banner_y") == 260
    assert recorded.get("quote_y") == 940


def test_edit_clip_no_face_data_keeps_fixed_positions(
        editor_setup, fake_tracker, monkeypatch):
    """No faces sampled: banner_y/quote_y stay None (fixed 800 /
    1240), exactly as before the feature."""
    monkeypatch.setitem(FAKE_PLAN, "boxes", [(0, []), (30, [])])
    editor, tmp_path = editor_setup
    _fake_ffmpeg(monkeypatch)
    recorded = _record_card_calls(editor, monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    editor.edit_clip(str(raw), str(tmp_path / "out.mp4"), streamer="xqc",
                     layout="speaker_tracking", mirror=False,
                     hook_banner="TEST BANNER",
                     quote_line="Bro, I cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    assert recorded.get("banner_y") is None
    assert recorded.get("quote_y") is None


def test_edit_clip_non_face_layouts_untouched(
        editor_setup, fake_tracker, monkeypatch):
    """blurred_backdrop / center_crop never run face analysis, so
    the placement overrides stay None — byte-identical behaviour."""
    editor, tmp_path = editor_setup
    _fake_ffmpeg(monkeypatch)
    recorded = _record_card_calls(editor, monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    for layout in ("blurred_backdrop", "center_crop"):
        recorded.clear()
        editor.edit_clip(str(raw), str(tmp_path / f"out_{layout}.mp4"),
                         streamer="xqc", layout=layout, mirror=True,
                         hook_banner="TEST BANNER",
                         quote_line="Bro, I cannot believe this",
                         quote_line_enabled=True,
                         generate_thumbnail=False)
        assert recorded.get("banner_y") is None, layout
        assert recorded.get("quote_y") is None, layout


def test_edit_clip_face_box_failure_falls_back_to_fixed(
        editor_setup, monkeypatch):
    """analyze_face_boxes raising must not fail the render: the
    banner keeps its fixed position."""
    class FlakyTracker(FakeTracker):
        def analyze_face_boxes(self, path, sample_fps=6):
            raise RuntimeError("detection exploded")

    mod = types.ModuleType("speaker_tracker")
    mod.SpeakerTracker = FlakyTracker
    monkeypatch.setitem(sys.modules, "speaker_tracker", mod)

    editor, tmp_path = editor_setup
    _fake_ffmpeg(monkeypatch)
    recorded = _record_card_calls(editor, monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    result = editor.edit_clip(str(raw), str(tmp_path / "out.mp4"),
                              streamer="xqc", layout="speaker_tracking",
                              mirror=False, hook_banner="TEST BANNER",
                              generate_thumbnail=False)
    assert result == str(tmp_path / "out.mp4")
    assert recorded.get("banner_y") is None


# -- real render ---------------------------------------------------------

@needs_ffmpeg
def test_edit_clip_face_avoidance_real_render(editor_setup, fake_tracker,
                                              monkeypatch):
    """End to end with real ffmpeg on a 2 s synthetic clip (mirror on,
    the default path): the render succeeds at 1080x1920 and the
    banner was placed at the face-dodging y, not the fixed 800."""
    editor, tmp_path = editor_setup
    recorded = _record_card_calls(editor, monkeypatch)
    src = tmp_path / "src.mp4"
    _make_synthetic_clip(src)
    out = tmp_path / "out.mp4"

    result = editor.edit_clip(str(src), str(out), streamer="xqc",
                              layout="speaker_tracking", mirror=True,
                              hook_banner="TEST BANNER",
                              quote_line="Bro, I cannot believe this",
                              quote_line_enabled=True,
                              generate_thumbnail=False)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0
    assert recorded.get("banner_y") == 260
    assert recorded.get("quote_y") == 940

    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "error",
         "-show_entries", "stream=codec_type,width,height",
         "-show_entries", "format=duration", "-of", "json", str(out)],
        capture_output=True, text=True, check=True, timeout=60)
    import json
    info = json.loads(probe.stdout)
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    assert len(video) == 1
    assert video[0]["width"] == 1080 and video[0]["height"] == 1920
    assert abs(float(info["format"]["duration"]) - 2.0) < 0.5
