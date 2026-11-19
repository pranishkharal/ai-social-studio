"""Tests for thumbnail.py (feature 10: auto-thumbnails) + ClipEditor wiring."""

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# dotenv is not installed in this VM; stub it before importing repo modules
# that load config (clip_editor -> config -> dotenv).
if "dotenv" not in sys.modules:
    _dotenv = types.ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = _dotenv

import thumbnail
from thumbnail import (
    estimate_peak_time,
    generate_thumbnail,
    parse_astats_rms_by_window,
    pick_thumbnail_time,
)

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def make_test_video(path, duration=4, with_audio=True, loud_second_half=False):
    """Generate a tiny mp4 with ffmpeg (testsrc + optional sine audio)."""
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=640x360:rate=10"]
    if with_audio:
        if loud_second_half:
            # Quiet first half, loud second half: peak should be ~3s.
            cmd += ["-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={duration}",
                    "-af", f"volume='if(lt(t\\,{duration / 2}),0.05,1)':eval=frame"]
        else:
            cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-300:]
    return path


# --- astats parsing ---

SAMPLE_ASTATS = """\
[Parsed_ametadata_1 @ 0x123] frame:0    pts:0       pts_time:0
[Parsed_ametadata_1 @ 0x123] lavfi.astats.Overall.RMS_level=-30.123456
[Parsed_ametadata_1 @ 0x123] frame:100  pts:44100   pts_time:1
[Parsed_ametadata_1 @ 0x123] lavfi.astats.Overall.RMS_level=-10.5
[Parsed_ametadata_1 @ 0x123] frame:200  pts:88200   pts_time:2
[Parsed_ametadata_1 @ 0x123] lavfi.astats.Overall.RMS_level=-inf
[Parsed_ametadata_1 @ 0x123] frame:300  pts:132300  pts_time:2.5
[Parsed_ametadata_1 @ 0x123] lavfi.astats.Overall.RMS_level=-20.0
"""


def test_parse_astats_rms_by_window():
    result = parse_astats_rms_by_window(SAMPLE_ASTATS)
    assert result == {0.0: -30.123456, 1.0: -10.5, 2.0: -20.0}


def test_parse_astats_skips_silence_and_garbage():
    assert parse_astats_rms_by_window("") == {}
    assert parse_astats_rms_by_window("no astats here\n") == {}
    assert parse_astats_rms_by_window(None) == {}


def test_estimate_peak_time_picks_loudest_window(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")  # only needs to exist; ffmpeg call is mocked
    monkeypatch.setattr(
        thumbnail, "_run",
        lambda cmd, timeout=120: subprocess.CompletedProcess(cmd, 0, "", SAMPLE_ASTATS))
    # Loudest window is 1.0-2.0s (-10.5 dB) -> midpoint 1.5s
    assert estimate_peak_time(str(video)) == 1.5


def test_estimate_peak_time_returns_none_on_failure(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(
        thumbnail, "_run",
        lambda cmd, timeout=120: subprocess.CompletedProcess(cmd, 1, "", "boom"))
    assert estimate_peak_time(str(video)) is None
    assert estimate_peak_time(str(tmp_path / "missing.mp4")) is None


# --- frame-time picking ---

def test_pick_thumbnail_time_explicit_peak_wins(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(thumbnail, "estimate_peak_time", lambda p: 9.9)
    assert pick_thumbnail_time(str(video), peak_time=3.25) == 3.25


def test_pick_thumbnail_time_falls_back_to_words_then_middle(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(thumbnail, "estimate_peak_time", lambda p: None)
    monkeypatch.setattr(thumbnail, "_largest_face_time", lambda p, d: None)
    monkeypatch.setattr(thumbnail, "probe_duration", lambda p: 10.0)
    words = [{"word": "a", "start": 6.1, "end": 6.4},
             {"word": "b", "start": 6.5, "end": 6.8},
             {"word": "c", "start": 1.0, "end": 1.3}]
    # Densest 1s window is 6-7s -> midpoint 6.5s
    assert pick_thumbnail_time(str(video), words=words) == 6.5
    # No words -> middle frame
    assert pick_thumbnail_time(str(video)) == 5.0


def test_largest_face_time_skipped_without_cv2(tmp_path, monkeypatch):
    # Simulate a missing cv2 explicitly: another test module installs a
    # fake cv2 in this pytest process, so absence must not be assumed.
    # (sys.modules["cv2"] = None makes `import cv2` raise ImportError.)
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert thumbnail._largest_face_time(str(tmp_path / "v.mp4"), 10.0) is None


# --- generate_thumbnail (real ffmpeg) ---

@needs_ffmpeg
def test_generate_thumbnail_real_video(tmp_path):
    video = make_test_video(tmp_path / "clip.mp4", duration=4)
    out = tmp_path / "clip_thumb.jpg"
    result = generate_thumbnail(str(video), "XQC was SHOCKED after this happened",
                                str(out))
    assert result == str(out)
    assert out.exists()
    from PIL import Image
    img = Image.open(out)
    assert img.format == "JPEG"
    # Source frame is 640x360 landscape; the thumbnail is cropped to
    # 9:16 portrait like the clip (round(360 * 9 / 16) = 202 wide).
    assert img.size == (202, 360)


@needs_ffmpeg
def test_generate_thumbnail_picks_loud_moment(tmp_path):
    video = make_test_video(tmp_path / "clip.mp4", duration=4, loud_second_half=True)
    peak = estimate_peak_time(str(video))
    assert peak is not None
    assert peak >= 2.0  # loud half wins
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(str(video), "LOUD MOMENT", str(out))
    assert out.exists()


@needs_ffmpeg
def test_generate_thumbnail_no_audio_falls_back(tmp_path):
    video = make_test_video(tmp_path / "silent.mp4", duration=4, with_audio=False)
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(str(video), "", str(out))  # empty hook text: clean frame
    assert out.exists()


def test_generate_thumbnail_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_thumbnail(str(tmp_path / "nope.mp4"), "hook", str(tmp_path / "t.jpg"))


# --- 9:16 thumbnail crop: the clean source frame is 16:9, the clip is 9:16 ---

def _striped_image():
    """640x360 image: red left third, green middle, blue right third."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 360), (0, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 212, 359], fill=(255, 0, 0))
    draw.rectangle([427, 0, 639, 359], fill=(0, 0, 255))
    return img


def test_crop_to_vertical_centers_without_face():
    # 640x360 -> 202x360 centred: left edge at (640-202)//2 = 219,
    # which is inside the green middle stripe.
    cropped = thumbnail._crop_to_vertical(_striped_image())
    assert cropped.size == (202, 360)
    assert cropped.getpixel((0, 180)) == (0, 255, 0)


def test_crop_to_vertical_centers_on_face_and_clamps():
    img = _striped_image()
    # Face at x=100 (left/red stripe): window centred on it, clamped
    # to the left edge -> leftmost pixel is red.
    cropped = thumbnail._crop_to_vertical(img, center_x=100)
    assert cropped.size == (202, 360)
    assert cropped.getpixel((0, 180)) == (255, 0, 0)
    # Face at x=600 (right/blue stripe): window clamped to the right
    # edge -> rightmost pixel is blue.
    cropped = thumbnail._crop_to_vertical(img, center_x=600)
    assert cropped.getpixel((201, 180)) == (0, 0, 255)


def test_crop_to_vertical_leaves_portrait_frames_alone():
    from PIL import Image
    img = Image.new("RGB", (360, 640), (255, 0, 0))
    assert thumbnail._crop_to_vertical(img).size == (360, 640)


def test_face_box_none_without_cv2(tmp_path, monkeypatch):
    # Same simulation as test_largest_face_time_skipped_without_cv2:
    # no OpenCV means no face centring (the crop falls back to centre)
    # and no text push-down (the band stays at the top).
    monkeypatch.setitem(sys.modules, "cv2", None)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"not a real image")
    assert thumbnail._largest_face_box(str(frame)) is None


def test_overlay_hook_text_stays_below_face():
    """With min_y set, the text band starts below the face, not over it."""
    from PIL import Image
    img = Image.new("RGB", (608, 1080), (255, 255, 255))
    out = thumbnail._overlay_hook_text(img, "HOOK TEXT HERE", min_y=300)
    # Above the band: still white (the face area stays clear).
    assert out.getpixel((304, 100)) == (255, 255, 255)
    # Inside the band (just below min_y, away from the glyphs):
    # the semi-transparent black band darkens the white frame.
    r, g, b = out.getpixel((304, 315))
    assert r < 150 and g < 150 and b < 150


def test_overlay_hook_text_top_when_no_room_below_face():
    """When the face is too low for the band to fit below it, the band
    stays at the top instead of running off the bottom."""
    from PIL import Image
    img = Image.new("RGB", (608, 1080), (255, 255, 255))
    out = thumbnail._overlay_hook_text(img, "HOOK TEXT HERE", min_y=1000)
    # Top placement (6% of 1080 = 64): the band darkens y ~70.
    r, g, b = out.getpixel((304, 70))
    assert r < 150 and g < 150 and b < 150


@needs_ffmpeg
def test_generate_thumbnail_from_16x9_source_is_portrait(tmp_path):
    # The clean-frame path (bug #4 fix): the source is 16:9 but the
    # thumbnail must still be 9:16 portrait, with the hook text as the
    # only text on it.
    source = _make_solid_video(tmp_path / "source.mp4", "red")
    rendered = _make_solid_video(tmp_path / "rendered.mp4", "blue")
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(str(rendered), "HOOK TEXT HERE", str(out),
                       peak_time=1.0, frame_source_path=str(source))
    from PIL import Image
    img = Image.open(out)
    assert img.size == (202, 360)
    assert _corner_pixel_is(out, "red")


# --- clean-frame thumbnails: one frame, one text box ---

def _make_solid_video(path, color, duration=2):
    """A solid-colour mp4 (lavfi color + sine audio)."""
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i", f"color=c={color}:duration={duration}:size=640x360:rate=10",
           "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-shortest", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-300:]
    return path


def _corner_pixel_is(image_path, channel):
    """Bottom-left pixel (far from the top text box) dominant channel."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    r, g, b = img.getpixel((10, img.height - 10))
    if channel == "red":
        return r > 150 and r > b + 50
    return b > 150 and b > r + 50


@needs_ffmpeg
def test_generate_thumbnail_uses_clean_frame_source(tmp_path):
    # Rendered clip is blue; the clean pre-overlay source is red. The
    # thumbnail frame must come from the red source, with the hook text
    # box as the only text on it.
    source = _make_solid_video(tmp_path / "source.mp4", "red")
    rendered = _make_solid_video(tmp_path / "rendered.mp4", "blue")
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(str(rendered), "HOOK TEXT HERE", str(out),
                       peak_time=1.0, frame_source_path=str(source))
    assert out.exists()
    assert _corner_pixel_is(out, "red")


@needs_ffmpeg
def test_generate_thumbnail_without_frame_source_uses_render(tmp_path):
    rendered = _make_solid_video(tmp_path / "rendered.mp4", "blue")
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(str(rendered), "HOOK TEXT HERE", str(out), peak_time=1.0)
    assert out.exists()
    assert _corner_pixel_is(out, "blue")


@needs_ffmpeg
def test_generate_thumbnail_missing_frame_source_falls_back(tmp_path):
    rendered = _make_solid_video(tmp_path / "rendered.mp4", "blue")
    out = tmp_path / "thumb.jpg"
    generate_thumbnail(str(rendered), "HOOK TEXT HERE", str(out), peak_time=1.0,
                       frame_source_path=str(tmp_path / "gone.mp4"))
    assert out.exists()
    assert _corner_pixel_is(out, "blue")


# --- hook banner position: face-tracking layouts sit below the facecam ---

def test_caption_card_lower_for_face_tracking_layouts(editor_setup):
    editor, raw, tmp_path, _, _ = editor_setup
    from PIL import Image

    def card_top(layout):
        card = tmp_path / f"card_{layout}.png"
        editor.generate_screen_caption_card(
            "XQC was SHOCKED after this happened live", str(card),
            position="top", layout=layout)
        bbox = Image.open(card).getbbox()
        assert bbox is not None
        return bbox[1]

    # Classic / blurred backdrop: banner near the top (below TikTok tabs).
    assert card_top("blurred_backdrop") < 500
    assert card_top("center_crop") < 500
    # Face-tracking layouts: banner below the facecam zone (y 0-760).
    assert card_top("speaker_tracking") >= 750
    assert card_top("camera_switch") >= 750
    assert card_top("split_screen") >= 750


# --- ClipEditor wiring (feature 10 + 9 + 11 integration) ---

@pytest.fixture
def editor_setup(tmp_path, monkeypatch):
    """ClipEditor with ffmpeg, logging and watermark stubbed out."""
    import clip_editor

    logs = []
    monkeypatch.setattr(clip_editor, "log_event",
                        lambda level, src, msg: logs.append((level, src, msg)))
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)

    watermark = tmp_path / "wm.png"
    from PIL import Image
    Image.new("RGBA", (1082, 108), (0, 0, 0, 255)).save(watermark)
    monkeypatch.setattr(clip_editor, "generate_kick_watermark",
                        lambda streamer: str(watermark))

    ffmpeg_calls = []

    def fake_run(cmd, capture_output=True, text=True, creationflags=0, timeout=180):
        ffmpeg_calls.append(list(cmd))
        # Pretend ffmpeg wrote the output — but only for real render
        # calls; encoder probes (`ffmpeg -encoders`) have no output file
        # and must not create a stray file named after their last arg.
        if str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(clip_editor.subprocess, "run", fake_run)

    editor = clip_editor.ClipEditor()
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    return editor, raw, tmp_path, ffmpeg_calls, logs


def _render_calls(ffmpeg_calls):
    """The actual render invocations, excluding encoder-detection probes
    (edit_clip runs `ffmpeg -encoders` once per process to auto-detect
    the hardware encoder, so the probe may or may not appear depending
    on whether the encoder cache is already warm)."""
    return [c for c in ffmpeg_calls if "-filter_complex" in c]


def test_edit_clip_profile_layout_when_none(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls, _ = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="westcol", layout=None,
                     generate_thumbnail=False)
    renders = _render_calls(ffmpeg_calls)
    assert len(renders) == 1
    filter_str = " ".join(renders[0])
    # westcol profile layout is "split" -> vstack in the filter chain
    assert "vstack" in filter_str


def test_edit_clip_default_layout_for_unknown_streamer(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls, _ = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="somebody_new", layout=None,
                     generate_thumbnail=False)
    filter_str = " ".join(_render_calls(ffmpeg_calls)[0])
    # Default profile layout is blurred_backdrop -> boxblur in the filter
    assert "boxblur" in filter_str


def test_edit_clip_explicit_layout_ignores_profile(editor_setup, monkeypatch):
    import streamer_profiles
    editor, raw, tmp_path, ffmpeg_calls, _ = editor_setup

    def _boom(streamer, path=None):
        raise AssertionError("profile must not be consulted for explicit layout")

    monkeypatch.setattr(streamer_profiles, "get_profile", _boom)
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="westcol", layout="center_crop",
                     generate_thumbnail=False)
    filter_str = " ".join(_render_calls(ffmpeg_calls)[0])
    assert "vstack" not in filter_str  # not the westcol "split" profile layout


def test_edit_clip_silence_trim_called_with_words(editor_setup, monkeypatch):
    import silence_cutter
    editor, raw, tmp_path, ffmpeg_calls, logs = editor_setup

    trimmed = tmp_path / "trimmed.mp4"
    trimmed.write_bytes(b"trimmed")
    calls = []

    def fake_cut(input_path, words, output_path, max_gap=None, pad=None):
        calls.append((input_path, words, output_path))
        return output_path, 2.5

    monkeypatch.setattr(silence_cutter, "cut_silence", fake_cut)
    out = tmp_path / "edited.mp4"
    words = [{"word": "a", "start": 0.0, "end": 1.0}]
    editor.edit_clip(str(raw), str(out), streamer="adinross", layout="center_crop",
                     words=words, silence_trim=True, generate_thumbnail=False)
    assert len(calls) == 1
    assert calls[0][0] == str(raw)
    assert calls[0][1] == words
    # ffmpeg renders from the trimmed file, not the raw clip
    render = _render_calls(ffmpeg_calls)[0]
    assert render[render.index("-i") + 1] == calls[0][2]
    assert any("removed 2.5s" in msg for _, _, msg in logs)


def test_edit_clip_no_words_means_no_silence_call(editor_setup, monkeypatch):
    import silence_cutter
    editor, raw, tmp_path, ffmpeg_calls, _ = editor_setup

    def _boom(*a, **k):
        raise AssertionError("cut_silence must not run without words")

    monkeypatch.setattr(silence_cutter, "cut_silence", _boom)
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross", layout="center_crop",
                     generate_thumbnail=False)
    render = _render_calls(ffmpeg_calls)[0]
    assert render[render.index("-i") + 1] == str(raw)


def test_edit_clip_silence_trim_disabled(editor_setup, monkeypatch):
    import silence_cutter
    editor, raw, tmp_path, ffmpeg_calls, _ = editor_setup

    def _boom(*a, **k):
        raise AssertionError("cut_silence must not run when silence_trim=False")

    monkeypatch.setattr(silence_cutter, "cut_silence", _boom)
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross", layout="center_crop",
                     words=[{"word": "a", "start": 0.0, "end": 1.0}],
                     silence_trim=False, generate_thumbnail=False)
    render = _render_calls(ffmpeg_calls)[0]
    assert render[render.index("-i") + 1] == str(raw)


def test_edit_clip_generates_thumbnail_by_default(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls, logs = editor_setup
    calls = []
    monkeypatch.setattr(
        thumbnail, "generate_thumbnail",
        lambda video, hook, out, words=None, peak_time=None,
        frame_source_path=None: calls.append(
            (video, hook, out, frame_source_path)))
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross", layout="center_crop",
                     hook_banner="HOOK TEXT HERE")
    assert len(calls) == 1
    assert calls[0][0] == str(out)  # thumbnail is made for the rendered clip
    assert calls[0][1] == "HOOK TEXT HERE"
    assert calls[0][2] == str(tmp_path / "edited_thumb.jpg")
    # ...but its frame comes from the clean pre-overlay source clip, so
    # the thumbnail's text box never lands on the banner already burned
    # into the render (one frame, one text box).
    assert calls[0][3] == str(raw)
    assert any("Thumbnail saved" in msg for _, _, msg in logs)


def test_edit_clip_thumbnail_opt_out(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls, _ = editor_setup

    def _boom(*a, **k):
        raise AssertionError("thumbnail must not run when generate_thumbnail=False")

    monkeypatch.setattr(thumbnail, "generate_thumbnail", _boom)
    out = tmp_path / "edited.mp4"
    result = editor.edit_clip(str(raw), str(out), streamer="adinross",
                              layout="center_crop", generate_thumbnail=False)
    assert result == str(out)


def test_edit_clip_thumbnail_failure_never_fails_render(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls, logs = editor_setup
    monkeypatch.setattr(
        thumbnail, "generate_thumbnail",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("ffmpeg exploded")))
    out = tmp_path / "edited.mp4"
    result = editor.edit_clip(str(raw), str(out), streamer="adinross",
                              layout="center_crop", generate_thumbnail=True)
    assert result == str(out)
    assert any("Thumbnail generation failed" in msg for _, _, msg in logs)


def test_caption_card_renders_on_linux(editor_setup):
    editor, raw, tmp_path, _, _ = editor_setup
    card = tmp_path / "card.png"
    result = editor.generate_screen_caption_card(
        "XQC was SHOCKED after this happened live 😳", str(card),
        position="top", layout="blurred_backdrop")
    assert result == str(card)
    assert card.exists()
    from PIL import Image
    img = Image.open(card)
    assert img.size == (1080, 1920)


def test_font_helpers_never_crash(monkeypatch):
    import clip_editor
    font = clip_editor.load_font(38, bold=True)
    assert font is not None
    emoji_font = clip_editor.load_font(36, emoji=True)
    assert emoji_font is not None
    # On this Linux box the resolver finds a real Linux font for the
    # Windows font names (that is the intended cross-platform fallback).
    found = clip_editor.resolve_font_path(["arialbd.ttf", "arial.ttf"])
    assert found is not None and Path(found).exists()
    # With no font file available at all, load_font falls back to
    # Pillow's built-in default instead of crashing.
    monkeypatch.setattr(clip_editor, "resolve_font_path", lambda names: None)
    fallback = clip_editor.load_font(38, bold=True)
    assert fallback is not None
