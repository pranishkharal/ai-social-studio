"""Tests for render_modes.py — pure string builders, no ffmpeg needed."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from render_modes import (  # noqa: E402
    build_camera_switch_filter,
    build_split_screen_filter,
)

TRAJ_A = {"x": 100, "crop_w": 607, "raw_h": 1080}
TRAJ_B = {"x": 1200, "crop_w": 607, "raw_h": 1080}
TRAJ_MOVING = {"x": list(range(0, 900, 10)), "crop_w": 607, "raw_h": 1080}


def _trim_windows(filt):
    """Extract (start, end) pairs from every trim= in the filter string."""
    return [(float(s), float(e)) for s, e in
            re.findall(r"trim=start=([\d.]+):end=([\d.]+)", filt)]


# ---------------------------------------------------------------- camera switch

def test_camera_switch_basic_two_speakers():
    segments = [
        {"speaker": "alice", "start": 0.0, "end": 5.0},
        {"speaker": "bob", "start": 5.0, "end": 10.0},
    ]
    filt, label = build_camera_switch_filter(
        segments, {"alice": TRAJ_A, "bob": TRAJ_B})
    assert label == "[camswitch_out]"
    assert "crop=607:1080:100:0,scale=1080:1920[cam_alice]" in filt
    assert "crop=607:1080:1200:0,scale=1080:1920[cam_bob]" in filt
    assert "trim=start=0.00:end=5.00" in filt
    assert "trim=start=5.00:end=10.00" in filt
    assert "concat=n=2:v=1:a=0[camswitch_out]" in filt


def test_camera_switch_min_hold_merges_short_interjection():
    # Bob's 0.8 s interjection is shorter than the 2 s hold: it must be
    # absorbed into Alice's turn instead of causing a flicker cut.
    segments = [
        {"speaker": "alice", "start": 0.0, "end": 5.0},
        {"speaker": "bob", "start": 5.0, "end": 5.8},
        {"speaker": "alice", "start": 5.8, "end": 10.0},
    ]
    filt, _ = build_camera_switch_filter(
        segments, {"alice": TRAJ_A, "bob": TRAJ_B})
    windows = _trim_windows(filt)
    assert all(end - start >= 2.0 for start, end in windows)
    assert "cam_bob" not in filt  # bob never gets a camera of his own
    assert "concat=n=1" not in filt


def test_camera_switch_short_first_segment_merges_into_next():
    segments = [
        {"speaker": "bob", "start": 0.0, "end": 1.0},
        {"speaker": "alice", "start": 1.0, "end": 6.0},
    ]
    filt, _ = build_camera_switch_filter(
        segments, {"alice": TRAJ_A, "bob": TRAJ_B})
    windows = _trim_windows(filt)
    assert windows == [(0.0, 6.0)]
    assert "cam_bob" not in filt


def test_camera_switch_merges_consecutive_same_speaker_turns():
    segments = [
        {"speaker": "alice", "start": 0.0, "end": 3.0},
        {"speaker": "alice", "start": 3.2, "end": 6.0},  # 0.2 s gap
        {"speaker": "bob", "start": 6.0, "end": 9.0},
    ]
    filt, _ = build_camera_switch_filter(
        segments, {"alice": TRAJ_A, "bob": TRAJ_B})
    assert _trim_windows(filt) == [(0.0, 6.0), (6.0, 9.0)]


def test_camera_switch_moving_trajectory_uses_piecewise_expression():
    segments = [{"speaker": "alice", "start": 0.0, "end": 3.0}]
    filt, _ = build_camera_switch_filter(segments, {"alice": TRAJ_MOVING})
    assert "if(lt(t," in filt
    assert "crop=607:1080:'" in filt


def test_camera_switch_single_segment_no_concat():
    segments = [{"speaker": "alice", "start": 0.0, "end": 5.0}]
    filt, label = build_camera_switch_filter(segments, {"alice": TRAJ_A})
    assert "concat=" not in filt
    assert filt.endswith("[camswitch_out]")
    assert label == "[camswitch_out]"


def test_camera_switch_custom_min_hold():
    segments = [
        {"speaker": "alice", "start": 0.0, "end": 5.0},
        {"speaker": "bob", "start": 5.0, "end": 6.0},  # 1 s turn
        {"speaker": "alice", "start": 6.0, "end": 10.0},
    ]
    filt, _ = build_camera_switch_filter(
        segments, {"alice": TRAJ_A, "bob": TRAJ_B}, min_hold=0.5)
    assert "cam_bob" in filt  # 1 s is fine with a 0.5 s hold


def test_camera_switch_unknown_speaker_raises():
    with pytest.raises(ValueError, match="No trajectory provided for speaker 'bob'"):
        build_camera_switch_filter(
            [{"speaker": "bob", "start": 0.0, "end": 3.0}], {"alice": TRAJ_A})


def test_camera_switch_empty_segments_raises():
    with pytest.raises(ValueError, match="segments must not be empty"):
        build_camera_switch_filter([], {"alice": TRAJ_A})


def test_camera_switch_centred_crop_when_x_omitted():
    segments = [{"speaker": "alice", "start": 0.0, "end": 3.0}]
    filt, _ = build_camera_switch_filter(
        segments, {"alice": {"crop_w": 607, "raw_h": 1080}})
    assert "crop=607:1080:(iw-607)/2:0" in filt


# ---------------------------------------------------------------- split screen

def _panel(name, x):
    return {"speaker": name, "x": x, "crop_w": 607, "raw_h": 1080}


def test_split_screen_two_panels_with_active_highlight():
    filt, label = build_split_screen_filter(
        [_panel("alice", 100), _panel("bob", 1200)], active_speaker="alice")
    assert label == "[split_out]"
    # Panels are (1920 - 4) // 2 = 958 px tall.
    assert "scale=1080:958[split_p0]" in filt
    assert "scale=1080:958[split_p1]" in filt
    # Active panel: 3 px cyan border. Inactive: darkened 15%.
    assert "drawbox=x=0:y=0:w=iw:h=ih:color=0x00FFFF:t=3" in filt
    assert "eq=brightness=-0.15" in filt
    # Grey divider + vstack of top/divider/bottom.
    assert "color=c=0x808080:s=1080x4[split_div]" in filt
    assert "vstack=inputs=3[split_out]" in filt


def test_split_screen_no_active_speaker_is_neutral():
    filt, _ = build_split_screen_filter([_panel("alice", 100), _panel("bob", 1200)])
    assert "drawbox" not in filt
    assert "eq=brightness" not in filt
    assert "vstack=inputs=3[split_out]" in filt


def test_split_screen_single_speaker_full_screen():
    filt, label = build_split_screen_filter([_panel("alice", 100)])
    assert "vstack" not in filt
    assert "crop=607:1080:100:0,scale=1080:1920[split_out]" in filt
    assert label == "[split_out]"


def test_split_screen_ignores_third_speaker():
    filt, _ = build_split_screen_filter(
        [_panel("alice", 100), _panel("bob", 1200), _panel("carol", 600)])
    assert filt.count("crop=607:1080") == 2


def test_split_screen_empty_raises():
    with pytest.raises(ValueError, match="speakers must not be empty"):
        build_split_screen_filter([])
