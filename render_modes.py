"""
render_modes.py — Camera-switch and split-screen ffmpeg filter builders.

These functions are PURE STRING BUILDERS: they build ffmpeg ``filter_complex``
strings and never execute ffmpeg, so they are fully testable offline. The
coordinator wires the returned filter into ClipEditor.edit_clip().

Patterns adapted from opensource-clipping
(https://github.com/NaufalRizqullah/opensource-clipping, MIT licence):
clipping/studio/render_camera_switch.py (minimum 2.0 s hold before switching
speakers, canonical per-speaker crop) and
clipping/studio/render_split_screen.py (vertically stacked panels, 4 px
divider, 3 px active-speaker border, inactive panel darkened 15%).

Shared input contract — a *trajectory* describes how to crop one speaker
from the source frame into a 9:16 panel:

    trajectory = {
        "x": int | list[int],   # crop_x: a static pixel value, or one
                                # crop_x per source frame (as returned by
                                # SpeakerTracker.analyze_video_trajectory).
                                # Optional: omit for a centred crop.
        "crop_w": int,          # crop width in source pixels (required)
        "raw_h": int,           # source frame height in pixels (required)
    }

Both builders return ``(filter_complex_string, output_label)``. The filter
is VIDEO ONLY: audio is untouched, so the caller still maps it separately
(e.g. ``-map 0:a``). The returned output label (e.g. ``[camswitch_out]``)
is the video stream to ``-map`` in the ffmpeg command.
"""

import re

DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_MIN_HOLD = 2.0       # seconds; matches opensource-clipping's switch hold
DEFAULT_FPS = 30.0
DIVIDER_HEIGHT = 4           # px divider between split-screen panels
ACTIVE_BORDER = 3            # px highlight border for the active speaker
ACTIVE_BORDER_COLOR = "0x00FFFF"  # cyan, as in opensource-clipping
INACTIVE_DARKEN = 0.15       # inactive panel darkened by 15%
SAME_SPEAKER_MERGE_GAP = 0.5  # seconds; merge same-speaker turns this close
KEYFRAME_STEP_SECONDS = 1.5  # piecewise crop keyframes, as in SpeakerTracker


def _safe_label(name: str) -> str:
    """Make a string safe to use inside an ffmpeg filter label."""
    label = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not label or label[0].isdigit():
        label = "s_" + label
    return label


def _fmt_time(t: float) -> str:
    """Format a timestamp for ffmpeg trim/xfade (2 decimals, no trailing junk)."""
    return f"{float(t):.2f}"


def _piecewise_x_expr(x_values, fps: float) -> str:
    """
    Build a nested ``if(lt(t,...))`` expression that pans crop_x through the
    per-frame trajectory, sampling keyframes every KEYFRAME_STEP_SECONDS.
    Same algorithm as SpeakerTracker.get_ffmpeg_crop_filter(). A nearly
    static trajectory collapses to a single constant.
    """
    values = [int(v) for v in x_values]
    if not values:
        raise ValueError("Trajectory x list must not be empty")
    if len(values) == 1 or (max(values) - min(values)) < 25:
        return str(int(sum(values) / len(values)))
    step = max(1, int(fps * KEYFRAME_STEP_SECONDS))
    keyframes = []
    for i in range(0, len(values), step):
        keyframes.append((round(i / fps, 2), values[i]))
    if keyframes[-1][0] < (len(values) - 1) / fps:
        keyframes.append((round((len(values) - 1) / fps, 2), values[-1]))
    expr = f"{keyframes[-1][1]}"
    for k in range(len(keyframes) - 2, -1, -1):
        t_start, x_start = keyframes[k]
        t_end, x_end = keyframes[k + 1]
        slope = (x_end - x_start) / max(0.01, (t_end - t_start))
        current_seg = f"({x_start}+{slope:.4f}*(t-{t_start}))"
        expr = f"if(lt(t,{t_end}),{current_seg},{expr})"
    return expr


def _crop_filter(source_label: str, trajectory: dict, out_label: str,
                 width: int, height: int, fps: float) -> str:
    """
    Build one ``[src]crop=...:...,scale=WxH[out]`` chain for a trajectory.
    The crop uses absolute source timestamps, so it must run BEFORE any
    trim/setpts in the chain.
    """
    crop_w = int(trajectory["crop_w"])
    raw_h = int(trajectory["raw_h"])
    x = trajectory.get("x")
    if x is None:
        x_expr = f"(iw-{crop_w})/2"
    elif isinstance(x, (list, tuple)):
        x_expr = f"'{_piecewise_x_expr(x, fps)}'"
    else:
        x_expr = str(int(x))
    return f"{source_label}crop={crop_w}:{raw_h}:{x_expr}:0,scale={width}:{height}{out_label}"


def _enforce_min_hold(segments, min_hold: float):
    """
    Merge consecutive same-speaker turns, then enforce the minimum hold:
    a turn shorter than min_hold is absorbed into the previous turn (or
    into the next turn when it is the first). This is what stops the
    camera flickering between speakers on short interjections.
    Returns a new list of {"speaker", "start", "end"} dicts.
    """
    ordered = sorted(
        ({"speaker": str(s["speaker"]), "start": float(s["start"]), "end": float(s["end"])}
         for s in segments),
        key=lambda s: s["start"],
    )
    merged = []
    for seg in ordered:
        if seg["end"] <= seg["start"]:
            continue
        if (merged and merged[-1]["speaker"] == seg["speaker"]
                and seg["start"] - merged[-1]["end"] <= SAME_SPEAKER_MERGE_GAP):
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))
    held = []
    for seg in merged:
        if seg["end"] - seg["start"] >= min_hold or not held:
            held.append(dict(seg))
        else:
            held[-1]["end"] = seg["end"]
    if len(held) > 1 and held[0]["end"] - held[0]["start"] < min_hold:
        held[1]["start"] = held[0]["start"]
        held.pop(0)
    return held


def build_camera_switch_filter(segments, trajectories, width: int = DEFAULT_WIDTH,
                               height: int = DEFAULT_HEIGHT, min_hold: float = DEFAULT_MIN_HOLD,
                               fps: float = DEFAULT_FPS, source_label: str = "[0:v]"):
    """
    Build a camera-switch filter: the full 9:16 crop cuts to whoever is
    speaking, holding each speaker for at least ``min_hold`` seconds.

    Args:
        segments: list of {"speaker": str, "start": float, "end": float}
            voice-activity turns (e.g. from diarization), in seconds on the
            source timeline. Turns shorter than min_hold are merged away.
        trajectories: dict mapping speaker name -> trajectory dict (see
            module docstring). Every speaker in ``segments`` must have one.
        width / height: output canvas size (default 1080x1920).
        min_hold: minimum seconds before switching speakers (default 2.0).
        fps: source frame rate, used to interpret per-frame trajectories.
        source_label: ffmpeg label of the source video stream (default "[0:v]").

    Returns:
        (filter_complex_string, "[camswitch_out]").

    Wiring: pass the filter as ``-filter_complex`` and ``-map
    "[camswitch_out]"`` for video plus ``-map 0:a`` for audio (audio is
    untouched — the video cuts are sample-accurate to the segment times).

    A speaker used in more than one segment gets their cropped stream
    ``split`` into one copy per segment: an ffmpeg filter output may
    only be consumed once, so feeding the same ``[cam_<speaker>]``
    label into two trims fails at runtime on ffmpeg 8 ("Error
    reinitializing filters"). Each trim also carries ``setsar=1`` so
    the concat sees identical sample aspect ratios.
    """
    if not segments:
        raise ValueError("segments must not be empty")
    held = _enforce_min_hold(segments, min_hold)
    if not held:
        raise ValueError("segments contain no usable (positive-length) turns")
    for seg in held:
        if seg["speaker"] not in trajectories:
            raise ValueError(f"No trajectory provided for speaker {seg['speaker']!r}")

    # How many segments each speaker appears in (after min-hold merging).
    use_count = {}
    for seg in held:
        use_count[seg["speaker"]] = use_count.get(seg["speaker"], 0) + 1

    parts = []
    cam_labels = {}
    for seg in held:
        speaker = seg["speaker"]
        if speaker not in cam_labels:
            label = f"[cam_{_safe_label(speaker)}]"
            parts.append(_crop_filter(source_label, trajectories[speaker], label,
                                      width, height, fps))
            if use_count[speaker] > 1:
                # One split copy per segment that uses this camera.
                copies = [f"[cam_{_safe_label(speaker)}_{i}]"
                          for i in range(use_count[speaker])]
                parts.append(
                    f"{label}split={use_count[speaker]}{''.join(copies)}"
                )
                cam_labels[speaker] = copies
            else:
                cam_labels[speaker] = [label]

    seg_labels = []
    use_index = {}
    for i, seg in enumerate(held):
        speaker = seg["speaker"]
        idx = use_index.get(speaker, 0)
        use_index[speaker] = idx + 1
        label = f"[csw_seg{i}]"
        parts.append(
            f"{cam_labels[speaker][idx]}trim=start={_fmt_time(seg['start'])}"
            f":end={_fmt_time(seg['end'])},setpts=PTS-STARTPTS,setsar=1{label}"
        )
        seg_labels.append(label)

    out_label = "[camswitch_out]"
    if len(seg_labels) == 1:
        # Single turn: rename the trimmed stream to the output label.
        parts[-1] = parts[-1].replace(seg_labels[0], out_label)
    else:
        parts.append(f"{''.join(seg_labels)}concat=n={len(seg_labels)}:v=1:a=0{out_label}")
    return ";".join(parts), out_label


def build_split_screen_filter(speakers, active_speaker=None, width: int = DEFAULT_WIDTH,
                              height: int = DEFAULT_HEIGHT, divider: int = DIVIDER_HEIGHT,
                              fps: float = DEFAULT_FPS, source_label: str = "[0:v]"):
    """
    Build a split-screen filter: two speakers in vertically stacked panels
    on the 9:16 canvas, with the active speaker highlighted.

    Args:
        speakers: list of speaker dicts, each a trajectory dict (see module
            docstring) plus a "speaker" name key:
            {"speaker": str, "x": ..., "crop_w": int, "raw_h": int}.
            The first entry is the TOP panel, the second the BOTTOM panel.
            Only the first two are used; extra speakers are ignored.
            With a single speaker this degrades gracefully to a full-screen
            crop of that speaker.
        active_speaker: name of the currently speaking speaker (or None).
            The active panel gets a 3 px cyan border; the inactive panel is
            darkened 15%. With None, both panels are neutral.
        width / height: output canvas size (default 1080x1920).
        divider: divider height in px between the panels (default 4).
        fps: source frame rate, used to interpret per-frame trajectories.
        source_label: ffmpeg label of the source video stream (default "[0:v]").

    Returns:
        (filter_complex_string, "[split_out]").

    Wiring: pass the filter as ``-filter_complex`` and ``-map "[split_out]"``
    for video plus ``-map 0:a`` for audio. The highlight is static per
    render; for a highlight that follows the conversation, render one
    split-screen pass per active-speaker turn and concat the passes.
    """
    if not speakers:
        raise ValueError("speakers must not be empty")
    panels = list(speakers)[:2]
    out_label = "[split_out]"

    if len(panels) == 1:
        filt = _crop_filter(source_label, panels[0], out_label, width, height, fps)
        return filt, out_label

    panel_h = (height - divider) // 2
    parts = []
    panel_labels = []
    for i, panel in enumerate(panels):
        label = f"[split_p{i}]"
        crop = _crop_filter(source_label, panel, label, width, panel_h, fps)
        parts.append(crop)
        name = panel.get("speaker")
        if active_speaker is not None and name == active_speaker:
            parts.append(
                f"{label}drawbox=x=0:y=0:w=iw:h=ih:color={ACTIVE_BORDER_COLOR}"
                f":t={ACTIVE_BORDER}[split_p{i}_hl]"
            )
            panel_labels.append(f"[split_p{i}_hl]")
        elif active_speaker is not None:
            parts.append(
                f"{label}eq=brightness=-{INACTIVE_DARKEN}[split_p{i}_dim]"
            )
            panel_labels.append(f"[split_p{i}_dim]")
        else:
            panel_labels.append(label)

    parts.append(
        f"color=c=0x808080:s={width}x{divider}[split_div]"
    )
    parts.append(
        f"{panel_labels[0]}[split_div]{panel_labels[1]}vstack=inputs=3{out_label}"
    )
    return ";".join(parts), out_label
