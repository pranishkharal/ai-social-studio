"""
clip_formats.py — Multi-part Short formats: callback, compilation,
escalation, then-vs-now, reveal, chat, wait-for-it.

Sara's app makes one-moment clips only. Callbacks and compilations are
the formats that go most viral on clip channels — and they reuse moments
the engine already found, so a format is assembled from the pipeline's
moment list instead of hunting for new footage.

    moment      {start, end}                      one clip, the default
    callback    {parts: [claim 3-6 s, payoff 8-15 s]}
                                                  hard cut; small EARLIER
                                                  tag when the gap is over
                                                  10 min; "jump_card": true
                                                  adds a full "40 MINUTES
                                                  LATER" card
    compilation {parts: [3-5]}                    #3 card, clip, #2 card,
                                                  clip, #1 card, best last
    escalation  {parts: [3-6], word: "STRIKE"}    STRIKE 1/3 ... 3/3, badge
                                                  and number warm up from
                                                  white to red
    thenvsnow   {parts: [2], labels: ["THEN", "NOW"]}
                                                  label card + corner badge
                                                  on each part
    reveal      {start, end, reveal: 1.6}         the face is blurred for
                                                  `reveal` s while the game
                                                  plays, then it clears
                                                  with a flash
    chat        {start, end, chat: [...]}         top chat lines pop up as
                                                  bubbles after the payoff
    waitforit   {start, end, payoff: s}           "WAIT FOR IT" tag + a bar
                                                  that fills until the
                                                  payoff, then both vanish
                                                  with a flash

A *moment* is a dict the pipeline already produces: {"start", "end",
"title" (or "hook"), "label", "score"}. :func:`plan_parts` turns a moment
list into a format spec; :func:`render_format` renders it. Each part is
rendered through the normal render path (``ClipEditor.edit_clip``) and
cached by content hash (:class:`PartCache`), so two formats on the same
moment render it once. Cards are Pillow images over a blurred frame (the
same card style ``clip_editor`` uses); badges and tags are ffmpeg
drawtext overlays built by ``render_modes``.

``render_format`` takes an optional ``part_renderer`` callable — tests
inject a fake, the pipeline uses the default (window extract +
``edit_clip``). Single-part formats (reveal, chat, waitforit) can also be
applied to an already-rendered clip with :func:`apply_single_clip_format`
(that is how the pipeline renders one moment *as* a format).

Technique adapted from gta6-clipper (https://gitlab.com/nocodehack.io/
gta6-clipper, MIT licence), formats.py — reimplemented here on this
repo's ClipEditor / render_modes stack; no code copied.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

YELLOW = "0xFFD400"
RED = "0xFF3B30"
WHITE = "white"
BADGE_Y = 215       # corner badges / tags sit under the platform's top UI
CHAT_Y = 1440       # chat rows sit above the bottom UI
CHAT_ROW_H = 112    # vertical slot per chat row
CHAT_ROW_LIFE = 2.2  # seconds a chat row stays on screen
CARD_FIRST_SECONDS = 0.9
CARD_SECONDS = 0.6
EARLIER_GAP_SECONDS = 600  # callback gap over 10 min gets the EARLIER tag
CACHE_MAX_AGE_DAYS = 7
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

FORMATS = {
    "moment": "one strong clip, the default",
    "callback": "a setup early and its payoff much later (a promise, a prediction, a bet)",
    "compilation": "the same thing happens 3-5 times in a stream (rage, fails, jokes): numbered, best last",
    "escalation": "the same thing keeps getting worse in order (strike 1, 2, 3)",
    "thenvsnow": "a claim or plan, then what really happened (prediction vs reality)",
    "reveal": "a reaction to something on screen: the face is hidden for 1-2 s while the thing plays, then it clears",
    "chat": "the moment is funny but chat's lines are the joke: chat pops up over it",
    "waitforit": "a quiet or slow build into one big payoff: tag and bar tease it",
}
NEEDS_PARTS = {"callback": 2, "compilation": 3, "escalation": 3, "thenvsnow": 2}


def _creationflags():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ---------------------------------------------------------------------------
# specs: planning + validation (pure)
# ---------------------------------------------------------------------------


def format_of(spec):
    """Format name of a spec: explicit "format", else "callback" when it
    has parts, else "moment"."""
    return spec.get("format") or ("callback" if spec.get("parts") else "moment")


def validate_spec(spec):
    """
    Check a format spec; raises ValueError with a plain message when it
    cannot be rendered (unknown format, too few parts, a part without a
    usable span). Returns the format name.
    """
    fmt = format_of(spec)
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}; "
                         f"formats: {', '.join(sorted(FORMATS))}")
    parts = spec.get("parts") or []
    need = NEEDS_PARTS.get(fmt, 0)
    if len(parts) < need:
        raise ValueError(f"{fmt} needs at least {need} parts, "
                         f"got {len(parts)}")
    if fmt in ("reveal", "chat", "waitforit", "moment") and not parts:
        for key in ("start", "end"):
            if key not in spec:
                raise ValueError(f"{fmt} needs a start/end or one part")
    for i, part in enumerate(parts):
        try:
            start = float(part["start"])
            end = float(part["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"part {i + 1} needs numeric start/end")
        if end - start < 0.5:
            raise ValueError(f"part {i + 1} is too short "
                             f"({start}-{end}s, need >= 0.5 s)")
    if fmt == "compilation" and len(parts) > 5:
        raise ValueError("compilation takes at most 5 parts")
    if fmt == "escalation" and len(parts) > 6:
        raise ValueError("escalation takes at most 6 parts")
    return fmt


def _moment_span(moment, max_len=None, min_len=None):
    """(start, end) of a moment, clipped to a max length from its start."""
    start = float(moment["start"])
    end = float(moment["end"])
    if max_len is not None:
        end = min(end, start + max_len)
    if min_len is not None and end - start < min_len:
        end = start + min_len
    return round(start, 2), round(end, 2)


def plan_parts(format_name, moments, hook="", **options):
    """
    Turn a moment list into a format spec (pure — no rendering).

    Args:
        format_name: one of FORMATS.
        moments: [{"start", "end", "title"/"hook", "label", "score"}]
            in stream order. Compilation sorts by score (best last);
            escalation keeps stream order.
        hook: headline for the first card / first part.
        options: format extras — word (escalation, default "STRIKE"),
            labels (thenvsnow, default ["THEN", "NOW"]), reveal seconds,
            chat lines, payoff second, tag, jump_card.

    Returns a spec dict for :func:`render_format`.
    """
    if format_name not in FORMATS:
        raise ValueError(f"Unknown format {format_name!r}; "
                         f"formats: {', '.join(sorted(FORMATS))}")
    moments = list(moments or [])
    spec = {"format": format_name, "hook": hook}
    spec.update(options)

    if format_name == "moment":
        if not moments:
            raise ValueError("moment needs at least 1 moment")
        start, end = _moment_span(moments[0])
        spec["parts"] = [{"start": start, "end": end,
                          "label": moments[0].get("label", "")}]
        return spec

    if format_name == "callback":
        if len(moments) < 2:
            raise ValueError("callback needs 2 moments (claim + payoff)")
        claim = moments[0]
        payoff = moments[1]
        c_start, c_end = _moment_span(claim, max_len=6.0)
        p_start, p_end = _moment_span(payoff, max_len=15.0)
        spec["parts"] = [
            {"start": c_start, "end": c_end,
             "label": claim.get("label", "THE CLAIM")},
            {"start": p_start, "end": p_end,
             "label": payoff.get("label", "THE PAYOFF")},
        ]
        return spec

    if format_name == "compilation":
        if len(moments) < 3:
            raise ValueError("compilation needs at least 3 moments")
        # Best last: sort by score ascending so the highest score is #1.
        ordered = sorted(moments[:5],
                         key=lambda m: float(m.get("score", 0) or 0))
        spec["parts"] = []
        for moment in ordered:
            start, end = _moment_span(moment, max_len=12.0)
            spec["parts"].append({"start": start, "end": end,
                                  "label": moment.get("label", "")})
        return spec

    if format_name == "escalation":
        if len(moments) < 3:
            raise ValueError("escalation needs at least 3 moments")
        spec.setdefault("word", "STRIKE")
        spec["parts"] = []
        for moment in moments[:6]:
            start, end = _moment_span(moment, max_len=12.0)
            spec["parts"].append({"start": start, "end": end,
                                  "label": moment.get("label", "")})
        return spec

    if format_name == "thenvsnow":
        if len(moments) < 2:
            raise ValueError("thenvsnow needs 2 moments")
        spec.setdefault("labels", ["THEN", "NOW"])
        spec["parts"] = []
        for moment in moments[:2]:
            start, end = _moment_span(moment, max_len=15.0)
            spec["parts"].append({"start": start, "end": end,
                                  "label": moment.get("label", "")})
        return spec

    # Single-moment formats: reveal / chat / waitforit.
    if not moments:
        raise ValueError(f"{format_name} needs 1 moment")
    start, end = _moment_span(moments[0])
    spec["parts"] = [{"start": start, "end": end,
                      "label": moments[0].get("label", "")}]
    return spec


# ---------------------------------------------------------------------------
# per-format render cache (content hash)
# ---------------------------------------------------------------------------


def cache_key(*fields):
    """12-hex-char content hash of the render inputs, so two formats on
    the same moment (same source, span, hook, layout, theme) share one
    part render."""
    joined = "|".join(str(f) for f in fields)
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


class PartCache:
    """
    Disk cache for rendered format parts, keyed by content hash.

    A part is rendered once and reused across formats; a second render
    of the same part (same source file, span, hook, layout, theme)
    returns the cached file. Files older than ``max_age_days`` are
    swept on first use. Publish is atomic (render to ``*_tmp.mp4``,
    then ``os.replace``), so a half-written file is never "cached",
    and a per-key lock means parallel renders of the same part run the
    render once.
    """

    def __init__(self, cache_dir, max_age_days=CACHE_MAX_AGE_DAYS):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self._locks = {}
        self._locks_guard = threading.Lock()
        self._swept = False

    def _lock_for(self, key):
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def path_for(self, key):
        return self.dir / f"{key}.mp4"

    def sweep(self):
        """Delete cached files older than max_age_days (once per instance,
        plus whenever called directly)."""
        if self._swept:
            return
        self._swept = True
        cutoff = time.time() - self.max_age_days * 86400
        for path in self.dir.glob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

    def get_or_render(self, key, render_fn):
        """
        Return (path, cached): the cached part, or the result of
        ``render_fn(tmp_path)`` published atomically. ``render_fn``
        must write a complete file to the tmp path it is given.
        """
        self.sweep()
        target = self.path_for(key)
        with self._lock_for(key):
            if target.exists() and target.stat().st_size > 0:
                return target, True
            tmp = self.dir / f"{key}_tmp.mp4"
            render_fn(tmp)
            if not tmp.exists() or tmp.stat().st_size == 0:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise RuntimeError(f"Part render for {key} produced no file")
            os.replace(tmp, target)
            return target, False


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _run_ffmpeg(cmd, timeout=600):
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
        creationflags=_creationflags())
    if res.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + res.stderr.strip()[-300:])
    return res


def probe_duration(path):
    """Duration in seconds via ffprobe, or None when unavailable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=_creationflags())
        if res.returncode == 0:
            return float(res.stdout.strip())
    except Exception:
        pass
    return None


def extract_frame(video_path, out_png, at="0.3", from_end=False):
    """Grab one frame (default 0.3 s in, or 0.15 s before the end)."""
    seek = ["-sseof", "-0.15"] if from_end else ["-ss", str(at)]
    _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", *seek,
                 "-i", str(video_path), "-frames:v", "1", str(out_png)])
    return Path(out_png)


def extract_window(video_path, start, end, out_path):
    """
    Cut [start, end) out of a video, frame-accurate.

    Re-encodes (instead of stream copy): a stream-copy cut can only
    start on a keyframe, so a window with no keyframe inside it comes
    out with no video at all (a 2-8 s window of a clip with keyframes
    every 8.3 s is audio-only). Input seeking keeps it fast — ffmpeg
    seeks to the keyframe before the window, then decodes and drops
    frames until the exact start.
    """
    start = float(start)
    duration = max(float(end) - start, 0.1)
    _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error",
                 "-ss", f"{start:.2f}", "-t", f"{duration:.2f}",
                 "-i", str(video_path),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                 "-c:a", "aac", "-b:a", "192k", str(out_path)])
    return Path(out_path)


def _probe_audio(path):
    """(sample_rate, channels) of the first audio stream, or None."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        fields = res.stdout.strip().split(",")
        if len(fields) >= 2 and fields[0].isdigit():
            return int(fields[0]), int(fields[1])
    except Exception:
        pass
    return None


def _normalise_join_audio(clip, tmp_dir, index):
    """One join input with 48 kHz stereo AAC audio (video copied).

    The concat demuxer needs identical audio parameters across
    inputs: a 96 kHz part next to a 48 kHz card makes the AAC
    decoder fail and the whole join exits 69. Clips that already
    match pass through untouched; a clip with no audio gets a
    silent track so the join still works.
    """
    audio = _probe_audio(clip)
    if audio == (48000, 2):
        return clip
    out = tmp_dir / f"join_norm_{index}.mp4"
    if audio is None:
        _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error",
                     "-i", str(clip),
                     "-f", "lavfi", "-i",
                     "anullsrc=r=48000:cl=stereo",
                     "-map", "0:v", "-map", "1:a",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-shortest", str(out)])
    else:
        _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error",
                     "-i", str(clip),
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                     "-ar", "48000", "-ac", "2", str(out)])
    return out


def join_clips(clips, out_path):
    """Join clips into one file (re-encode: parts may differ in codec
    parameters, and the concat demuxer with stream copy would fail)."""
    clips = [Path(c).resolve() for c in clips]
    if len(clips) == 1:
        shutil.copyfile(clips[0], out_path)
        return Path(out_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="format_join_"))
    try:
        # Uniform audio first: the concat demuxer fails hard on
        # mixed sample rates (see _normalise_join_audio).
        clips = [_normalise_join_audio(c, tmp_dir, k)
                 for k, c in enumerate(clips)]
        concat_list = tmp_dir / "concat.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for clip in clips:
                safe = str(clip).replace("'", "'\\''")
                f.write(f"file '{safe}'\n")
        _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error",
                     "-f", "concat", "-safe", "0", "-i", str(concat_list),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                     "-c:a", "aac", "-b:a", "192k", str(out_path)])
        return Path(out_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def apply_video_filter(src, video_filter, out_path):
    """Re-encode one clip with a video filter chain (audio copied)."""
    _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error",
                 "-i", str(src), "-vf", video_filter,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                 "-c:a", "copy", str(out_path)])
    return Path(out_path)


# ---------------------------------------------------------------------------
# cards (Pillow, same style as clip_editor's caption cards)
# ---------------------------------------------------------------------------


def _load_font(size):
    """Bold font for cards; falls back to Pillow's default font."""
    from PIL import ImageFont
    try:
        from platform_utils import find_font
        path = find_font(["arialbd.ttf", "arial.ttf",
                          "DejaVuSans-Bold.ttf", "impact.ttf"])
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _hex_to_rgb(hex_color):
    h = str(hex_color).strip().lstrip("#").lstrip("0x")
    if len(h) != 6:
        return (255, 255, 255)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def render_card_png(frame_png, lines, out_png,
                    size=(VIDEO_WIDTH, VIDEO_HEIGHT)):
    """
    Punchy card: the frame blurred and darkened, ``lines`` =
    [(text, size, colour_hex)] stacked in the middle. Returns the PNG path.
    """
    from PIL import Image, ImageDraw, ImageFilter

    frame = Image.open(frame_png).convert("RGB")
    frame = frame.resize(size)
    frame = frame.filter(ImageFilter.GaussianBlur(20))
    # Darken + slight saturation lift so white text pops.
    dark = Image.new("RGB", size, (0, 0, 0))
    frame = Image.blend(frame, dark, 0.35)
    draw = ImageDraw.Draw(frame)

    total_h = sum(s * 1.15 for _, s, _ in lines)
    y = (size[1] - total_h) / 2
    for text, font_size, colour in lines:
        font = _load_font(int(font_size))
        text_w = draw.textlength(text, font=font)
        x = (size[0] - text_w) / 2
        rgb = _hex_to_rgb(colour)
        # Black outline (drawn as offsets) so text reads over any frame.
        outline = max(2, int(font_size) // 14)
        for dx in range(-outline, outline + 1, 2):
            for dy in range(-outline, outline + 1, 2):
                if dx or dy:
                    draw.text((x + dx, y + dy), text, font=font,
                              fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=rgb)
        y += font_size * 1.15
    frame.save(out_png, "PNG")
    return Path(out_png)


def probe_video_size(path):
    """(width, height) of a video via ffprobe, or None when unavailable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json",
             str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=_creationflags())
        stream = json.loads(res.stdout)["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception:
        return None


def probe_video_fps(path):
    """Frame rate of a video via ffprobe, or 30.0 when unavailable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
             str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=_creationflags())
        num, den = res.stdout.strip().split("/")
        fps = float(num) / float(den)
        return fps if fps > 0 else 30.0
    except Exception:
        return 30.0


def card_clip(frame_png, lines, out_path, duration=CARD_SECONDS,
              with_sound=True, size=(VIDEO_WIDTH, VIDEO_HEIGHT), fps=30):
    """
    A short video card: the card PNG looped for ``duration`` seconds,
    with a soft whoosh underneath (synthesised, no samples) unless
    ``with_sound`` is False.

    ``size`` / ``fps`` must match the parts the card is joined with —
    the concat demuxer needs uniform resolution and frame rate, or
    the joined clip comes out the wrong length. The audio is padded
    to the full card duration (the whoosh is shorter than a 0.9 s
    card, and ``-shortest`` would truncate the video to it).
    """
    png = Path(out_path).with_suffix(".png")
    render_card_png(frame_png, lines, png, size=size)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-framerate", str(fps), "-loop", "1", "-t", f"{duration:.2f}",
           "-i", str(png)]
    if with_sound:
        try:
            from meme_sounds import sound_path
            whoosh = sound_path("whoosh")
        except Exception:
            whoosh = None
    else:
        whoosh = None
    if whoosh is not None:
        cmd += ["-i", str(whoosh),
                "-filter_complex",
                "[1:a]atrim=0:{:.2f},apad=whole_dur={:.2f},"
                "afade=t=out:st={:.2f}:d=0.15[aout]".format(
                    duration, duration, max(duration - 0.15, 0)),
                "-map", "0:v", "-map", "[aout]"]
    else:
        cmd += ["-f", "lavfi", "-t", f"{duration:.2f}", "-i",
                "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a"]
    cmd += ["-t", f"{duration:.2f}",
            "-c:v", "libx264", "-crf", "18", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            str(out_path)]
    _run_ffmpeg(cmd)
    try:
        png.unlink()
    except OSError:
        pass
    return Path(out_path)


# ---------------------------------------------------------------------------
# chat bubbles
# ---------------------------------------------------------------------------


def chat_row_timings(lines, payoff_t, duration, max_rows=5):
    """
    [(show_time, user, colour, text)] for chat bubble overlays: the
    pick's own lines, first at payoff + 0.2 s, then one every 0.55 s.
    Rows that would start in the last 1.2 s of the clip are dropped.
    """
    rows = []
    for i, line in enumerate(list(lines or [])[:max_rows]):
        at = payoff_t + 0.2 + i * 0.55
        if at >= duration - 1.2:
            break
        if isinstance(line, dict):
            rows.append((round(at, 2), line.get("user", ""),
                         line.get("colour", "#FFFFFF"),
                         str(line.get("text", ""))))
        else:
            rows.append((round(at, 2), "", "#FFFFFF", str(line)))
    return rows


def render_chat_row_png(user, colour, text, out_png, size=56, max_width=980):
    """One Twitch-style chat row: dark rounded bar, coloured name, white
    message. Returns (width, height)."""
    from PIL import Image, ImageDraw, ImageFont

    while True:
        font = _load_font(size)
        name = (user[:12] + ": ") if user else ""
        # Measure with a throwaway draw (Pillow needs a draw for textlength
        # on some versions; ImageFont.getlength works on all).
        name_w = font.getlength(name)
        text_w = font.getlength(text)
        total_w = int(name_w + text_w + 56)
        if total_w <= max_width or size <= 44:
            break
        size -= 4
    height = int(size * 1.0 + 40)
    image = Image.new("RGBA", (total_w, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, total_w - 1, height - 1),
                           radius=height // 3, fill=(14, 14, 20, 178))
    y = (height - size) // 2 - 6
    draw.text((28, y), name, font=font, fill=_hex_to_rgb(colour))
    draw.text((28 + name_w, y), text, font=font, fill=(255, 255, 255))
    image.save(out_png, "PNG")
    return total_w, height


def chat_overlay_filter(rows, duration, work_dir):
    """
    Build the ffmpeg inputs + filter_complex that pops chat rows over a
    clip. Returns (input_args, filter_complex, output_label). Each row
    lives CHAT_ROW_LIFE seconds and scrolls up one slot for every newer
    row, like a real chat feed.
    """
    inputs = []
    chains = []
    last = "[0:v]"
    for i, (at, user, colour, text) in enumerate(rows):
        png = Path(work_dir) / f"chat_row{i}.png"
        render_chat_row_png(user, colour, text, png)
        life = min(CHAT_ROW_LIFE, duration - at)
        inputs += ["-loop", "1", "-framerate", "30", "-t", f"{life:.2f}",
                   "-i", str(png)]
        newer = "+".join(f"gt(t,{b:.2f})" for b, *_ in rows[i + 1:]) or "0"
        out_label = f"[chat{i}]"
        chains.append(
            f"[{i + 1}:v]format=rgba,setpts=PTS+{at:.2f}/TB[ci{i}];"
            f"{last}[ci{i}]overlay=x=40:"
            f"y='{CHAT_Y}-overlay_h/2-({newer})*{CHAT_ROW_H}':"
            f"enable='between(t,{at:.2f},{at + life:.2f})':"
            f"eof_action=pass{out_label}")
        last = out_label
    return inputs, ";".join(chains), last


# ---------------------------------------------------------------------------
# part rendering (default: window extract + ClipEditor)
# ---------------------------------------------------------------------------


def default_part_renderer(source_path, part, hook, out_path, work_dir,
                          layout=None, theme=None, streamer=""):
    """
    Render one part span through the normal render path: extract the
    window, then ``ClipEditor.edit_clip`` (9:16, watermark, hook banner
    when ``hook`` is given). No thumbnail per part — the finished format
    clip gets one thumbnail of its own.
    """
    from clip_editor import ClipEditor

    window = Path(work_dir) / f"window_{part['start']:.2f}_{part['end']:.2f}.mp4"
    extract_window(source_path, part["start"], part["end"], window)
    editor = ClipEditor()
    return editor.edit_clip(
        raw_clip_path=str(window),
        output_path=str(out_path),
        streamer=streamer,
        layout=layout,
        hook_banner=hook or None,
        generate_thumbnail=False,
        theme=theme,
    )


def _part_cache_key(source_path, part, hook, layout, theme):
    """Content-hash inputs for one part render: source identity (path +
    size + mtime) plus everything that shapes the picture."""
    source = Path(source_path)
    try:
        stat = source.stat()
        source_id = f"{source}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        source_id = str(source)
    return cache_key(source_id, part["start"], part["end"], hook or "",
                     layout or "", theme or "")


def _render_part(source_path, part, hook, cache, work_dir, part_renderer,
                 layout, theme, streamer, log):
    """Render one part (cached); returns the part mp4 path."""
    key = _part_cache_key(source_path, part, hook, layout, theme)
    if cache is None:
        out = Path(work_dir) / f"part_{key}.mp4"
        part_renderer(source_path, part, hook, out, work_dir,
                      layout=layout, theme=theme, streamer=streamer)
        return out
    def _render(tmp_path):
        part_renderer(source_path, part, hook, tmp_path, work_dir,
                      layout=layout, theme=theme, streamer=streamer)
    path, cached = cache.get_or_render(key, _render)
    log.append(f"part {part['start']}-{part['end']}s: "
               f"{'cached' if cached else 'rendered'}")
    return path


# ---------------------------------------------------------------------------
# format assembly
# ---------------------------------------------------------------------------


def _has_facecam(video_path):
    """
    True when a face is visible in the first second (Haar cascade on 3
    frames). Best effort: when OpenCV or the cascade is unavailable the
    answer is True (render the reveal as asked) — the reveal format is
    only picked for clips that have a facecam anyway.
    """
    try:
        import cv2
        cascade = Path(__file__).resolve().parent / "models" / (
            "haarcascade_frontalface_default.xml")
        if not cascade.exists():
            return True
        detector = cv2.CascadeClassifier(str(cascade))
        cap = cv2.VideoCapture(str(video_path))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
        hits = 0
        for t in (0.1, 0.6, 1.1):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 4)
            if len(faces) and max(f[2] for f in faces) / width >= 0.025:
                hits += 1
        cap.release()
        return hits >= 2
    except Exception:
        return True


def _numbered_cards(spec, parts, rendered, work_dir, big_of, badge_of,
                    color_of, log, card_size=(VIDEO_WIDTH, VIDEO_HEIGHT),
                    card_fps=30):
    """
    Shared by compilation, escalation, thenvsnow: a card (big text,
    small label) then the part with a corner badge. Returns the ordered
    clip list for joining. Cards are rendered at ``card_size`` /
    ``card_fps`` so they match the parts they are joined with.
    """
    from render_modes import build_badge_filter

    out = []
    n = len(parts)
    # Card text sizes are tuned for 1080x1920; scale them when the
    # parts are a different resolution.
    text_scale = card_size[0] / VIDEO_WIDTH
    for k, (part, rendered_part) in enumerate(zip(parts, rendered)):
        colour = color_of(k, n)
        big = big_of(k, n)
        label = str(part.get("label", "")).strip()
        lines = []
        if k == 0 and spec.get("hook"):
            lines.append((str(spec["hook"]), int(78 * text_scale), "FFD400"))
        lines.append((big, int((330 if len(big) < 5 else 190) * text_scale),
                      colour.lstrip("0x")))
        if label:
            lines.append((label, int(80 * text_scale), "FFFFFF"))
        frame_png = extract_frame(rendered_part, work_dir / f"card_frame{k}.png")
        card = card_clip(frame_png, lines, work_dir / f"card{k}.mp4",
                         CARD_FIRST_SECONDS if k == 0 else CARD_SECONDS,
                         size=card_size, fps=card_fps)
        out.append(card)
        badged = apply_video_filter(
            rendered_part,
            build_badge_filter(badge_of(k, n), colour),
            work_dir / f"badged{k}.mp4")
        out.append(badged)
        log.append(f"part {k + 1}: card '{big}' + badge '{badge_of(k, n)}'")
    return out


def _peak_time(video_path):
    """Loudest moment in the last two thirds of a clip: where the payoff
    is. Falls back to 70% through the clip."""
    try:
        from meme_sounds import find_payoff
        duration = probe_duration(video_path)
        if duration:
            return find_payoff(video_path, duration)
    except Exception:
        pass
    return None


def render_format(source_path, spec, out_path, part_renderer=None,
                  work_dir=None, cache_dir=None, layout=None, theme=None,
                  streamer=""):
    """
    Render one format spec into a finished 9:16 clip.

    Args:
        source_path: the source video the parts are cut from (a raw
            stream capture or VOD).
        spec: from :func:`plan_parts` / :func:`validate_spec` — must
            carry "format" and "parts" (single-part formats may carry
            "start"/"end" instead).
        out_path: destination mp4.
        part_renderer: (source, part, hook, out, work_dir, layout,
            theme, streamer) -> path. Default renders through
            ``ClipEditor.edit_clip``.
        work_dir / cache_dir: scratch + part cache. Defaults: temp dir /
            <edited>/.partcache.
        layout / theme / streamer: passed through to the part renderer.

    Returns {"out", "format", "duration", "parts", "log"}.
    """
    fmt = validate_spec(spec)
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    work = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="format_"))
    work.mkdir(parents=True, exist_ok=True)
    own_work_dir = work_dir is None
    if cache_dir is None:
        try:
            from config import EDITED_DIR
            cache_dir = Path(EDITED_DIR) / ".partcache"
        except Exception:
            cache_dir = work / ".partcache"
    cache = PartCache(cache_dir)
    log = [f"format {fmt}"]
    renderer = part_renderer or default_part_renderer

    parts = list(spec.get("parts") or [])
    if not parts:  # single-part format given as start/end
        parts = [{"start": spec["start"], "end": spec["end"],
                  "label": spec.get("label", "")}]
    try:
        hook = str(spec.get("hook") or "")

        if fmt in ("reveal", "chat", "waitforit"):
            # Single-part formats: render the part, then overlay.
            rendered = _render_part(source_path, parts[0], hook, cache,
                                    work, renderer, layout, theme,
                                    streamer, log)
            final = apply_single_clip_format(
                rendered, fmt, out_path, work_dir=work,
                hook=hook, spec=spec, log=log)
            duration = probe_duration(final)
            log.append(f"ok {final} "
                       f"{duration:.1f}s" if duration else f"ok {final}")
            return {"out": str(final), "format": fmt, "duration": duration,
                    "parts": [str(rendered)], "log": log}

        if fmt == "moment":
            rendered = _render_part(source_path, parts[0], hook, cache,
                                    work, renderer, layout, theme,
                                    streamer, log)
            shutil.copyfile(rendered, out_path)
            duration = probe_duration(out_path)
            log.append(f"ok {out_path} {duration:.1f}s" if duration else
                       f"ok {out_path}")
            return {"out": str(out_path), "format": fmt, "duration": duration,
                    "parts": [str(rendered)], "log": log}

        if fmt == "callback":
            from render_modes import build_badge_filter

            rendered = []
            for k, part in enumerate(parts):
                rendered.append(_render_part(
                    source_path, part, hook if k == 0 else "", cache,
                    work, renderer, layout, theme, streamer, log))
            gap = float(parts[1]["start"]) - float(parts[0]["end"])
            clips = []
            first = rendered[0]
            if gap > EARLIER_GAP_SECONDS or spec.get("tag"):
                first = apply_video_filter(
                    rendered[0],
                    build_badge_filter(spec.get("tag", "EARLIER")),
                    work / "earlier.mp4")
                log.append(f"callback gap {gap / 60:.0f} min: EARLIER tag")
            clips.append(first)
            if spec.get("jump_card"):
                jump = str(spec.get("jump") or
                           f"{gap / 60:.0f} MINUTES LATER")
                frame_png = extract_frame(rendered[0],
                                          work / "jump_frame.png",
                                          from_end=True)
                part_size = (probe_video_size(rendered[0])
                             or (VIDEO_WIDTH, VIDEO_HEIGHT))
                part_fps = probe_video_fps(rendered[0])
                clips.append(card_clip(
                    frame_png,
                    [(jump, int(120 * part_size[0] / VIDEO_WIDTH), "FFD400")],
                    work / "jump.mp4", CARD_SECONDS, size=part_size,
                    fps=part_fps))
                log.append(f"jump card '{jump}'")
            clips.append(rendered[1])
            join_clips(clips, out_path)
        elif fmt in ("compilation", "escalation", "thenvsnow"):
            rendered = [_render_part(source_path, part, "", cache, work,
                                     renderer, layout, theme, streamer, log)
                        for part in parts]
            # Cards must match the parts' resolution (the concat
            # demuxer needs uniform size + fps).
            part_size = (probe_video_size(rendered[0])
                         or (VIDEO_WIDTH, VIDEO_HEIGHT))
            part_fps = probe_video_fps(rendered[0])
            if fmt == "compilation":
                clips = _numbered_cards(
                    spec, parts, rendered, work,
                    lambda k, n: f"#{n - k}",
                    lambda k, n: f"#{n - k}",
                    lambda k, n: YELLOW, log, card_size=part_size,
                    card_fps=part_fps)
            elif fmt == "escalation":
                word = str(spec.get("word", "STRIKE")).strip() or "STRIKE"
                heat = ["0xFFFFFF", "0xFFD400", "0xFF9500", "0xFF3B30"]

                def _heat(k, n):
                    return heat[min(3, round(3 * k / max(n - 1, 1)))]

                clips = _numbered_cards(
                    spec, parts, rendered, work,
                    lambda k, n: f"{word} {k + 1}",
                    lambda k, n: f"{word} {k + 1}/{n}",
                    _heat, log, card_size=part_size,
                    card_fps=part_fps)
            else:
                labels = spec.get("labels", ["THEN", "NOW"])
                clips = _numbered_cards(
                    spec, parts, rendered, work,
                    lambda k, n: labels[min(k, 1)],
                    lambda k, n: labels[min(k, 1)],
                    lambda k, n: YELLOW if k == 0 else RED, log,
                    card_size=part_size, card_fps=part_fps)
            join_clips(clips, out_path)
        else:  # pragma: no cover — validate_spec guards this
            raise ValueError(f"unhandled format {fmt!r}")

        duration = probe_duration(out_path)
        log.append(f"ok {out_path} {duration:.1f}s" if duration else
                   f"ok {out_path}")
        if duration and not 8 <= duration <= 120:
            log.append("WARN: outside 8-120 s")
        return {"out": str(out_path), "format": fmt, "duration": duration,
                "parts": [str(p) for p in rendered], "log": log}
    finally:
        if own_work_dir:
            shutil.rmtree(work, ignore_errors=True)


def apply_single_clip_format(src, format_name, out_path, work_dir=None,
                             hook="", spec=None, log=None):
    """
    Apply a single-part format (reveal / chat / waitforit) to an
    already-rendered clip. ``spec`` carries the format extras: reveal
    seconds + tag, chat lines, payoff second + tag. Returns the output
    path. Used by the pipeline to render one moment *as* a format.
    """
    from render_modes import (build_reveal_filter,
                              build_waitforit_filter)

    spec = spec or {}
    log = log if log is not None else []
    src = Path(src)
    out_path = Path(out_path)
    work = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="format_single_"))
    work.mkdir(parents=True, exist_ok=True)
    own_work_dir = work_dir is None
    try:
        duration = probe_duration(src) or 0.0
        if format_name == "reveal":
            reveal_t = float(spec.get("reveal", 1.6))
            if not _has_facecam(src):
                log.append("reveal: no facecam found, rendered as a plain "
                           "moment (nothing blurred)")
                shutil.copyfile(src, out_path)
                return out_path
            tag = str(spec.get("tag", "WATCH HIS FACE"))
            apply_video_filter(
                src, build_reveal_filter(reveal_t, tag=tag), out_path)
            log.append(f"reveal: face blurred {reveal_t}s, tag '{tag}'")
            return out_path

        if format_name == "waitforit":
            payoff = spec.get("payoff")
            payoff_t = float(payoff) if payoff is not None else (
                _peak_time(src) or duration * 0.7)
            payoff_t = min(max(payoff_t, 2.5), max(duration - 0.5, 2.5))
            tag = str(spec.get("tag", "WAIT FOR IT"))
            if payoff_t < 2.5:
                log.append("waitforit WARN: payoff in the first 2.5 s, "
                           "nothing to wait for; use format moment")
            apply_video_filter(
                src, build_waitforit_filter(payoff_t, tag=tag), out_path)
            log.append(f"waitforit: payoff at {payoff_t:.1f}s")
            return out_path

        if format_name == "chat":
            payoff = spec.get("payoff")
            payoff_t = float(payoff) if payoff is not None else (
                _peak_time(src) or duration * 0.7)
            lines = spec.get("chat") or spec.get("chat_rows") or []
            if not lines:
                raise ValueError("chat format needs chat lines "
                                 '("chat": ["KEKW", ...])')
            rows = chat_row_timings(lines, payoff_t, duration)
            if not rows:
                raise ValueError("no usable chat lines in this span")
            log.append(f"chat {len(rows)} rows from {payoff_t:.1f}s: " +
                       "; ".join(f"{u}: {t}" for _, u, _, t in rows))
            inputs, filt, out_label = chat_overlay_filter(
                rows, duration, work)
            _run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error",
                         "-i", str(src), *inputs,
                         "-filter_complex", filt,
                         "-map", out_label, "-map", "0:a",
                         "-t", f"{duration:.2f}",
                         "-c:v", "libx264", "-preset", "veryfast",
                         "-crf", "17", "-c:a", "copy", str(out_path)])
            return out_path

        raise ValueError(f"{format_name} is not a single-clip format; "
                         "use render_format")
    finally:
        if own_work_dir:
            shutil.rmtree(work, ignore_errors=True)
