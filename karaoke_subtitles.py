"""
karaoke_subtitles.py — Animated kinetic karaoke subtitles (.ass).

Word-by-word animated ASS subtitles in the Alex Hormozi / Veed style: each
word highlights (sweeps to the accent colour) exactly as it is spoken, using
libass ``{\\k}`` karaoke timing tags, with an optional subtle bounce/scale pop
on the active word via ``\\t`` transforms.

Karaoke approach adapted from opensource-clipping
(https://github.com/NaufalRizqullah/opensource-clipping, MIT licence),
clipping/studio/subtitles.py (buat_file_ass).

Input words use the same shape ai_story_engine.transcribe_clip() produces:
    {"word": str, "start": float_seconds, "end": float_seconds}

The output is a plain .ass file that ffmpeg's ``subtitles`` filter can burn
in, e.g. ``subtitles='clip.ass':fontsdir='C\\:/Windows/Fonts'``. It targets
the existing 1080x1920 canvas (config.VIDEO_WIDTH / VIDEO_HEIGHT).
"""

import os
from pathlib import Path

PLAY_RES_X = 1080
PLAY_RES_Y = 1920

MAX_WORDS_PER_LINE = 3       # rapid 2-3 word pacing, like the static subtitles
MAX_GAP_WITHIN_LINE = 0.8    # seconds; a longer pause starts a new line
MIN_EVENT_DURATION = 0.35    # seconds; matches generate_ass_subtitles()
END_PADDING = 0.08           # seconds of hold after the last word
MIN_KARAOKE_CS = 10          # minimum karaoke duration per word (centiseconds)

# ASS colours are &HBBGGRR (blue/green/red, alpha in the high byte).
# With {\k} tags libass sweeps each word from SecondaryColour (unsung) to
# PrimaryColour (sung), so Primary = the highlight colour.
STYLES = {
    "hormozi": {
        "fontname": "Impact",
        "fontsize": 64,
        "primary": "&H0000FFFF",    # yellow highlight (sung)
        "secondary": "&H00FFFFFF",  # white (unsung)
        "outline": "&H00000000",
        "back": "&H90000000",
        "bold": 1,
        "outline_w": 6,
        "shadow": 4,
        "margin_l": 60,
        "margin_r": 60,
        "margin_v": 280,
        "uppercase": True,
        "bounce": True,
    },
    "minimal": {
        "fontname": "Arial",
        "fontsize": 56,
        "primary": "&H00FFFFFF",    # white (sung)
        "secondary": "&H00D8D8D8",  # light grey (unsung)
        "outline": "&H00000000",
        "back": "&H80000000",
        "bold": 0,
        "outline_w": 3,
        "shadow": 2,
        "margin_l": 60,
        "margin_r": 60,
        "margin_v": 280,
        "uppercase": False,
        "bounce": False,
    },
}


def _fmt_time(t: float) -> str:
    """ASS timestamp H:MM:SS.cc (centiseconds, always 2 digits)."""
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:  # rounding carried over a second boundary
        s += 1
        cs = 0
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    """Escape characters that are special in ASS event text."""
    return (
        text.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
    )


def _chunk_words(words):
    """
    Group words into display lines of at most MAX_WORDS_PER_LINE words,
    breaking the line early when the gap between two words exceeds
    MAX_GAP_WITHIN_LINE seconds (a natural pause).
    """
    lines = []
    current = []
    prev_end = None
    for w in words:
        word = str(w.get("word", "")).strip()
        if not word:
            continue
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))
        if end < start:
            end = start
        item = {"word": word, "start": start, "end": end}
        if current and (
            len(current) >= MAX_WORDS_PER_LINE
            or (prev_end is not None and start - prev_end > MAX_GAP_WITHIN_LINE)
        ):
            lines.append(current)
            current = []
        current.append(item)
        prev_end = end
    if current:
        lines.append(current)
    return lines


def _bounce_tags(line, line_start, style):
    """
    Build the leading override block for a line: a subtle scale pop
    (100 -> 112 -> 100 over ~200 ms) timed at each word's start, so the
    active word "bounces" as it is spoken. Returns "" for styles without
    bounce. Pops closer than 200 ms apart are skipped so \\t transforms
    never overlap.
    """
    if not style.get("bounce"):
        return ""
    tags = []
    last_pop_ms = None
    for w in line:
        pop_ms = int(round((w["start"] - line_start) * 1000))
        if last_pop_ms is not None and pop_ms - last_pop_ms < 200:
            continue
        tags.append(f"\\t({pop_ms},{pop_ms + 90},\\fscx112\\fscy112)")
        tags.append(f"\\t({pop_ms + 90},{pop_ms + 190},\\fscx100\\fscy100)")
        last_pop_ms = pop_ms
    if not tags:
        return ""
    return "{" + "".join(tags) + "}"


def generate_karaoke_ass(words: list, output_ass_path: str, style: str = "hormozi") -> str:
    """
    Write animated word-by-word karaoke subtitles to an .ass file.

    Args:
        words: list of {"word": str, "start": float, "end": float} dicts
            (Whisper word timestamps, as produced by
            AIStoryEngine.transcribe_clip).
        output_ass_path: destination .ass file path.
        style: "hormozi" (bold Impact, yellow highlight, bounce pop) or
            "minimal" (white Arial, no bounce).

    Returns:
        The output path (as given). Raises ValueError for an unknown style.
    """
    if style not in STYLES:
        raise ValueError(f"Unknown karaoke style {style!r}; expected one of {sorted(STYLES)}")
    s = STYLES[style]

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {PLAY_RES_X}\n"
        f"PlayResY: {PLAY_RES_Y}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Karaoke,{s['fontname']},{s['fontsize']},{s['primary']},"
        f"{s['secondary']},{s['outline']},{s['back']},{s['bold']},0,0,0,"
        f"100,100,2,0,1,{s['outline_w']},{s['shadow']},2,"
        f"{s['margin_l']},{s['margin_r']},{s['margin_v']},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    events = []
    for line in _chunk_words(words or []):
        line_start = line[0]["start"]
        line_end = max(line_start + MIN_EVENT_DURATION, line[-1]["end"] + END_PADDING)

        # Karaoke durations: each word is "sung" until the next word starts
        # (last word until its own end), in centiseconds.
        parts = []
        for i, w in enumerate(line):
            if i + 1 < len(line):
                dur = line[i + 1]["start"] - w["start"]
            else:
                dur = w["end"] - w["start"]
            cs = max(MIN_KARAOKE_CS, int(round(dur * 100)))
            text = w["word"].upper() if s["uppercase"] else w["word"]
            parts.append(f"{{\\k{cs}}}{_escape_ass_text(text)}")

        bounce = _bounce_tags(line, line_start, s)
        events.append(
            f"Dialogue: 0,{_fmt_time(line_start)},{_fmt_time(line_end)},"
            f"Karaoke,,0,0,0,,{bounce}{' '.join(parts)}"
        )

    output_path = Path(output_ass_path)
    if output_path.parent and str(output_path.parent):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        if events:
            f.write("\n".join(events) + "\n")
    return output_ass_path
