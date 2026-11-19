"""
encoder.py — Hardware encoder auto-detection for ffmpeg.

Priority (adapted from opensource-clipping, MIT licence,
clipping/studio/ffmpeg_utils.py detect_video_encoder):
    h264_nvenc (NVIDIA) > h264_amf (AMD) > h264_vaapi (Intel/AMD Linux)
    > libx264 (CPU fallback, current behaviour).

``ffmpeg -encoders`` is run once per process and cached. The selection itself
is a pure function of that output (select_encoder), so tests can pass fake
output without ffmpeg installed. If ffmpeg is missing entirely, libx264
defaults are returned.

Wiring (coordinator): ``codec, extra_args = detect_encoder()`` then
``cmd += ["-c:v", codec, *extra_args]``.

Note: listing an encoder in ``ffmpeg -encoders`` does not guarantee the
hardware is present (e.g. NVENC listed on a machine with no NVIDIA GPU
fails at encode time). Callers that need certainty should verify with a
short test encode, as opensource-clipping does. VAAPI additionally needs
input-side hardware setup (see VAAPI_INPUT_ARGS) and an
``format=nv12,hwupload`` step in the filter chain — its extra_args here are
output-side only.
"""

import subprocess

ENCODER_PRIORITY = ("h264_nvenc", "h264_amf", "h264_vaapi", "libx264")

# Output-side ffmpeg args per encoder (used AFTER "-c:v <codec>").
ENCODER_ARGS = {
    # NVENC: p4 preset (good quality/speed balance), VBR with CQ 23.
    "h264_nvenc": ["-preset", "p4", "-rc", "vbr", "-cq", "23"],
    # AMF: balanced quality preset, VBR 8M target (1080p vertical).
    "h264_amf": ["-quality", "balanced", "-rc", "vbr", "-b:v", "8M"],
    # VAAPI: bitrate-controlled; see VAAPI_INPUT_ARGS for the required
    # input-side device setup.
    "h264_vaapi": ["-b:v", "8M"],
    # CPU fallback — matches the current ClipEditor behaviour exactly.
    "libx264": ["-preset", "veryfast", "-crf", "21"],
}

# VAAPI input-side args (must appear BEFORE the -i they apply to) plus the
# filter step "format=nv12,hwupload" at the end of the filter chain.
VAAPI_INPUT_ARGS = ["-init_hw_device", "vaapi=va:/dev/dri/renderD128",
                    "-filter_hw_device", "va"]

_cache = {"output": None, "done": False}
_verified_cache = {"codec": None, "args": None, "done": False}


def _run_ffmpeg_encoders() -> str:
    """Run `ffmpeg -encoders` once (cached); "" when ffmpeg is unavailable."""
    if _cache["done"]:
        return _cache["output"]
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout or ""
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        output = ""
    _cache["output"] = output
    _cache["done"] = True
    return output


def _test_encode(codec, extra_args, timeout=30):
    """
    Run a 1-second test encode with the given codec.

    Listing an encoder in `ffmpeg -encoders` does not guarantee the
    hardware is present (e.g. NVENC listed on a machine with no NVIDIA
    GPU fails at encode time), so a short real encode is the certain
    check. Returns True only when ffmpeg exits 0.
    """
    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if codec == "h264_vaapi":
        cmd += VAAPI_INPUT_ARGS
    cmd += ["-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30"]
    if codec == "h264_vaapi":
        cmd += ["-vf", "format=nv12,hwupload"]
    cmd += ["-c:v", codec, *extra_args, "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        return result.returncode == 0
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False


def reset_cache():
    """Clear the cached `ffmpeg -encoders` output (mainly for tests)."""
    _cache["output"] = None
    _cache["done"] = False
    _verified_cache["codec"] = None
    _verified_cache["args"] = None
    _verified_cache["done"] = False


def _available_encoders(encoders_output):
    """Parse the set of encoder names from `ffmpeg -encoders` text."""
    available = set()
    for line in (encoders_output or "").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            available.add(parts[1])
    return available


def select_encoder(encoders_output):
    """
    Pure function: pick (codec, extra_args) from `ffmpeg -encoders` text.

    Encoder names are matched as whole tokens (the second column of each
    line), so e.g. "libx264rgb" does not count as "libx264". Unknown or
    empty output falls back to libx264 defaults.
    """
    available = _available_encoders(encoders_output)
    for codec in ENCODER_PRIORITY:
        if codec in available:
            return codec, list(ENCODER_ARGS[codec])
    return "libx264", list(ENCODER_ARGS["libx264"])


def detect_encoder(encoders_output=None, verify=False):
    """
    Detect the best available H.264 encoder.

    Args:
        encoders_output: optional `ffmpeg -encoders` text to select from
            (used by tests). When None, ffmpeg is run once and cached.
        verify: when True (and encoders_output is None), each hardware
            candidate is confirmed with a 1-second test encode before it
            is picked, so a listed-but-broken encoder (e.g. NVENC with no
            GPU) falls through to the next working one. The verified
            result is cached per process. Pass verify=False (or explicit
            encoders_output) for a pure list-based selection.

    Returns:
        (codec, extra_args), e.g. ("h264_nvenc", ["-preset", "p4", ...]).
        Returns ("libx264", ["-preset", "veryfast", "-crf", "21"]) when
        ffmpeg is missing or no hardware encoder is listed (or verified).
    """
    if encoders_output is not None:
        return select_encoder(encoders_output)
    if verify and _verified_cache["done"]:
        return _verified_cache["codec"], list(_verified_cache["args"])
    output = _run_ffmpeg_encoders()
    if not verify:
        return select_encoder(output)
    available = _available_encoders(output)
    for codec in ENCODER_PRIORITY:
        if codec not in available:
            continue
        args = list(ENCODER_ARGS[codec])
        # libx264 is the CPU fallback: trust the listing, no test needed.
        if codec == "libx264" or _test_encode(codec, args):
            _verified_cache.update({"codec": codec, "args": args,
                                    "done": True})
            return codec, list(args)
    # Nothing verified — last resort is still libx264 (ffmpeg is broken
    # or missing; the render will surface the real error).
    fallback = list(ENCODER_ARGS["libx264"])
    _verified_cache.update({"codec": "libx264", "args": fallback,
                            "done": True})
    return "libx264", list(fallback)
