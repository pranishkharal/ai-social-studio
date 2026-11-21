"""
meme_sounds.py — Meme sounds + visual gags at the payoff of a clip.

Nine moods (embarrassing, fail, shock, hype, win, awkward, cringe, funny,
suspense), each a *synthesised* sound — vine boom, airhorn, sad trombone,
crickets, heartbeat — made with numpy, no samples, so there is nothing to
licence and nothing to download. Each mood also carries a visual gag
(freeze-frame, slow push-in, shake, flash, vignette, desaturation) applied
with ffmpeg at the payoff point.

OPT-IN ONLY, default OFF: the TikTok study found the top clipping
accounts use no effects, so a meme layer is never applied unless a clip
asks for it (``edit_clip(meme_mood="shock")`` or the ``MEME_SOUNDS_ENABLED``
env var). Guardrails, enforced in :func:`apply_meme`:
  * at most one meme per clip (one mood, one call);
  * never in the first 3 s (``HOOK_SAFE``) — the hook is sacred;
  * a meme ending may extend the clip by at most ``MAX_EXT`` (3 s).

Sounds are generated on first use into a cache directory (``MEME_SOUNDS_DIR``
env var, else a temp-dir folder) — never committed to the repo.

Technique adapted from gta6-clipper (https://gitlab.com/nocodehack.io/
gta6-clipper, MIT licence), meme.py — the synthesis recipes and the
mood table, reimplemented here for this repo's render path.
"""

import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 48000
HOOK_SAFE = 3.0   # never put a gag in the first seconds (the hook)
MAX_EXT = 3.0     # a meme ending may extend the clip by at most this
ENV_ENABLED = "MEME_SOUNDS_ENABLED"
ENV_SOUNDS_DIR = "MEME_SOUNDS_DIR"

# ---------------------------------------------------------------------------
# synthesis helpers
# ---------------------------------------------------------------------------


def _time(dur):
    return np.arange(int(dur * SR)) / SR


def _envelope(n, attack=0.005, release=0.05):
    """Fade in/out envelope so a sound never clicks at its edges."""
    env = np.ones(n)
    na = max(int(attack * SR), 1)
    nr = max(int(release * SR), 1)
    env[:na] = np.linspace(0, 1, na)
    env[-nr:] = np.minimum(env[-nr:], np.linspace(1, 0, nr))
    return env


def _lowpass(x, cutoff, order=3):
    """FFT low-pass (fast, no scipy needed)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= 1 / (1 + (freqs / cutoff) ** (2 * order))
    return np.fft.irfft(spec, len(x))


def _bandpass(x, lo, hi):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    spec *= (freqs > lo) & (freqs < hi)
    return np.fft.irfft(spec, len(x))


def _reverb(x, tail=0.5, mix=0.25, seed=1):
    """Cheap exponential-decay noise reverb, normalised to the input peak."""
    rng = np.random.default_rng(seed)
    n = int(tail * SR)
    impulse = rng.standard_normal(n) * np.exp(-np.arange(n) / (SR * tail / 5))
    impulse = _lowpass(impulse, 3000)
    wet = np.fft.irfft(
        np.fft.rfft(x, len(x) + n) * np.fft.rfft(impulse, len(x) + n),
        len(x) + n)
    wet = wet / (np.abs(wet).max() + 1e-9)
    out = np.zeros(len(wet))
    out[:len(x)] = x
    return out * (1 - mix) + wet * mix * np.abs(x).max()


def _normalise(x, peak_db=-3.0):
    return x / (np.abs(x).max() + 1e-9) * 10 ** (peak_db / 20)


# ---------------------------------------------------------------------------
# the sounds (all synthesised, royalty-free by construction)
# ---------------------------------------------------------------------------


def vine_boom():
    """Sub-bass thump: a sine sweeping 158 Hz -> 38 Hz with an impact burst."""
    t = _time(1.8)
    freq = 38 + 120 * np.exp(-t * 14)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    x = np.tanh(2.2 * np.sin(phase)) * np.exp(-t * 2.6)
    x += 0.35 * np.sin(2 * phase) * np.exp(-t * 6)
    x += 0.5 * _bandpass(np.random.default_rng(3).standard_normal(len(t)),
                         80, 900) * np.exp(-t * 30)
    return _normalise(_reverb(x * _envelope(len(t), 0.002, 0.2), 0.6, 0.18))


def sad_trombone():
    """Wah-wah-wah-wahhh: four descending brass notes, the last one drooping."""
    notes = [(233.1, 0.55), (220.0, 0.55), (207.7, 0.55), (196.0, 1.5)]
    out = []
    for i, (f0, dur) in enumerate(notes):
        t = _time(dur)
        last = i == len(notes) - 1
        vibrato = 1 + (0.004 * np.sin(2 * np.pi * 5.5 * t) if last else 0)
        droop = 1 - 0.06 * np.clip((t - 0.5) / 1.0, 0, 1) if last else 1
        phase = 2 * np.pi * np.cumsum(f0 * vibrato * droop) / SR
        wah = 0.25 + 0.75 * np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
        x = sum(np.sin(k * phase) * k ** -1.0 * np.exp(-k / (1 + 9 * wah))
                for k in range(1, 22))
        x = x + 0.04 * np.random.default_rng(i).standard_normal(len(t))
        out.append(x * _envelope(len(t), 0.04, 0.35 if last else 0.12)
                   * (0.6 + 0.4 * wah))
        out.append(np.zeros(int(0.05 * SR)))
    x = np.concatenate(out)
    return _normalise(_reverb(_bandpass(_lowpass(x, 2200), 80, 9000), 0.4, 0.15))


def record_scratch():
    """Vinyl scratch: a falling sawtooth + band noise + crackle."""
    t = _time(0.75)
    rng = np.random.default_rng(5)
    freq = 900 * np.exp(-t * 2.2) * (1 + 0.5 * np.sin(2 * np.pi * 9 * t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    saw = 2 * ((phase / (2 * np.pi)) % 1) - 1
    noise = _bandpass(rng.standard_normal(len(t)), 1500, 9000)
    x = (0.6 * saw + 0.9 * noise) * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 14 * t)))
    x += 0.6 * (rng.random(len(t)) > 0.9985) * rng.standard_normal(len(t))
    return _normalise(x * _envelope(len(t), 0.003, 0.03) * np.exp(-t * 1.5))


def airhorn():
    """MLG airhorn: three short blasts + one long one, Ab-C-Eb chord."""
    out = []
    for dur in (0.28, 0.28, 0.28, 1.1):
        t = _time(dur)
        x = 0
        for f in (415.3, 523.3, 622.3):
            phase = 2 * np.pi * np.cumsum(
                f * (1 + 0.003 * np.sin(2 * np.pi * 7 * t))) / SR
            x = x + sum(np.sin(k * phase) * k ** -0.65 for k in range(1, 16))
        x = (np.tanh(0.25 * x)
             * (1 + 0.15 * np.random.default_rng(1).standard_normal(len(t)))
             * _envelope(len(t), 0.012, 0.3 if dur > 1 else 0.06))
        out += [x, np.zeros(int(0.07 * SR))]
    return _normalise(_reverb(_lowpass(np.concatenate(out), 5000), 0.5, 0.15))


def bass_drop():
    """Riser hiss, then a sub drop that holds at 38 Hz."""
    t = _time(2.2)
    freq = np.where(t < 0.9, 220 * np.exp(-t * 2.4) + 38, 38)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    x = (np.tanh(1.6 * np.sin(phase)) * np.where(t < 0.9, 0.7, 1.0)
         * np.exp(-np.maximum(t - 0.9, 0) * 1.4))
    x += (0.5 * _bandpass(np.random.default_rng(2).standard_normal(len(t)),
                          2000, 9000) * np.exp(-t * 4) * (t < 0.9))
    return _normalise(x * _envelope(len(t), 0.01, 0.3))


def crickets():
    """Awkward silence: three chirping crickets at ~4 kHz + night air."""
    t = _time(3.4)
    x = np.zeros(len(t))
    for freq, rate, phase0 in [(4300, 2.6, 0.0), (4750, 2.1, 0.3),
                               (3900, 3.0, 0.6)]:
        gate = (((t * rate + phase0) % 1) < 0.35).astype(float)
        pulse = (np.sin(2 * np.pi * 38 * t) > 0.2).astype(float)
        x += np.sin(2 * np.pi * freq * t) * gate * pulse * 0.4
    x += 0.012 * _lowpass(np.random.default_rng(7).standard_normal(len(t)), 1500)
    return _normalise(x * _envelope(len(t), 0.2, 0.5), -8)


def heartbeat():
    """Slow double-thump heartbeat for suspense."""
    dur = 3.4
    x = np.zeros(int(dur * SR))
    for beat in np.arange(0.0, dur - 0.6, 0.85):
        for offset, gain, f0 in [(0, 1.0, 62), (0.2, 0.7, 52)]:
            n = int(0.22 * SR)
            t = np.arange(n) / SR
            i = int((beat + offset) * SR)
            x[i:i + n] += (np.sin(2 * np.pi * (f0 + 30 * np.exp(-t * 30)) * t)
                           * np.exp(-t * 18) * gain)
    return _normalise(_reverb(x, 0.3, 0.1), -4)


def whoosh():
    """Swept-noise whoosh for card transitions."""
    t = _time(0.7)
    noise = np.random.default_rng(4).standard_normal(len(t))
    y = np.zeros(len(t))
    centre = 400 + 5500 * np.sin(np.pi * t / 0.7) ** 2
    for lo in range(0, len(t), 480):
        seg = noise[lo:lo + 960]
        k = len(seg)
        y[lo:lo + k] += (_bandpass(seg, centre[lo] * 0.6, centre[lo] * 1.5)
                         * np.hanning(k))
    return _normalise(y * np.sin(np.pi * t / 0.7) ** 1.5)


def ding():
    """Bright bell ding (win / correct answer)."""
    t = _time(1.6)
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d)
            for f, a, d in [(1318.5, 1, 3), (2637, 0.35, 5), (3951, 0.15, 8)])
    return _normalise(x * _envelope(len(t), 0.002, 0.05))


def rimshot():
    """Ba-dum-tss: two drums + a cymbal splash."""
    out = np.zeros(int(1.7 * SR))
    rng = np.random.default_rng(6)

    def drum(at, freq, dur, noise_gain):
        t = _time(dur)
        i = int(at * SR)
        out[i:i + len(t)] += (
            np.sin(2 * np.pi * (freq + 90 * np.exp(-t * 40)) * t)
            * np.exp(-t * 14)
            + noise_gain * _bandpass(rng.standard_normal(len(t)), 1500, 9000)
            * np.exp(-t * 25))

    drum(0.0, 190, 0.25, 0.9)
    drum(0.16, 110, 0.35, 0.1)
    t = _time(1.2)
    i = int(0.42 * SR)
    out[i:i + len(t)] += (0.7 * _bandpass(rng.standard_normal(len(t)),
                                          5000, 14000) * np.exp(-t * 4))
    return _normalise(out)


SOUNDS = {
    "vine_boom": vine_boom,
    "sad_trombone": sad_trombone,
    "record_scratch": record_scratch,
    "airhorn": airhorn,
    "bass_drop": bass_drop,
    "crickets": crickets,
    "heartbeat": heartbeat,
    "whoosh": whoosh,
    "ding": ding,
    "rimshot": rimshot,
}

# ---------------------------------------------------------------------------
# moods
# ---------------------------------------------------------------------------

# mood -> {sound, effect, volume dB, offset s, hold s, tail s | "auto"}.
# effect names: zoom (slow push-in), punch (quick zoom), shake, flash,
# vignette, desat. hold = freeze-frame seconds at the payoff. tail =
# extra seconds the clip is held at the end so the sound finishes
# ("auto" = just long enough for the sound, capped by MAX_EXT).
MOODS = {
    "embarrassing": {"sound": "sad_trombone", "effect": ["zoom", "vignette", "desat"],
                     "volume": -3, "offset": 0.0, "hold": 1.2, "tail": "auto"},
    "fail":         {"sound": "sad_trombone", "effect": ["punch", "desat"],
                     "volume": -2, "offset": 0.1, "hold": 0.0, "tail": "auto"},
    "shock":        {"sound": "vine_boom", "effect": ["punch", "flash"],
                     "volume": 0, "offset": 0.0, "hold": 0.0, "tail": 0},
    "hype":         {"sound": "airhorn", "effect": ["punch", "shake", "flash"],
                     "volume": -2, "offset": 0.0, "hold": 0.0, "tail": 0},
    "win":          {"sound": "bass_drop", "effect": ["punch", "flash"],
                     "volume": -2, "offset": -0.15, "hold": 0.0, "tail": 0},
    "awkward":      {"sound": "crickets", "effect": ["zoom", "desat"],
                     "volume": -4, "offset": 0.0, "hold": 0.8, "tail": "auto"},
    "cringe":       {"sound": "record_scratch", "effect": ["flash", "desat"],
                     "volume": -2, "offset": 0.0, "hold": 1.0, "tail": 0.4},
    "funny":        {"sound": "rimshot", "effect": ["punch"],
                     "volume": -3, "offset": 0.1, "hold": 0.0, "tail": 0.6},
    "suspense":     {"sound": "heartbeat", "effect": ["zoom", "vignette"],
                     "volume": -3, "offset": -0.3, "hold": 0.0, "tail": 0},
}


def meme_enabled():
    """True when meme sounds are switched on (env MEME_SOUNDS_ENABLED).

    Default OFF — a clip must opt in per clip (``meme_mood=``) or via
    this env var."""
    return os.environ.get(ENV_ENABLED, "").strip().lower() in ("1", "true", "yes", "on")


def default_mood():
    """Mood from the MEME_MOOD env var, when meme sounds are enabled.

    Returns None when memes are off or no mood is set. An invalid mood
    name logs a warning and returns None instead of failing a render.
    """
    if not meme_enabled():
        return None
    mood = os.environ.get("MEME_MOOD", "").strip().lower()
    if not mood:
        return None
    if mood not in MOODS:
        print(f"[MemeSounds] Unknown MEME_MOOD {mood!r}; ignoring it.")
        return None
    return mood


def sounds_dir():
    """Where synthesised WAVs live (env MEME_SOUNDS_DIR, else temp)."""
    override = os.environ.get(ENV_SOUNDS_DIR, "").strip()
    if override:
        return Path(override)
    return Path(tempfile.gettempdir()) / "ai-social-studio-meme-sounds"


def write_wav(path, samples):
    """Write mono 16-bit 48 kHz WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def sound_path(name, directory=None):
    """Path of a synthesised sound WAV, generating it on first use."""
    if name not in SOUNDS:
        raise ValueError(f"Unknown meme sound {name!r}; "
                         f"expected one of {sorted(SOUNDS)}")
    d = Path(directory) if directory else sounds_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.wav"
    if not path.exists() or path.stat().st_size == 0:
        tmp = path.with_name(f"{name}.{os.getpid()}.tmp")
        write_wav(tmp, SOUNDS[name]())
        os.replace(tmp, path)
    return path


def make_all_sounds(directory=None):
    """(Re)synthesise every sound; returns the sorted sound names."""
    d = Path(directory) if directory else sounds_dir()
    for name in SOUNDS:
        sound_path(name, d)
    return sorted(SOUNDS)


# ---------------------------------------------------------------------------
# payoff + planning (pure, testable without ffmpeg)
# ---------------------------------------------------------------------------


def _creationflags():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def probe_video(path):
    """(width, height, duration_seconds) via ffprobe."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60,
        creationflags=_creationflags())
    data = json.loads(res.stdout)
    stream = data["streams"][0]
    return (int(stream["width"]), int(stream["height"]),
            float(data["format"]["duration"]))


def probe_audio_duration(path):
    """Duration of an audio file in seconds via ffprobe."""
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60,
        creationflags=_creationflags())
    return float(res.stdout.strip())


def find_payoff(path, duration):
    """Loudest 100 ms in the last 60% of the clip, never in the hook zone.

    Falls back to 70% through the clip when the audio cannot be read.
    """
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
             "-ar", "16000", "-f", "s16le", "-"],
            capture_output=True, timeout=300,
            creationflags=_creationflags()).stdout
        samples = np.frombuffer(raw, "<i2").astype(float)
        window = 1600  # 100 ms at 16 kHz
        n = len(samples) // window
        if n < 5:
            raise ValueError("audio too short")
        rms = np.sqrt((samples[:n * window].reshape(n, window) ** 2).mean(1))
        lo = int(max(duration * 0.4, HOOK_SAFE) * 10)
        hi = max(int((duration - 0.6) * 10), lo + 1)
        return float(np.argmax(rms[lo:hi]) + lo) / 10
    except Exception:
        return max(HOOK_SAFE, duration * 0.7)


def plan_meme(mood, duration, at=None, sound_len=None):
    """
    Work out the timing of a meme layer without running ffmpeg.

    Returns {"at", "sound_start", "hold", "tail", "total", "effects",
    "sound", "volume"}: "at" is the payoff second (clamped to
    [HOOK_SAFE, duration - 0.4]), "sound_start" is where the sound
    begins (payoff + the mood's offset, also hook-safe), "hold" the
    freeze-frame seconds, "tail" the extra hold at the end, "total"
    the final clip duration (never more than duration + MAX_EXT).
    """
    if mood not in MOODS:
        raise ValueError(f"Unknown meme mood {mood!r}; "
                         f"expected one of {sorted(MOODS)}")
    m = MOODS[mood]
    payoff = float(at) if at is not None else duration * 0.7
    payoff = min(max(payoff, HOOK_SAFE), max(duration - 0.4, HOOK_SAFE))
    hold = float(m.get("hold", 0))
    sound_start = max(HOOK_SAFE, payoff + float(m.get("offset", 0)))
    tail = m.get("tail", 0)
    if tail == "auto":
        length = float(sound_len) if sound_len else 0.0
        tail = max(0.0, sound_start + length - (duration + hold)) + 0.3
    tail = max(0.0, min(float(tail), MAX_EXT - hold))
    return {
        "at": round(payoff, 2),
        "sound_start": round(sound_start, 2),
        "hold": hold,
        "tail": round(tail, 2),
        "total": round(duration + hold + tail, 2),
        "effects": list(m.get("effect", [])),
        "sound": m["sound"],
        "volume": m.get("volume", 0),
    }


def meme_filter_complex(width, height, duration, plan, sound_len):
    """
    Build the ffmpeg filter_complex for a meme layer (pure string
    builder — no ffmpeg run, so it is fully testable offline).

    Video: pre-payoff | freeze-frame (when the mood holds) | post-payoff
    with the transient effects (zoom/punch/shake/flash) riding the first
    segment after the payoff, and the static effects (desat/vignette)
    on everything after it. Audio: the original (paused during the
    freeze) ducked under the meme sound, loudness-normalised to
    -14 LUFS at the end.
    """
    payoff = plan["at"]
    hold = plan["hold"]
    tail = plan["tail"]
    total = plan["total"]
    sound_start = plan["sound_start"]
    effects = set(plan["effects"])

    # --- transient zoom/shake expression (first segment after payoff) ---
    zoom_terms = []
    dx = dy = "0"
    if "zoom" in effects:
        zoom_terms.append(f"0.14*min(t/{hold if hold else 1.5:.3f},1)")
    if "punch" in effects:
        zoom_terms.append("0.12*max(0,1-t/0.35)")
    if "shake" in effects:
        zoom_terms.append("0.05")
        dx = "20*sin(t*70)*max(0,1-t/0.5)"
        dy = "14*cos(t*83)*max(0,1-t/0.5)"
    zoom = "1+" + "+".join(zoom_terms) if zoom_terms else None
    transient = ""
    if zoom:
        transient = (f"scale=w='2*trunc(iw*({zoom})/2)':"
                     f"h='2*trunc(ih*({zoom})/2)':eval=frame,"
                     f"crop={width}:{height}:'(iw-{width})/2+{dx}':"
                     f"'(ih-{height})*0.4+{dy}',")
    flash = "fade=t=in:st=0:d=0.22:color=white," if "flash" in effects else ""
    static = ""
    if "desat" in effects:
        static += "eq=saturation=0.25:contrast=1.05,"
    if "vignette" in effects:
        static += "vignette=angle=0.95,"
    base = "fps=30,setsar=1,format=yuv420p,"

    video = []
    video.append(f"[0:v]split={3 if hold else 2}[a]"
                 f"{'[b]' if hold else ''}[c]")
    video.append(f"[a]trim=0:{payoff:.3f},setpts=PTS-STARTPTS,{base}null[pre]")
    if hold:
        video.append(
            f"[b]trim=start={payoff:.3f}:end={payoff + 0.1:.3f},"
            f"setpts=PTS-STARTPTS,fps=30,trim=end_frame=1,"
            f"tpad=stop_mode=clone:stop_duration={hold:.3f},"
            f"setpts=PTS-STARTPTS,{transient}{flash}{static}{base}null[hold]")
    post_effects = ("" if hold else transient + flash) + static
    pad = (f"tpad=stop_mode=clone:stop_duration={tail:.3f},"
           if tail else "")
    video.append(f"[c]trim=start={payoff:.3f},setpts=PTS-STARTPTS,"
                 f"{post_effects}{base}{pad}null[post]")
    if hold:
        video.append("[pre][hold][post]concat=n=3:v=1:a=0[vo]")
    else:
        video.append("[pre][post]concat=n=2:v=1:a=0[vo]")

    # --- audio: original (paused during the freeze) + the sound ---
    audio = [
        f"[0:a]aresample={SR},aformat=channel_layouts=stereo,asplit=2[x][y]",
        f"[x]atrim=0:{payoff:.3f},asetpts=PTS-STARTPTS[ap]",
        f"[y]atrim=start={payoff:.3f},asetpts=PTS-STARTPTS[aq]",
    ]
    if hold:
        audio.append(f"anullsrc=r={SR}:cl=stereo,atrim=0:{hold:.3f},"
                     f"asetpts=PTS-STARTPTS[ah]")
        audio.append("[ap][ah][aq]concat=n=3:v=0:a=1[ao]")
    else:
        audio.append("[ap][aq]concat=n=2:v=0:a=1[ao]")
    sound_play = min(sound_len, total - sound_start)
    audio.append(
        f"[ao]apad=whole_dur={total:.3f},"
        f"volume=0.3:enable='between(t,{sound_start:.3f},"
        f"{sound_start + sound_play:.3f})'[ad]")
    audio.append(
        f"[1:a]aresample={SR},aformat=channel_layouts=stereo,"
        f"atrim=0:{sound_play:.3f},"
        f"afade=t=out:st={max(sound_play - 0.25, 0):.3f}:d=0.25,"
        f"volume={plan['volume']}dB,"
        f"adelay={int(sound_start * 1000)}|{int(sound_start * 1000)}[as]")
    audio.append(
        f"[ad][as]amix=inputs=2:normalize=0:duration=first,"
        f"loudnorm=I=-14:TP=-1.5:LRA=9,atrim=0:{total:.3f}[ao2]")
    return ";".join(video + audio)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def apply_meme(src, mood, out=None, at=None, sounds_directory=None):
    """
    Apply one meme layer (sound + visual gag) to a finished clip.

    Args:
        src: finished 9:16 clip.
        mood: one of the 9 MOODS keys.
        out: destination (default: <src stem>_<mood>.mp4 next to src).
        at: payoff second; None means the loudest moment (find_payoff).
        sounds_directory: where the synthesised WAVs live (default
            sounds_dir()).

    Returns {"out", "at", "sound", "hold", "tail", "duration"}.

    The payoff is always clamped out of the hook zone (first 3 s) and
    the clip grows by at most MAX_EXT seconds. Raises ValueError for an
    unknown mood, RuntimeError when ffmpeg fails.
    """
    if mood not in MOODS:
        raise ValueError(f"Unknown meme mood {mood!r}; "
                         f"expected one of {sorted(MOODS)}")
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"Source clip not found: {src}")
    out = Path(out) if out else src.with_name(f"{src.stem}_{mood}.mp4")
    width, height, duration = probe_video(src)
    sound_file = sound_path(MOODS[mood]["sound"], sounds_directory)
    sound_len = probe_audio_duration(sound_file)
    payoff = float(at) if at is not None else find_payoff(src, duration)
    plan = plan_meme(mood, duration, at=payoff, sound_len=sound_len)
    filt = meme_filter_complex(width, height, duration, plan, sound_len)

    filter_file = Path(tempfile.mkdtemp(prefix="meme_")) / "graph.txt"
    filter_file.write_text(filt, encoding="utf-8")
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(src), "-i", str(sound_file),
             "-filter_complex_script", str(filter_file),
             "-map", "[vo]", "-map", "[ao2]",
             "-t", f"{plan['total']:.3f}",
             "-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-movflags", "+faststart", str(out)],
            capture_output=True, text=True, timeout=900,
            creationflags=_creationflags())
    finally:
        import shutil
        shutil.rmtree(filter_file.parent, ignore_errors=True)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError("ffmpeg meme render failed:\n" + res.stderr[-1500:])
    return {
        "out": str(out),
        "at": plan["at"],
        "sound": sound_file.name,
        "hold": plan["hold"],
        "tail": plan["tail"],
        "duration": plan["total"],
    }
