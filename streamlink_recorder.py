"""Streamlink-based live recording (gta6-clipper feature 15).

A second way to capture a live stream, alongside the ffmpeg HLS
capture in kick_monitor.py: streamlink negotiates the stream itself
and pipes the video into ffmpeg, which writes rolling 4 s segments
into a 10-minute buffer. Streamlink is the standard tool for this
job, and it is worth having as a fallback because the app's own
ffmpeg HLS capture has failed before (403 / truncated streams in
this VM).

Behaviour:
- 4 s segments, named by wall-clock start time, so restarts are
  harmless; segments older than 10 minutes are deleted.
- Stall detection: 30 s with no new segment restarts the recorder
  (a stream can go silent without closing the connection).
- Disk-full guard: stops cleanly under 2 GB free instead of filling
  the disk.
- Exponential back-off when the stream is offline (3 s, doubling to
  a 5-minute cap) instead of hammering the platform every few
  seconds.

Honest limit: streamlink talks to the same network as ffmpeg, so it
may hit the same VM network limits (403 / truncated streams). It was
live-tested in this VM — see the build report for the result. Do not
claim it works without that test.

Usage:
    python streamlink_recorder.py https://kick.com/adinross --out buffer/adinross
    python streamlink_recorder.py https://www.twitch.tv/xqc --out hunt/twitch_xqc
"""

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Seconds of video kept in the rolling buffer (matches
# config.STREAM_BUFFER_MINUTES = 10).
KEEP_SECONDS = 600
# Seconds without a new segment before the recorder restarts.
STALL_SECONDS = 30
# Stop recording when less than this much disk is free.
MIN_FREE_DISK_BYTES = 2 * 1024 ** 3
# Back-off when the stream is offline: 3 s, doubling after 10 idle
# restarts, capped at 5 minutes.
BACKOFF_BASE_SECONDS = 3
BACKOFF_MAX_SECONDS = 300
BACKOFF_IDLE_RESTARTS = 10
# Segment length and the streamlink quality ladder.
SEGMENT_SECONDS = 4
STREAMLINK_QUALITIES = "720p60,720p,best"
# Override the quality ladder (e.g. STREAMLINK_QUALITY=360p on a slow
# connection — in this VM the 720p60 stream stalls while 160p/360p
# flows, so a lower ladder is the difference between recording and
# nothing at all).
QUALITY_ENV_VAR = "STREAMLINK_QUALITY"


def backoff_seconds(idle_restarts):
    """Wait before the next restart attempt: 3 s while the stream
    might just be hiccuping, then doubling (up to 5 min) when it
    looks offline."""
    if idle_restarts < BACKOFF_IDLE_RESTARTS:
        return BACKOFF_BASE_SECONDS
    return min(BACKOFF_MAX_SECONDS,
               BACKOFF_BASE_SECONDS * 2 ** (idle_restarts - BACKOFF_IDLE_RESTARTS + 1))


def streamlink_command(url, python=sys.executable, quality=None):
    """streamlink piped to stdout: ffmpeg reads the stream from the
    pipe, so streamlink handles the negotiation and ffmpeg only
    segments. ``quality`` defaults to the STREAMLINK_QUALITY env var
    or the 720p60 ladder."""
    quality = quality or os.environ.get(QUALITY_ENV_VAR) or STREAMLINK_QUALITIES
    return [python, "-m", "streamlink", "--stdout",
            "--twitch-disable-ads", url, quality]


def ffmpeg_segment_command(segment_dir):
    """ffmpeg reads the piped stream on stdin and writes 4 s
    stream-copy segments named by wall-clock start time."""
    return ["ffmpeg", "-loglevel", "error", "-i", "-",
            "-c", "copy", "-f", "segment",
            "-segment_time", str(SEGMENT_SECONDS),
            "-strftime", "1", "-reset_timestamps", "1",
            str(Path(segment_dir) / "s_%Y%m%d_%H%M%S.ts")]


def disk_full(path, minimum=MIN_FREE_DISK_BYTES):
    """True when less than ``minimum`` bytes are free at ``path``."""
    try:
        return shutil.disk_usage(os.path.abspath(path)).free < minimum
    except OSError:
        return False


def _mtime(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0  # deleted by cleanup between glob and stat


def list_segments(segment_dir):
    """[(start_timestamp, path)] of the recorded segments, oldest
    first. Segment names carry the wall-clock start (s_YYYYMMDD_HHMMSS)."""
    out = []
    for path in Path(segment_dir).glob("s_*.ts"):
        try:
            start = dt.datetime.strptime(path.stem[2:], "%Y%m%d_%H%M%S").timestamp()
        except ValueError:
            continue
        out.append((start, path))
    return sorted(out)


def prune_segments(segment_dir, keep_seconds=KEEP_SECONDS, now=None):
    """Delete segments older than ``keep_seconds``. Returns how many
    were removed."""
    now = now or time.time()
    removed = 0
    for start, path in list_segments(segment_dir):
        if start < now - keep_seconds:
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass  # still open (Windows lock): next pass
    return removed


def latest_window(segment_dir, seconds, output_path, timeout=120):
    """Concatenate the last ``seconds`` of recorded segments into one
    MP4 at ``output_path`` (stream copy, no re-encode). Returns the
    output path, or None when there are no segments. The newest
    segment is skipped — it is still being written."""
    done = list_segments(segment_dir)[:-1]
    if not done:
        return None
    cutoff = time.time() - seconds
    selected = [(t, p) for t, p in done if t >= cutoff] or done[-1:]
    list_path = Path(segment_dir) / f"window_{int(time.time())}.txt"
    list_path.write_text("".join(f"file '{p.name}'\n" for _, p in selected),
                         encoding="utf-8")
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", str(list_path), "-c", "copy",
             "-bsf:a", "aac_adtstoasc", str(output_path)],
            capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0 or not Path(output_path).exists():
            return None
        return str(output_path)
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except OSError:
            pass


def cut_window(segment_dir, output_path, at_timestamp, seconds_before=25,
               seconds_after=10, timeout=180):
    """Cut [at - before, at + after) out of the recorded segments into
    ``output_path`` (re-encoded, so the cut is frame-accurate).
    Returns the output path, or None when no segments cover it."""
    start, end = at_timestamp - seconds_before, at_timestamp + seconds_after
    done = list_segments(segment_dir)[:-1]  # newest is still being written
    first = max([i for i, (t, _) in enumerate(done) if t <= start], default=0)
    selected = [(t, p) for t, p in done[first:] if t <= end]
    if not selected:
        return None
    list_path = Path(segment_dir) / f"cut_{int(time.time())}.txt"
    list_path.write_text("".join(f"file '{p.name}'\n" for _, p in selected),
                         encoding="utf-8")
    try:
        res = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
             "-safe", "0", "-i", str(list_path),
             "-ss", f"{max(start - selected[0][0], 0):.2f}",
             "-t", str(seconds_before + seconds_after),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", str(output_path)],
            capture_output=True, text=True, timeout=timeout)
        if res.returncode != 0 or not Path(output_path).exists():
            return None
        return str(output_path)
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except OSError:
            pass


class StreamlinkRecorder:
    """Records one live stream into rolling 4 s segments.

    ``run()`` loops forever (restart on stall or crash) until
    ``stop()`` is called; ``run_once()`` runs a single record attempt
    and returns why it ended — the unit-testable core of the loop.
    """

    def __init__(self, url, segment_dir, stall_seconds=STALL_SECONDS,
                 keep_seconds=KEEP_SECONDS, python=sys.executable,
                 sleep=time.sleep, quality=None):
        self.url = url
        self.segment_dir = Path(segment_dir)
        self.stall_seconds = stall_seconds
        self.keep_seconds = keep_seconds
        self.python = python
        self.sleep = sleep
        self.quality = quality
        self.stop_requested = False
        self.idle_restarts = 0
        self.log_path = self.segment_dir.parent / "streamlink.log"

    def stop(self):
        self.stop_requested = True

    def _newest_segment_mtime(self):
        segments = list_segments(self.segment_dir)
        return max((_mtime(p) for _, p in segments), default=0)

    def run_once(self, max_seconds=None):
        """One record attempt: start streamlink + ffmpeg, watch for a
        stall, kill both when it ends. Returns "stalled", "ended",
        "disk_full", "ffmpeg_failed" or "stopped"."""
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        if disk_full(self.segment_dir):
            print("disk almost full (under 2 GB free): recording stopped",
                  flush=True)
            return "disk_full"
        err = open(self.log_path, "a", encoding="utf-8")
        streamlink = subprocess.Popen(
            streamlink_command(self.url, self.python, self.quality),
            stdout=subprocess.PIPE, stderr=err)
        try:
            ffmpeg = subprocess.Popen(
                ffmpeg_segment_command(self.segment_dir),
                stdin=streamlink.stdout)
        except OSError as e:
            streamlink.kill()
            err.close()
            print(f"ffmpeg could not start: {e}", flush=True)
            return "ffmpeg_failed"
        streamlink.stdout.close()
        started = time.time()
        got_segment = False
        reason = "ended"
        try:
            while ffmpeg.poll() is None and not self.stop_requested:
                if max_seconds is not None and time.time() - started > max_seconds:
                    reason = "stopped"
                    break
                self.sleep(5)
                newest = self._newest_segment_mtime()
                got_segment = got_segment or newest > started
                if time.time() - max(newest, started) > self.stall_seconds:
                    print(f"stream stalled {self.stall_seconds} s: "
                          "restarting the recorder", flush=True)
                    err.write(time.strftime("%H:%M:%S") + " stall: restart\n")
                    reason = "stalled"
                    break
        finally:
            for proc in (ffmpeg, streamlink):
                try:
                    proc.kill()
                except OSError:
                    pass
            err.close()
        if self.stop_requested:
            reason = "stopped"
        self.idle_restarts = 0 if got_segment else self.idle_restarts + 1
        return reason

    def run(self):
        """Record forever, restarting on stall/crash with back-off
        when the stream looks offline. Returns when stop() is called
        or the disk fills up."""
        while not self.stop_requested:
            prune_segments(self.segment_dir, self.keep_seconds)
            reason = self.run_once()
            if reason in ("disk_full", "stopped"):
                return reason
            self.sleep(backoff_seconds(self.idle_restarts))
        return "stopped"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Record a live stream with streamlink + ffmpeg "
                    "(rolling 4 s segments, 10-minute buffer).")
    parser.add_argument("url", help="Stream URL (Kick, Twitch, YouTube live, ...)")
    parser.add_argument("--out", required=True,
                        help="Directory for the segments (a 'seg' subfolder is used)")
    parser.add_argument("--seconds", type=int, default=0,
                        help="Stop after N seconds (0 = run until stopped)")
    args = parser.parse_args(argv)
    segment_dir = Path(args.out) / "seg"
    recorder = StreamlinkRecorder(args.url, segment_dir)
    if args.seconds:
        import threading
        threading.Timer(args.seconds, recorder.stop).start()
    return recorder.run()


if __name__ == "__main__":
    sys.exit(0 if main() in ("stopped", None) else 2)
