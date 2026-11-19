# Changelog

All notable changes to ai-social-studio are documented here.

## [Unreleased] — Hook banner shows for 4 seconds only

- **Hook banner timing** (`clip_editor.py`): the white hook banner now
  shows for the first 4 seconds of a clip only
  (`BANNER_DURATION_SECONDS = 4.0`), fades out over 0.5 s
  (`BANNER_FADE_SECONDS`), and stays hidden for the rest of the clip.
  Applies to every layout (classic, face-view, camera-switch,
  split-screen) — the banner is composited in one place, after the
  layout transform. The Kick watermark bar is branding and stays
  visible throughout. Implementation: the still card image is looped
  (finite loop — `loop=-1` stalls the filter graph) and trimmed to
  the banner window, with a `fade=t=out:alpha=1` plus an overlay
  `enable='between(t,0,4.50)'`. The trim matters: an untrimmed loop
  stretches the output to the loop length. Covered by
  `tests/test_banner_timing.py` (5 tests).

## [1.1.1] — 2026-11-19 — Live-demo bug fixes + no subtitles by default

Seven bugs found during a live demo run on adinross (19 Nov 2026), plus
the no-subtitles default for Kick clips. Verified: full test suite
484 passed, 1 skipped (the pre-existing OpenCV skip); every fix has a
targeted repro check and a regression test.

### Behaviour change — no subtitles by default
- Kick live-stream clips are now rendered **without burned-in
  subtitles** by default. Whisper still transcribes every clip —
  silence cutting depends on the word timestamps — the words just are
  not drawn on the video. Opt back in with `ENABLE_SUBTITLES=true`,
  `pipeline.process_moment(subtitles=True)`, or the dashboard's
  auto-captions checkbox. Karaoke style (`SUBTITLE_STYLE=karaoke`)
  only applies when subtitles are enabled.

### Bug fixes
- **Hype detector** (`hype_detector.py`): ffmpeg 8 prints
  `mean_volume` before `max_volume`, so the old max-then-mean regex
  never matched and every clip fell back to hype score 10.0. The
  parser now searches for each value independently (both orders work).
  The analysis is also audio-only now (`-vn`): decoding the 1080p60
  video stream to null was slow enough to hit the 10 s timeout and
  fall back to 10.0 on a loaded machine — audio-only takes under a
  second (timeout raised to 30 s as a safety margin).
- **Whisper transcription** (`requirements.txt`): faster-whisper
  1.2.1 (the latest release) passes `metadata_errors="ignore"` to
  `av.open()`, which PyAV 19 removed — transcription failed outright
  on a fresh install. `av` is now pinned to `>=15,<19` (15–18 accept
  the keyword and ship wheels for Windows, macOS and Linux).
- **Camera-switch rendering** (`render_modes.py`): the filter failed
  on ffmpeg 8 when the same camera was used in two or more segments
  (one filter output fed into two trims). The builder now emits an
  explicit `split` per repeated camera plus `setsar=1` on every
  segment. Covered by a test that actually runs ffmpeg.
- **Auto-thumbnails** (`thumbnail.py`, `clip_editor.py`): the
  thumbnail's hook-text box used to land on top of the white hook
  banner already burned into the frame. The thumbnail frame is now
  grabbed from the clean pre-overlay source clip, so the thumbnail's
  text box is the only text on the thumbnail — never two text boxes
  on one frame. Because that source frame is 16:9 while the clip is
  9:16, the thumbnail is cropped to portrait first, centred on the
  streamer's face when one is detected (else centred), and the
  hook-text band sits below the facecam instead of covering the face.
- **Hook banner vs facecam** (`clip_editor.py`): the top banner
  covered the streamer's face in speaker-tracking / camera-switch /
  split-screen layouts (the facecam sits at the top of the frame).
  The banner is now layout-aware: it drops below the facecam zone
  (y 800) for face-tracking layouts; the classic layout keeps the
  top banner.
- **OpenCV pin** (`requirements.txt`): OpenCV 5.x removed
  `CascadeClassifier`, which face tracking relies on.
  `opencv-python-headless` is now pinned to `>=4.9,<5`.
- **Live capture** (`kick_monitor.py`, `requirements.txt`): the v2
  channel endpoint no longer reliably returns `playback_url` — the
  monitor now reads it from `GET /api/v2/channels/{slug}/livestream`
  (falling back to the channel payload when present), and ffmpeg gets
  the `Referer: https://kick.com/` header plus a browser User-Agent
  that Kick's IVS playback requires (bare ffmpeg requests get a 400).
  API fetches also retry a few times on transient failures, and the
  monitor now fetches via `requests` instead of the stdlib urllib:
  urllib's http.client got truncated responses (`IncompleteRead`)
  from Kick's CDN/proxies on this API while requests handled the
  same calls reliably (0/13 urllib attempts vs 10/10 requests
  attempts in live testing).

## [1.1.0] — 2026-11-19 — Live Clipping Upgrades

All 17 features from the suggestions doc, plus a Tier 5 of rendering
upgrades borrowed from
[opensource-clipping](https://github.com/NaufalRizqullah/opensource-clipping)
(MIT licence). Every new behaviour is off by default; the engine behaves
exactly as before until you opt in via `.env`.

### Tier 1 — Live core
- **Auto-clip watcher** (`auto_clip_watcher.py`): background thread that
  samples the newest buffer chunk, blends audio hype (0.6) with chat hype
  (0.4), and fires `process_moment` when the score clears
  `AUTO_CLIP_MIN_SCORE`. Off by default.
- **Live chat hype** (`chat_hype.py`): Kick chat over the Pusher websocket
  (`chatrooms.{id}.v2`), emote/keyword + velocity scoring.
- **Kick webhooks** (`kick_webhooks.py`): `POST /api/webhooks/kick` receiver
  with pure-stdlib RSA-SHA256 signature verification, subscription helpers,
  and `GET /api/webhooks/status` + `POST /api/webhooks/subscribe` endpoints.
  Webhook events reuse the monitor's new `handle_stream_started` /
  `handle_stream_ended` hooks.
- **Clip cooldown + duplicate protection** (`clip_dedup.py`): per-streamer
  cooldown and Jaccard word-overlap dedup on title + summary + caption.

### Tier 2 — Reach & money
- **YouTube Shorts uploader** (`youtube_uploader.py`): real YouTube Data
  API v3 uploads (OAuth installed-app flow, resumable 8 MiB chunks).
- **Instagram Reels uploader** (`instagram_uploader.py`): Meta Graph API
  container → poll → publish. Needs a publicly reachable video URL.
- **Upload scheduler** (`upload_scheduler.py`): `upload_queue` table with
  per-platform spacing, daily caps, retry/backoff; runs as a background
  thread from the dashboard.
- **Earnings tracker** (`earnings.py`): `submissions` table, CPM earnings
  (views/100k × CPM), dashboard Earnings view.
- **Analytics feedback loop** (`analytics.py`): `clip_performance` table,
  AI-score bucket/category/streamer breakdowns, threshold recommendation,
  monthly report; dashboard Analytics view.

### Tier 3 — Better clips
- **Dead-space removal** (`silence_cutter.py`): Whisper word timestamps
  cut silent gaps before rendering; the pipeline remaps subtitle words to
  trimmed-clip time so captions never drift.
- **Auto-thumbnails** (`thumbnail.py`): best frame (peak audio → loudest
  RMS → largest face → densest speech → middle) + hook text, saved as
  `<clip>_thumb.jpg`. Never fails a render.
- **Per-streamer profiles** (`streamer_profiles.py` + `streamer_profiles.json`):
  clip length and layout per streamer; `process_moment(duration=None)` and
  `edit_clip(layout=None)` resolve from the profile.
- **Whisper config** (`ai_story_engine.py`): lazy model loading,
  `WHISPER_MODEL` (tiny/small/medium/…), `WHISPER_DEVICE` (cpu/cuda with
  automatic CPU fallback), beam size, compute type.
- **Multi-streamer detection** (`streamer_detection.py`): transcript
  mentions (primary) + face evidence (supporting) + optional
  `face_recognition`; `pick_submission_streamer` applies the Kick
  program's lowest-CPM rule. Results are stored in the new
  `clips.detected_streamers` column.

### Tier 4 — Reliability
- **Docker** (`Dockerfile`, `docker-compose.yml`, `.dockerignore`):
  python:3.11-slim with ffmpeg + fonts on port 8500.
- **Cross-platform fonts/paths**: no hardcoded `C:/Windows/Fonts` left;
  font resolution checks Windows first, then Linux font dirs.
- **Dashboard login** (`auth.py`): password or precomputed hash, HMAC-SHA256
  session cookies (7-day expiry), `/login` + `/logout`, middleware gate
  in `web_dashboard.py`. The Kick webhook endpoint stays public by design.
- **SSL verification fixed**: `CERT_NONE` removed from `kick_monitor.py`.
- **Discord / Telegram alerts** (`alerts.py`): streamer-live, clip-ready,
  upload done/failed. Daemon-threaded, never raises.
- **Retry + resume** (`retry_utils.py`): `@retry` decorator and a JSON
  `CheckpointStore`; the pipeline checkpoints footage/transcribe/analyze/
  render, and `pipeline.resume_clip` (via `POST /api/clips/{id}/retry`)
  reuses completed steps instead of starting over.
- **Tests + CI** (`.github/workflows/ci.yml`): 459 tests passing
  (1 skipped: the OpenCV smoke test needs a real cv2 install).

### Tier 5 — Rendering upgrades
- **Face tracking** (`speaker_tracker.py`): MediaPipe primary → YOLO
  optional → Haar fallback (`FACE_TRACKER_BACKEND`); Haar path is
  byte-identical to before.
- **Kinetic karaoke subtitles** (`karaoke_subtitles.py`): `{\k}` per-word
  highlight + bounce, hormozi/minimal styles; `SUBTITLE_STYLE=karaoke`.
- **Camera-switch / split-screen layouts** (`render_modes.py`): pure
  filter builders wired into `edit_clip` as `layout="camera_switch"` /
  `"split_screen"`; both fall back to speaker tracking when the speaker
  data is missing (the pipeline does not produce diarization yet).
- **B-roll** (`broll.py`): Pexels stock cutaways with Ken Burns zoom,
  overlaid mid-clip with fades (duration-preserving, audio stays in sync).
- **Background music** (`bgm.py`): local MP3 pool in `assets/bgm/<mood>/`,
  sidechain ducking under speech; a failed BGM render retries once
  without music.
- **Encoder auto-detect** (`encoder.py`): NVENC > AMF > VAAPI > libx264,
  each hardware candidate confirmed with a real 1-second test encode so
  a listed-but-GPU-less encoder falls through; `VIDEO_ENCODER` override.
- **LLM provider fallback** (`llm_providers.py`): Gemini →
  OpenAI-compatible local model (LM Studio / vLLM / NVIDIA NIM) →
  heuristic; the winning provider is tagged as `llm_provider`.

### Cross-platform & Android phone (added in the same release)
- **Platform helpers** (`platform_utils.py`): single source of truth for
  OS detection, font directories (Windows `%WINDIR%\Fonts`, macOS
  `~/Library/Fonts` + `/Library/Fonts` + `/System/Library/Fonts`,
  Linux truetype dirs), Windows-name → platform-equivalent font
  fallback (DejaVu / Liberation / Noto on Linux, Arial / Helvetica on
  macOS), colour-emoji font lookup, LAN IP detection, desktop-dir
  resolution.
- **No hardcoded Windows paths left**: `clip_editor.py`, `thumbnail.py`
  and `kick_program.py` all delegate font lookup to `platform_utils`;
  `generate_app_icons.py`, `test_card.py`, `test_emoji.py` and
  `test_safe_zone.py` use the same helpers; `parse_prestige_videos.py`
  takes its log path from argv / `PRESTIGE_LOG_PATH` and writes next to
  the script; `scratch_parse_tiktok.py`, `sync_overlays.py` and the
  `speaker_tracker.py` demo block resolve paths relative to the repo;
  the dashboard's Activity Logs header no longer shows a Windows path.
- **macOS / Linux launcher** (`start.sh`, executable): mirrors
  `start_engine.bat` — checks python3 + ffmpeg, installs deps on first
  run, passes `--host`/`--port`/`--no-browser` through to `main.py`.
- **Launcher** (`main.py`): `--host`/`--port`/`--no-browser` flags plus
  `DASHBOARD_HOST`/`DASHBOARD_PORT` env vars; prints the LAN URL for
  phone access when bound to `0.0.0.0`.
- **Desktop app + shortcut** (`desktop_app.py`,
  `create_desktop_shortcut.py`): pywebview window, else a browser
  app-mode window (Edge/Chrome on Windows, Chrome on macOS,
  Chrome/Chromium on Linux), else the default browser; shortcuts are a
  `.lnk` on Windows, a `.command` on macOS, a `.desktop` file on Linux.
- **Phone-friendly dashboard** (`templates/index.html`): viewport +
  `mobile-web-app-capable` metas (Add to Home screen), wrapping header
  with the subtitle hidden on small screens, `p-3` padding on phones,
  notification panel sized to the viewport, and a ≤640 px media query
  with 44 px minimum tap targets and no sideways scrolling.
- **Tests** (`tests/test_platform.py`): 34 tests covering per-OS font
  dirs, font fallback mapping, emoji fonts, `load_font` never crashing,
  the launcher flags, the mobile-friendly template markers, and the
  absence of hardcoded Windows font/repo paths in the core modules.
- Docker is unchanged (it already binds `0.0.0.0:8500`).

### Changed
- `process_moment(duration=None)` resolves clip length from the
  streamer's profile; the dashboard trigger accepts a null duration.
- `edit_clip` accepts `words` (silence trim), `bgm`/`broll`/`category`,
  `speaker_segments`/`trajectories`/`split_speakers`, and `encoder`.
- `ClipTriggerReq.duration` is now optional in the dashboard API.
- README architecture diagram corrected: live detection is 15 s polling
  (+ instant webhooks when configured), not "< 2 s" for polling.

### Known limitations (not verifiable without live access)
- Whether a user token can subscribe to `livestream.status.updated` for
  *other* broadcasters is unverified; webhooks may only cover your own
  channel, with polling remaining the mechanism for monitored streamers.
- YouTube OAuth / real upload, Instagram publish, Pexels, LM Studio,
  Discord/Telegram delivery, and the Docker image build were not
  exercised live (no keys / no Docker in the build VM); all are covered
  by mocked unit tests.
- Whisper small/medium + CUDA, MediaPipe/YOLO tracking quality, and VAAPI
  encoding are unverified on real hardware (stubbed/mocked in tests).
- TikTok scheduler dispatch still needs a manual Post click by design.
- `tarik` and `westcol` are monitored but have no Kick-program CPM rate
  yet, so the submission picker returns `("other", "/other")` for them.

## [1.0.0] — 2026-11-18

Initial release: Kick live monitoring with a rolling buffer, manual
1-click clipping, hype scoring, Whisper + Gemini story/metadata, 9:16
editor with speaker tracking and ASS subtitles, TikTok upload via
Playwright, FastAPI dashboard, SQLite storage, and the Discord bot
command generator.
