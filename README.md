# 🚀 Kick AI Clip Engine & Autonomous Social Publisher

An ultra-lightweight, automated system to monitor **Kick.com** live streamers, detect viral hype moments and story arcs using AI, auto-crop into **9:16 vertical TikTok/Reels format** with hook banners (subtitles off by default — opt in with `ENABLE_SUBTITLES=true`), and auto-publish to social media.

---

## ✨ What's new in v1.4.1 — Quote Line Off by Default

The quote line (the white spoken line under the headline) is now
**off by default** — Sara's call after watching the demo clips: it
reads as captioning under the headline. The headline in the white box
is unchanged. The quote is still picked and stored for every clip; it
is just not drawn on the video unless you opt in: tick the
**Quote line** checkbox on the dashboard (clip trigger or the clip's
re-render panel), pass `quote_line_enabled=True`, or set
`QUOTE_LINE_ENABLED=true`. See [CHANGELOG.md](CHANGELOG.md).

## ✨ What's new in v1.4.0 — gta6-clipper Upgrade

21 features borrowed from the open-source GTA 6 auto-clipper (MIT
licence), integrated into the existing features instead of bolted on:
the hype detector now also watches the video, every clip is
quality-checked before it is marked ready, and clips can be callbacks,
compilations and multi-part formats. See [CHANGELOG.md](CHANGELOG.md)
for the full list.

1. **Visual moment detection** — the hype detector now sees chases,
   crashes, fights, stunts, reactions and funny moments (optical flow
   + audio bands + laughter wobble, no AI), and the moment type feeds
   the story headlines (`visual_moments.py`). Turn it off with
   `VISUAL_DETECTION_ENABLED=0`.
2. **Game-event OCR** — reads WASTED / BUSTED / MISSION PASSED banners
   off the screen (RapidOCR, optional) and adds up to 15 hype points
   (`game_ocr.py`).
3. **Automatic clip QC** — every finished clip gets PASS / WARN / FAIL
   (duration, −14 LUFS loudness, black/frozen frames, cards, silence)
   before it is marked ready; FAIL gets one automatic re-cut
   (`clip_qc.py`).
4. **Loudness normalisation** — every render is normalised to −14 LUFS
   (two-pass loudnorm) with an audio-only safety pass (`loudness.py`).
5. **Learn from your own channel** — YouTube Shorts view counts are
   auto-filled from the channel's public Shorts (yt-dlp, no API key;
   set `YOUTUBE_CHANNEL_URL`) and the learned rules nudge moment
   scoring. YouTube only; TikTok stays manual (`channel_learn.py`).
6. **Multi-part formats** — callback, compilation, escalation,
   then-vs-now, reveal, chat bubbles and wait-for-it, built from
   moments the engine already found (`clip_formats.py`,
   `callback_finder.py`).
7. **Meme sounds** (opt-in, default off) — numpy-synthesised sounds +
   ffmpeg visual gags at the payoff, never in the first 3 s
   (`meme_sounds.py`, `meme_mood=` / `MEME_SOUNDS_ENABLED=1`).
8. **Active-speaker framing** — camera-switch and split-screen layouts
   now work for real: lip motion + speech loudness pick who is talking
   (`speaker_framing.py`).
9. **Whisper vocabulary prompt** — game + streamer names and chat slang
   are passed to Whisper so names transcribe correctly
   (`WHISPER_VOCAB_PROMPT` to override).
10. **Smarter chat spikes + Kick key negotiation** — spikes count
    distinct people (bots and floods cannot fake them), and the Kick
    Pusher key is negotiated fresh on every connect instead of
    hard-coded (`chat_hype.py`).
11. **VOD signals** — Twitch viewer clips and YouTube most-replayed
    peaks now score VOD windows (`vod_signals.py`).
12. **Peek, voice-over, clip rules, live hunt, streamlink recorder,
    lite mode** — a 4-frame strip per pick (`peek.py`); record your own
    line on a clip (`voiceover.py`); per-streamer clip-permission
    badges (`clip_rules.json`); one panel for who's live on Kick,
    Twitch and YouTube (`live_hunt.py`); a streamlink fallback recorder
    (`STREAMLINK_RECORDER=1`); lite mode auto-detects PCs under 12 GB
    RAM (`lite_mode.py`).
13. **Translation, YouTube analytics, review UX, self-update** —
    offline caption translation (Argos, 16 languages); YouTube
    analytics with retention (needs your API key / OAuth client);
    A/X keyboard shortcuts, Shorts-by-day view, drag-onto-week
    scheduling with a dry-run preview; self-update with backup and
    rollback (`UPDATE_MANIFEST_URL`).
14. **Rendering refinements** — sentence-boundary snapping, profanity
    masking in karaoke captions, per-format render cache, and named
    themes (`themes.json`, `CLIP_THEME`).

## ✨ What's new in v1.3.0 — Face Tracking Upgrade

The banner never covers the streamer's face anymore, MediaPipe face
tracking works for real, and face-view clips can follow the whole body
instead of just the face. See [CHANGELOG.md](CHANGELOG.md) for the full
list.

1. **Face-aware banner placement** — on face-tracking layouts the hook
   banner and quote line are placed where they do not cover any face in
   the rendered clip (sampled face boxes mapped through the crop
   trajectory, including the mirror). The quote placement only applies
   when the quote line is enabled (off by default since v1.4.1). No
   faces detected means the fixed
   positions (banner y 800, quote centred on y 1240). Works with both
   banner timings (4 s and whole-clip). Non-face-tracking layouts are
   unchanged (`face_avoidance.py`).
2. **MediaPipe face tracking works for real** — the backend was
   rewritten from the dead `mp.solutions` API (removed in mediapipe 1.x
   — the old code always silently fell back to Haar) to the Tasks API
   (`mp.tasks.vision.FaceDetector`). The ~230 KB BlazeFace model
   auto-downloads into `models/` on first use. Known limit, verified on
   a real adinross capture: BlazeFace sees the frame at 128×128, so it
   misses small facecam faces (~75 px in a 1920 px frame — 2% detection
   vs Haar's 100%). When a backend barely detects anything on a video,
   the tracker now re-analyses it with Haar automatically, so face-view
   clips stay correct on small-facecam streams.
3. **Full-body tracking option** — the speaker_tracking layout can
   follow the streamer's whole body (MediaPipe Pose) instead of the
   face. Pick it in the dashboard (Layout & Overlay Setup → Tracking
   Target), per clip via `tracking_target="body"`, or for all clips
   with `TRACKING_TARGET=body`. Best for IRL / standing / dancing
   streams; on gameplay streams it can lock onto game characters, so
   face tracking stays the default. Falls back to face tracking when no
   body is found. The pose model (~5.8 MB) auto-downloads into `models/`
   on first use.
4. **Mirrored face-view fix** — the tracker analyses the unmirrored
   source but the crop runs after `hflip`, so mirrored face-view clips
   used to frame the mirror image of the face. The trajectory is now
   mirrored too, so the crop lands on the face as rendered.

On Linux the MediaPipe Tasks API needs the system EGL/GLES libraries
(`sudo apt install libegl1 libgles2`); without them MediaPipe init fails
and the tracker falls back to Haar. `mediapipe>=1.0` is now a real
dependency in `requirements.txt`.

## ✨ What's new in v1.2.0 — TikTok Clipping Upgrades

Seven features learned from studying three TikTok clipping accounts
([@zysno](https://www.tiktok.com/@zysno),
[@godsclipper](https://www.tiktok.com/@godsclipper),
[@kicksclipping](https://www.tiktok.com/@kicksclipping) — 16 clips
watched): one peak moment, one bold headline that tells the whole
story, original audio, no captions, no effects, clips that end and
loop. See [CHANGELOG.md](CHANGELOG.md) for the full list.

1. **Story headlines** — titles now name the specific moment from the
   transcript ("Adinross finds his calling") instead of generic
   "COMPLETELY SPEECHLESS" lines. A smarter local builder does this with
   no API key; with a Gemini key the LLM prompt demands the same
   specificity (`story_headlines.py`).
2. **Quote line** — the best spoken line from the clip as a second
   caption under the headline (white bold text, black outline, lower
   third). One static line, not word-by-word captions — subtitles stay
   off by default. Off by default since v1.4.1: opt in with
   `QUOTE_LINE_ENABLED=true` or the dashboard checkbox.
3. **Short-clip mode** — tight 10–15 s clips cut around the single
   loudest moment (`SHORT_CLIP_ENABLED=true`, `SHORT_CLIP_SECONDS=12`),
   with a dashboard checkbox on the clip trigger.
4. **Auto hashtags** — the post caption is written as
   `title #<streamer> #fyp #kick`, plus a game tag (`#gta6`) when the
   game is known.
5. **Soft background music** — a quiet track mixed low under the
   original audio (base level 0.12, ducked further while someone talks).
   A gentle ambient track is generated on first use
   (`assets/bgm/chill/soft_pad.mp3`, synthesized with ffmpeg, royalty-free), so it
   works out of the box; drop your own MP3s into `assets/bgm/<mood>/`
   to use them instead. `BGM_ENABLED=true` or the dashboard checkbox.
6. **VOD / multi-source clipping** — clip past videos, not just live
   Kick streams: paste a Kick VOD, Twitch VOD/clip or YouTube link
   (podcasts included) or a local file, and the engine finds the
   loudest moment and clips it (`vod_clipper.py`, dashboard "Clip a
   Past Video" panel, or `python3 vod_clipper.py <url-or-file>`; URL
   downloads need `yt-dlp`, `pip install yt-dlp`).
7. **Headline timing setting** — the headline (and the quote line,
   when enabled) can now
   stay for the whole clip, like the TikTok accounts do, or keep Sara's
   4-second rule: `BANNER_TIMING=whole_clip` or `4s` (default `4s`),
   with a dashboard select on the clip trigger.

Also fixed: the render timeout now scales with clip length (1080p60
renders no longer fail at a fixed 180 s), and renders are verified to
end abruptly on the moment with no fade-out, so clips loop cleanly.

## ✨ What's new in v1.1.0 — Live Clipping Upgrades

Seventeen new features plus a Tier 5 of rendering upgrades borrowed from
[opensource-clipping](https://github.com/NaufalRizqullah/opensource-clipping).
Everything is off by default — the engine behaves exactly as before until
you opt in via `.env`. See [CHANGELOG.md](CHANGELOG.md) for the full list.

**Tier 1 — Live core**
1. **Auto-clip watcher** — clips hype moments automatically, no button click (`AUTO_CLIP_ENABLED=true`).
2. **Live chat hype** — Kick chat via Pusher websocket feeds the hype score (`CHAT_HYPE_ENABLED=true`).
3. **Kick webhooks** — instant live detection via `livestream.status.updated` (`/api/webhooks/kick`).
4. **Clip cooldown + duplicate protection** — no double-clips of the same moment.

**Tier 2 — Reach & money**
5. **YouTube Shorts uploader** — real YouTube Data API v3 uploads.
6. **Instagram Reels uploader** — Meta Graph API (needs a publicly reachable video URL).
7. **Upload scheduler** — spaced, capped, retrying upload queue per platform.
8. **Earnings tracker + analytics** — Kick-program submissions, CPM earnings, and a feedback loop that recommends your best AI-score threshold.

**Tier 3 — Better clips**
9. **Dead-space removal** — Whisper word timestamps cut silent gaps; subtitles are remapped to match when subtitles are enabled.
10. **Auto-thumbnails** — best frame + hook text, saved next to every render.
11. **Per-streamer profiles** — clip length and layout per streamer (`streamer_profiles.json`).
12. **Whisper config** — tiny/small/medium, CPU/GPU switch (`WHISPER_MODEL`, `WHISPER_DEVICE`).
13. **Multi-streamer detection** — picks the lowest-CPM streamer for Kick-program submission (POV rule).

**Tier 4 — Reliability**
14. **Docker** — `Dockerfile` + `docker-compose.yml`; cross-platform fonts/paths (no more Windows-only paths).
15. **Dashboard login** — password + session cookies; SSL verification fixed (no more `CERT_NONE`).
16. **Discord / Telegram alerts** — live, clip-ready, upload done/failed.
17. **Retry + resume + CI** — checkpoints let a failed clip resume mid-pipeline (`/api/clips/{id}/retry`); GitHub Actions runs the test suite.

**Tier 5 — Rendering upgrades**
- MediaPipe/YOLO face tracking with adaptive Haar fallback (a backend that barely detects anything on a video is re-analysed with Haar), face-aware banner placement (the banner dodges the face, and the quote line does too when it is enabled — off by default since v1.4.1), full-body tracking option (MediaPipe Pose), kinetic karaoke subtitles (`SUBTITLE_STYLE=karaoke`, only when subtitles are enabled via `ENABLE_SUBTITLES=true`), camera-switch + split-screen layouts, Pexels B-roll, background music with sidechain ducking (`BGM_ENABLED=true`), NVENC/AMF/VAAPI encoder auto-detect (verified with a test encode), and an LLM fallback chain (Gemini → local OpenAI-compatible model → heuristic).

**Cross-platform + Android phone**
- Runs on **Windows, macOS and Linux** (same code, `platform_utils.py` handles fonts/paths per OS) — `start.sh` for Mac/Linux, `start.bat`/`start_engine.bat` for Windows, Docker unchanged. The dashboard is **phone-friendly**: open it from your Android browser on the same Wi-Fi (`--host 0.0.0.0`, see below).

## ⚡ Why is it so "lightweight" on your PC?

| Approach | Traditional Screen Capture (Heavy) | Kick AI Clip Engine (Ultra-Lightweight) |
| :--- | :--- | :--- |
| **Method** | Keep a browser and OBS/screen recorder open 24/7 | **Direct Headless Ingestion**: Kick API + AWS IVS `.m3u8` stream-copy |
| **CPU/GPU load** | **60% - 90%** (loud fans, lag) | **< 1% CPU** (stream copy involves no video re-encoding) |
| **Screen restriction** | Screen must stay open all the time, can't minimise | **0% screen disturbance** (runs quietly in the background) |
| **Quality** | Depends on screen resolution, lag and frame drops | **Crystal-clear 1080p 60fps** direct source video |
| **AI detection** | Process a full 6 hours of heavy video frame-by-frame | **Smart event triggers**: audio decibel peaks (screaming/laughing) + chat spikes |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["Kick.com Live Streamers\n(e.g. Adin Ross, xQc)"] --> B["Kick Monitor\n(15 s polling + webhooks)"]
    A -->|Chat events| CH["Chat Hype\n(Pusher websocket)"]
    B -->|Live detected| C["AWS IVS Master Stream\n(Zero-Reencode Buffer)"]
    C --> D["Hype & Event Sniffer\n(Audio Peaks & Screams)"]
    CH -->|Chat velocity + emotes| W["Auto-Clip Watcher\n(0.6 audio + 0.4 chat)"]
    D --> W
    W -->|Viral Spike Detected| E["Clip Extraction\n(profile-length slice)"]
    E --> F["AI Story Engine\n(Whisper STT + Gemini / local LLM)"]
    F -->|Viral Title, Story, Captions, Punchline| G["Autonomous Video Editor\n(FFmpeg: silence trim, karaoke,\nBGM ducking, B-roll, HW encode)"]
    G -->|9:16 Vertical Crop, Animated Subtitles, Hook Banner| H["Upload Scheduler\n(TikTok / Reels / Shorts)"]
    G --> I["Interactive Web Dashboard\n(http://localhost:8500)"]
    I -->|Earnings, Analytics, Queue| H
```

---

## 📂 Project File Structure

- `main.py`: Master runner that starts the dashboard and background monitor in 1 click.
- `kick_monitor.py`: Detects live streams with no browser load and maintains a 10-minute rolling buffer.
- `hype_detector.py`: Measures audio volume spikes in the stream (screams, laughter, hype) in 0.2 seconds.
- `ai_story_engine.py`: Uses Google Gemini and local Whisper to extract the clip's full story, punchline, viral title, hook banner and captions.
- `clip_editor.py`: Converts 16:9 horizontal video to 9:16 vertical TikTok format, adds a top hook banner that shows for the first 4 seconds only (then fades out; the Kick bar stays visible), and word-by-word subtitles only when subtitles are enabled. The quote line (one spoken line under the headline) is off by default since v1.4.1 — opt in with `QUOTE_LINE_ENABLED=true`, `quote_line_enabled=True`, or the dashboard checkbox. Face-tracking layouts place the banner around the rendered face (and the quote line too, when enabled — see `face_avoidance.py`) and can follow the whole body instead of the face (`tracking_target="body"`).
- `speaker_tracker.py` / `face_avoidance.py` / `body_tracker.py`: Face-tracked auto-framing for 9:16 clips (MediaPipe Tasks API, YOLO or Haar backends, with adaptive Haar fallback when a backend barely detects anything); pure helpers that map tracker face boxes to output coordinates and pick banner/quote positions that do not overlap any face; full-body tracking via MediaPipe Pose as an alternative framing source.
- `screen_controller.py`: Playwright-based screen/browser agent that can scrub the Kick player back and forth and auto-upload to TikTok.
- `web_dashboard.py`: Local FastAPI web control centre (`http://localhost:8500`).
- `database.py`: SQLite database storing all streamers, detected clips and system logs.
- `config.py`: The system's central settings.
- `auto_clip_watcher.py` (Tier 1): Background thread that automatically creates clips on hype moments.
- `chat_hype.py` (Tier 1): Live hype score from Kick chat (Pusher websocket).
- `kick_webhooks.py` (Tier 1): Kick webhook receiver + subscription helper (instant live detection).
- `clip_dedup.py` (Tier 1): Clip cooldown and duplicate protection.
- `youtube_uploader.py` / `instagram_uploader.py` (Tier 2): YouTube Shorts and Instagram Reels uploaders.
- `upload_scheduler.py` (Tier 2): Spaced, capped, retrying upload queue.
- `earnings.py` / `analytics.py` / `dashboard_routers.py` (Tier 2): Kick programme earnings, performance feedback and their API routes.
- `silence_cutter.py` / `thumbnail.py` / `streamer_profiles.py` / `streamer_detection.py` (Tier 3): Dead-space cutting, auto-thumbnails, per-streamer profiles, multi-streamer detection.
- `auth.py` / `alerts.py` / `retry_utils.py` (Tier 4): Dashboard login, Discord/Telegram alerts, retry + checkpoint resume.
- `karaoke_subtitles.py` / `render_modes.py` / `broll.py` / `bgm.py` / `encoder.py` / `llm_providers.py` (Tier 5): Kinetic karaoke subtitles, camera-switch/split-screen, B-roll, background music, hardware encoder detection, LLM fallback.
- `Dockerfile` / `docker-compose.yml` (Tier 4): For running in a container.

---

## 🛠️ Quick Start

### 1. Go to the folder
```powershell
cd <path-to-this-repo>   # where you cloned ai-social-studio
```

### 2. (Optional) Add a Gemini API Key
Copy `.env.example` to `.env` and add your free Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
VIDEO_LAYOUT=blurred_backdrop
```
*(Note: even without an API key, the system still works 100% with the built-in local Whisper and smart heuristic engine!)*

### 3. Start the system
```powershell
# Windows:
start_engine.bat        # or: python main.py
# macOS / Linux:
chmod +x start.sh       # only needed once after cloning
./start.sh              # or: python3 main.py
```
This command immediately opens your web dashboard at `http://localhost:8500` in your default browser!

---

## 💻 Windows / macOS / Linux — Setup per platform

The same code runs on all three OSs; fonts and paths come from `platform_utils.py`, per OS.

| Need | Windows | macOS (MacBook) | Linux (Ubuntu) |
|---|---|---|---|
| Python | 3.11+ ([python.org](https://www.python.org/downloads/)) — tick "Add to PATH" during install | `brew install python@3.11` | `sudo apt install python3 python3-pip python3-venv` |
| ffmpeg | Install from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH (or `winget install ffmpeg`) | `brew install ffmpeg` | `sudo apt install ffmpeg` |
| Fonts | Windows fonts are found automatically | macOS fonts are found automatically (Impact / Arial Bold) | `sudo apt install fonts-dejavu-core fonts-liberation` (if not already installed) |
| Start | `start_engine.bat` (or `python main.py`) | `./start.sh` (or `python3 main.py`) | `./start.sh` (or `python3 main.py`) |
| 1-click shortcut | `python create_desktop_shortcut.py` → Desktop `.lnk` | → Desktop `.command` (you may need to allow it in Gatekeeper the first time) | → Desktop `.desktop` + app menu (choose "Allow Launching" the first time) |

On first run the launcher installs dependencies itself (if fastapi is not found, it runs `pip install -r requirements.txt`). `start.sh` passes through the `--host`, `--port` and `--no-browser` flags. The desktop app (`desktop_app.py`) also runs on all three OSs: pywebview window first, otherwise browser app-mode (Windows: Edge/Chrome, macOS: Chrome, Linux: Chrome/Chromium).

## 📱 Open the Dashboard from Your Android Phone (Phone browser)

No native app needed — the clipping engine runs on your computer/server and the dashboard opens in your phone's browser (mobile-friendly layout: large tap targets, a header and notifications panel that fit the phone screen).

**Option A — same Wi-Fi (at home/in the office):**
1. On the computer, put `DASHBOARD_HOST=0.0.0.0` and `DASHBOARD_PASSWORD=<your-password>` in `.env` (a password is essential — the dashboard will now be visible to the whole local network).
2. Start the app on the computer — the "On your phone (same Wi-Fi)" line at startup shows the IP, e.g. `http://192.168.1.5:8500`.
3. Open the same address in Chrome on your Android phone (phone and computer on the same Wi-Fi) and log in with the password.

**Option B — on a VPS / server (from anywhere):**
1. Run it on the server with Docker (`docker compose up --build`, already binds to `0.0.0.0:8500`) or `python main.py --host 0.0.0.0 --no-browser`.
2. Open the server's IP/domain + port in the browser (e.g. `http://<your-vps-ip>:8500`) — in production use an HTTPS reverse proxy (nginx/Caddy) + `DASHBOARD_PASSWORD`.

**Tip:** In Chrome, ⋮ → "Add to Home screen" turns the dashboard into a home-screen icon — it opens like a full-screen app.

---

## 🐳 Run with Docker (Docker Quickstart)

```powershell
copy .env.example .env
# put your keys in .env, then:
docker compose up --build
```
The dashboard will be at `http://localhost:8500`. Before the first `up`, the `kick_engine.db` file must exist (an empty file is fine) — otherwise Docker creates a directory in its place. GPU encoding (NVENC) does not work in Docker on Windows; keep `VIDEO_ENCODER=libx264`.

---

## 🔔 Kick Webhooks Setup (Instant Live Detection)

Polling checks the live status every 15 seconds; webhooks tell you the moment a stream goes live.
1. Create an app at [kick.com/settings/developer](https://kick.com/settings/developer) — note down `KICK_CLIENT_ID` and `KICK_CLIENT_SECRET`.
2. Put `KICK_CLIENT_ID`, `KICK_CLIENT_SECRET`, `KICK_ACCESS_TOKEN` and `KICK_WEBHOOK_URL` (your server's public HTTPS URL) in `.env`.
3. Check the checklist at `GET /api/webhooks/status` on the dashboard, then subscribe with `POST /api/webhooks/subscribe`.
4. Webhook endpoint: `POST /api/webhooks/kick` (signature verified with RSA-SHA256).

> Note: it has not been verified whether a user token can subscribe for *other* broadcasters. If Kick refuses, webhooks will only work for your own channel and polling will continue for other streamers — which already works.

---

## ⚙️ Configuration (.env)

All settings are grouped in `.env.example` — with every Tier 1–5 feature. The most important:
- `AUTO_CLIP_ENABLED=true` — turn on auto-clipping (default: off).
- `CHAT_HYPE_ENABLED=true` — turn on chat hype (default: off).
- `DASHBOARD_PASSWORD` or `DASHBOARD_PASSWORD_HASH` — dashboard login (default: open).
- `ENABLE_SUBTITLES=true` — burn subtitles into clips (default: off — Kick clips render with no subtitles; Whisper still transcribes for silence cutting).
- `SUBTITLE_STYLE=karaoke` — kinetic karaoke subtitles (default: classic; only applies when subtitles are enabled).
- `BGM_ENABLED=true` + MP3s in `assets/bgm/<mood>/` — background music (default: off).
- `VIDEO_ENCODER=auto` — NVENC/AMF/VAAPI/libx264 auto-detect (default: auto).
- `DASHBOARD_HOST=0.0.0.0` + `DASHBOARD_PORT=8500` — open the dashboard from the local network / Android phone (default: this computer only).

---

## 🎮 What Can You Do in the Dashboard?
1. **Streamer Watchlist**: Add or remove streamers. Their live status, viewers and title show up instantly.
2. **Instant Clip Trigger**: Extract the last 30s, 45s or 60s clip of any live streamer in 1 click.
3. **Player Scrub**: Rewind or forward the streamer's stream by 30 seconds.
4. **Vertical Video Studio**: Watch the finished 9:16 TikTok video directly in the web player (with top hook banner; subtitles only if enabled).
5. **Publish to TikTok**: Publish in 1 click with video and captions auto-filled in TikTok Studio.
6. **Resource Telemetry**: See real-time CPU and RAM usage at the top of the dashboard to see how light it runs.
7. **Earnings & Analytics**: Kick programme submissions, CPM earnings and AI-score feedback (which threshold performs best).
8. **Upload Queue**: Spaced, capped queue for YouTube / Instagram / TikTok uploads.
9. **Clip Retry**: Resume a failed clip from its last checkpoint (`/api/clips/{id}/retry`).
10. **Webhooks Status**: Kick webhook configuration and setup checklist (`/api/webhooks/status`).
