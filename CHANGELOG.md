# Changelog

All notable changes to ai-social-studio are documented here.

## [1.4.1] — 2026-11-21 — Quote Line Off by Default

Sara watched the TikTok short-story demo clip and asked to remove the
quote line ("captioning") — the white spoken line under the headline.
The headline in the white box stays; only the quote line goes.

- **Quote line off by default** (`config.py`, `clip_editor.py`,
  `pipeline.py`, `web_dashboard.py`, `templates/index.html`,
  `.env.example`): `QUOTE_LINE_ENABLED` now defaults to false, the
  same pattern as subtitles. The quote is still picked and stored for
  every clip — it just is not drawn on the video unless opted in:
  `quote_line_enabled=True` through the pipeline / editor, the
  dashboard's Quote line checkbox (clip trigger, and the clip's
  re-render panel — a re-render with the box unchecked drops the line
  even for clips that had one), or `QUOTE_LINE_ENABLED=true` in `.env`.
  The face-avoidance quote placement still applies when the line is
  enabled; the story-headline builder is untouched (headlines that
  contain a quoted clause are part of the headline, not the quote
  line). Covered by `tests/test_quote_line.py`,
  `tests/test_banner_timing.py`, `tests/test_face_avoidance.py` and
  `tests/test_tiktok_pipeline.py` (default-off, config opt-in,
  explicit-argument-wins, and pipeline pass-through tests).

## [1.4.0] — 2026-11-21 — gta6-clipper Upgrade

21 features from the open-source GTA 6 auto-clipper
(nocodehack.io/gta6-clipper, MIT), integrated into the existing
features: visual detection joins the audio hype detector, QC runs
inside the clip pipeline, VOD signals join VOD scoring, and
multi-part formats reuse moments the engine already found. Not
copied: the Claude Code workflow, the 32 caption styles (subtitles
stay off on Kick clips), Blotato auto-posting (paid), the
multi-language UI, and GTA-specific branding.

### New features
- **Visual moment detection** (`visual_moments.py`, `hype_detector.py`,
  `story_headlines.py`, `pipeline.py`): labels chase / crash / fight /
  stunt / reaction / funny with no AI — 2 fps 160 px frames, Farneback
  optical flow (radial outflow = speed, rotation = stunt), audio bands,
  and laughter from the 3.5–8 Hz voice-envelope wobble. The best
  moment's confidence blends into the hype score (50% audio / 30%
  lexical / 20% visual with a transcript) and its type reaches the
  story headlines via `moment_type`. Runs in the pipeline by default;
  `VISUAL_DETECTION_ENABLED=0` skips the scan. Covered by
  `tests/test_visual_moments.py`.
- **Game-event OCR** (`game_ocr.py`): reads big centred on-screen
  banners (WASTED, BUSTED, MISSION PASSED/FAILED, VICTORY, GAME OVER
  and more) with RapidOCR + fuzzy matching, plus colour cues, and adds
  up to 15 hype points naming the banner. RapidOCR is an optional lazy
  import (`pip install rapidocr-onnxruntime`); without it only
  hard-cut events are reported. Covered by `tests/test_game_ocr.py`.
- **Automatic clip QC** (`clip_qc.py`, `pipeline.py`, `database.py`):
  every finished clip is checked before it is marked ready_to_upload —
  duration 12–60 s, loudness −14 LUFS ± 2, black/frozen frames,
  letterbox, opens/ends on a logo or black card, face or motion in the
  first second (Haar), silence % — and gets PASS / WARN / FAIL. FAIL
  gets one automatic re-cut, then the clip is marked failed with the
  reasons; the report is stored in the new `clips.qc_report` column.
  No captions check, deliberately (subtitles are off by default).
  Covered by `tests/test_clip_qc.py`.
- **Loudness normalisation** (`loudness.py`, `clip_editor.py`): every
  render is normalised to −14 LUFS (two-pass loudnorm, `linear=true`),
  with an audio-only safety pass (video copied) when the finished
  clip lands more than 1.5 LU off. Covered by `tests/test_loudness.py`.
- **Learn from your own channel** (`channel_learn.py`,
  `analytics.py`): fetches a channel's Shorts + public view counts
  with yt-dlp (one request, cached a day, no API key), matches them to
  the clips the app made, and writes rules (streamer / length / layout
  / post-time / hook) that nudge moment scoring by at most ±3.
  Auto-fills YouTube view counts in the analytics table (no duplicate
  snapshots). YouTube only — TikTok has no public view-count feed, so
  TikTok stays manual. Needs `YOUTUBE_CHANNEL_URL` and 8+ Shorts.
  Covered by `tests/test_channel_learn.py`.
- **Multi-part formats** (`clip_formats.py`, `callback_finder.py`,
  `render_modes.py`, `pipeline.py`): callback, compilation,
  escalation, then-vs-now, reveal, chat bubbles and wait-for-it, built
  from moments the engine already found (`pipeline.process_format`).
  The callback finder proposes setup/payoff pairs from the transcript
  (claim regexes + IDF-weighted rare shared words). Parts are rendered
  through the normal `ClipEditor` path and reused across formats by
  content hash (`PartCache`). Covered by `tests/test_clip_formats.py`
  and `tests/test_callback_finder.py`.
- **Meme sounds + visual gags** (`meme_sounds.py`, `clip_editor.py`):
  9 moods with numpy-synthesised sounds (vine boom, airhorn, sad
  trombone, crickets, heartbeat — royalty-free by construction) plus
  ffmpeg visual gags (freeze-frame, push-in, shake, flash, vignette).
  OPT-IN ONLY, default off (`meme_mood=` / `MEME_SOUNDS_ENABLED=1`);
  at most one per clip, never in the first 3 s. Covered by
  `tests/test_meme_sounds.py`.
- **Active-speaker framing** (`speaker_framing.py`, `pipeline.py`,
  `render_modes.py`, `clip_editor.py`): produces `speaker_segments` +
  `trajectories` so the camera-switch / split-screen layouts work for
  real instead of falling back to plain face tracking. Lip motion from
  MediaPipe FaceLandmarker jaw-open data when the model is available
  (downloaded at runtime on first use, size-checked — no binary in the
  repo), else mouth-region motion + per-face speech loudness; 1.5 s
  minimum hold, split screen when two people trade lines fast, blurred
  fill when unsure. Covered by `tests/test_speaker_framing.py`.
- **Whisper vocabulary prompt** (`ai_story_engine.py`, `pipeline.py`):
  game + streamer names and chat slang are passed to Whisper as
  `initial_prompt`, with the VAD filter on, so names transcribe
  correctly (`WHISPER_VOCAB_PROMPT` overrides). Covered by
  `tests/test_whisper_vocab.py`.
- **Smarter chat spikes + Kick key negotiation** (`chat_hype.py`,
  `auto_clip_watcher.py`): spikes now count distinct chatters (≥3× the
  5-minute median and ≥15 people, 60 s cooldown; bots, ad links and
  `!commands` filtered; local-median fallback for huge chats), adding
  +20 hype while active. The Kick Pusher app key + cluster are
  negotiated from Kick on every connect, falling back to the built-in
  key. Covered by `tests/test_chat_spikes.py`.
- **VOD signals** (`vod_signals.py`, `vod_clipper.py`): Twitch viewer
  clips (`[CLIPPED x2 6.0k views]`, public GQL) and YouTube
  most-replayed peaks (`[REPLAYED x8]`, yt-dlp heatmap) score VOD
  windows (combined bonus cap 35). `--no-vod-signals` disables.
  Covered by `tests/test_vod_signals.py`.
- **Peek before render** (`peek.py`, `pipeline.py`): a 4-frame strip
  per pick (start, third, two-thirds, end), saved next to the render
  as `*_peek.jpg`, so wrong-game / menu / logo picks are caught before
  rendering. Failures never block a render. Covered by
  `tests/test_peek.py`.
- **Voice-over** (`voiceover.py`, `clip_editor.py`, `pipeline.py`,
  `gta6_routers.py`): Sara's recorded line placed at any second, game
  audio ducked ~15 dB under it, video copied untouched; too-long lines
  are trimmed. Dashboard upload in the clip modal (`POST
  /api/voiceover`). Covered by `tests/test_voiceover.py`.
- **Clip permission rules** (`clip_rules.py`, `clip_rules.json`,
  `pipeline.py`): per-streamer badges (allowed / ask / no / unknown)
  with notes and sources, including the Kick Clipping Program note for
  adinross. The pipeline refuses "no" streamers unless permission is
  confirmed (`CLIP_PERMISSION_OVERRIDE=1`); the badge shows in the
  dashboard. Covered by `tests/test_clip_rules.py`.
- **Multi-platform live hunt** (`live_hunt.py`, `gta6_routers.py`):
  one dashboard panel for who's live on Kick, Twitch and YouTube
  (public endpoints, no keys), with viewers, title, start/stop
  recording, and each streamer's last 5 past streams. Covered by
  `tests/test_live_hunt.py`.
- **Streamlink recorder** (`streamlink_recorder.py`,
  `kick_monitor.py`): fallback live recorder — streamlink piped into
  ffmpeg, 4 s segments, 10-minute rolling buffer, 30 s stall restart,
  disk-full guard, exponential back-off when offline. Enable with
  `STREAMLINK_RECORDER=1` (quality via `STREAMLINK_QUALITY`).
  Live-tested in this VM at 360p (720p60 needs a normal home network).
  Covered by `tests/test_streamlink_recorder.py`.
- **Lite mode** (`lite_mode.py`, `main.py`, `pipeline.py`):
  auto-detects PCs under 12 GB RAM (1 render at a time, 30 fps, first
  hour of long streams only, audio downloads deleted after
  transcribing) plus a disk-space guard that stops a run early with a
  clear message. `LITE_MODE=1/0` overrides. Covered by
  `tests/test_lite_mode.py`.
- **Offline translation** (`translate_captions.py`): Argos Translate
  (OPUS models, 16 languages) with RTL handling, CJK chunking and word
  timings re-spread over the translated words. Optional lazy import;
  needs Argos models to run live. Covered by
  `tests/test_translate_captions.py` (mocked model).
- **YouTube analytics** (`ytstats.py`, `gta6_routers.py`): Level 1
  (Data API key: views/likes/comments per Short) and Level 2 (own
  OAuth client: retention curve, average view %) with a rule-based
  diagnosis of why a Short underperformed. Needs Sara's API key /
  OAuth client to run live. Covered by `tests/test_ytstats.py`
  (mocked API).
- **Review UX** (`templates/index.html`, `gta6_routers.py`): A/X
  keyboard shortcuts (approve/reject), Shorts-by-day view, and
  drag-onto-week scheduling with a dry-run preview that lists exactly
  what would be sent and warns about every skipped item. Covered by
  `tests/test_review_ux.py`.
- **Self-update** (`self_update.py`): manifest check, sha256 verify,
  backup to `backups/<version>/`, replaces only shipped files
  (clips/edited/buffer/.env/DB/profiles/tokens/templates/static
  protected), rolls back on failure, dry-run mode. `UPDATE_MANIFEST_URL`
  (reports "no_info" when unset). Covered by
  `tests/test_self_update.py`.
- **Rendering refinements** (`silence_cutter.py`,
  `karaoke_subtitles.py`, `clip_formats.py`, `themes.py`,
  `themes.json`, `clip_editor.py`): sentence-boundary snapping
  (`snap_clip_bounds`, 4-rule snap); profanity masking in karaoke
  captions (`f***`, audio untouched, `KARAOKE_MASK_PROFANITY` default
  on — only matters when subtitles are enabled); per-format render
  cache (`PartCache`, content-hash); 5 named themes (`clean` =
  current default look, `bold-yellow`, `neon`, `minimal-white`,
  `warm-ink`; `CLIP_THEME` / `theme=`). Covered by
  `tests/test_render_refinements.py`.

### Fixes
- Loudness normalisation no longer changes the audio sample rate:
  linear loudnorm upsamples to 192 kHz internally and the AAC
  encoder caps at 96 kHz, so renders came out at 96 kHz — and the
  format joiner (concat demuxer) failed hard (exit 69, AAC decode
  errors) when a 96 kHz part met a 48 kHz card. `loudnorm_filter`
  now ends with `aresample=48000`, so every render leaves at a
  uniform 48 kHz; `join_clips` also normalises each input's audio
  to 48 kHz stereo before the concat (video copied, already-matching
  clips untouched, video-only clips get a silent track), so stale
  cached parts and foreign-rate inputs cannot break a join again.
  Found by rendering a real compilation; covered by new tests in
  `tests/test_loudness.py` and `tests/test_clip_formats.py`.
- QC card test no longer fails dark game scenes: it required the
  panel to be dark *or* flat, so a real dusk GTA scene (panel mean
  ~24, textured) failed as "ends on a card". It now requires dark
  *and* flat, or perfectly flat at any brightness (a solid-colour
  card). The single-dark-band "black strip" warn was removed for the
  same reason — dark road at the bottom of a dusk scene is
  near-black for the whole clip; a picture that actually goes black
  is still caught by the game-panel-black FAIL. Found by running a
  real clip through the gate; covered by new `_is_card` regression
  tests in `tests/test_clip_qc.py`.
- Headline verb bug: "happened" became "happenes" and "killed" became
  "kils" (`_ed_to_present` mishandled -en verbs and stripped genuine
  double consonants). Fixed with an -en verb list and a restricted
  dedup set (`tests/test_gta6_integration.py`).
- Voice-over temp file now goes next to the output instead of
  `EDITED_DIR`, so `os.replace` cannot fail across filesystems (same
  latent bug the meme pass had).
- Format rendering: `extract_window` re-encodes instead of stream
  copy (windows with no keyframe came out audio-only); `join_clips`
  resolves absolute paths; overlay cards match the parts' size and
  fps instead of inflating the join.

### Verified (2026-11-21, checked by the coordinator — not the feature authors)
- Full test suite: **1077 passed, 11 skipped, 0 failed** (baseline
  before this upgrade: 696 passed, 1 skipped). The 11 skips are the
  pre-existing face-tracker OpenCV skip + 10 tests that need real
  OpenCV (8 game-OCR, 2 QC frame checks); those 10 pass with real
  OpenCV (60/60 in `test_game_ocr.py` + `test_clip_qc.py` +
  `test_visual_moments.py` under the venv's OpenCV).
- Visual detection on a real stream: scanned a 75 s adinross GTA 6
  capture and found 5 moments (stunt 0.85, crash 0.80, crash 0.78,
  reaction 0.76, reaction 0.74) and located the facecam top-left.
  Scan took 67 s (1.1x realtime) with the test suite sharing the CPU.
- Game-event OCR: the capture window holds no WASTED/BUSTED banner
  (scan ran, RapidOCR available, 0 events), so OCR was proven on
  synthetic frames instead — WASTED and BUSTED both read at
  confidence 0.99, a banner-free control frame correctly empty.
- Demo clips (rendered through the real pipeline from a 75 s adinross
  GTA 6 capture, all 1080x1920, no subtitles, verified frame by frame):
  `gta6-visual-stunt-9x16.mp4` (clip #24, 15.0 s, hype 57.7,
  trigger "Sudden scream / laughter / audio volume spike + visual
  moment: stunt", QC PASS — 15.0 s, -14.04 LUFS, no fail, no warn);
  `gta6-compilation-9x16.mp4` (clip #26, 29.9 s, 3 parts best last
  with #3/#2/#1 badges, 48 kHz audio);
  `gta6-ocr-wasted-proof.jpg` (OCR proof frame).

### Honest limits
- Visual moment detection found real moments on a 75 s GTA 6 capture
  (see Verified), but accuracy against hand-labelled ground truth is
  still unmeasured, and the reference's "~85x realtime" claim was not
  reproduced (a 75 s scan took 67 s here, ffmpeg startup dominates).
- Kick key negotiation, Twitch viewer clips and the YouTube heatmap
  were verified with mocks/injected fakes only — no live calls made.
- Channel learning was never run against a real channel (needs
  `YOUTUBE_CHANNEL_URL`); translation and YouTube analytics need
  Argos models / Sara's keys and are mocked-tested only.
- Streamlink recording works in this VM at 360p and below; 720p60
  needs a normal home network (untested there).
- Active-speaker framing's 2-person positive path is verified only
  via stubbed tests (no real 2-person talking video available); on
  small-facecam Kick streams MediaPipe rarely sees faces, so Haar +
  the fallback path does the real work there. FaceLandmarker needs
  system libEGL/libGLESv2 (apt) to load.
- QC adds roughly 10–20 s of ffmpeg passes per clip, and the QC
  re-cut re-renders with identical settings, so it only rescues
  transient failures.

## [1.3.0] — 2026-11-21 — Face Tracking Upgrade

The banner never covers the streamer's face anymore, MediaPipe face
tracking works for real, and face-view clips can follow the whole body.
Verified: full test suite 696 passed, 1 skipped (the pre-existing
OpenCV skip); three demo clips rendered from a real adinross capture
and frame-verified (face-aware banner, MediaPipe backend, body
tracking).

### New features
- **Face-aware banner placement** (`face_avoidance.py`, `clip_editor.py`,
  `speaker_tracker.py`): for `speaker_tracking`, `smart_crop`,
  `camera_switch` and `split_screen`, sampled face boxes
  (`SpeakerTracker.analyze_face_boxes`, 6 fps) are mapped through the
  crop trajectory actually used for the render, and the banner is
  placed at the first of top (260) / below-face (800) / bottom
  (1410 − box_h) that does not overlap any face (20 px safety gap),
  falling back to the least-overlapping slot; the quote line moves
  the same way around the face and the banner. With no faces detected
  both keep their fixed positions (banner 800, quote centred on
  y 1240), and non-face-tracking layouts are untouched. Placement is
  baked into the card PNGs, so it works with both banner timings
  (4 s and whole-clip). Covered by `tests/test_face_avoidance.py`
  (40 tests, including a real ffmpeg render).
- **MediaPipe face tracking works for real** (`speaker_tracker.py`):
  the backend was rewritten from the dead `mp.solutions` API (removed
  in mediapipe 1.x — the old code always silently fell back to Haar)
  to the Tasks API (`mp.tasks.vision.FaceDetector`, IMAGE mode,
  confidence 0.5, one detector per tracker, `close()` releases it).
  The ~230 KB BlazeFace model auto-downloads into `models/` on first
  use (3 retries, size-validated, temp file renamed into place only
  when valid); a failed download falls back to Haar. A legacy
  `mp.solutions` path is kept for mediapipe < 1.0 installs.
  `mediapipe>=1.0` is now a real dependency in `requirements.txt`
  (it pulls `opencv-contrib-python` 5.x; CascadeClassifier was
  verified working on 5.1.0.95 — the `<5` pin is kept for installs
  without mediapipe). Model files (`*.tflite`, `*.task`) are now
  git-ignored. On Linux the Tasks API needs the system EGL/GLES
  libraries (`sudo apt install libegl1 libgles2`); without them
  MediaPipe init fails and the tracker falls back to Haar. New dev
  tool `scripts/compare_trackers.py` compares backends on a real
  clip. Covered by `tests/test_face_tracker_mp.py` (10 new tests:
  Tasks-API fakes, model download success/failure/validation,
  close, legacy fallback).
- **Adaptive backend fallback** (`speaker_tracker.py`): when the
  active non-Haar backend detects faces in fewer than 20% of sampled
  frames (`BACKEND_MIN_DETECTION_RATE`), the video is re-analysed
  with Haar. BlazeFace short-range sees the whole frame at 128×128,
  so on a small-facecam gaming stream it can miss the streamer
  entirely — verified on a real 75 s adinross capture (501 samples):
  MediaPipe detected 2% of samples vs Haar's 100%, because the
  facecam face is ~75 px in a 1920 px frame (the same face scores
  0.94 in a 300×300 crop, proving the integration is correct). Without
  the fallback, auto-selection would park the camera at the centre
  and cut the facecam out of the clip. Pass
  `fallback_to_haar=False` to measure a backend's raw behaviour
  (`scripts/compare_trackers.py` does this). Covered by 3 new tests.
- **Full-body tracking option** (`body_tracker.py`, `config.py`,
  `pipeline.py`, `web_dashboard.py`, `templates/index.html`,
  `clip_editor.py`): `BodyTracker` (MediaPipe PoseLandmarker, Tasks
  API) as an alternative tracking target for the speaker_tracking
  layout, with automatic fallback to face tracking when no body is
  found; `TRACKING_TARGET` setting (`face`/`body`, default `face`);
  dashboard Tracking Target select; `tracking_target` parameter on
  `process_moment` / `resume_clip` / `rerender_clip` / `edit_clip`.
  Largest detected body wins when several people are on screen. The
  pose model (`pose_landmarker_lite.task`, ~5.8 MB) auto-downloads
  into `models/` on first use. Verified on the 75 s adinross capture:
  body found on 71.9% of sampled frames (63% streamer facecam, 37%
  game characters) — body tracking is best for IRL streams where the
  streamer is the biggest body on screen; face tracking stays the
  default for facecam streams. Covered by
  `tests/test_body_tracker.py` (37 tests).

### Bug fixes
- **Mirrored face-view crop framed the wrong side of the stream
  (`clip_editor.py`)**: the tracker analyses the unmirrored source
  but the crop runs after `hflip` when `mirror=True` (the default),
  so the crop centred on the mirror image of the face — for an
  off-centre facecam the face was not in the rendered frame at all.
  The crop trajectory and the face boxes are now both mirrored, so
  the crop lands on the face as rendered. Verified on a real
  mirrored render: the facecam is fully visible.
- **MediaPipe detector shutdown noise (`clip_editor.py`)**: the
  speaker-tracking and camera-switch/split-screen branches now call
  `tracker.close()` when done, so the process no longer logs an
  ignored `FaceDetector.__del__` exception at exit.
- **T=12 facecam cut-off diagnosed (not a tracking bug)**: in the
  2026-11-19 live run the face-view clip cut the facecam off at
  t=12 because Haar detected Adin in 0 of 180 samples at 720p — his
  facecam face is ~55–60 px and Haar's minimum detectable face is
  60 px at that resolution, so only large game faces set the crop
  target. On the 1080p capture the same face is ~72 px and Haar
  detects it in 502/533 samples with the facecam fully in frame.
  MediaPipe does not fix this either (BlazeFace found zero Adin
  detections at 720p). Prefer 1080p captures for face-view clips.

### Known limits
- Face-aware placement only dodges *detected* faces; with no
  detections the banner and quote keep their fixed positions.
- The multi-face "largest wins" rule has no temporal filtering: a
  single-sample game face can yank the crop target for one sample
  (EMA blips the crop; MediaPipe's snap rule would jump it). A
  persistence/continuity rule is planned for a tracker-quality pass.
- Face-view renders run face detection twice (trajectory + boxes);
  a single combined pass is a planned optimisation.
- Body tracking is slow on weak machines (~3.3× realtime on a
  1.5-CPU VM); frame downscaling before Pose detection is planned.

## [1.2.0] — 2026-11-20 — TikTok clipping upgrades

Seven features learned from studying three TikTok clipping accounts
(@zysno, @godsclipper, @kicksclipping — 16 clips watched, November
2026). Verified: full test suite 602 passed, 1 skipped (the
pre-existing OpenCV skip); four demo clips rendered from a real adinross
capture and frame-verified (story headline + quote line, whole-clip
timing, soft background music, VOD path).

### New features
- **Story headlines** (`story_headlines.py`, `ai_story_engine.py`):
  titles now name the specific moment from the transcript instead of
  generic "COMPLETELY SPEECHLESS" lines — e.g. the real adinross clip
  transcript produces "Adinross finds his calling — "Chaz is not
  clean"". A local builder (verb + object + quoted phrases, no API key
  needed) powers the heuristic fallback, and the Gemini/LLM prompt now
  demands the same specificity. Covered by
  `tests/test_story_headlines.py` (25 tests).
- **Quote line** (`clip_editor.py`, `pipeline.py`, `database.py`): the
  best spoken line from the clip is shown as a second caption under the
  headline — white bold text with black outline in the lower third.
  One static line, NOT word-by-word captions (subtitles stay off by
  default). The line is picked from the transcript
  (`story_headlines.pick_quote_line`), stored in the clips table
  (`quote_line` column, safe migration) and reused on re-render.
  When the headline itself is a quote, the quote card is dropped —
  showing the same line twice looked like a bug.
  `QUOTE_LINE_ENABLED=true` by default. Covered by
  `tests/test_quote_line.py` (6 tests) and
  `tests/test_tiktok_pipeline.py`.
- **Short-clip mode** (`pipeline.py`, `hype_detector.py`): tight clips
  (default 12 s) cut around the single loudest moment inside the
  footage — the TikTok accounts' most viral clips are 9–17 s.
  `HypeDetector.find_peak_window` scans audio-only volumedetect
  windows; the pipeline cuts the peak window and renders that.
  Silence trimming is skipped in short-clip mode — the window is
  already tight, and trimming sparse-speech moments collapsed a 12 s
  window to 3.6 s in the first demo run.
  `SHORT_CLIP_ENABLED` / `SHORT_CLIP_SECONDS` in `.env`, plus a
  dashboard checkbox. A failed peak cut falls back to the full clip.
  Covered by `tests/test_peak_window.py` (4 tests) and
  `tests/test_tiktok_pipeline.py`.
- **Auto hashtags** (`story_headlines.py`, `pipeline.py`): the post
  caption is written as `title #<streamer> #fyp #kick`, plus a game tag
  (`#gta6` for "GTA 6") when the game is known — pass `game=` to
  `process_moment` or type it in the dashboard's game box.
- **Soft background music** (`bgm.py`, `assets/bgm/`): a quiet track
  mixed low under the original audio (base level 0.12, about −18 dB,
  ducked further while someone talks via the existing sidechain
  compressor). A gentle ambient pad is generated on first use when the
  file is missing (`assets/bgm/chill/soft_pad.mp3`, 60 s, synthesized
  with ffmpeg from sine tones — royalty-free by construction; the MP3
  itself is not stored in the repo), so BGM works out of the
  box; drop your own MP3s into `assets/bgm/<mood>/` to use them
  instead (see `assets/bgm/README.md`). `BGM_ENABLED=true` or the
  dashboard checkbox. Measured on a real clip: the speech keeps its
  level with BGM on (full-clip mean −28.0 dB vs −29.7 dB without),
  and the music is audible in a speech-free gap (−41.3 dB vs −47.2 dB
  without — about 13 dB under the speech). Covered by
  `tests/test_bgm_mix.py` (9 tests).
- **VOD / multi-source clipping** (`vod_clipper.py`): clip past videos,
  not just live Kick streams — Kick VOD, Twitch VOD/clip, YouTube
  (podcasts included) or a local file. Downloads with yt-dlp (720p
  max, `pip install yt-dlp` — optional, local files work without it),
  scans windows with the hype detector, extracts the best moment and
  runs it through the normal pipeline. The extracted window is kept in
  `vod/windows/` so resume/re-render keep working. Dashboard: "Clip a
  Past Video" panel (`POST /api/vod/clip`); CLI:
  `python3 vod_clipper.py <url-or-file>`. Covered by
  `tests/test_vod_clipper.py` (34 tests). A real YouTube download was
  verified in the dev VM (19 s public video, 720p).
- **Headline timing setting** (`clip_editor.py`, `config.py`): the
  headline (and quote line) can stay for the whole clip, like the
  TikTok accounts do, instead of Sara's 4-second rule:
  `BANNER_TIMING=whole_clip` or `4s` (default `4s`, unchanged
  behaviour). Whole-clip mode loops the card for the full clip
  duration with no fade; the 4 s mode is byte-identical to before.
  Dashboard select on the clip trigger. Covered by
  `tests/test_banner_timing.py` (20 tests, was 5).

### Fixes
- **Render timeout scales with clip length** (`clip_editor.py`):
  `render_timeout_seconds(duration)` = `max(180, min(1200,
  duration * 20))` — the fixed 180 s timeout failed 1080p60 renders
  twice in the 2026-11-19 live run (clip #12).
- **Loop ending verified** (`clip_editor.py`): renders end abruptly on
  the moment — no fade-out on the main video/audio streams (the only
  fades are on the banner/quote card streams in 4 s mode), so clips
  loop cleanly. Covered by tests asserting no `afade` and no main-chain
  fade in the filter graph.
- **BGM mix no longer quiets the original audio** (`bgm.py`): ffmpeg's
  `amix` divides by the input count by default, which dropped the
  speech ~4–6 dB whenever BGM was on (measured on a real clip:
  full-clip mean −29.7 dB without BGM vs −34.0 dB with it). Both mix
  filters now use `amix=...:normalize=0`, so the speech keeps its
  level and the music adds underneath. Covered by
  `test_bgm_does_not_quiet_original_audio`.

### Dashboard
- Clip trigger gains headline-timing select, short-clip checkbox, soft
  music checkbox and a game input (hashtags). New "Clip a Past Video
  (VOD)" panel. Re-render reuses the stored quote line and accepts a
  banner-timing override.

### Honest limits
- Gemini story headlines are still untested with a real key (no key in
  this environment) — the local builder is verified; the LLM path only
  has its prompt + fallback tests.
- yt-dlp downloads of Kick/Twitch VODs are untested live (only a
  public YouTube video was downloaded); Kick VODs may need login.
- Real TikTok/YouTube/Instagram publishing is still manual-by-design
  for TikTok and mock-tested for the others (no keys).

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
