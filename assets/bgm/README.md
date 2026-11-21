# Background Music Pool

Drop your own **royalty-free MP3s** into the mood folders below. The BGM
picker (`bgm.py: pick_bgm`) chooses one at random per clip, based on the
clip's highlight category (see `mood_for_category`).

| Folder | Used for |
|---|---|
| `chill/` | Interesting Conversations (also the fallback when a mood folder is empty) |
| `epic/` | Exciting Gameplay, Big Wins |
| `upbeat/` | Funny Moments, Streamer Reactions, Viral Potential |
| `sad/` | Angry or Emotional Reactions |
| `suspense/` | Unexpected Moments |

## Built-in track

One track is generated automatically on first use (ffmpeg synthesises it
in about a second) when the file is missing, so BGM works with no setup:

- `chill/soft_pad.mp3` — a gentle ambient pad (a warm A-major chord:
  220 / 277 / 330 Hz sine tones with a slow tremolo and fade in/out),
  60 seconds, synthesised with ffmpeg — no downloaded audio, so it is
  royalty-free by construction. Because it lives in `chill/`, it is the
  fallback for **every** mood: any clip with BGM on gets this track
  unless its mood folder has its own MP3s.

The track is mixed soft under the speech (`BGM_BASE_VOLUME = 0.12`,
about -18 dB under the voice) and the sidechain ducking
(`bgm.py: build_ducking_filter`) dips it further whenever someone
talks. Raise or lower the music level with the `BGM_BASE_VOLUME` env
var (for example `BGM_BASE_VOLUME=0.2` for louder music).

## Adding your own music

1. Find royalty-free tracks (see sources below) and save them as
   `.mp3` files — only `.mp3` files are picked up (case-insensitive
   extension).
2. Drop each file into the mood folder it fits (for example a hype
   track into `epic/`). Any track in a mood folder joins that mood's
   pool and may be picked instead of the built-in pad.
3. Short tracks are fine: the track is looped (`-stream_loop -1`) to
   cover the whole clip, and mixed under the speech with sidechain
   ducking, so the music dips whenever someone talks.
4. With no MP3s in a mood folder, the picker falls back to `chill/`;
   with no MP3s anywhere, BGM is skipped entirely.
5. The pool location can be overridden with the `BGM_DIR` env var.

Good royalty-free sources: YouTube Audio Library, Pixabay Music,
Free Music Archive — always check the licence allows reuse in monetised
short-form video before publishing.
