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

Notes:

- **No music is bundled with the repo.** Add your own files — only `.mp3`
  files are picked up (case-insensitive extension).
- Short tracks are fine: the track is looped (`-stream_loop -1`) to cover
  the whole clip, and mixed under the speech with sidechain ducking
  (`bgm.py: build_ducking_filter`), so the music dips whenever someone
  talks.
- With no MP3s in a mood folder, the picker falls back to `chill/`; with
  no MP3s anywhere, BGM is skipped entirely.
- The pool location can be overridden with the `BGM_DIR` env var.

Good royalty-free sources: YouTube Audio Library, Pixabay Music,
Free Music Archive — always check the licence allows reuse in monetised
short-form video before publishing.
