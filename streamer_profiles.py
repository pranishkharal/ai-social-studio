"""
Per-streamer profiles for ai-social-studio (feature 11).

Every threshold used to be global -- same hype sensitivity, same clip
length, same layout for xQc, Adin Ross, Tarik and everyone else. An IRL
streamer needs different settings from a Valorant pro, so each streamer
gets a profile:

    {
        "hype_db_threshold": float,  # audio dB level that counts as hype
        "clip_seconds": int,         # default clip length for this streamer
        "layout": str,               # speaker_tracking | split |
                                     #   blurred_backdrop | center_crop
        "caption_style": str,        # caption voice, e.g. "prestige"
        "platforms": [str],          # platforms to post this streamer's clips to
        "auto_clip_enabled": bool    # whether the auto-clip watcher may clip them
    }

Profiles live in ``streamer_profiles.json`` next to this module (override
the location with the STREAMER_PROFILES_PATH env var). ``get_profile``
merges the streamer's entry over the "default" profile, case-insensitively,
so a profile only needs to list the keys it changes. Unknown streamers get
the default profile. A missing or broken JSON file falls back to the
built-in defaults rather than crashing the pipeline.
"""

import json
import os
from pathlib import Path

DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "streamer_profiles.json"

# Built-in fallback when streamer_profiles.json is missing or unreadable.
# The "default" layout matches ClipEditor's historical default
# (blurred_backdrop) so existing behaviour is unchanged.
DEFAULT_PROFILE = {
    "hype_db_threshold": -15.0,
    "clip_seconds": 45,
    "layout": "blurred_backdrop",
    "caption_style": "prestige",
    "platforms": ["tiktok", "youtube", "instagram"],
    "auto_clip_enabled": True,
}


def _profiles_path(path=None):
    if path is not None:
        return Path(path)
    env_path = os.environ.get("STREAMER_PROFILES_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_PROFILES_PATH


def load_profiles(path=None):
    """
    Load the profiles file as a dict {name_lower: profile_dict}.

    Returns {"default": DEFAULT_PROFILE copy} when the file is missing,
    unreadable, or not valid JSON -- profiles must never break clipping.
    Keys are lowercased so lookups are case-insensitive.
    """
    profiles_path = _profiles_path(path)
    profiles = {"default": dict(DEFAULT_PROFILE)}
    try:
        with open(profiles_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        return profiles
    if not isinstance(raw, dict):
        return profiles
    for name, profile in raw.items():
        if not isinstance(profile, dict):
            continue
        key = str(name).lower().strip()
        if key == "default":
            merged = dict(DEFAULT_PROFILE)
            merged.update(profile)
            profiles["default"] = merged
        else:
            profiles[key] = dict(profile)
    return profiles


def get_profile(streamer, path=None):
    """
    Merged profile for a streamer: their entry over the "default" profile.

    Lookup is case-insensitive and ignores a leading "@". Unknown
    streamers (and a None/empty name) get a copy of the default profile.
    The returned dict is a fresh copy -- mutating it never affects later
    calls.
    """
    profiles = load_profiles(path)
    merged = dict(profiles.get("default", DEFAULT_PROFILE))
    if streamer:
        key = str(streamer).lower().strip().lstrip("@")
        if key in profiles:
            merged.update(profiles[key])
    return merged
