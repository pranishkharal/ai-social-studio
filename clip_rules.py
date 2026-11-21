"""Clip permission rules per streamer (gta6-clipper feature 13).

Each streamer gets a badge — allowed / ask / no / unknown — with a
note and a source link, stored in clip_rules.json next to this module.
Clipping the wrong streamer risks strikes and takedowns, so the
pipeline checks the badge before it clips:

- "allowed": clip freely (credit the streamer).
- "ask": clip only through the streamer's own program or with
  permission — e.g. adinross pays clippers through Kick's Clipping
  Program, so joining it is the way in.
- "no": the pipeline refuses to clip unless the user confirms they
  have permission (explicit ``permission_confirmed=True`` or the
  CLIP_PERMISSION_OVERRIDE=1 env var).
- "unknown": no rule on file — the pipeline clips but logs a warning
  so Sara knows the streamer has not been checked yet.

The dashboard shows the badge next to the active streamer
(gta6_routers.py serves it at /api/clip-rules/...).
"""

import json
import os
from pathlib import Path

RULES_FILE = Path(__file__).resolve().parent / "clip_rules.json"

# Badge values, in order of how safe they are to clip.
RULES = ("allowed", "ask", "no", "unknown")

# Env var that confirms permission for a "no" streamer for the whole
# run (e.g. Sara has written permission and does not want to confirm
# every clip).
OVERRIDE_ENV_VAR = "CLIP_PERMISSION_OVERRIDE"


class PermissionDeniedError(Exception):
    """Raised when the pipeline is asked to clip a "no" streamer
    without confirmed permission."""


def _default_rules_path():
    return RULES_FILE


def load_rules(path=None):
    """The rules dict from clip_rules.json ({} when the file is
    missing or broken — unknown streamers are then all "unknown")."""
    try:
        data = json.loads(Path(path or _default_rules_path()).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _key(platform, streamer):
    return f"{(platform or 'kick').strip().lower()}:{streamer.strip().lower().lstrip('@')}"


def get_rule(platform, streamer, rules=None):
    """The badge for one streamer: {"rule", "note", "source"}.

    Unknown streamers (or a missing rules file) come back as
    "unknown" with an empty note and source.
    """
    rules = load_rules() if rules is None else rules
    entry = rules.get(_key(platform, streamer)) or {}
    rule = str(entry.get("rule", "unknown")).strip().lower()
    if rule not in RULES:
        rule = "unknown"
    return {
        "rule": rule,
        "note": entry.get("note", "") or "",
        "source": entry.get("source", "") or "",
    }


def permission_is_confirmed(explicit=False):
    """True when the user confirmed they have permission — either the
    explicit flag or the CLIP_PERMISSION_OVERRIDE=1 env var."""
    return bool(explicit) or os.environ.get(OVERRIDE_ENV_VAR, "") == "1"


def check_clip_allowed(platform, streamer, permission_confirmed=False,
                       rules=None):
    """Guard the pipeline: returns the badge dict, or raises
    PermissionDeniedError for a "no" streamer without confirmed
    permission. "allowed", "ask" and "unknown" pass through — "ask"
    and "unknown" are warnings the caller logs, not blocks.
    """
    badge = get_rule(platform, streamer, rules=rules)
    if badge["rule"] == "no" and not permission_is_confirmed(permission_confirmed):
        raise PermissionDeniedError(
            f"{streamer} is marked 'no' for clipping"
            + (f" ({badge['note']})" if badge["note"] else "")
            + ". Confirm you have permission (pass permission_confirmed=True "
            f"or set {OVERRIDE_ENV_VAR}=1) to clip them."
        )
    return badge


def list_rules(rules=None):
    """All rules on file, sorted by key: [{"platform", "streamer",
    "rule", "note", "source"}]."""
    rules = load_rules() if rules is None else rules
    out = []
    for key, entry in sorted(rules.items()):
        platform, _, streamer = key.partition(":")
        out.append({
            "platform": platform,
            "streamer": streamer,
            "rule": entry.get("rule", "unknown"),
            "note": entry.get("note", "") or "",
            "source": entry.get("source", "") or "",
        })
    return out
