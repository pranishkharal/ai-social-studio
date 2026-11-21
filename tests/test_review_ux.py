"""Tests for the review UX (gta6-clipper feature 21): the schedule
dry-run preview logic in gta6_routers.py, plus markers proving the
dashboard template carries the keyboard shortcuts, Shorts-by-day
view, drag-onto-week schedule and voice-over upload UI.
"""

from datetime import date
from pathlib import Path

import gta6_routers

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "index.html"

CLIPS = {
    1: {"title": "Clip one", "approval_status": "approved",
        "edited_path": "/edited/one.mp4"},
    2: {"title": "Clip two", "approval_status": "pending",
        "edited_path": "/edited/two.mp4"},
    3: {"title": "Clip three", "approval_status": "rejected",
        "edited_path": "/edited/three.mp4"},
    4: {"title": "Clip four", "approval_status": "approved",
        "edited_path": None},
}


def test_resolve_schedule_day_date():
    assert gta6_routers.resolve_schedule_day("2026-11-23") == date(2026, 11, 23)


def test_resolve_schedule_day_weekday():
    # 2026-11-21 is a Saturday.
    today = date(2026, 11, 21)
    assert gta6_routers.resolve_schedule_day("Sat", today=today) == date(2026, 11, 21)
    assert gta6_routers.resolve_schedule_day("Monday", today=today) == date(2026, 11, 23)
    assert gta6_routers.resolve_schedule_day("wednesday", today=today) == date(2026, 11, 25)
    assert gta6_routers.resolve_schedule_day("Funday", today=today) is None


def test_dry_run_plans_valid_items():
    result = gta6_routers.dry_run_schedule(
        [{"clip_id": 1, "day": "2026-11-23", "time": "18:00", "platform": "tiktok"},
         {"clip_id": 2, "day": "Mon", "time": "19:00", "platform": "youtube"}],
        CLIPS, set(), today=date(2026, 11, 21))
    assert len(result["planned"]) == 2
    assert result["planned"][0]["scheduled_for"] == "2026-11-23T18:00"
    assert result["planned"][1]["platform"] == "youtube"
    assert result["warnings"] == []


def test_dry_run_warns_for_every_bad_item():
    result = gta6_routers.dry_run_schedule(
        [{"clip_id": 99, "day": "2026-11-23", "time": "18:00", "platform": "tiktok"},
         {"clip_id": 3, "day": "2026-11-23", "time": "18:00", "platform": "tiktok"},
         {"clip_id": 4, "day": "2026-11-23", "time": "18:00", "platform": "tiktok"},
         {"clip_id": 1, "day": "Funday", "time": "18:00", "platform": "tiktok"},
         {"clip_id": 1, "day": "2026-11-23", "time": "25:99", "platform": "tiktok"},
         {"clip_id": 1, "day": "2026-11-23", "time": "18:00", "platform": "tiktok"},
         {"clip_id": 2, "day": "2026-11-23", "time": "18:00", "platform": "tiktok"}],
        CLIPS, set(), today=date(2026, 11, 21))
    assert len(result["planned"]) == 1  # only the first valid clip 1
    assert len(result["warnings"]) == 6
    joined = " ".join(result["warnings"])
    assert "not found" in joined
    assert "rejected" in joined
    assert "not rendered" in joined
    assert "not a date or weekday" in joined
    assert "not HH:MM" in joined
    assert "two uploads" in joined  # clip 2 clashes with clip 1's slot


def test_dry_run_warns_already_queued():
    result = gta6_routers.dry_run_schedule(
        [{"clip_id": 1, "day": "2026-11-23", "time": "18:00", "platform": "tiktok"}],
        CLIPS, {1})
    assert result["planned"] == []
    assert "already in the upload queue" in result["warnings"][0]


def test_dry_run_endpoint(fresh_db):
    clip_id = fresh_db["seed_clip"](edited_path="/edited/one.mp4")
    req = gta6_routers.ScheduleDryRun(items=[
        {"clip_id": clip_id, "day": "2026-11-23", "time": "18:00",
         "platform": "tiktok"},
        {"clip_id": 99999, "day": "2026-11-23", "time": "18:00",
         "platform": "tiktok"},
    ])
    result = gta6_routers.schedule_dry_run_endpoint(req)
    assert len(result["planned"]) == 1
    assert result["planned"][0]["clip_id"] == clip_id
    assert len(result["warnings"]) == 1


def test_template_has_keyboard_shortcuts():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "submitApproval('approved')" in html
    assert "e.key === 'a'" in html
    assert "e.key === 'x'" in html
    assert "Keys: A approve" in html


def test_template_has_by_day_view():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "toggleByDayView()" in html
    assert "renderClipsByDay" in html
    assert 'id="btn-by-day"' in html


def test_template_has_week_schedule():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="week-schedule-grid"' in html
    assert "dropClipOnDay" in html
    assert "previewSchedule()" in html
    assert "/api/schedule/dry-run" in html
    assert "confirmSchedule()" in html
    assert "card.draggable = true" in html


def test_template_has_voiceover_upload():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="voiceover-file-input"' in html
    assert "uploadVoiceover()" in html
    assert "/api/voiceover" in html


def test_template_has_hunt_panel_and_clip_rule_badge():
    html = TEMPLATE.read_text(encoding="utf-8")
    assert 'id="hunt-list"' in html
    assert "fetchHunt()" in html
    assert 'id="clip-rule-badge"' in html
    assert "fetchClipRule" in html
