"""Tests for earnings.py (Kick clipping-program earnings tracker)."""

import database
import earnings
from earnings import (
    earnings_summary, get_submission, list_submissions, record_submission,
    update_views,
)


def test_record_submission_uses_clip_streamer_cpm(fresh_db, tmp_path):
    clip_id = fresh_db["seed_clip"](streamer="xqc")
    sub = record_submission(clip_id, "tiktok", "https://tiktok.com/@x/123")
    assert sub["streamer"] == "xqc"
    assert sub["cpm"] == 15  # xqc rate from kick_program
    assert sub["views"] == 0
    assert sub["estimated_earnings"] == 0.0
    assert sub["platform"] == "tiktok"


def test_record_submission_explicit_streamer(fresh_db):
    sub = record_submission(None, "youtube", "https://youtube.com/watch?v=ABC",
                            streamer="adinross")
    assert sub["streamer"] == "adinross"
    assert sub["cpm"] == 30


def test_record_submission_unknown_creator_defaults_to_40(fresh_db):
    sub = record_submission(None, "instagram", "https://instagram.com/reel/X",
                            streamer="some_random_creator")
    assert sub["cpm"] == 40


def test_record_submission_multi_streamer_pov_uses_lowest_cpm(fresh_db):
    # adinross $30 vs xqc $15: the POV rule submits under xqc.
    sub = record_submission(None, "tiktok", "https://tiktok.com/@x/9",
                            streamers=["adinross", "xqc"])
    assert sub["streamer"] == "xqc"
    assert sub["cpm"] == 15


def test_record_submission_rejects_bad_platform(fresh_db):
    try:
        record_submission(None, "twitch", "https://twitch.tv/x")
    except ValueError as e:
        assert "Unknown platform" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_update_views_computes_earnings(fresh_db):
    clip_id = fresh_db["seed_clip"](streamer="xqc")
    sub = record_submission(clip_id, "tiktok", "https://tiktok.com/@x/1")
    updated = update_views(sub["id"], views=200_000, likes=5_000)
    # 200k views / 100k * $15 = $30
    assert updated["views"] == 200_000
    assert updated["likes"] == 5_000
    assert updated["estimated_earnings"] == 30.0


def test_update_views_missing_submission(fresh_db):
    try:
        update_views(999, views=100)
    except ValueError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_update_views_youtube_pull_when_no_views_given(fresh_db, monkeypatch):
    sub = record_submission(None, "youtube",
                            "https://www.youtube.com/watch?v=VID12345678",
                            streamer="adinross")
    monkeypatch.setattr(earnings, "_try_youtube_pull", lambda url: (50_000, 1_200))
    updated = update_views(sub["id"])
    assert updated["views"] == 50_000
    assert updated["likes"] == 1_200
    assert updated["estimated_earnings"] == 15.0  # 50k/100k * $30
    assert updated["views_pulled"] is True


def test_update_views_pull_failure_leaves_values_untouched(fresh_db, monkeypatch):
    sub = record_submission(None, "youtube",
                            "https://www.youtube.com/watch?v=VID12345678",
                            streamer="adinross")
    update_views(sub["id"], views=10_000)
    monkeypatch.setattr(earnings, "_try_youtube_pull", lambda url: None)
    updated = update_views(sub["id"])  # no views given, pull fails
    assert updated["views"] == 10_000
    assert updated["views_pulled"] is False


def test_get_and_list_submissions(fresh_db):
    s1 = record_submission(None, "tiktok", "https://tiktok.com/@x/1", streamer="xqc")
    record_submission(None, "youtube", "https://youtube.com/watch?v=A",
                      streamer="adinross")
    assert get_submission(s1["id"])["streamer"] == "xqc"
    assert get_submission(999) is None
    assert len(list_submissions()) == 2
    assert len(list_submissions(streamer="xqc")) == 1
    assert len(list_submissions(platform="youtube")) == 1


def test_earnings_summary_totals_and_breakdowns(fresh_db):
    s1 = record_submission(None, "tiktok", "https://tiktok.com/@x/1", streamer="xqc")
    s2 = record_submission(None, "youtube", "https://youtube.com/watch?v=A",
                           streamer="adinross")
    record_submission(None, "instagram", "https://instagram.com/reel/Z",
                      streamer="xqc")
    update_views(s1["id"], views=100_000)   # $15
    update_views(s2["id"], views=200_000)   # $60

    summary = earnings_summary()
    assert summary["total_earnings"] == 75.0
    assert summary["total_views"] == 300_000
    assert summary["submission_count"] == 3
    assert summary["by_streamer"]["xqc"]["earnings"] == 15.0
    assert summary["by_streamer"]["adinross"]["earnings"] == 60.0
    assert summary["by_platform"]["tiktok"]["views"] == 100_000
    assert summary["by_platform"]["youtube"]["views"] == 200_000
    assert summary["by_platform"]["instagram"]["views"] == 0


def test_earnings_summary_filters(fresh_db):
    s1 = record_submission(None, "tiktok", "https://tiktok.com/@x/1", streamer="xqc")
    record_submission(None, "tiktok", "https://tiktok.com/@x/2", streamer="adinross")
    update_views(s1["id"], views=100_000)

    by_xqc = earnings_summary(streamer="xqc")
    assert by_xqc["total_earnings"] == 15.0
    assert by_xqc["submission_count"] == 1

    # Month filter: backdate one submission to a past month.
    conn = database.get_connection()
    conn.execute("UPDATE submissions SET submitted_at = '2026-01-15 10:00:00' "
                 "WHERE id = ?", (s1["id"],))
    conn.commit()
    conn.close()
    assert earnings_summary(month="2026-01")["submission_count"] == 1
    assert earnings_summary(month="2026-02")["submission_count"] == 0


def test_earnings_summary_empty_db(fresh_db):
    summary = earnings_summary()
    assert summary["total_earnings"] == 0.0
    assert summary["total_views"] == 0
    assert summary["by_streamer"] == {}
