"""ProgressModel (FINAL_SPEC §5.5.15, §7.1): weights, phases, byte cap, ETA visibility and rounding, stall, drop()."""
from __future__ import annotations

import re
import time

import pytest

from yfs.ui.runpanel import GIB, ProgressModel

T0 = 1_800_000_000.0                      # a fixed wall-clock start (time.time() scale)


def done_around(now: float, seconds: float) -> str:
    return time.strftime("%H:%M", time.localtime(now + seconds))


def test_runpanel_imports_without_tkinter_side_effects():
    import yfs.ui.runpanel as rp
    assert rp.ProgressModel.PHASE["writing"] == (0.50, 0.40)


def test_overall_is_weighted_by_raw_size():
    m = ProgressModel({"a": 3 * GIB, "b": GIB})
    assert m.overall() == 0.0 and m.slide() is None
    m.on_progress("a", "reading", 0, 0, T0)
    m.on_finished("a", T0 + 10)
    assert m.overall() == pytest.approx(0.75)
    m.on_progress("b", "stitching", 50, 100, T0 + 20)
    assert m.slide() == pytest.approx(0.05 + 0.45 * 0.5)
    assert m.overall() == pytest.approx((3 + 0.275) / 4)


def test_missing_weight_counts_as_one_gib():
    m = ProgressModel({"a": 0, "b": None, "c": 2 * GIB})
    m.on_finished("a", T0)
    assert m.overall() == pytest.approx(1 / 4)
    m.on_progress("x", "stitching", 1, 1, T0 + 1)      # a row nobody weighed still counts, as 1 GiB
    assert m.overall() == pytest.approx((1 + 0.5) / 5)


@pytest.mark.parametrize("phase,cur,tot,frac", [
    ("reading", 0, 0, 0.0), ("reading", 5, 10, 0.025), ("stitching", 0, 64, 0.05), ("stitching", 64, 64, 0.5),
    ("preparing", 0, 0, 0.5), ("writing", 0, 0, 0.5), ("pyramid", 3, 6, 0.95), ("finishing", 0, 0, 1.0),
    ("stitching", 200, 100, 0.5),                       # current > total is capped
])
def test_phase_mapping(phase, cur, tot, frac):
    m = ProgressModel({"a": GIB})
    m.on_progress("a", phase, cur, tot, T0)
    assert m.slide() == pytest.approx(frac)


def test_bytes_phase_is_capped_at_099_and_never_moves_backwards():
    m = ProgressModel({"a": GIB})
    m.on_progress("a", "writing", 0, 0, T0)
    m.on_bytes("a", 50, 100, T0 + 2)
    assert m.slide() == pytest.approx(0.5 + 0.4 * 0.5)
    m.on_bytes("a", 500, 100, T0 + 4)                   # the estimate of the level-0 size was too low
    assert m.slide() == pytest.approx(0.5 + 0.4 * 0.99)
    m.on_progress("a", "stitching", 1, 2, T0 + 5)       # a late, older event does not pull the bar back
    assert m.slide() == pytest.approx(0.5 + 0.4 * 0.99)
    m.on_progress("a", "pyramid", 0, 7, T0 + 6)
    assert m.slide() == pytest.approx(0.9)


def run_linear(m: ProgressModel, row: str, seconds: float, fraction: float, start: float = T0) -> float:
    """Feed stitching progress so the row's fraction grows linearly to `fraction` over `seconds`."""
    steps = int(seconds)
    for i in range(steps + 1):
        f = fraction * i / steps
        m.on_progress(row, "stitching", int(round((f - 0.05) / 0.45 * 10000)) if f > 0.05 else 0, 10000, start + i)
    return start + steps


def test_eta_hidden_before_5_percent_and_60_seconds():
    m = ProgressModel({"a": GIB})
    now = run_linear(m, "a", 50, 0.5)                    # 50 % but only 50 s in
    assert m.eta_text(now) == ProgressModel.WORKING_OUT
    assert m.remaining_s(now) is None
    m2 = ProgressModel({"a": GIB})
    now2 = run_linear(m2, "a", 300, 0.049)               # 5 min in, still below 5 %
    assert m2.eta_text(now2) == ProgressModel.WORKING_OUT
    m3 = ProgressModel({"a": GIB})
    now3 = run_linear(m3, "a", 120, 0.2)
    assert m3.eta_text(now3) != ProgressModel.WORKING_OUT


def test_eta_rate_is_an_ema_of_5_second_samples():
    m = ProgressModel({"a": GIB})
    now = run_linear(m, "a", 120, 0.2)                   # a steady 0.2 / 120 s of the batch per second
    left = m.remaining_s(now)
    assert left == pytest.approx(0.8 / (0.2 / 120), rel=0.02)          # 480 s
    assert m.eta_text(now) == ProgressModel.eta_words(left, now)
    assert m.eta_text(now).startswith(("about 8 min left", "about 9 min left"))
    assert m.remaining_s(now, "a") == pytest.approx(left, rel=1e-6)
    # no progress for a while: each 5 s sample pulls the rate down with weight 0.3
    for k in range(1, 4):
        m.eta_text(now + 5 * k)
    assert m.remaining_s(now + 15) == pytest.approx(left / 0.7 ** 3, rel=0.02)


@pytest.mark.parametrize("seconds,words,minutes", [
    (5, "less than a minute", 0), (59.9, "less than a minute", 0),
    (60, "about 1 min", 1), (61, "about 2 min", 2), (9 * 60 + 1, "about 10 min", 10),
    (10 * 60, "about 10 min", 10), (12.4 * 60, "about 10 min", 10), (12.6 * 60, "about 15 min", 15),
    (57 * 60, "about 55 min", 55), (57.5 * 60, "about 1 h", 60), (59.9 * 60, "about 1 h", 60),
    (60 * 60, "about 1 h", 60), (64 * 60, "about 1 h", 60), (66 * 60, "about 1 h 10 min", 70),
    (119 * 60, "about 2 h", 120), (185 * 60, "about 3 h 10 min", 190),
])
def test_eta_rounding_table(seconds, words, minutes):
    """MW-23: 'done around' is the clock time of the rounded minutes, so the two halves always agree; 57.5-65 min is
    'about 1 h', never 'about 60 min' or 'about 1 h 00 min'."""
    text = ProgressModel.eta_words(seconds, T0)
    if words == "less than a minute":
        assert text == "less than a minute left"
    else:
        assert text == f"{words} left (done around {done_around(T0, minutes * 60)})"


def test_the_minutes_and_the_clock_time_always_agree():
    """MW-23, over six hours of estimates: the clock time is the stated time from now; no '0 min' or '60 min'."""
    now = time.mktime((2026, 10, 10, 22, 0, 0, 0, 0, -1))
    for seconds in range(60, 6 * 3600, 7):
        text = ProgressModel.eta_words(seconds, now)
        m = re.fullmatch(r"about (?:(\d+) h)? ?(?:([1-9]\d*) min)? left \(done around (\d\d):(\d\d)\)", text)
        assert m, text
        hours, mins, hh, mm = (int(x or 0) for x in m.groups())
        assert mins < 60 and (hh * 60 + mm - 22 * 60) % (24 * 60) == hours * 60 + mins, text


def test_stall_seconds_follow_changes_of_current_or_bytes():
    m = ProgressModel({"a": GIB, "b": GIB})
    assert m.stalled_for(T0) == 0.0                      # nothing running
    m.on_progress("a", "stitching", 10, 100, T0)
    assert m.stalled_for(T0 + 61) == pytest.approx(61)
    m.on_progress("a", "stitching", 10, 100, T0 + 30)   # the same value again is not progress
    assert m.stalled_for(T0 + 61) == pytest.approx(61)
    m.on_progress("a", "stitching", 11, 100, T0 + 40)
    assert m.stalled_for(T0 + 61) == pytest.approx(21)
    m.on_progress("a", "writing", 0, 0, T0 + 50)
    m.on_bytes("a", 1000, 10_000, T0 + 52)
    m.on_bytes("a", 1000, 10_000, T0 + 54)               # bytes unchanged
    assert m.stalled_for(T0 + 100) == pytest.approx(48)
    m.on_bytes("a", 2000, 10_000, T0 + 90)
    assert m.stalled_for(T0 + 100) == pytest.approx(10)
    m.on_finished("a", T0 + 100)
    assert m.stalled_for(T0 + 500) == 0.0
    m.on_progress("b", "reading", 0, 0, T0 + 501)        # a new row starts its own clock
    assert m.stalled_for(T0 + 511) == pytest.approx(10)


def test_drop_removes_an_unticked_waiting_row():
    m = ProgressModel({"a": GIB, "b": GIB, "c": 2 * GIB})
    m.on_finished("a", T0)
    assert m.overall() == pytest.approx(0.25)
    m.drop("c")
    assert m.overall() == pytest.approx(0.5)
    m.drop("b")
    assert m.overall() == pytest.approx(1.0)
    m.drop("missing")                                    # harmless
    assert m.overall() == pytest.approx(1.0)


def test_pyramid_bytes_are_activity_so_a_slow_pyramid_level_is_not_a_stall():
    """MW-9: during 'pyramid' the partial keeps growing; its size changes count as activity, not as progress."""
    m = ProgressModel({"a": GIB})
    m.on_progress("a", "pyramid", 1, 3, T0)
    m.on_bytes("a", 360_000, 10_000_000, T0 + 30, "pyramid")
    m.on_bytes("a", 1_340_000, 10_000_000, T0 + 59, "pyramid")
    assert m.stalled_for(T0 + 65) < 60 and m.stall_notice(T0 + 65, "LabShare", "Stitched") is None
    assert m.slide() == pytest.approx(0.9 + 0.1 / 3)                   # the bytes did not move the bar
    assert m.stall_notice(T0 + 59 + 61, "LabShare", "Stitched") == \
        "No progress for 1 min — writing to Stitched seems slow or stalled"


@pytest.mark.parametrize("phase,notice", [
    ("reading", "No progress for 3 min — LabShare may be stalled"),
    ("stitching", "No progress for 3 min — LabShare may be stalled"),
    ("writing", "No progress for 3 min — writing to Stitched seems slow or stalled"),
    ("pyramid", "No progress for 3 min — writing to Stitched seems slow or stalled"),
    ("preparing", None), ("finishing", None)])
def test_the_stall_notice_blames_the_place_the_engine_waits_for(phase, notice):
    """MW-9: tiles are read from the tile folder, the image is written to the save folder; while the engine prepares
    or finishes it reports nothing, so a quiet minute is not a stall."""
    m = ProgressModel({"a": GIB})
    m.on_progress("a", phase, 1, 3, T0)
    assert m.stall_notice(T0 + 59, "LabShare", "Stitched") is None
    assert m.stall_notice(T0 + 200, "LabShare", "Stitched") == notice


def test_the_stall_notice_has_words_for_unknown_places_and_waits_for_a_phase():
    m = ProgressModel({"a": GIB, "b": GIB})
    assert m.stall_notice(T0 + 500, "LabShare", "Stitched") is None        # nothing running
    m.on_progress("a", "stitching", 1, 9, T0)
    assert m.stall_notice(T0 + 61, "", "") == "No progress for 1 min — the tile folder may be stalled"
    m.on_progress("a", "writing", 0, 0, T0 + 100)
    assert m.stall_notice(T0 + 161, "", "") == "No progress for 1 min — writing to the save folder seems slow or " \
                                               "stalled"
    m.on_bytes("b", 10, 100, T0 + 200)                     # bytes of another row, before its first phase
    assert m.stall_notice(T0 + 400, "LabShare", "Stitched") is None
