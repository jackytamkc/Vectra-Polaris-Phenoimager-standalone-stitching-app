"""Run panel: the progress model (overall fraction, ETA, stall), the run bar (idle, running, stopping, completion
banner) and the log drawer. FINAL_SPEC §2.4.1, §2.4.4-§2.4.8, §4.9, §5.5.15."""
from __future__ import annotations

import math
import os
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from yfs.core.events import LogLine
from yfs.core.model import Issue, RunSummary
from yfs.core.names import human_bytes, human_duration
from yfs.core.osutil import display_text, open_path
from yfs.core.settings import log_dir
from yfs.ui.dialogs import IssuesDialog
from yfs.ui.theme import LOG_TAGS, TOKENS as T, ui_thread
from yfs.ui.widgets import Tooltip

if TYPE_CHECKING:
    from yfs.core.plan import Totals
    from yfs.ui.app import AppContext

GIB = float(1 << 30)


class ProgressModel:
    """Pure (testable): overall fraction, ETA text, stall detection.
    Each row's fraction runs through the phases below; the overall fraction weights rows by their estimated raw
    size. Times are time.time() values (event timestamps and `now`). Texts use Unicode: display them through
    Theme.text()."""
    PHASE = {"reading": (0.0, 0.05), "stitching": (0.05, 0.45), "preparing": (0.50, 0.0),
             "writing": (0.50, 0.40), "pyramid": (0.90, 0.10), "finishing": (1.0, 0.0)}   # (start, span)
    SAMPLE_S = 5.0                  # rate sampling period
    ALPHA = 0.3                     # EMA weight of the newest rate sample
    WORKING_OUT = "working out time left…"
    STALL_READING = "No progress for {} min — {} may be stalled"
    STALL_WRITING = "No progress for {} min — writing to {} seems slow or stalled"

    def __init__(self, weights: Mapping[str, float]) -> None:
        """weight = est.raw_bytes or 1 GiB."""
        self._w: Dict[str, float] = {r: float(w) if w else GIB for r, w in weights.items()}
        self._f: Dict[str, float] = dict.fromkeys(self._w, 0.0)
        self._current: Optional[str] = None
        self._phase: Optional[str] = None                     # the engine phase of the current row
        self._marks: Tuple[object, object] = (None, None)     # last (phase, current) and last bytes written
        self._changed_at: Optional[float] = None
        self._t0: Optional[float] = None
        self._sample: Optional[Tuple[float, float]] = None    # (time, done weight) of the last rate sample
        self._rate: Optional[float] = None                    # EMA of done weight per second

    # ---- events
    def on_progress(self, row_id: str, phase: str, current: int, total: int, t: float) -> None:
        start, span = self.PHASE.get(phase, (0.0, 0.0))
        self._advance(row_id, start + span * (min(1.0, current / total) if total > 0 else 0.0), t, (phase, current), 0)
        self._phase = phase

    def on_bytes(self, row_id: str, written: int, expected: int, t: float, phase: str = "writing") -> None:
        """frac = min(0.99, written/expected) while level 0 is written; growth in a later phase ('pyramid') only
        counts as activity."""
        start, span = self.PHASE["writing"]
        frac = start + span * (min(0.99, written / expected) if expected > 0 else 0.0) if phase == "writing" else 0.0
        self._advance(row_id, frac, t, written, 1)

    def on_finished(self, row_id: str, t: float) -> None:
        if row_id in self._f:
            self._f[row_id] = 1.0
        if self._current == row_id:
            self._current = None
        self._observe(t)

    def drop(self, row_id: str) -> None:
        """Unticked while waiting."""
        self._w.pop(row_id, None)
        self._f.pop(row_id, None)

    # ---- queries
    def overall(self) -> float:
        total = sum(self._w.values())
        return self._done() / total if total else 0.0

    def slide(self) -> Optional[float]:
        return self._f.get(self._current) if self._current is not None else None

    def remaining_s(self, now: float, row_id: Optional[str] = None) -> Optional[float]:
        """Estimated seconds left for the batch (or for one row), or None while there is no estimate yet."""
        self._observe(now)
        if not self._rate or self._t0 is None or now - self._t0 < 60 or self.overall() < 0.05:
            return None
        if row_id is not None:
            return self._w.get(row_id, 0.0) * (1.0 - self._f.get(row_id, 1.0)) / self._rate
        return (sum(self._w.values()) - self._done()) / self._rate

    def eta_text(self, now: float) -> str:
        """rate = EMA (alpha 0.3) of done-weight per second sampled every 5 s; shown only when overall >= 0.05 and
        elapsed >= 60 s, else 'working out time left…'. <60 s 'less than a minute'; <10 min 'about N min' (ceil);
        <60 min rounded to 5 min; else 'about H h MM min' (MM rounded to 10, 'about H h' when MM is 0) +
        ' (done around HH:MM)'."""
        left = self.remaining_s(now)
        return self.WORKING_OUT if left is None else self.eta_words(left, now)

    @staticmethod
    def eta_words(seconds: float, now: float) -> str:
        """The rounding table of eta_text() for a known number of seconds left. The clock time is that of the
        rounded minutes, so the two always agree (MW-23)."""
        if seconds < 60:
            return "less than a minute left"
        minutes = seconds / 60.0
        step = 1 if minutes < 10 else 5 if minutes < 60 else 10
        rounded = math.ceil(minutes) if step == 1 else int(minutes / step + 0.5) * step
        hours, mins = divmod(rounded, 60)
        text = (f"about {hours} h {mins} min" if mins else f"about {hours} h") if hours else f"about {rounded} min"
        return f"{text} left (done around {time.strftime('%H:%M', time.localtime(now + rounded * 60))})"

    def stalled_for(self, now: float) -> float:
        """Seconds since the last change of current or written bytes."""
        if self._current is None or self._changed_at is None:
            return 0.0
        return max(0.0, now - self._changed_at)

    def stall_notice(self, now: float, tiles_place: str, save_place: str) -> Optional[str]:
        """The run bar's warning after a minute without progress, blaming the place the engine is waiting for
        (MW-9): the tile folder while it reads and stitches tiles, the save folder while it writes the image or its
        pyramid. None before a minute, and while it prepares or finishes (no progress is reported then)."""
        stalled = self.stalled_for(now)
        if stalled < 60 or self._phase not in ("reading", "stitching", "writing", "pyramid"):
            return None
        if self._phase in ("reading", "stitching"):
            return self.STALL_READING.format(int(stalled // 60), tiles_place or "the tile folder")
        return self.STALL_WRITING.format(int(stalled // 60), save_place or "the save folder")

    # ---- internals
    def _done(self) -> float:
        return sum(self._w[r] * self._f[r] for r in self._w)

    def _advance(self, row_id: str, frac: float, t: float, mark: object, kind: int) -> None:
        if row_id not in self._w:                         # a row the caller did not weigh: 1 GiB
            self._w[row_id], self._f[row_id] = GIB, 0.0
        self._f[row_id] = max(self._f[row_id], frac)      # never moves backwards
        if row_id != self._current:
            self._current, self._phase, self._marks, self._changed_at = row_id, None, (None, None), t
        if self._marks[kind] != mark:
            self._marks = (mark, self._marks[1]) if kind == 0 else (self._marks[0], mark)
            self._changed_at = t
        self._observe(t)

    def _observe(self, t: float) -> None:
        """Start the clock and take a rate sample every SAMPLE_S seconds."""
        if self._t0 is None:
            self._t0 = t
        done = self._done()
        if self._sample is None:
            self._sample = (t, done)
        elif t - self._sample[0] >= self.SAMPLE_S:
            rate = max(0.0, (done - self._sample[1]) / (t - self._sample[0]))
            self._rate = rate if self._rate is None else self.ALPHA * rate + (1 - self.ALPHA) * self._rate
            self._sample = (t, done)


# ---------------------------------------------------------------------------------------------------- text helpers
def fit_text(font: tkfont.Font, text: str, width: int, ell: str, keep: str = "") -> str:
    """`text` cut at the right with `ell` so that it is at most `width` px wide (unchanged while width is unknown).
    keep: a tail of `text` that stays whole while there is room for it; the cut is made before it instead."""
    if width <= 1 or font.measure(text) <= width:
        return text
    if not (keep and text.endswith(keep) and font.measure(ell + keep) < width):
        keep = ""
    head = text[:len(text) - len(keep)]
    lo, hi = 0, len(head)
    while lo < hi:                                         # the longest prefix that fits together with the ellipsis
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if font.measure(head[:mid].rstrip() + ell + keep) <= width else (lo, mid - 1)
    return head[:lo].rstrip() + ell + keep


def fit_place(font: tkfont.Font, lead: str, place: str, tail: str, width: int, ell: str, least: int = 4) -> str:
    """'<lead><place><tail>' in at most `width` px (unchanged while width is unknown). The place's name is cut first,
    down to its first `least` characters, so that the words around it stay whole ("Can't save to 2026-09…: Permission
    denied."); when that is not enough, the end is cut too ("Can't save to 2026…: its .stitcher-in-progress fol…")."""
    def cut(n: int) -> str:
        return lead + place[:n].rstrip() + ell + tail
    text = lead + place + tail
    if width <= 1 or font.measure(text) <= width:
        return text
    if len(place) <= least:                                # a short name stays whole: the end is cut
        return fit_text(font, text, width, ell)
    if font.measure(cut(least)) > width:                   # not even the name's start fits with the rest
        return fit_text(font, cut(least), width, ell)
    lo, hi = least, len(place) - 1
    while lo < hi:                                         # the most characters of the name that fit
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if font.measure(cut(mid)) <= width else (lo, mid - 1)
    return cut(lo)


def _font(label: ttk.Label) -> tkfont.Font:
    name = str(label.cget("font")) or ttk.Style(label).lookup(str(label.cget("style")) or "TLabel", "font")
    return tkfont.nametofont(name or "TkDefaultFont")


def _padding(label: ttk.Label) -> int:
    """What a label adds to its text's width (its border and padding)."""
    return max(0, label.winfo_reqwidth() - _font(label).measure(str(label.cget("text"))))


def _lines(font: tkfont.Font, text: str, width: int) -> int:
    """How many lines `text` takes when it wraps at spaces to `width` px, as a label with that wraplength does."""
    lines, line = 1, ""
    for word in text.split(" "):
        if line and font.measure(f"{line} {word}") > width:
            lines, line = lines + 1, word
        else:
            line = f"{line} {word}" if line else word
    return lines


def _whole_parts(font: tkfont.Font, text: str, width: int, sep: str = " · ") -> str:
    """As many of the first parts of `text` (separated by `sep`) as fit in `width` px, each followed by `sep`: whole
    parts or nothing, never a word cut short ('Stitching 2 of 5 · PC-02_Scan1 · ')."""
    parts = text.split(sep)
    while parts and font.measure(sep.join(parts) + sep) > width:
        parts.pop()
    return sep.join(parts) + sep if parts else ""


def _refit(label: ttk.Label, ell: str) -> None:
    label.configure(text=fit_text(_font(label), label._yfs_full, label.winfo_width(), ell))


def fit_label(label: ttk.Label, text: str, ell: str) -> None:
    """Show `text` in a label made with width=1 (so it never widens its row), shortened with `ell` to the width the
    label gets and fitted again whenever it is resized."""
    if not hasattr(label, "_yfs_full"):
        label.bind("<Configure>", lambda _e: _refit(label, ell), add="+")
    label._yfs_full = text
    _refit(label, ell)


def _full_if_cut(label: ttk.Label) -> str:
    """For a Tooltip: the whole text of a label that fit_text() has shortened, else ''."""
    full = getattr(label, "_yfs_full", "")
    return full if str(label.cget("text")) != full else ""


def totals_text(t: "Totals", free: Optional[int], volume: str = "") -> str:
    """Row 1 of the idle run bar (§2.4.1): '5 slides · ~46 GB to write · 1.3 TB free on Data · all fit in memory'.
    Slides whose size can't be estimated at all (t.unsized) are counted apart, and never said to fit (MWV-2)."""
    if not t.ticked:
        return ""
    sized = t.ticked - t.unsized
    parts = [f"{t.ticked:,} slide{'s' if t.ticked != 1 else ''}"]
    if t.unknown_estimates:
        parts.append("sizes are estimated once the first tiles are read")
    elif sized:
        parts.append(f"{human_bytes(t.out_typical, True)} to write" +
                     (f" (+{t.unsized:,} of unknown size)" if t.unsized else ""))
    else:
        parts.append("size unknown" if t.ticked == 1 else "sizes unknown")
    if free is not None:
        parts.append(f"{human_bytes(free)} free" + (f" on {volume}" if volume else ""))
    if not t.unknown_estimates and sized:
        parts.append(f"{t.disk_mode} need{'s' if t.disk_mode == 1 else ''} temporary disk space (slower)"
                     if t.disk_mode else f"{sized:,} fit{'s' if sized == 1 else ''} in memory" if t.unsized
                     else "all fit in memory" if t.ticked > 1 else "fits in memory")
    return " · ".join(parts)


def _stall_parts(notice: str) -> Tuple[str, str, str]:
    """A ProgressModel stall notice as (the words before the place, the place, the words after it), so that a cut
    falls inside the place's name."""
    for template in (ProgressModel.STALL_WRITING, ProgressModel.STALL_READING):
        before, after = template.split("{}")[1:]
        at = notice.find(before)
        if at > 0 and notice.endswith(after):
            start = at + len(before)
            return notice[:start], notice[start:len(notice) - len(after)], after
    return "", notice, ""


_PROBLEMS = ("failed", "skipped", "finalize", "done_gaps", "cancelled")
_STITCHED = ("done", "done_gaps", "done_generic")


def problem_lines(summary: RunSummary, name_of: Callable[[str], str]) -> List[str]:
    """One plain sentence per row of a finished run that needs a look (§2.4.7), in run order."""
    return [f"{name_of(rid)} {o.message[:1].lower()}{o.message[1:]}" for rid, o in summary.rows if o.status in _PROBLEMS]


def banner_headline(summary: RunSummary, short: bool = False) -> str:
    """'Finished at 22:41 (took 46 min): 5 stitched (1 with gaps), 1 failed.' or why the batch stopped (§2.4.6,
    §2.4.7). short (compact mode): without the time, 'Finished: 5 stitched (1 with gaps), 1 failed.'"""
    if summary.stop_reason == "output_unavailable":
        why = next((o.message for _, o in reversed(summary.rows) if o.output_side or o.output_unavailable), "")
        why = why.split(": ", 1)[-1].rstrip(".") if why.startswith("Failed") else why.rstrip(".")
        return "Stopped: the save folder became unavailable" + (f" ({why})." if why else ".")
    c = summary.counts
    gaps = f" ({c['done_gaps']:,} with gaps)" if c.get("done_gaps") else ""
    parts = ((sum(c.get(k, 0) for k in _STITCHED), "stitched" + gaps), (c.get("failed", 0), "failed"),
             (c.get("skipped", 0), "skipped"), (c.get("cancelled", 0), "stopped"), (c.get("finalize", 0), "to finalize"))
    what = ", ".join(f"{n:,} {word}" for n, word in parts if n) or "nothing was stitched"
    if short:
        return f"{'Stopped' if summary.stop_reason else 'Finished'}: {what}."
    when = time.strftime("%H:%M", time.localtime(summary.finished))
    took = human_duration(summary.finished - summary.started)
    return f"{'Stopped' if summary.stop_reason else 'Finished'} at {when} (took {took}): {what}."


# ---------------------------------------------------------------------------------------------------- run bar
_ACTIONS = {"browse_input": (("Browse…", "browse_input"),), "browse_output": (("Browse…", "browse_output"),),
            "stop_scan": (("Stop searching", "stop_scan"),),
            "resume_scan": (("Keep searching", "resume_scan"), ("Stop here", "stop_scan")),
            "fix_channels": (("Fix names…", "fix_channels"),), "settings": (("Settings…", "settings"),),
            "delete_leftovers": (("Delete them", "delete_leftovers"),), "retry_check": (("Try again", "retry_check"),),
            "rescan": (("Rescan", "rescan"),)}
_NOT_ERRORS = {"no_source": "info", "no_output": "info", "none_ticked": "info", "paused": "info",
               "checking_output": "wait", "scanning": "wait"}   # blocking, but nothing went wrong
# "Can't save to <place>: <reason>." and the like: a long place name is cut, not the words around it (MW-24)
_PLACE_LEAD = {"output_error": "Can't save to ", "pdir_problem": "Can't save to ", "final_check": "Couldn't check "}


def _issue_look(issue: Issue) -> Tuple[bool, str]:
    """(quiet, icon name): information, and blocking issues where nothing went wrong, are shown quietly."""
    quiet = issue.severity == "info" or (issue.severity == "block" and issue.code in _NOT_ERRORS)
    return quiet, _NOT_ERRORS.get(issue.code, "info") if quiet else "warn" if issue.severity == "warn" else "error"


def primary_issue(issues: Sequence[Issue]) -> Optional[Issue]:
    """Show a blocking error before a quiet prerequisite, then the first warning."""
    return next((i for i in issues if i.severity == "block" and i.code not in _NOT_ERRORS),
                next((i for i in issues if i.severity == "block"), issues[0] if issues else None))


class RunBar(ttk.Frame):
    """The bar under the plan (§2.4.1, §2.4.4, §2.4.5, §2.4.7). Layouts: idle (issue line, totals, Stitch button),
    running (status line, overall and per-slide bars, Stop…), stopping, and the completion banner.
    on_issue_action(action) receives Issue.action values plus 'quit_anyway' and 'finalize'. The Stitch button shows
    'Checking…' from the click until the next show_*(); Ctrl+Enter can call `bar.button.invoke()`. The banner stays
    until it is dismissed ([x], dismiss(), its own buttons, or show_running()); show_idle() meanwhile only stores
    the idle data that dismissing it shows."""

    def __init__(self, master: tk.Misc, ctx: "AppContext", *, on_start: Callable[[], None],
                 on_stop: Callable[[], None], on_issue_action: Callable[[str], None],
                 on_open_output: Callable[[], None], on_retry_failed: Callable[[], None],
                 on_new_batch: Callable[[], None]) -> None:
        px, t = ctx.theme.px, ctx.theme.text
        super().__init__(master, padding=(px(12), px(6)))
        self.ctx, self.mode = ctx, "idle"
        self._cb = {"start": on_start, "action": on_issue_action, "open": on_open_output,
                    "retry": on_retry_failed, "new": on_new_batch}
        self._idle: Tuple[int, str, Optional[Issue], int] = (0, "", None, 0)
        self._hidden: Tuple[Issue, ...] = ()
        self._compact = False
        self._issue_widgets: List[tk.Widget] = []
        self.columnconfigure(0, weight=1)
        self.issue = ttk.Frame(self, padding=(px(8), px(3)))
        self.issue.bind("<Configure>", lambda _e: self._fit_row(self.issue, self.issue_text), add="+")
        self.issue_icon = ttk.Label(self.issue)
        self.issue_text = ttk.Label(self.issue)
        self.issue_icon.pack(side="left", padx=(0, px(6)))
        self.issue_text.pack(side="left", padx=(0, px(6)))
        self.totals = ttk.Label(self, style="Bg.Muted.TLabel", width=1)
        self.line = ttk.Frame(self)                         # running/stopping text, then the stall notice
        self.line.bind("<Configure>", lambda _e: self._fit_line(), add="+")
        self.status = ttk.Label(self.line)
        self.stall = ttk.Label(self.line, style="Bg.Warn.TLabel")
        self.status._yfs_full, self.stall._yfs_parts = "", ()
        self.status.pack(side="left")
        self.bars = ttk.Frame(self)
        self.overall = ttk.Progressbar(self.bars, style="Big.Horizontal.TProgressbar", maximum=1.0)
        self.slide = ttk.Progressbar(self.bars, style="Slim.Horizontal.TProgressbar", maximum=1.0)
        self.overall.pack(fill="x")
        self.slide.pack(fill="x", pady=(px(3), 0))
        self.stop_button = ttk.Button(self, text=t("Stop…"), style="Danger.TButton", command=on_stop)
        self.quit_button = ttk.Button(self, text="Quit anyway", style="Danger.TButton",
                                      command=lambda: on_issue_action("quit_anyway"))
        self._run_button: Optional[ttk.Button] = None      # the one beside the bars: Stop…, Quit anyway or none
        self.button = ttk.Button(self, style="Accent.TButton", command=self._start)
        self.banner: Optional[ttk.Frame] = None
        self._finished: Optional[Tuple[RunSummary, Sequence[str]]] = None
        Tooltip(self.issue_text, self._issue_tip)
        for label in (self.status, self.stall):
            Tooltip(label, self._line_tip)
        self.show_idle(0, "", None, 0)

    # ---- public API (§5.5.15)
    @ui_thread
    def show_idle(self, n_ticked: int, totals_text: str, first_issue: Optional[Issue], more: int,
                  hidden: Sequence[Issue] = ()) -> None:
        """Issue line (the first blocking issue, else the first warning, '+N more…'), totals and the Stitch button,
        which is disabled when nothing is ticked or the issue blocks. hidden: the other block/warn issues; the
        '+N more…' link lists them with their actions (MW-11). Without them it is a plain '(+N more)'."""
        self._idle = (n_ticked, totals_text, first_issue, more)
        self._hidden = tuple(hidden)
        if self.mode != "finished":
            self.mode = "idle"
            self._render_idle()

    @ui_thread
    def show_running(self, text: str, overall: float, slide: Optional[float], stall: Optional[str]) -> None:
        """text: 'Stitching 2 of 5 · <slide> · <phase>' plus ' · <ETA>' unless stalled; `stall` is shown after it in
        the warn colour. While stopping, only the bars move."""
        self.overall["value"] = max(0.0, min(1.0, overall))
        self.slide["value"] = max(0.0, min(1.0, slide or 0.0))
        if self.mode == "stopping":
            return
        if self.mode != "running":
            self.mode = "running"
            self._run_layout(self.stop_button)
        self._set_line(text, stall)

    @ui_thread
    def show_stopping(self, text: str, quit_anyway: bool) -> None:
        """'Stopping after <slide>…' or 'Stopping… finishing the current write step', with [Quit anyway]."""
        self.mode = "stopping"
        self._run_layout(self.quit_button if quit_anyway else None)
        self._set_line(text, None)

    @ui_thread
    def show_finished(self, summary: RunSummary, lines: Sequence[str]) -> None:
        """The completion banner (§2.4.7): the headline (wrapped, never cut), up to 3 problem lines, then a row with
        the summary CSV, the next steps and [x]. Compact mode keeps it about as low as the idle bar, so the plan keeps
        its four rows (MW-15, D4): the headline leaves out the time, the next steps stand beside it (under it when it
        would take more than two lines there), and its tooltip has the rest of the full banner."""
        th, px, icons = self.ctx.theme, self.ctx.theme.px, self.ctx.icons
        t, c = th.text, summary.counts
        retry = sum(c.get(k, 0) for k in ("failed", "skipped", "cancelled"))
        kind = "Warn" if retry or c.get("finalize") or c.get("done_gaps") or summary.stop_reason else "Ok"   # DS-1
        bg, label = T[f"{kind.lower()}_bg"], f"{kind}.Banner.TLabel"
        self.mode, self._finished = "finished", (summary, lines)
        if self.banner is not None:
            self.banner.destroy()
        b = self.banner = ttk.Frame(self, style=f"{kind}.TFrame", padding=(px(10), px(1 if self._compact else 6)))
        steps = tk.Frame(b, background=bg)                 # the next steps' row (the banner's first child)
        buttons = [("Open output folder", "open", ())] + ([("Finalize", "action", ("finalize",))]
                                                          if c.get("finalize") else [])
        if summary.stop_reason:
            buttons.append(("Retry remaining" if summary.stop_reason == "output_unavailable" else "Continue",
                            "start", ()))
        elif retry:
            buttons.append(("Retry failed", "retry", ()))
        for text, key, args in reversed(buttons + [("Start a new batch", "new", ()), ("", "dismiss", ())]):
            command = ((lambda k=key, a=args: self._cb[k](*a)) if key in ("open", "action") else
                       (lambda k=key: self._leave(k)))         # opening a folder or finalizing keeps the banner
            button = ttk.Button(steps, text=t(text), image=icons.get("close") if key == "dismiss" else "",
                                style="Primary.TButton" if key == "open" else f"{kind}.Banner.TButton",
                                command=command)
            if self._compact and key == "open":
                button.configure(padding=(px(12), px(2)))   # the same height as the other banner actions
            button.pack(side="right", padx=(px(6), 0))     # packed from the right
        headline = banner_headline(summary, short=self._compact)
        top = tk.Frame(b, background=bg)                   # the headline's row, or the left of the next steps' row
        icon = ttk.Label(top, image=icons.get(kind.lower()), style=label)
        icon.pack(side="left", anchor="n", padx=(0, px(8)))
        head = ttk.Label(top, text=t(headline), style=label, font="YfsBold", anchor="w", justify="left")
        # it wraps at the width it gets (less its border and padding); until then, at the room the bar seems to leave
        edge, width = _padding(head), self.winfo_width()
        room = width - px(44) - icon.winfo_reqwidth() - px(8) - edge
        beside = room - sum(w.winfo_reqwidth() + px(6) for w in steps.pack_slaves())
        one_row = self._compact and (width <= 1 or _lines(th.fonts["YfsBold"], t(headline), beside) <= 2)
        if one_row:                                        # (top is made after steps, so it is drawn over it)
            steps.pack(fill="x")
            top.pack(in_=steps, side="left", fill="x", expand=True)
        else:                                              # (the next steps are packed first: they keep their row)
            steps.pack(side="bottom", fill="x", pady=(0 if self._compact else px(4), 0))
            top.pack(fill="x")
        head.configure(wraplength=max(px(120), beside if one_row else room) if width > 1 else 0)
        head.bind("<Configure>", lambda e: head.configure(wraplength=max(1, e.width - edge)), add="+")
        head.pack(side="left", fill="x", expand=True)
        csv = f"Summary saved: {os.path.basename(summary.csv_path)}" if summary.csv_path else ""
        shown = list(lines[:3]) + ([f"…and {len(lines) - 3} more — see the list."] if len(lines) > 3 else [])
        if self._compact:
            full = banner_headline(summary)
            tip = "\n".join(x for x in (full if full != headline else "", *shown, csv) if x)
            Tooltip(head, lambda: tip)
        else:
            for text in shown:
                line = ttk.Label(b, style=label, width=1)
                line.pack(fill="x", padx=(icons.size + px(8), 0))
                fit_label(line, t(text), th.G["ell"])
                Tooltip(line, lambda w=line: _full_if_cut(w))
            if csv:
                note = self._note(steps, csv, link=("Open", lambda: open_path(str(summary.csv_path))))
                Tooltip(note, lambda: _full_if_cut(note))
        self._place((b, dict(row=0, column=0, columnspan=3, sticky="ew")))

    # ---- extras
    @ui_thread
    def dismiss(self) -> None:
        """Close the completion banner and show the latest idle data."""
        if self.mode == "finished":
            self.mode = "idle"
            self._render_idle()

    @ui_thread
    def show_open_error(self, path: str) -> None:
        """[Open output folder] failed: show the path with a [Copy] button in the banner (§2.4.7)."""
        if self.mode == "finished":
            self._banner_row(f"Couldn't open the folder: {path}", "Copy",
                             lambda: (self.clipboard_clear(), self.clipboard_append(path)))

    @ui_thread
    def set_compact(self, on: bool) -> None:
        """Compact mode (§2.1.4): the issue line and the totals share one line (the totals move to a tooltip), and
        the running bar and completion banner stay low enough for the plan to keep its rows (MW-15, D4)."""
        self._compact = bool(on)
        if self.mode == "idle":
            self._render_idle()
        elif self.mode == "finished" and self._finished is not None:
            self.show_finished(*self._finished)
        else:
            self._run_layout(self._run_button)

    # ---- internals
    def _start(self) -> None:
        if self.mode == "idle":
            self.button.configure(text=self.ctx.theme.text("Checking…"))
            self.button.state(["disabled"])
            self._cb["start"]()

    def _leave(self, key: str) -> None:
        self.dismiss()
        if key == "start":
            self._start()
        elif key != "dismiss":
            self._cb[key]()

    def _place(self, *specs: Tuple[tk.Widget, dict]) -> None:
        px = self.ctx.theme.px
        pad = 0 if self._compact and self.mode == "finished" else px(3 if self._compact else 6)
        self.configure(padding=(px(12), pad))
        for w in self.grid_slaves():
            w.grid_forget()
        for w, kw in specs:
            w.grid(**kw)

    def _render_idle(self) -> None:
        px, t = self.ctx.theme.px, self.ctx.theme.text
        n, totals, issue, more = self._idle
        self._set_issue(issue, more)
        fit_label(self.totals, t(totals), self.ctx.theme.G["ell"])
        self.button.configure(text=f"Stitch {n:,} slide{'s' if n != 1 else ''}")
        blocked = n == 0 or (issue is not None and issue.severity == "block")
        self.button.state(["disabled" if blocked else "!disabled"])
        specs = [(self.button, dict(row=0, column=2, rowspan=2, sticky="e", padx=(px(12), 0)))]
        if issue is not None:
            specs.append((self.issue, dict(row=0, column=0, sticky="ew", pady=(0, px(2)))))
        if issue is None or not self._compact:
            specs.append((self.totals, dict(row=1, column=0, sticky="ew", padx=(px(9), 0))))
        self._place(*specs)

    def _set_issue(self, issue: Optional[Issue], more: int) -> None:
        for w in self._issue_widgets:
            w.destroy()
        self._issue_widgets = []
        if issue is None:
            return
        t, px = self.ctx.theme.text, self.ctx.theme.px
        quiet, icon = _issue_look(issue)
        kind = "Warn" if issue.severity == "warn" else "Error"
        frame, label, button = (("TFrame", "Bg.Muted.TLabel", "LinkBg.TButton") if quiet else
                                (f"{kind}.TFrame", f"{kind}.Banner.TLabel", f"{kind}.Banner.TButton"))
        self.issue.configure(style=frame)
        self.issue_icon.configure(image=self.ctx.icons.get(icon), style=label)
        self.issue_text.configure(style=label)
        lead = _PLACE_LEAD.get(issue.code, "")
        end = issue.text.find(": ", len(lead)) if lead and issue.text.startswith(lead) else -1
        self.issue_text._yfs_full = t(issue.text)
        self.issue_text._yfs_parts = (tuple(t(p) for p in (lead, issue.text[len(lead):end], issue.text[end:]))
                                      if end > len(lead) else None)
        for text, action in _ACTIONS.get(issue.action or "", ()):
            self._issue_widgets.append(ttk.Button(self.issue, text=t(text), style=button,
                                                  command=lambda a=action: self._cb["action"](a)))
        if self._hidden:
            link = ttk.Button(self.issue, text=t(f"+{len(self._hidden)} more…"), style=button,
                              command=self._show_hidden)
            Tooltip(link, lambda: "\n".join(i.text for i in self._hidden))
            self._issue_widgets.append(link)
        elif more:
            self._issue_widgets.append(ttk.Label(self.issue, text=f"(+{more} more)", style=label))
        for w in self._issue_widgets:
            w.pack(side="left", padx=(0, px(6)))
        self._fit_row(self.issue, self.issue_text)

    def _issue_tip(self) -> str:
        """The whole issue while the line is cut short (MW-24), and the totals in compact mode (§2.1.4)."""
        return "\n".join(x for x in (_full_if_cut(self.issue_text), self._idle[1] if self._compact else "") if x)

    def _show_hidden(self) -> None:
        """'+N more…' (MW-11): every other blocking issue and warning, each with its action buttons."""
        rows = [(_issue_look(i)[1], i.text, _ACTIONS.get(i.action or "", ())) for i in self._hidden]
        IssuesDialog.open(self.winfo_toplevel(), self.ctx, rows, lambda action: action and self._cb["action"](action))

    def _run_layout(self, button: Optional[ttk.Button]) -> None:
        """The text line over the bars, with `button` (Stop…, Quit anyway or None) and the disabled accent button
        beside the bars. Compact mode keeps it as low as the idle bar (MW-15): the buttons stand beside both rows,
        and only the overall bar is shown."""
        px, self._run_button = self.ctx.theme.px, button
        row, span, text_span = (0, 2, 1) if self._compact else (1, 1, 3)    # the buttons' row and span; the text's
        self.button.configure(text=self.ctx.theme.text("Stitching…"))
        self.button.state(["disabled"])
        if self._compact:
            self.slide.pack_forget()
        else:
            self.slide.pack(fill="x", pady=(px(3), 0))
        specs = [(self.line, dict(row=0, column=0, columnspan=text_span, sticky="ew", pady=(0, px(4)))),
                 (self.bars, dict(row=1, column=0, sticky="ew")),
                 (self.button, dict(row=row, rowspan=span, column=2, sticky="e"))]
        if button is not None:
            specs.append((button, dict(row=row, rowspan=span, column=1, padx=(px(12), px(8)))))
        self._place(*specs)

    def _set_line(self, text: str, stall: Optional[str]) -> None:
        t = self.ctx.theme.text
        self.status._yfs_full = t(text)
        self.stall._yfs_parts = tuple(t(p) for p in _stall_parts(stall)) if stall else ()
        self._fit_line()

    def _fit_line(self) -> None:
        """Fit the running line to its width. A stall notice is its news (MW-9): it comes first, cut inside its
        place's name if it must be, and the status before it keeps the whole parts ('Stitching 2 of 5', the slide, the
        phase) that the room left can hold. Without a notice the status is cut with an ellipsis. The tooltip of the
        line has it all."""
        ell, width = self.ctx.theme.G["ell"], self.line.winfo_width()
        status, parts = self.status._yfs_full, self.stall._yfs_parts
        notice, font = "".join(parts), _font(self.status)
        if width <= 1:                                      # not laid out yet: both whole
            status += " · " if parts else ""
        elif parts:
            warn, room = _font(self.stall), width - _padding(self.status) - _padding(self.stall)
            notice = fit_place(warn, *parts, max(2, room), ell)
            status = _whole_parts(font, status, room - warn.measure(notice))
        else:
            status = fit_text(font, status, max(2, width - _padding(self.status)), ell)
        self.status.configure(text=status)
        self.stall.configure(text=notice)
        shown = [w for w in (self.status, self.stall) if str(w.cget("text"))]
        if self.line.pack_slaves() != shown:                # (an empty label would still take its border's room)
            for w in self.line.pack_slaves():
                w.pack_forget()
            for w in shown:
                w.pack(side="left")

    def _line_tip(self) -> str:
        """The whole running line while some of it is cut short or left out."""
        whole = self.status._yfs_full + (" · " + "".join(self.stall._yfs_parts) if self.stall._yfs_parts else "")
        return whole if str(self.status.cget("text")) + str(self.stall.cget("text")) != whole else ""

    def _fit_row(self, frame: ttk.Frame, label: ttk.Label) -> None:
        """Shorten `label` (natural width, packed first) so that the widgets packed after it stay visible; a label
        with _yfs_parts (lead, place, tail) is cut inside the place's name (MW-24)."""
        full, font, width = getattr(label, "_yfs_full", ""), _font(label), frame.winfo_width()
        others = [w for w in frame.pack_slaves() if w is not label]
        used = sum(w.winfo_reqwidth() for w in others) + self.ctx.theme.px(6) * (len(others) + 1)
        used += self.ctx.theme.px(16) + 2 if frame is self.issue else 0     # the issue box: padding and border
        room = max(2, width - used - _padding(label)) if width > 1 else 0   # (0: not laid out yet, keep it whole)
        parts, ell = getattr(label, "_yfs_parts", None), self.ctx.theme.G["ell"]
        label.configure(text=fit_place(font, *parts, room, ell) if parts else fit_text(font, full, room, ell))

    def _banner_row(self, text: str, button: str, command: Callable[[], None]) -> None:
        assert self.banner is not None
        row = tk.Frame(self.banner, background=T[f"{str(self.banner.cget('style')).split('.')[0].lower()}_bg"])
        row.pack(fill="x")
        self._note(row, text, link=(button, command))

    def _note(self, row: tk.Frame, text: str, link: Tuple[str, Callable[[], None]]) -> ttk.Label:
        """A banner text at the left of `row`, under the headline, cut to the room the row leaves it, with a link
        right after it."""
        assert self.banner is not None
        t, kind = self.ctx.theme.text, str(self.banner.cget("style")).split(".")[0]
        holder = tk.Frame(row, background=T[f"{kind.lower()}_bg"])
        holder.pack(side="left", fill="x", expand=True, padx=(self.ctx.icons.size + self.ctx.theme.px(8), 0))
        label = ttk.Label(holder, text=t(text), style=f"{kind}.Banner.TLabel")
        label.pack(side="left")
        label._yfs_full = t(text)
        ttk.Button(holder, text=link[0], style=f"{kind}.Banner.TButton", command=link[1]).pack(side="left")
        holder.bind("<Configure>", lambda _e: self._fit_row(holder, label), add="+")
        return label


# ---------------------------------------------------------------------------------------------------- log drawer
EVERYTHING, PROBLEMS_ONLY = "Everything", "Warnings and errors"


class LogDrawer:
    """§2.4.8: a bar (chevron button, warning/error pills, the last line, Copy all, Log folder) and a read-only
    tk.Text body that becomes the lower pane of `pane` when opened. Lines are inserted once per pump tick, at most
    MAX_LINES are kept, and the view follows new lines only when it was already at the bottom.
    The caller places `drawer.bar` (grid or pack) in bar_parent."""
    MAX_LINES = 5000

    def __init__(self, bar_parent: tk.Misc, pane: tk.PanedWindow, ctx: "AppContext") -> None:
        px = ctx.theme.px
        self.ctx, self.pane = ctx, pane
        self._pending: List[LogLine] = []
        self._flush_id: Optional[str] = None
        self._warnings = self._errors = 0
        self.bar = ttk.Frame(bar_parent, padding=(0, px(4), 0, px(2)))
        self.bar.columnconfigure(3, weight=1)
        self.toggle_button = ttk.Button(self.bar, text="Log", image=ctx.icons.get("chev_right"), compound="left",
                                        style="FlatBg.TButton", command=self.toggle)
        self.toggle_button.grid(row=0, column=0)
        self.warn_pill = ttk.Label(self.bar, style="Pill.Warn.TLabel")
        self.error_pill = ttk.Label(self.bar, style="Pill.Error.TLabel")
        for pill in (self.warn_pill, self.error_pill):
            pill.bind("<Button-1>", lambda _e: self.set_open(True))
        self.last = ttk.Label(self.bar, style="Bg.Small.TLabel", width=1)
        self.last.grid(row=0, column=3, sticky="ew", padx=(px(8), px(8)))
        self.level = tk.StringVar(self.bar, EVERYTHING)
        self.filter = ttk.Combobox(self.bar, textvariable=self.level, values=(EVERYTHING, PROBLEMS_ONLY),
                                   state="readonly", width=len(PROBLEMS_ONLY) + 1)
        self.filter.bind("<<ComboboxSelected>>", lambda _e: self._apply_filter())
        ttk.Button(self.bar, text="Copy all", style="FlatBg.TButton", command=self.copy_all).grid(row=0, column=5)
        ttk.Button(self.bar, text="Log folder", style="FlatBg.TButton", command=self.open_folder).grid(row=0, column=6)
        self.body = ttk.Frame(pane)
        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)
        self.text = tk.Text(self.body, wrap="none", font="YfsMono", state="disabled", width=1, height=1,
                            highlightthickness=0, padx=px(6), pady=px(4), cursor="arrow")
        ys = ttk.Scrollbar(self.body, orient="vertical", command=self.text.yview)
        xs = ttk.Scrollbar(self.body, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        for tag, opts in LOG_TAGS.items():
            self.text.tag_configure(tag, **opts)
        self.text.tag_raise("time")                         # the time and slide parts keep their own look
        self.text.tag_raise("slide")
        self.text.bind("<Map>", lambda _e: self.text.see("end"), add="+")    # opening shows the newest lines
        self.is_open = False

    # ---- public API (§5.5.15)
    @ui_thread
    def append(self, lines: Sequence[LogLine]) -> None:
        """Queue lines; they are inserted together once the current pump tick is over."""
        self._pending.extend(lines)
        if self._flush_id is None and self._pending:
            self._flush_id = self.text.after_idle(self._flush)

    @ui_thread
    def set_open(self, open_: bool) -> None:
        """Show or hide the body as the lower pane; its height is remembered in ui.log_height."""
        open_, px = bool(open_), self.ctx.theme.px
        if open_ == self.is_open:
            return
        self.is_open = open_
        settings, scale = self.ctx.settings, self.ctx.theme.scale
        if open_:
            height = settings.get("ui.log_height", 180) if settings is not None else 180
            height = height if isinstance(height, (int, float)) and height > 0 else 180
            self.pane.add(self.body, minsize=px(80), height=px(height), stretch="never")
            self.filter.grid(row=0, column=4, padx=(0, px(6)))
        else:
            if settings is not None and self.body.winfo_height() > 1:
                settings.set("ui.log_height", int(round(self.body.winfo_height() / scale)))
            self.pane.forget(self.body)
            self.filter.grid_remove()
        if settings is not None:
            settings.set("ui.log_open", open_)
        self.toggle_button.configure(image=self.ctx.icons.get("chev_down" if open_ else "chev_right"))

    @ui_thread
    def counts(self) -> Tuple[int, int]:
        """(warnings, errors) logged since the last reset_counts()."""
        return self._warnings, self._errors

    @ui_thread
    def reset_counts(self) -> None:
        """Start counting warnings and errors afresh (a new batch, MW-22); the log body keeps every line."""
        self._warnings = self._errors = 0
        self._update_pills()

    # ---- extras
    @ui_thread
    def toggle(self) -> None:
        """Ctrl+J / the Log button."""
        self.set_open(not self.is_open)

    @ui_thread
    def copy_all(self) -> None:
        self.text.clipboard_clear()
        self.text.clipboard_append(self.text.get("1.0", "end-1c"))

    @ui_thread
    def open_folder(self) -> None:
        open_path(log_dir())                               # a worker; failures end up in the log itself

    # ---- internals
    def _flush(self) -> None:
        self._flush_id = None
        lines, self._pending = self._pending, []
        if not lines or not self.text.winfo_exists():
            return
        xft, chunks, last = self.ctx.theme.xft, [], ""
        for ln in lines:
            self._warnings += ln.tag == "warn"
            self._errors += ln.tag == "error"
            stamp = time.strftime("%H:%M:%S", time.localtime(ln.created))
            slide = display_text(ln.slide, xft)
            message = display_text(ln.text, xft).replace("\n", "\n          ")
            chunks += [stamp + "  ", ("time", ln.tag)]
            if slide:
                chunks += [slide, ("slide", ln.tag), "  ", (ln.tag,)]
            chunks += [message + "\n", (ln.tag,)]
            last = f"{stamp}  {slide + '  ' if slide else ''}{message.splitlines()[0] if message else ''}"
        at_end = not self.text.winfo_viewable() or self.text.yview()[1] >= 0.999   # (hidden: yview is meaningless)
        self.text.configure(state="normal")
        self.text.insert("end", *chunks)
        extra = int(self.text.index("end-1c").split(".")[0]) - 1 - self.MAX_LINES
        if extra > 0:
            self.text.delete("1.0", f"{extra + 1}.0")
        self.text.configure(state="disabled")
        if at_end:
            self.text.see("end")
        fit_label(self.last, last, self.ctx.theme.G["ell"])
        self._update_pills()

    def _update_pills(self) -> None:
        px = self.ctx.theme.px
        for pill, n, word, col in ((self.warn_pill, self._warnings, "warning", 1),
                                   (self.error_pill, self._errors, "error", 2)):
            if n:
                pill.configure(text=f"{n:,} {word}{'s' if n != 1 else ''}")
                pill.grid(row=0, column=col, padx=(px(6), 0))
            else:
                pill.grid_remove()

    def _apply_filter(self) -> None:
        hide = self.level.get() == PROBLEMS_ONLY
        for tag in ("info", "ok", "debug"):
            self.text.tag_configure(tag, elide=hide)
        self.text.see("end")
