"""Dialogs: new folder, stop, the *Ready to stitch?* sheet, channel names, slide details, settings and the
*How it works* / *About* window (FINAL_SPEC §2.2.9, §2.3.7, §2.4.2-§2.4.5, §2.5, §5.5.16).

Every dialog is a tk.Toplevel, transient for its parent and centred over it; modal ones grab with safe_grab and hand
the grab back to whoever had it. Results go to on_done callbacks (no nested event loops). No dialog touches the
filesystem: the settings are read from memory and saved by the settings writer thread.
"""
from __future__ import annotations

import gc
import ntpath
import os
import platform
import re
import sys
import tkinter as tk
from collections import Counter
from tkinter import ttk
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from yfs import APP_NAME, __version__
from yfs.core import osutil
from yfs.core.model import Issue
from yfs.core.names import human_duration, norm_key
from yfs.core.plan import Plan, PlanRow, display_name
from yfs.core.settings import log_dir
from yfs.ui.theme import TOKENS as T
from yfs.ui.widgets import SlideFacts, center_over, safe_grab

if TYPE_CHECKING:
    from yfs.ui.app import AppContext

GITHUB_URL = "https://github.com/jackytamkc/Vectra-Polaris-Phenoimager-standalone-stitching-app"
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = frozenset(["con", "prn", "aux", "nul"] + [f"{d}{i}" for d in ("com", "lpt") for i in range(1, 10)])
_SCALES = (("Automatic", "auto"), ("90%", 0.9), ("100%", 1.0), ("125%", 1.25), ("150%", 1.5), ("175%", 1.75),
           ("200%", 2.0))
_UNCHANGED_PLAN = object()
# Appendix B: the v1 welcome text, word for word (as in v1's StitcherApp.show_welcome_message).
WELCOME_V1 = (
    "Welcome to your FAVOURITE Stitcher!\n\n"
    "It is important to note that this Stitcher only works with AKOYA VECTRA POLARIS/PHENOIMAGER unmixed tiles\n"
    "Reason to use this:\n"
    "1. You can run this in background \n"
    "2. You can stitched multiple folders sequentially without you keep clicking \n"
    "3. You don't have to pay £5/hr\n"
    "Note: This app uses your RAM/disk to store tmp files, make sure you have some space left \n"
    "If your input/output file is in network drive, the stitching speed is 99% depends on your network speed \n"
    "App developed by Jacky@Ramachandran Lab\n\n"
    "Logic adopted from Pete Bankhead AKA Qupath Founder\n"
    "Enjoy!")
_HOW_BULLETS = (
    "Nothing is stitched until you press the button: you first see every slide that was found, and you can untick "
    "any of them.",
    "Each slide is written into a hidden .stitcher-in-progress folder and moved into place only when it is "
    "complete, so a half-written file never looks finished.",
    "Slides that already have a stitched file are skipped unless you tick them.",
    "A slide that fits in memory is stitched in RAM; a bigger one uses temporary disk space next to the stitched "
    "files (or the folder chosen in Settings), which is slower.",
    "Network drives stitch at network speed: a local disk is fastest for the tiles and for the stitched files.")
_KEYS_MAIN = (("Ctrl+O / Ctrl+Shift+O", "Browse for tiles / for the output folder"), ("F5", "Rescan"),
              ("Ctrl+Enter", "Stitch (only when enabled)"),
              ("Space / Delete", "Toggle / untick the selected plan rows"), ("F2", "Rename the output of the focused row"),
              ("Shift+F10 or the Menu key", "The menu of the focused plan row (Try again, Finalize, Search deeper…)"),
              ("Enter or double-click", "Slide details (a double-click on the Output file cell renames instead)"),
              ("Ctrl+A (in the plan)", "Select all rows (selection only, not ticks)"), ("Ctrl+J", "Show or hide the log"),
              ("Ctrl+,", "Settings"), ("F1", "How it works (this window)"),
              ("Esc", "Close the topmost dialog, or cancel an inline edit"))
_KEYS_BROWSER = (("Enter / double-click", "Open the folder"), ("Ctrl+Enter", "Use the folder (or Save here)"),
                 ("Backspace, Alt+Up", "Parent folder"), ("Alt+Left / Alt+Right", "Back / forward"),
                 ("Ctrl+L, F6; typing / ~ \\", "Edit the path"),
                 ("Tab / Down (in path edit)", "Complete / open the suggestion list"),
                 ("F5, Ctrl+R", "Refresh (also retries a drive that isn't answering)"), ("Ctrl+F", "Filter"),
                 ("Ctrl+H", "Show or hide hidden items"), ("Ctrl+Shift+N", "New folder (when saving)"),
                 ("Home / End / PgUp / PgDn, letters", "Move in the list / jump to a name"),
                 ("Esc", "Leave path edit, clear the filter, clear the selection, then Cancel"))


def folder_name_error(name: str, existing_keys: FrozenSet[str]) -> Optional[str]:
    """Why `name` can't be used for a new folder (§2.2.9), or None. Keys are norm_key() of the existing names."""
    if not name.strip():
        return "Type a name for the new folder."
    if _ILLEGAL.search(name):
        return 'A name can\'t contain < > : " / \\ | ? * or control characters.'
    if name[-1] in ". ":
        return "A name can't end with a dot or a space."
    if name.split(".")[0].strip().casefold() in _RESERVED:
        return f"{name} is a reserved name on Windows."
    if norm_key(name) in existing_keys:
        return "Something with this name already exists here."
    return None


class _Dialog(tk.Toplevel):
    """Common frame: transient, a Surface body with padding, Esc and the close button cancel, centred when shown.
    A modal dialog grabs, and gives the grab back to the window that had it when it closes."""

    def __init__(self, master: tk.Misc, ctx: "AppContext", title: str, modal: bool = True) -> None:
        super().__init__(master)
        self.withdraw()
        self.ctx, self.modal, self._done_cb = ctx, modal, None
        self._closed = False
        self.px, self.G, self.T = ctx.theme.px, ctx.theme.G, ctx.theme.text
        try:
            holder = self.grab_current() if modal else None
        except KeyError:                                 # a grab held by a Tk-internal window
            holder = None
        self._regrab = str(holder) if holder is not None else ""
        self.title(self.T(title))
        self.transient(master.winfo_toplevel())
        self.configure(background=T["surface"])
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.body = ttk.Frame(self, style="Surface.TFrame", padding=self.px(16))
        self.body.pack(fill="both", expand=True)
        self.body.columnconfigure(0, weight=1)

    def q(self, s: str) -> str:
        """`s` in single quotes."""
        return f"{self.G['lq']}{s}{self.G['rq']}"

    def _show(self, focus: Optional[tk.Misc] = None) -> None:
        center_over(self, self.master.winfo_toplevel())
        self.deiconify()
        if self.modal:
            safe_grab(self)
        (focus or self).focus_set()

    def _buttons(self, row: int, specs: Sequence[Tuple[str, Callable[[], None], str]],
                 left: Optional[Tuple[str, Callable[[], None]]] = None) -> Dict[str, ttk.Button]:
        """A button row (the last spec is the default action), with an optional button at the left."""
        frame = ttk.Frame(self.body, style="Surface.TFrame")
        frame.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(self.px(16), 0))
        out = {}
        if left is not None:
            out[left[0]] = ttk.Button(frame, text=self.T(left[0]), command=left[1])
            out[left[0]].pack(side="left")
        for text, cmd, style in reversed(specs):
            out[text] = ttk.Button(frame, text=self.T(text), command=cmd, style=style)
            out[text].pack(side="right", padx=(self.px(8), 0))
        return out

    def _label(self, parent: tk.Misc, text: str, style: str = "Surface.TLabel", wrap: int = 520, **grid: Any) -> ttk.Label:
        label = ttk.Label(parent, text=self.T(text), style=style, wraplength=self.px(wrap), justify="left")
        label.grid(sticky="w", **grid)
        return label

    def _fit_list(self, items: ttk.Frame) -> None:
        """Size the warning list: all of it, or as much as keeps the dialog within MAX_SCREEN of the screen, with a
        scroll bar, the mouse wheel and Up/Down/PgUp/PgDn to see the rest."""
        self.update_idletasks()
        w, h = items.winfo_reqwidth(), items.winfo_reqheight()
        others = self.winfo_reqheight() - self.listing.winfo_reqheight()     # padding and the button row
        room = max(self.px(120), int(self.winfo_screenheight() * self.MAX_SCREEN) - others)
        self.listing.configure(width=w, height=min(h, room), scrollregion=(0, 0, w, h))
        if h <= room:
            return
        bar = ttk.Scrollbar(self.body, orient="vertical", command=self.listing.yview)
        bar.grid(row=0, column=1, sticky="ns", padx=(self.px(8), 0))
        self.listing.configure(yscrollcommand=bar.set)
        for seq, units in (("<Button-4>", -1), ("<Button-5>", 1), ("<Up>", -1), ("<Down>", 1)):
            self.bind(seq, lambda _e, n=units: self.listing.yview_scroll(n, "units"), add="+")
        for seq, pages in (("<Prior>", -1), ("<Next>", 1)):
            self.bind(seq, lambda _e, n=pages: self.listing.yview_scroll(n, "pages"), add="+")
        self.bind("<MouseWheel>", lambda e: self.listing.yview_scroll(-1 if e.delta > 0 else 1, "units"), add="+")

    def _see(self, widget: tk.Misc) -> None:
        """Scroll a list control reached with Tab into view."""
        top, parent = 0, widget
        while parent.master is not self.listing:
            top += parent.winfo_y()
            parent = parent.master
        bottom = top + widget.winfo_height()
        shown, height = self.listing.canvasy(0), self.listing.winfo_height()
        total = max(1, int(self.listing.cget("scrollregion").split()[3]))
        if top < shown or bottom > shown + height:
            step = max(1, int(self.listing.cget("yscrollincrement")))
            offset = max(0, min(top, bottom - height))
            self.listing.yview_moveto(((offset + step - 1) // step * step) / total)

    def _close(self, result: Any) -> None:
        if self._closed:
            return
        cb, self._done_cb = self._done_cb, None
        regrab = self._regrab
        self.destroy()
        if regrab and int(self.tk.call("winfo", "exists", regrab)):
            safe_grab(self.nametowidget(regrab))
        if cb is not None:
            cb(result)

    def _on_destroy(self, event: Any) -> None:
        if event.widget is not self or self._closed:
            return
        self._closed = True
        self._done_cb, self._regrab = None, ""
        if type(self).__dict__.get("instance") is self:
            type(self).instance = None
        self._free_tk_objects()

    def _free_tk_objects(self) -> None:
        """Free this dialog's Tk variables and images now, on the Tk thread (CT-6). Left to the garbage collector,
        they would be finalised on whichever thread collects next, and from a worker that Tcl call can freeze the
        app. The collection also clears the dialog's own reference cycles here rather than on a worker."""
        tk_objects = (tk.Variable, tk.Image)
        for name, value in list(vars(self).items()):
            if isinstance(value, tk_objects) or (isinstance(value, list) and value and
                                                 all(isinstance(v, tk_objects) for v in value)):
                delattr(self, name)
        gc.collect()

    @classmethod
    def _alive(cls) -> Optional["_Dialog"]:
        """The single instance of a non-modal dialog class, if its window still exists."""
        dlg = cls.__dict__.get("instance")
        try:
            return dlg if dlg is not None and dlg.winfo_exists() else None
        except tk.TclError:                              # its Tk application was destroyed
            return None

    def _cancel(self) -> None:
        self._close(None)


class NewFolderDialog(_Dialog):
    """Output-mode browser: name a new folder; on_done(name) or on_done(None) (§2.2.9)."""

    @classmethod
    def ask(cls, master: tk.Misc, ctx: "AppContext", existing_keys: FrozenSet[str],
            on_done: Callable[[Optional[str]], None]) -> "NewFolderDialog":
        return cls(master, ctx, existing_keys, on_done)

    def __init__(self, master: tk.Misc, ctx: "AppContext", existing_keys: FrozenSet[str],
                 on_done: Callable[[Optional[str]], None]) -> None:
        super().__init__(master, ctx, "New folder")
        self._done_cb, self._keys = on_done, existing_keys
        self._label(self.body, "Name of the new folder:", row=0, column=0)
        self.var = tk.StringVar(self, "" if norm_key("stitched") in existing_keys else "stitched")
        self.entry = ttk.Entry(self.body, textvariable=self.var, width=40)
        self.entry.grid(row=1, column=0, sticky="ew", pady=(self.px(6), 0))
        self.error = self._label(self.body, "", "Surface.Error.TLabel", 360, row=2, column=0, pady=(self.px(4), 0))
        self._buttons(3, (("Cancel", self._cancel, "TButton"), ("Create", self._ok, "Primary.TButton")))
        self.entry.bind("<Return>", lambda _e: self._ok())
        self.entry.select_range(0, "end")
        self._show(self.entry)

    def _ok(self) -> None:
        name = self.var.get()
        problem = folder_name_error(name, self._keys)
        if problem:
            self.error.configure(text=self.T(problem))
        else:
            self._close(name)


class StopDialog(_Dialog):
    """Stop (or quit) during a run: on_done('keep' | 'after_current' | 'now') (§2.4.5). quitting_after: the app
    already closes after the current slide; the dialog then offers on_done('cancel_quit' | 'now') (MW-19).
    `closing` tells the caller whether the answer was about quitting."""

    @classmethod
    def open(cls, master: tk.Misc, ctx: "AppContext", slide: str, eta_s: Optional[float], n_done: int,
             closing: bool, on_done: Callable[[str], None], quitting_after: bool = False,
             stopping_now: bool = False) -> "StopDialog":
        return cls(master, ctx, slide, eta_s, n_done, closing, on_done, quitting_after, stopping_now)

    def __init__(self, master: tk.Misc, ctx: "AppContext", slide: str, eta_s: Optional[float], n_done: int,
                 closing: bool, on_done: Callable[[str], None], quitting_after: bool = False,
                 stopping_now: bool = False) -> None:
        super().__init__(master, ctx, f"Quit {APP_NAME}?" if closing or quitting_after else "Stop stitching?")
        self.closing = closing
        self._done_cb = on_done
        if stopping_now:
            self._show(self._stopping(n_done, quitting_after))
        elif quitting_after:
            self._show(self._quitting(self.q(slide), eta_s, n_done))
        else:
            self._show(self._stop_choices(self.q(slide), eta_s, n_done, closing))

    def _stopping(self, n_done: int, quitting: bool) -> ttk.Button:
        """Stop now cannot be undone: waiting lets the write step end, never finishes or resumes the slide."""
        self._label(self.body, "The app will close once the current write step ends." if quitting else
                    "Stitching is stopping. The current write step must end before it can stop.", row=0, column=0)
        self._label(self.body, f"The current slide will be discarded and can be redone later.\n"
                    f"Finished slides ({n_done}) are kept. Keeping the app open will not resume stitching.",
                    "Surface.Muted.TLabel", row=1, column=0, pady=(self.px(8), 0))
        stay = "Don't quit" if quitting else "Keep app open"
        actions = [(stay, lambda: self._close("cancel_quit" if quitting else "keep"), "TButton")]
        if not quitting:
            actions.append(("Wait, then quit", lambda: self._close("after_current"), "TButton"))
        actions.append(("Quit now", lambda: self._close("now"), "Danger.TButton"))
        return self._buttons(2, actions)[stay]

    def _stop_choices(self, q: str, eta_s: Optional[float], n_done: int, closing: bool) -> ttk.Button:
        px, dash = self.px, self.G["dash"]
        self.choice = tk.StringVar(self, "after_current")
        options = ((f"Finish {q}, then quit" if closing else f"Finish {q} first, then stop", "after_current"),
                   (f"Quit now {dash} {q} is discarded" if closing
                    else f"Stop now {dash} {q} is discarded and can be redone later", "now"))
        eta = f"(about {human_duration(eta_s)})  " if eta_s is not None else ""
        ttk.Style(self).configure("Stop.Surface.TRadiobutton", wraplength=px(520))
        for row, (text, value) in enumerate(options):
            line = ttk.Frame(self.body, style="Surface.TFrame")
            line.grid(row=row, column=0, sticky="w", pady=(0, px(4)))
            ttk.Radiobutton(line, text=self.T(text), variable=self.choice, value=value,
                            style="Stop.Surface.TRadiobutton").pack(anchor="w")
            if not row:
                ttk.Label(line, text=eta + "recommended", style="Surface.Muted.TLabel").pack(
                    anchor="w", padx=(px(20), 0), pady=(0, px(4)))
        note = f"Finished slides ({n_done}) are kept either way."
        self._label(self.body, note + ("\nYou can minimise the window instead." if closing else ""),
                    "Surface.Muted.TLabel", row=2, column=0, pady=(px(8), 0))
        buttons = self._buttons(3, (("Keep stitching", lambda: self._close("keep"), "TButton"),
                                    ("Quit" if closing else "Stop", lambda: self._close(self.choice.get()),
                                     "Danger.TButton")))
        return buttons["Keep stitching"]

    def _quitting(self, q: str, eta_s: Optional[float], n_done: int) -> ttk.Button:
        """The app already quits after the current slide (MW-19): [Don't quit] -> 'cancel_quit', [Quit now] ->
        'now'; Esc changes nothing ('keep')."""
        eta = f" (about {human_duration(eta_s)})" if eta_s is not None else ""
        self._label(self.body, f"The app will close after {q} finishes{eta}.", row=0, column=0)
        self._label(self.body, f"Don't quit: the app stays open, and stitching still stops after {q}.\n"
                    f"Quit now: {q} is discarded and can be redone later.\n"
                    f"Finished slides ({n_done}) are kept either way.", "Surface.Muted.TLabel", row=1, column=0,
                    pady=(self.px(8), 0))
        buttons = self._buttons(2, (("Don't quit", lambda: self._close("cancel_quit"), "TButton"),
                                    ("Quit now", lambda: self._close("now"), "Danger.TButton")))
        return buttons["Don't quit"]

    def finished(self) -> None:
        """A closing question may outlast the batch: keep its quit choice, without obsolete stop promises."""
        for widget in self.body.winfo_children():
            widget.destroy()
        if hasattr(self, "choice"):
            del self.choice
        self._label(self.body, "The batch has ended. Quit the app?", row=0, column=0)
        buttons = self._buttons(1, (("Keep app open", lambda: self._close("keep"), "TButton"),
                                    ("Quit", lambda: self._close("now"), "Danger.TButton")))
        self._show(buttons["Keep app open"])

    def dismiss(self) -> None:
        """Close without an answer (the run ended while the dialog was open, CT-8)."""
        self._done_cb = None
        try:
            if self.winfo_exists():
                self._close(None)
        except tk.TclError:                              # its Tk application was destroyed
            pass

    def _cancel(self) -> None:
        self._close("keep")


class ConfirmStartDialog(_Dialog):
    """*Ready to stitch?*: the warnings, plus the generic-names choice for unresolved channel problems;
    on_done('start' | 'back' | 'fix_channels') (§2.4.2). The caller marks generic names accepted on 'start'.
    The warnings scroll, so the sheet stays within MAX_SCREEN of the screen height and its buttons in view (MW-5)."""
    MAX_SCREEN = 0.6

    @classmethod
    def open(cls, master: tk.Misc, ctx: "AppContext", warnings: List[Issue], unresolved_channel_rows: List[str],
             on_done: Callable[[str], None]) -> "ConfirmStartDialog":
        return cls(master, ctx, warnings, unresolved_channel_rows, on_done)

    def __init__(self, master: tk.Misc, ctx: "AppContext", warnings: List[Issue], unresolved_channel_rows: List[str],
                 on_done: Callable[[str], None]) -> None:
        super().__init__(master, ctx, "Ready to stitch?")
        self._done_cb, px = on_done, self.px
        self.choice = tk.StringVar(self, "generic")
        self.listing = tk.Canvas(self.body, highlightthickness=0, background=T["surface"], yscrollincrement=px(20))
        self.listing.grid(row=0, column=0, sticky="nsew")
        items = ttk.Frame(self.listing, style="Surface.TFrame")
        self.listing.create_window(0, 0, window=items, anchor="nw")
        row = 0
        for issue in warnings:
            ttk.Label(items, image=self.ctx.icons.get("warn" if issue.severity != "info" else "info"),
                      style="Surface.TLabel").grid(row=row, column=0, sticky="nw", padx=(0, px(8)), pady=(px(2), 0))
            self._label(items, issue.text, row=row, column=1, pady=(0, px(6)))
            row += 1
            if issue.code == "channels" and unresolved_channel_rows:
                dash = self.G["dash"]
                for text, value in ((f"Use generic names (Channel 0, 1, 2{self.G['ell']}) {dash} you can rename "
                                     "them in QuPath later", "generic"), (f"Fix the names now{self.G['ell']}", "fix")):
                    radio = ttk.Radiobutton(items, text=self.T(text), variable=self.choice, value=value,
                                            style="Surface.TRadiobutton")
                    radio.grid(row=row, column=1, sticky="w", pady=(0, px(4)))
                    radio.bind("<FocusIn>", lambda e: self._see(e.widget), add="+")
                    row += 1
        buttons = self._buttons(1, (("Go back", lambda: self._close("back"), "TButton"),
                                    ("Start stitching", self._start, "Accent.TButton")))
        self.start = buttons["Start stitching"]
        self.choice.trace_add("write", lambda *_: self.start.configure(
            text=self.T(f"Fix names{self.G['ell']}" if self.choice.get() == "fix" else "Start stitching")))
        self._unresolved = bool(unresolved_channel_rows)
        self._fit_list(items)
        self._show(self.start)

    def _start(self) -> None:
        self._close("fix_channels" if self._unresolved and self.choice.get() == "fix" else "start")

    def _cancel(self) -> None:
        self._close("back")


class IssuesDialog(_Dialog):
    """'+N more…' in the run bar (MW-11): the blocking issues and warnings that its issue line has no room for, each
    with its action buttons. rows: (icon name, text, ((button text, action), ...)). on_done(action) for a button,
    on_done(None) for Close or Esc."""
    MAX_SCREEN = 0.6

    @classmethod
    def open(cls, master: tk.Misc, ctx: "AppContext", rows: Sequence[Tuple[str, str, Sequence[Tuple[str, str]]]],
             on_done: Callable[[Optional[str]], None]) -> "IssuesDialog":
        return cls(master, ctx, rows, on_done)

    def __init__(self, master: tk.Misc, ctx: "AppContext", rows: Sequence[Tuple[str, str, Sequence[Tuple[str, str]]]],
                 on_done: Callable[[Optional[str]], None]) -> None:
        super().__init__(master, ctx, f"{len(rows)} more thing{'s' if len(rows) != 1 else ''} to check")
        self._done_cb, px = on_done, self.px
        self.listing = tk.Canvas(self.body, highlightthickness=0, background=T["surface"], yscrollincrement=px(20))
        self.listing.grid(row=0, column=0, sticky="nsew")
        items = ttk.Frame(self.listing, style="Surface.TFrame")
        self.listing.create_window(0, 0, window=items, anchor="nw")
        for row, (icon, text, actions) in enumerate(rows):
            ttk.Label(items, image=ctx.icons.get(icon), style="Surface.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, px(8)), pady=(px(2), 0))
            self._label(items, text, wrap=420, row=row, column=1, pady=(0, px(10)))
            line = ttk.Frame(items, style="Surface.TFrame")
            line.grid(row=row, column=2, sticky="ne", padx=(px(16), 0))
            for label, action in actions:
                button = ttk.Button(line, text=self.T(label), style="Link.TButton",
                                    command=lambda a=action: self._close(a))
                button.pack(side="left", padx=(px(4), 0))
                button.bind("<FocusIn>", lambda e: self._see(e.widget), add="+")
        close = self._buttons(1, (("Close", self._cancel, "TButton"),))["Close"]
        self._fit_list(items)
        self._show(close)


class ChannelNamesDialog(_Dialog):
    """Channel names for rows with a channel problem (R26); the first row is the reference.
    on_done(True) when names (or generic names) were set, False when cancelled (§2.4.3)."""

    @classmethod
    def open(cls, master: tk.Misc, ctx: "AppContext", plan: Plan, row_ids: List[str],
             on_done: Callable[[bool], None]) -> None:
        rows = [plan.row(i) for i in row_ids]
        rows = [r for r in rows if r.header is not None and r.header.channel_problem]
        if rows:
            cls(master, ctx, plan, rows, on_done)
        else:
            on_done(False)

    def __init__(self, master: tk.Misc, ctx: "AppContext", plan: Plan, rows: List[PlanRow],
                 on_done: Callable[[bool], None]) -> None:
        first, h = rows[0], rows[0].header
        self.same = [r for r in rows[1:] if r.header.n_channels == h.n_channels
                     and r.header.channel_problem == h.channel_problem]
        more = f" (and {len(self.same)} more with the same problem)" if self.same else ""
        super().__init__(master, ctx, f"Channel names · {display_name(first.cand)}{more}")
        self._done_cb, self.plan, self.first, px = on_done, plan, first, self.px
        n, got = h.n_channels, list(h.channels_extracted)
        if h.channel_problem == "duplicate" and len(set(got)) == 1:
            why = f"The tiles say every channel is called {self.G['ldq']}{got[0]}{self.G['rdq']}."
        elif h.channel_problem == "duplicate":
            why = "Some channel names in the tiles repeat."
        elif h.channel_problem == "count":
            why = f"The tiles name {len(got)} channels, but the images have {n}."
        else:
            why = "The tiles don't say what their channels are called."
        self._label(self.body, why + " Type the real names. They are stored in the stitched file, and QuPath / "
                    "Visiopharm show them.", row=0, column=0, columnspan=4, pady=(0, px(10)))
        grid = ttk.Frame(self.body, style="Surface.TFrame")
        grid.grid(row=1, column=0, columnspan=4, sticky="w")
        for col, head in enumerate(("#", "Colour", "Name in the tiles", "Use this name")):
            ttk.Label(grid, text=head, style="Surface.Small.TLabel").grid(row=0, column=col, sticky="w",
                                                                          padx=(0, px(14)), pady=(0, px(4)))
        start = list(first.channel_override or got)
        self.vars = [tk.StringVar(self, start[i] if i < len(start) and start[i] else f"Channel {i}") for i in range(n)]
        colors = list(h.channel_colors) + [None] * n
        self.entries = []
        for i in range(n):
            ttk.Label(grid, text=str(i), style="Surface.TLabel").grid(row=i + 1, column=0, sticky="w")
            dot = tk.Canvas(grid, width=px(12), height=px(12), background=T["surface"], highlightthickness=0)
            dot.create_oval(1, 1, px(12) - 1, px(12) - 1, fill=colors[i] or T["text_faint"],
                            outline=T["border_strong"])                  # a white channel stays visible
            dot.grid(row=i + 1, column=1, sticky="w")
            ttk.Label(grid, text=self.T(got[i] if i < len(got) else self.G["dash"]),
                      style="Surface.Muted.TLabel").grid(row=i + 1, column=2, sticky="w", padx=(0, px(14)))
            entry = ttk.Entry(grid, textvariable=self.vars[i], width=28)
            entry.grid(row=i + 1, column=3, sticky="w", pady=px(2))
            self.entries.append(entry)
        sources = [r for r in plan.rows() if r is not first and r.header is not None and r.header.n_channels == n
                   and (r.channel_override or r.header.channel_problem is None)]
        labels = [plan.label(r) for r in sources]
        counts = Counter(self.T(label) for label in labels)
        self.sources = []
        for r, label in zip(sources, labels):
            if counts[self.T(label)] > 1:
                label += f" {self.G['dash']} {r.cand.rel_folder}"
            self.sources.append((f"{label} ({n} channels)", tuple(r.channel_override or r.header.channels_extracted)))
        row = 2
        if self.sources:
            line = ttk.Frame(self.body, style="Surface.TFrame")
            line.grid(row=row, column=0, columnspan=4, sticky="w", pady=(px(10), 0))
            ttk.Label(line, text="Copy names from:", style="Surface.TLabel").pack(side="left", padx=(0, px(8)))
            self.copy = ttk.Combobox(line, state="readonly", width=40, values=[self.T(s[0]) for s in self.sources])
            self.copy.pack(side="left")
            self.copy.bind("<<ComboboxSelected>>", lambda _e: [v.set(name) for v, name in
                                                               zip(self.vars, self.sources[self.copy.current()][1])])
            row += 1
        self.also = tk.BooleanVar(self, True)
        if self.same:
            ttk.Checkbutton(self.body, text=f"Also use these names for the {len(self.same)} other slide"
                            f"{'s' if len(self.same) > 1 else ''} with the same problem", variable=self.also,
                            style="Surface.TCheckbutton").grid(row=row, column=0, columnspan=4, sticky="w",
                                                               pady=(px(8), 0))
            row += 1
        self._label(self.body, "Names that repeat are allowed, but make the channels hard to tell apart.",
                    "Surface.Muted.TLabel", row=row, column=0, columnspan=4, pady=(px(8), 0))
        self.error = self._label(self.body, "", "Surface.Error.TLabel", row=row + 1, column=0, columnspan=4)
        self._buttons(row + 2, (("Cancel", self._cancel, "TButton"), ("Save names", self._save, "Primary.TButton")),
                      left=(f"Use generic names (Channel 0-{n - 1})", self._generic))
        self._show(self.entries[0])

    def _targets(self) -> List[str]:
        return [self.first.row_id] + ([r.row_id for r in self.same] if self.also.get() else [])

    def _save(self) -> None:
        names = [v.get().strip() for v in self.vars]
        if not all(names):
            self.error.configure(text="Every channel needs a name.")
            return
        self.plan.set_channels(self._targets(), names)
        self._close(True)

    def _generic(self) -> None:
        self.plan.set_channels(self._targets(), None)
        self._close(True)

    def _cancel(self) -> None:
        self._close(False)


class SlideDetailsDialog(_Dialog):
    """Non-modal *Slide details* window with the tile map and facts of one plan row (§2.3.7); one instance."""
    instance: Optional["SlideDetailsDialog"] = None

    @classmethod
    def show(cls, master: tk.Misc, ctx: "AppContext", plan: Plan, row_id: str,
             on_fix_channels: Callable[[List[str]], None]) -> "SlideDetailsDialog":
        dlg = cls._alive() or cls(master, ctx)
        cls.instance = dlg
        dlg.plan, dlg.row_id, dlg.on_fix = plan, row_id, on_fix_channels
        dlg.refresh()
        return dlg

    @classmethod
    def refresh_rows(cls, row_ids: Optional[Sequence[str]] = None, *, plan: Any = _UNCHANGED_PLAN) -> None:
        """Refresh from MainWindow's current plan; None closes it. Omitting plan retains the row-only API."""
        dlg = cls._alive()
        if dlg is not None and (plan is not _UNCHANGED_PLAN or row_ids is None or dlg.row_id in row_ids):
            dlg.refresh(plan)

    def __init__(self, master: tk.Misc, ctx: "AppContext") -> None:
        super().__init__(master, ctx, "Slide details", modal=False)
        self.plan: Optional[Plan] = None
        self.row_id, self.on_fix, self._shown = "", None, False
        self.facts = SlideFacts(self.body, ctx, wide=True, surface="Surface", on_open=self._open_folder,
                                on_fix_channels=self._fix_channels)
        self.facts.grid(row=0, column=0, sticky="nw")
        self.extra = self._label(self.body, "", "Surface.Muted.TLabel", 600, row=1, column=0, pady=(self.px(8), 0))
        self.close_button = self._buttons(2, (("Close", self._cancel, "TButton"),))["Close"]

    def refresh(self, plan: Any = _UNCHANGED_PLAN) -> None:
        if self._closed:
            return
        if plan is not _UNCHANGED_PLAN:
            self.plan = plan
        row = self._current_row()
        if row is None:                                  # the plan was cleared or rebuilt without this row
            self._cancel()
            return
        self.title(self.T(f"Slide details · {display_name(row.cand)}"))
        self.facts.show_row(row)
        c = row.cand
        others = [r for r in self.plan.rows() if r is not row and r.section != "note" and r.cand.folder == c.folder
                  and r.cand.prefix == c.prefix and r.cand.kind == "other_suffix" and c.kind != "other_suffix"]
        self.extra.configure(text=self.T("; ".join(f"also {len(r.cand.tile_names):,} {r.cand.suffix} tiles (ignored)"
                                                   for r in others)))
        if not self._shown:
            self._shown = True
            self._show(self.close_button)
        else:
            self.deiconify()
            self.lift()

    def _current_row(self) -> Optional[PlanRow]:
        if self.plan is not None:
            try:
                return self.plan.row(self.row_id)
            except KeyError:
                pass
        return None

    def _open_folder(self, path: str) -> None:
        row = self._current_row()
        if row is not None and path == row.cand.folder:
            osutil.open_path(path)

    def _fix_channels(self, row_ids: List[str]) -> None:
        row = self._current_row()
        if row is not None and row_ids == [row.row_id] and self.on_fix is not None:
            self.on_fix(row_ids)

    def _on_destroy(self, event: Any) -> None:
        if event.widget is self:
            self.plan, self.on_fix = None, None
            self.facts._on_open, self.facts._on_fix = None, None
        super()._on_destroy(event)


class SettingsDialog(_Dialog):
    """Settings (§2.5). Changes apply when the dialog closes; on_done(the dotted keys that changed). During a run
    it is read-only."""

    @classmethod
    def open(cls, master: tk.Misc, ctx: "AppContext", read_only: bool,
             on_done: Callable[[Set[str]], None]) -> "SettingsDialog":
        return cls(master, ctx, read_only, on_done)

    def __init__(self, master: tk.Misc, ctx: "AppContext", read_only: bool,
                 on_done: Callable[[Set[str]], None]) -> None:
        super().__init__(master, ctx, "Settings")
        self._done_cb, px, s = on_done, self.px, ctx.settings
        scale = s.get("ui.scale", "auto")
        self.scale = tk.StringVar(self, next((label for label, v in _SCALES if v == scale), "Automatic"))
        self.naming = tk.StringVar(self, "slide" if s.get("discovery.naming") == "slide" else "folder")
        depth = s.get("discovery.max_depth", 3)
        self.depth = tk.StringVar(self, str(depth if isinstance(depth, int) and 1 <= depth <= 6 else 3))
        temp = s.get("run.temp_dir") or ""
        self.temp_mode, self.temp = tk.StringVar(self, "folder" if temp else "next"), tk.StringVar(self, temp)
        self.awake = tk.BooleanVar(self, bool(s.get("run.keep_awake", True)))
        self.csv = tk.BooleanVar(self, bool(s.get("run.summary_csv", True)))
        b = self.body
        if read_only:
            self._label(b, "You can change settings after this batch.", "Surface.Info.TLabel", row=0, column=0,
                        columnspan=3, pady=(0, px(10)))
        heads = ("Interface size", "Name stitched files after", "Search depth", "Temporary files for very large slides")
        for row, head in enumerate(heads, 1):
            ttk.Label(b, text=head, style="Surface.TLabel", font="YfsBold").grid(
                row=row, column=0, sticky="nw", padx=(0, px(16)), pady=(0, px(12)))
        cells = [ttk.Frame(b, style="Surface.TFrame") for _ in heads]
        for row, cell in enumerate(cells, 1):
            cell.grid(row=row, column=1, sticky="nw", pady=(0, px(12)))
        self.controls: List[Any] = [ttk.Combobox(cells[0], textvariable=self.scale, state="readonly", width=12,
                                                 values=[label for label, _v in _SCALES])]
        self.controls[0].pack(anchor="w")
        ttk.Label(cells[0], text="Takes effect the next time you open the app.", style="Surface.Small.TLabel").pack(
            anchor="w", pady=(px(2), 0))
        for text, value in (("the folder (like before)", "folder"), ("the slide name inside the tile files", "slide")):
            self.controls.append(ttk.Radiobutton(cells[1], text=text, variable=self.naming, value=value,
                                                 style="Surface.TRadiobutton"))
            self.controls[-1].pack(anchor="w")
        line = ttk.Frame(cells[2], style="Surface.TFrame")
        line.pack(anchor="w")
        self.controls.append(ttk.Spinbox(line, from_=1, to=6, width=4, textvariable=self.depth, state="readonly"))
        self.controls[-1].pack(side="left")
        ttk.Label(line, text="folder levels below the folder you choose", style="Surface.Muted.TLabel").pack(
            side="left", padx=(px(8), 0))
        self.controls.append(ttk.Radiobutton(cells[3], text="next to the stitched files", variable=self.temp_mode,
                                             value="next", style="Surface.TRadiobutton"))
        self.controls[-1].pack(anchor="w")
        line = ttk.Frame(cells[3], style="Surface.TFrame")
        line.pack(anchor="w", fill="x")
        self.controls.append(ttk.Radiobutton(line, text="this folder:", variable=self.temp_mode, value="folder",
                                             style="Surface.TRadiobutton"))
        self.controls[-1].pack(side="left")
        self.controls.append(ttk.Entry(line, textvariable=self.temp, width=34))
        self.controls[-1].pack(side="left", padx=(px(6), px(6)))
        self.controls[-1].bind("<KeyRelease>", lambda _e: self.temp_mode.set("folder"))
        self.controls.append(ttk.Button(line, text=self.T(f"Browse{self.G['ell']}"), command=self._browse_temp))
        self.controls[-1].pack(side="left")
        ttk.Label(cells[3], text="Used only when a slide doesn't fit in memory. A local disk is fastest.",
                  style="Surface.Small.TLabel").pack(anchor="w", pady=(px(2), 0))
        awake = "Keep the computer awake while stitching" + ("" if osutil.WINDOWS or osutil.MACOS
                                                            else " (Windows and macOS)")
        for row, (text, var) in enumerate(((awake, self.awake),
                                           ("Save a summary table (CSV) next to the stitched files", self.csv)), 5):
            self.controls.append(ttk.Checkbutton(b, text=text, variable=var, style="Surface.TCheckbutton"))
            self.controls[-1].grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, px(6)))
        if not (osutil.WINDOWS or osutil.MACOS):
            self.controls[-2].state(["disabled"])
        self.error = self._label(b, "", "Surface.Error.TLabel", row=7, column=0, columnspan=3)
        if read_only:
            for w in self.controls:
                w.state(["disabled"])
            buttons = self._buttons(8, (("Close", self._cancel, "Primary.TButton"),))
        else:
            buttons = self._buttons(8, (("Cancel", self._cancel, "TButton"), ("Save", self._save, "Primary.TButton")))
        self._show(buttons.get("Save") or buttons["Close"])

    def _browse_temp(self) -> None:
        from yfs.ui.browser import FolderBrowser, browser_initial   # static import; browser.py imports this module
        initial = browser_initial("temp", self.temp.get(), None, self.ctx.settings)
        FolderBrowser.open(self, self.ctx, mode="temp", initial=initial, on_done=self._temp_chosen)

    def _temp_chosen(self, path: Optional[str]) -> None:
        safe_grab(self)
        if path:
            self.temp.set(path)
            self.temp_mode.set("folder")

    def _save(self) -> None:
        temp = os.path.expanduser(self.temp.get().strip()) if self.temp_mode.get() == "folder" else ""
        if self.temp_mode.get() == "folder" and not temp:
            self.error.configure(text="Choose a folder for temporary files, or keep them next to the stitched files.")
            return
        if temp and not (os.path.isabs(temp) and (ntpath.splitdrive(temp)[0] or not osutil.WINDOWS)):
            self.error.configure(text=self.T(f"Type the full path of the folder for temporary files, or use "
                                             f"Browse{self.G['ell']}"))          # string test only (UD6)
            return
        new = {"ui.scale": dict(_SCALES).get(self.scale.get(), "auto"), "discovery.naming": self.naming.get(),
               "discovery.max_depth": min(6, max(1, int(self.depth.get() or 3))), "run.temp_dir": temp or None,
               "run.keep_awake": bool(self.awake.get()), "run.summary_csv": bool(self.csv.get())}
        s = self.ctx.settings
        changed = {k for k, v in new.items() if s.get(k) != v}
        for key in changed:
            s.set(key, new[key])
        if changed:
            s.save_async()
        self._close(changed)

    def _cancel(self) -> None:
        self._close(set())


def diagnostics(ctx: "AppContext") -> List[Tuple[str, str]]:
    """(label, value) lines for *About* and *Copy diagnostics*: versions, font system, scale, paths, last run."""
    import imagecodecs                                   # static imports, loaded only when asked for
    import numpy
    import psutil
    import tifffile
    th, s = ctx.theme, ctx.settings
    last = s.get("last_run") or {}
    fonts = "Xft" if th.xft and th.ws == "x11" else "core X fonts" if th.ws == "x11" else "native"
    return [("Version", __version__), ("Python", platform.python_version()),
            ("Tk", f"{ctx.root.tk.call('info', 'patchlevel')} ({fonts})"), ("numpy", numpy.__version__),
            ("tifffile", tifffile.__version__), ("imagecodecs", imagecodecs.__version__), ("psutil", psutil.__version__),
            ("OS", platform.platform()), ("Frozen", "yes" if getattr(sys, "frozen", False) else "no"),
            ("Interface size", f"{round(th.scale * 100)}%"), ("Settings", s.path), ("Logs", log_dir()),
            ("Last run", f"{last.get('state', 'none')}: {last.get('done', 0)} of {last.get('planned', 0)} stitched, "
                         f"{last.get('failed', 0)} failed {last.get('started', '')}".strip())]


class InfoDialog(_Dialog):
    """*How it works* (v1's welcome text word for word, what Stitch does, the keyboard shortcuts) and *About*;
    non-modal, one instance (§2.5)."""
    instance: Optional["InfoDialog"] = None

    @classmethod
    def show(cls, master: tk.Misc, ctx: "AppContext", page: str) -> "InfoDialog":
        dlg = cls._alive() or cls(master, ctx)
        cls.instance = dlg
        dlg.page(page)
        return dlg

    def __init__(self, master: tk.Misc, ctx: "AppContext") -> None:
        super().__init__(master, ctx, "How it works", modal=False)
        px, self._shown = self.px, False
        self.body.rowconfigure(1, weight=1)
        tabs = ttk.Frame(self.body, style="Surface.TFrame")
        tabs.grid(row=0, column=0, sticky="w", pady=(0, px(10)))
        self.tabs = {}
        for key, text in (("how", "How it works"), ("about", "About")):
            self.tabs[key] = ttk.Button(tabs, text=text, style="Flat.TButton", command=lambda k=key: self.page(k))
            self.tabs[key].pack(side="left", padx=(0, px(4)))
        self.frame: Optional[ttk.Frame] = None
        self.text: Optional[tk.Text] = None
        self.close_button = self._buttons(2, (("Close", self._cancel, "TButton"),))["Close"]

    def page(self, which: str) -> None:
        if self.frame is not None:
            self.frame.destroy()
        self.frame = ttk.Frame(self.body, style="Surface.TFrame")
        self.frame.grid(row=1, column=0, sticky="nsew")
        for key, button in self.tabs.items():
            button.configure(style="CrumbCurrent.TButton" if key == which else "Flat.TButton")
        self.title(self.T("About" if which == "about" else "How it works"))
        (self._about if which == "about" else self._how)(self.frame)
        self.update_idletasks()
        width = min(self.winfo_reqwidth(), self.winfo_screenwidth() - self.px(32))
        height = min(self.winfo_reqheight(), self.winfo_screenheight() - self.px(48))
        self.geometry(f"{width}x{height}")                 # the text scrolls; tabs and Close stay on screen
        if not self._shown:
            self._shown = True
            self._show(self.close_button)
        else:
            center_over(self, self.master.winfo_toplevel())
            self.deiconify()
            self.lift()

    def _how(self, frame: ttk.Frame) -> None:
        px = self.px
        key_width = max(self.ctx.theme.fonts["YfsBold"].measure(self.T(keys))
                        for keys, _ in _KEYS_MAIN + _KEYS_BROWSER) + px(24)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = self.text = tk.Text(frame, wrap="word", width=86, height=30, padx=px(8), pady=px(6),
                                   font="YfsBody", highlightthickness=0, cursor="arrow", tabs=(key_width,))
        bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=bar.set)
        text.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        text.tag_configure("h", font="YfsH2", spacing1=px(10), spacing3=px(4))
        text.tag_configure("key", font="YfsBold")
        text.tag_configure("action", lmargin2=key_width)
        text.tag_configure("muted", foreground=T["text_muted"])
        text.tag_configure("bullet", lmargin1=px(8), lmargin2=px(22), spacing3=px(3))
        put = lambda s, *tags: text.insert("end", self.T(s), tags)                   # noqa: E731
        put("From Jacky\n", "h")
        put(WELCOME_V1 + "\n")
        put("What happens when you press Stitch\n", "h")
        for line in _HOW_BULLETS:
            put("·  " + line + "\n", "bullet")
        put("If a network drive stops answering\n", "h")
        put("Only that drive is affected: the rest of the app keeps working, and closing the app is immediate even "
            "while the drive hangs.\n")
        for head, table in (("Keyboard shortcuts: main window", _KEYS_MAIN),
                            ("Keyboard shortcuts: choosing a folder", _KEYS_BROWSER)):
            put(head + "\n", "h")
            for keys, action in table:
                put(keys, "key")
                put("\t" + action + "\n", "action")
        if self.ctx.theme.ws == "aqua":
            put("On macOS, Command replaces Control and Option replaces Alt.\n", "muted")
        text.configure(state="disabled")

    def _about(self, frame: ttk.Frame) -> None:
        px = self.px
        self.text = None
        ttk.Label(frame, image=self.ctx.icons.logo(64), style="Surface.TLabel").grid(row=0, column=0, rowspan=3,
                                                                                    sticky="nw", padx=(0, px(16)))
        ttk.Label(frame, text=APP_NAME, style="Title.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(frame, text="It's your Favourite Stitcher", style="Tagline.TLabel").grid(row=1, column=1, sticky="w")
        ttk.Label(frame, text=f"Version {__version__}", style="Surface.Muted.TLabel").grid(row=2, column=1, sticky="nw")
        self._label(frame, "Made by Jacky @ Ramachandran Lab. Logic adapted from Pete Bankhead (QuPath founder).",
                    row=3, column=0, columnspan=2, pady=(px(14), px(4)))
        links = ttk.Frame(frame, style="Surface.TFrame")
        links.grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, px(10)))
        ttk.Button(links, text="GitHub page", style="Link.TButton",
                   command=lambda: osutil.open_url(GITHUB_URL)).pack(side="left")
        ttk.Button(links, text="Copy diagnostics", style="Link.TButton",
                   command=self._copy_diagnostics).pack(side="left", padx=(px(12), 0))
        table = ttk.Frame(frame, style="Surface.TFrame")
        table.grid(row=5, column=0, columnspan=2, sticky="w")
        for row, (label, value) in enumerate(diagnostics(self.ctx)):
            ttk.Label(table, text=label, style="Surface.Muted.TLabel").grid(row=row, column=0, sticky="nw",
                                                                            padx=(0, px(14)))
            ttk.Label(table, text=self.T(str(value)), style="Surface.TLabel", wraplength=px(460),
                      justify="left").grid(row=row, column=1, sticky="w")

    def _copy_diagnostics(self) -> None:
        self.clipboard_clear()
        self.clipboard_append("\n".join(f"{label}: {value}" for label, value in diagnostics(self.ctx)))

    def _close(self, result: Any) -> None:
        InfoDialog.instance = None
        super()._close(result)
