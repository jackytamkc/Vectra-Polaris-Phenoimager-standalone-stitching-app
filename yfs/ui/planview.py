"""The slide-review table (the plan): sections, tick+status images, pointer and keyboard map, inline rename,
the Select and context menus, the details strip and the welcome state. FINAL_SPEC §2.3, §2.5, §5.5.15.

PlanTable is the whole content of the step-3 card (it is the Card.TFrame itself): toolbar, table, details strip.
It changes the Plan only through its methods and leaves filesystem work to callbacks (or osutil workers)."""
from __future__ import annotations

import logging
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from yfs.core import osutil
from yfs.core.names import OME_SUFFIX, human_bytes, norm_key, sanitize_stem
from yfs.core.plan import (RUN_STATUSES, STITCHED, Plan, PlanRow, display_name, parent_rel, row_text,
                           top_warning)
from yfs.ui.runpanel import fit_label, fit_text
from yfs.ui.theme import T, tag_tree, ui_thread
from yfs.ui.widgets import Tooltip

if TYPE_CHECKING:
    from yfs.core.model import Note
    from yfs.ui.app import AppContext

log = logging.getLogger(__name__)
SECTIONS = ("new", "existing", "excluded")
COLUMNS = ("tiles", "size", "gap", "folder", "output", "what")   # gap: right-aligned numbers never touch text (MW-17)
_HEADINGS = (("#0", "Slide", "w"), ("tiles", "Tiles", "e"), ("size", "Size", "e"), ("gap", "", "w"),
             ("folder", "Folder", "w"), ("output", "Output file", "w"), ("what", "What will happen", "w"))
STATUS_ICON = {"new": "ready", "unfinished": "warn", "incomplete": "warn", "few_tiles": "skip", "sparse": "skip",
               "header_error": "error", "unsupported": "skip", "existing": "ok", "existing_checking": "wait",
               "replace": "warn", "unchecked": "warn", "ready": "ready", "busy": "wait", "other_suffix": "info",
               "unsuffixed": "info", "legacy_all": "info", "waiting": "wait", "running": "run", "done": "ok",
               "done_gaps": "warn", "done_generic": "warn", "failed": "error", "skipped": "skip", "cancelled": "skip",
               "finalize": "warn"}
_NOTE_ICON = {"outputs": "info", "unmatched": "info", "wsi": "info", "raw_scan": "info", "wide_pruned": "info",
              "not_searched_depth": "info", "large_dir": "skip", "system_skipped": "skip", "symlink_skipped": "skip",
              "cloud_skipped": "info", "not_mounted": "info", "no_images": "skip", "empty": "skip", "lnk": "skip",
              "no_access": "error", "unresponsive": "error", "error": "error"}
_NOTE_LABEL = {"raw_scan": "Raw scans", "not_searched_depth": "Deeper folders", "system_skipped": "System folders",
               "symlink_skipped": "Links", "no_images": "Folders without images", "lnk": "Windows shortcuts",
               "not_mounted": "Not connected"}
_NOTE_ACTION = {"not_searched_depth": ("Search deeper", "search_deeper"),
                "wide_pruned": ("Search these folders too", "search_wide"),
                "not_mounted": ("Rescan", "search_again"),
                "unresponsive": ("Search this folder again", "search_again"),
                "error": ("Search this folder again", "search_again")}
_SELECT = (("Tick all new slides", "all_new"), ("Untick all", "none"), ("Only slides not stitched yet", "not_stitched"),
           ("Invert ticks", "invert"), ("Untick smaller copies", "untick_smaller_copies"),
           ("Tick failed slides", "failed"))
_FAILED = ("failed", "skipped", "cancelled")
_FIELD = 4                                  # clam's Treeview field border, both sides (measured on both Tk builds)
_INTRO = ("Stitches AKOYA Vectra Polaris / PhenoImager unmixed tiles (…_component_data.tif) into pyramidal "
          "OME-TIFFs for QuPath and Visiopharm. It runs in the background, does many slides in a row, and you "
          "don't have to pay £5/hr.")


def status_icon(r: PlanRow) -> str:
    """The right-hand image of a row (§2.3.3): a ticked new row shows its top warning instead of 'ready'; a row
    that won't be stitched because it is unticked shows 'skip'."""
    if r.section == "note":
        return _NOTE_ICON.get(_kind(r.note), "info") if r.note else "info"
    if r.status in ("new", "few_tiles", "sparse"):
        return ("warn" if top_warning(r) else "ready") if r.checked else "skip"
    if r.status in ("waiting", "unfinished", "incomplete", "ready") and not r.checked:
        return "skip"                                       # "Won't be stitched"
    return STATUS_ICON.get(r.status, "info")


def _kind(n: "Note") -> str:
    """A note's kind for its icon, label and action: pseudo filesystems skipped as not mounted are system folders."""
    return "system_skipped" if n.kind == "not_mounted" and n.detail == "pseudo" else n.kind


def row_tags(r: PlanRow, icon: str) -> Tuple[str, ...]:
    """§4.6 Plan tags: running rows bold, errors red, ticked warnings amber, everything not ticked muted."""
    if r.section == "note":
        return ("note",)
    if r.status == "running":
        return ("running",)
    return ("error",) if icon == "error" else ("dim",) if not r.checked else ("warn",) if icon == "warn" else ()


def started(r: PlanRow) -> bool:
    return r.status in RUN_STATUSES and r.status != "waiting"


def stitched(r: PlanRow) -> bool:
    """Stitched in this plan: it isn't stitched again before a rescan (Plan.pending)."""
    return r.outcome is not None and r.outcome.status in STITCHED


def _channels(names: Sequence[str], n: int) -> str:
    if len(names) > 1 and len(set(names)) == 1:
        return f"{names[0]} ×{len(names)}"
    return ", ".join(names[:3]) + (", …" if len(names) > 3 else "") if names else "no names"


def details_text(r: PlanRow, what: Optional[str] = None) -> str:
    """The one-line details strip (§2.3.7): the name and what will happen (`what`: the text the table shows, as the
    cell may be shortened, MW-10), then each part once it is known; flags follow the tile count. Nothing is said
    twice: a ticked slide's top warning is its *what*, and a channel flag already names the channels (MW-25)."""
    c, h, e = r.cand, r.header, r.est
    what = row_text(r) if what is None else what
    if r.section == "note":
        return what
    parts = [display_name(c), what, f"{len(c.tile_names):,} tiles" if c.tile_names else "every TIFF in the folder"]
    parts += [f.text for f in r.flags if f.text != what]
    if h is not None:
        named = any(f.code == "channels" and f.severity == "warn" for f in r.flags)
        names = list(r.channel_override or h.channels_extracted)
        parts += [f"{h.n_channels} channels" + ("" if named else f" ({_channels(names, h.n_channels)})"),
                  f"{h.um_per_px_x:.3f} µm/px"]
    elif r.header_error and r.status != "header_error":
        parts.append(f"can't read the first tile ({r.header_error.rstrip('.')})")
    if e is not None:
        parts += [f"{e.canvas_w:,} × {e.canvas_h:,} px"] + (["fits in memory"] if e.memory_mode == "ram" else [])
    if c.prefix and not c.multi_slide_folder and norm_key(c.prefix) != norm_key(c.folder_name):
        parts.append(f"tiles named {c.prefix}")
    rel = parent_rel(c)
    return " · ".join(p for p in parts + ([f"in {rel}"] if rel else []) if p)


def fit_middle(font: tkfont.Font, text: str, width: int, ell: str) -> str:
    """`text` shortened in the middle with `ell` to at most `width` px, as much from each side; unchanged while the
    width is unknown."""
    if width <= 1 or font.measure(text) <= width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:                                         # the most characters that fit, half from each end
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if font.measure(_cut(text, mid, ell)) <= width else (lo, mid - 1)
    return _cut(text, lo, ell)


def _cut(text: str, keep: int, ell: str) -> str:
    head = keep // 2
    return text[:head] + ell + text[len(text) - (keep - head):]


def _cut_before(font: tkfont.Font, text: str, width: int, ell: str, end: str, head: int = 3) -> Optional[str]:
    """`text` cut short just before `end`, the part that tells it from its siblings, which stays whole; None when
    not even `head` characters of its start would fit before it."""
    if not end or font.measure(text[:head] + ell + end) > width:
        return None
    return fit_text(font, text[:-len(end)], width - font.measure(end), ell) + end


def _fit_folder(font: tkfont.Font, text: str, width: int, ell: str) -> str:
    """A folder keeps its last part whole ('du…/batchA', or '…/batchA'), else it is shortened in the middle."""
    if width <= 1 or font.measure(text) <= width:
        return text
    return _cut_before(font, text, width, ell, text[text.rfind("/"):] if "/" in text else "", 0) or \
        fit_middle(font, text, width, ell)


def _fit_output(font: tkfont.Font, text: str, width: int, ell: str) -> str:
    """An output name keeps what tells it from its copies ('Slide9_Sc…__batchA.ome.tif', D1), else it is cut at the
    end like the slide's name: a cut in the middle would drop the slide ID of 'SlideA1_Scan1'."""
    if width <= 1 or font.measure(text) <= width:
        return text
    stem = text[:-len(OME_SUFFIX)] if text.endswith(OME_SUFFIX) else text
    return _cut_before(font, text, width, ell, text[stem.rfind("__"):] if stem.rfind("__") > 0 else "") or \
        fit_text(font, text, width, ell)


_FITS = {"folder": _fit_folder, "output": _fit_output}   # their ends tell copies apart; other cells are cut at the end


def _new_label(ticked: int, rest: int, done: int) -> str:
    """'New slides  (5 of 6 ticked)', counting the slides Start would run: slides stitched in this plan are left out
    (MW-7) and counted in the section's What cell, or here once every one is: '(6 done)'."""
    return f"New slides  ({ticked} of {rest} ticked)" if rest or not done else f"New slides  ({done} done)"


def _entries(menu: tk.Menu) -> int:
    end = menu.index("end")
    return 0 if end is None else int(end) + 1


def _note_label(r: PlanRow) -> str:
    """Slide-column text of a note row: where it is, or what an aggregated row collects."""
    if r.row_id.endswith("|*"):
        return "More folders"
    if r.row_id.count("|") == 1:                            # aggregated: "note|<group>"
        return _NOTE_LABEL.get(_kind(r.note), r.note.kind) if r.note else ""
    return r.cand.folder_name if r.cand.rel_folder == "." else r.cand.rel_folder


def _many(label: str, n: int) -> str:
    return label if n < 2 else f"{label} ({n} slides)"


class _Ruler:
    """font.measure with a memory: after a resize every cell is shortened again (MW-10)."""

    def __init__(self, font: tkfont.Font) -> None:
        self.font, self.known = font, {}                   # type: tkfont.Font, Dict[str, int]

    def measure(self, text: str) -> int:
        width = self.known.get(text)
        if width is None:
            if len(self.known) > 50000:
                self.known.clear()
            width = self.known[text] = self.font.measure(text)
        return width


class _CellTip(Tooltip):
    """The full text of a shortened cell, just under the cell rather than under the whole table (MW-10)."""

    def __init__(self, tree: ttk.Treeview) -> None:
        super().__init__(tree, lambda: self.full)
        self.full, self.at = "", (0, 0)

    def _show(self) -> None:
        super()._show()
        tip = self._tip
        if tip is not None:
            x = min(self.at[0], tip.winfo_screenwidth() - tip.winfo_reqwidth() - 2)
            tip.wm_geometry(f"+{max(0, x)}+{self.at[1]}")


class PlanTable(ttk.Frame):
    """The step-3 card. Every tick change goes through plan.toggle* / plan.select; after a user change the
    table refreshes itself and calls on_changed(). Context-menu work that needs the filesystem or discovery is
    handed to on_note_action(action, row_id) with action in retry_header | search_deeper | search_wide |
    search_again | finalize. Extras: set_compact(), rename(row_id), min_height(), and the tree/select_menu."""

    def __init__(self, master: tk.Misc, ctx: "AppContext", *, on_focus: Callable[[Optional[str]], None],
                 on_details: Callable[[str], None], on_fix_channels: Callable[[List[str]], None],
                 on_note_action: Callable[[str, str], None], on_changed: Callable[[], None]) -> None:
        px = ctx.theme.px
        super().__init__(master, style="Card.TFrame", padding=(px(12), px(8)))
        self.ctx, self.plan = ctx, None                     # type: AppContext, Optional[Plan]
        self._on_focus, self._on_details, self._on_fix = on_focus, on_details, on_fix_channels
        self._on_note_action, self._on_changed = on_note_action, on_changed
        self._iid: Dict[str, str] = {}                      # row_id -> item id ("r<n>")
        self._rid: Dict[str, str] = {}                      # item id -> row_id
        self._full: Dict[str, tuple] = {}                   # item id -> what it shows, full texts
        self._shown: Dict[str, tuple] = {}                  # item id -> what it shows, shortened (skips unchanged)
        self._dimmed: Dict[tuple, tk.PhotoImage] = {}       # composite images with a dimmed tick box
        self._kids: Dict[str, List[str]] = {s: [] for s in SECTIONS}
        self._open: Dict[str, bool] = {}                    # sections the user opened or closed
        self._count = 0
        self._cols: Tuple[str, ...] = ()
        self._width: Dict[str, int] = {}                    # column -> the width the layout gave it
        self._need = dict.fromkeys(("#0", "folder", "output", "what"), 0)   # widest text seen, plus its room
        self._label_w = 0                                   # Slide column room for the section labels (MW-18)
        self._state, self._help_text, self._wide = "EMPTY", "", True
        self._running = self._compact = self._narrow = self._multi = self._slides = self._waiting = False
        self._run_row: Optional[str] = None
        self._edit: Optional[Dict[str, object]] = None
        self._welcome: Optional[ttk.Frame] = None
        self._tree_w, self._relayout_id = 0, None           # type: int, Optional[str]
        self._tip_cell: Optional[Tuple[str, str]] = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_toolbar()
        self._build_tree()
        self.details = ttk.Label(self, style="Surface.Small.TLabel", width=1)
        self.details.grid(row=2, column=0, sticky="ew", pady=(px(6), 0))
        self.menu = tk.Menu(self, tearoff=0)
        self.bind("<Configure>", self._on_configure, add="+")
        self._toolbar()

    # ---- construction
    def _build_toolbar(self) -> None:
        px = self.ctx.theme.px
        bar = self._bar = ttk.Frame(self, style="Surface.TFrame")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, px(6)))
        bar.columnconfigure(3, weight=1)
        ttk.Label(bar, text="3", style="StepNum.TLabel", width=2).grid(row=0, column=0, padx=(0, px(8)))
        ttk.Label(bar, text="Check the list, then stitch", style="Step.TLabel").grid(row=0, column=1)
        self.help = ttk.Label(bar, style="Surface.Small.TLabel")
        self.help.grid(row=0, column=2, padx=(px(10), 0))
        self.select_button = ttk.Menubutton(bar, text="Select")
        self.select_menu = tk.Menu(self.select_button, tearoff=0, postcommand=self._fill_select)
        self.select_button.configure(menu=self.select_menu)
        self.select_button.grid(row=0, column=4)

    def _build_tree(self) -> None:
        th, icons = self.ctx.theme, self.ctx.icons
        px, body = th.px, ttk.Frame(self, style="Surface.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self._body = body
        t = self.tree = ttk.Treeview(body, style="Plan.Treeview", columns=COLUMNS, show="tree headings",
                                     selectmode="extended", height=4)
        self._vsb = ttk.Scrollbar(body, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=self._yscroll)
        t.grid(row=0, column=0, sticky="nsew")
        self._vsb.grid(row=0, column=1, sticky="ns")
        tag_tree(t, "Plan")
        for col, title, anchor in _HEADINGS:
            t.heading(col, text=title, anchor=anchor)
        self._rulers = {name: _Ruler(th.fonts[name]) for name in ("YfsBody", "YfsBold")}
        m = lambda s: th.fonts["YfsBody"].measure(th.text(s)) + px(16)            # noqa: E731
        self._pair_w = 2 * icons.size + 2 * icons.gap + 1
        self._indent = int(ttk.Style(t).lookup("Plan.Treeview", "indent") or 0)   # a slide row's offset in #0
        self._pad = px(8)                                   # a cell's text inset, both sides together
        self._fixed_w = {"tiles": m("Tiles 0000"), "size": m("~99.9 GB"), "gap": px(10)}
        self._slide_cap = icons.size + icons.gap + self._pair_w + m("QA731104-05_Scan1 (tiles: …)")
        self._out_base, self._what_base = m("Output file"), m("Check channel names (DAPI ×5)")
        for col in ("#0",) + COLUMNS:                       # _layout() sets the widths; What takes what is left
            t.column(col, width=self._fixed_w.get(col, self._out_base), minwidth=px(20), stretch=col == "what",
                     anchor="e" if col in ("tiles", "size") else "w")
        for s in SECTIONS:
            t.insert("", "end", iid=f"sec:{s}", tags=("section",))
        t.detach(*[f"sec:{s}" for s in SECTIONS])
        self._tip = _CellTip(t)
        for seq, fn in (("<Button-1>", self._click), ("<Double-Button-1>", self._double), ("<Button-3>", self._context),
                        ("<space>", self._key_space), ("<Delete>", self._key_delete), ("<Return>", self._key_enter),
                        ("<KP_Enter>", self._key_enter), ("<F2>", lambda e: self.rename(self.focused_row()) or "break"),
                        ("<Control-a>", self._select_all), ("<<TreeviewSelect>>", self._selection_changed),
                        ("<<TreeviewOpen>>", lambda e: self._remember_open(True)),
                        ("<<TreeviewClose>>", lambda e: self._remember_open(False)),
                        ("<Configure>", self._tree_configured), ("<Motion>", self._hover),
                        ("<MouseWheel>", lambda e: self._tip.hide()), ("<FocusIn>", self._focus_in)):
            t.bind(seq, fn)
        t.bind("<Leave>", self._left, add="+")
        for seq in ("<Shift-F10>", "<Menu>", "<App>"):      # the context menu from the keyboard (App: Windows)
            try:
                t.bind(seq, self._key_menu)
            except tk.TclError:                             # a key this windowing system doesn't have
                pass
        if self.ctx.theme.ws == "aqua":                     # the Mac delete key sends BackSpace (PL-14)
            for seq, fn in (("<Button-2>", self._context), ("<Control-Button-1>", self._context),
                            ("<Command-a>", self._select_all), ("<BackSpace>", self._key_delete),
                            ("<Command-BackSpace>", self._key_delete)):
                t.bind(seq, fn)

    # ---- public API (§5.5.15)
    @ui_thread
    def bind_plan(self, plan: Optional[Plan]) -> None:
        """Show `plan` (None: empty). Rows that keep their row_id keep their selection, focus and position."""
        self._close_editor()
        self.plan, self._run_row = plan, None
        self._need = dict.fromkeys(self._need, 0)
        for ruler in self._rulers.values():
            ruler.known.clear()
        if plan is None:
            self.tree.delete(*self._rid)
            self._iid.clear()
            self._rid.clear()
            self._full.clear()
            self._shown.clear()
            self._kids = {s: [] for s in SECTIONS}
            self.tree.detach(*[f"sec:{s}" for s in SECTIONS])
            self._toolbar()
            self._show_details()
            return
        if self._welcome is not None:
            self._welcome.destroy()
            self._welcome = None
            self._body.grid()
            self._show_details()
        self.refresh(None)

    @ui_thread
    def refresh(self, row_ids: Optional[Iterable[str]] = None) -> None:
        """None = full rebuild (keeps open/selection). Otherwise only the given rows are redrawn, plus any row that
        appeared, moved or went away, and the section rows."""
        plan, t = self.plan, self.tree
        if plan is None:
            self._toolbar()
            return
        want = None if row_ids is None else set(row_ids)
        with plan.lock:                                     # one consistent snapshot; Tk calls happen after it
            rows, pending, self._multi = plan.rows(), {r.row_id for r in plan.pending()}, plan.multi_parent()
            show = {r.row_id: self._display(r) for r in rows
                    if want is None or r.row_id in want or r.row_id not in self._iid}
            groups: Dict[str, List[PlanRow]] = {s: [] for s in SECTIONS}
            for r in rows:
                groups["excluded" if r.section == "note" else r.section].append(r)
            heads = self._heads(groups, pending)
            running = next((r.row_id for r in rows if r.status == "running"), None)
            run = {r.row_id for r in rows if r.status in RUN_STATUSES}
            live = {r.row_id for r in rows}
            focused = self._rid.get(t.focus())
            strip = details_text(plan.row(focused), self._what(plan.row(focused))) if focused in show else \
                "" if focused not in live else None
            self._slides = any(r.section != "note" for r in rows)
            self._waiting = any(r.status == "waiting" and r.checked for r in rows)
            self._help_text = self._help(rows)
        gone = [rid for rid in self._iid if rid not in live]
        if gone:
            t.delete(*[self._iid[rid] for rid in gone])
            for rid in gone:
                iid = self._iid.pop(rid)
                del self._rid[iid]
                self._full.pop(iid, None)
                self._shown.pop(iid, None)
        for sec in SECTIONS:
            for r in groups[sec]:
                if r.row_id not in self._iid:
                    self._count += 1
                    iid = self._iid[r.row_id] = f"r{self._count}"
                    self._rid[iid] = r.row_id
                    t.insert(f"sec:{sec}", "end", iid=iid)
            kids = [self._iid[r.row_id] for r in groups[sec]]
            if kids != self._kids[sec]:
                t.set_children(f"sec:{sec}", *kids)
                self._kids[sec] = kids
        for rid, d in show.items():
            self._full[self._iid[rid]] = d
            self._grow(d, rid in run)
        for sec, (visible, is_open, d) in heads.items():
            t.item(f"sec:{sec}", open=is_open)
            self._full[f"sec:{sec}"] = d
        visible = [f"sec:{s}" for s in SECTIONS if heads[s][0]]
        if list(t.get_children("")) != visible:
            t.set_children("", *visible)
        self._write(list(self._full) if self._layout() else [self._iid[rid] for rid in show] +
                    [f"sec:{s}" for s in SECTIONS])
        if running != self._run_row:
            self._run_row = running
            if running is not None and t.item(t.parent(self._iid[running]), "open"):
                t.see(self._iid[running])                   # (see() would also open a closed section)
        if strip is not None:
            fit_label(self.details, self.ctx.theme.text(strip), self.ctx.theme.G["ell"])
        self._toolbar()
        self._place_editor()

    @ui_thread
    def set_state(self, state: str) -> None:
        """The main window's state: 'EMPTY', 'CHECKING', 'SCANNING', 'PAUSED', 'READY', 'RUNNING', 'STOPPING' or
        'FINISHED'. The help text, the Select menu, the tick boxes and the note actions follow it (MW-16, MW-21);
        once a stop is asked for, the waiting slides read "Won't be stitched (stopping)" (MW-19)."""
        self._state = state
        self.set_running(state in ("RUNNING", "STOPPING"))

    @ui_thread
    def set_running(self, on: bool) -> None:
        """During a run only waiting rows can be unticked, so every other tick box is dimmed; the Select menu
        offers only *Untick all waiting*."""
        self._running = bool(on)
        if on and self._edit is not None and started(self.plan.row(str(self._edit["row"]))):
            self._close_editor()
        self.refresh(None)

    @ui_thread
    def show_welcome(self, first_run: bool, on_choose: Callable[[], None], on_dismiss: Callable[[], None]) -> None:
        """§2.5: the inline welcome (first_run) or the one-line empty state, in place of the table."""
        px, t = self.ctx.theme.px, self.ctx.theme.text
        self._close_editor()
        if self._welcome is not None:
            self._welcome.destroy()
        w = self._welcome = ttk.Frame(self, style="Surface.TFrame")
        self._body.grid_remove()
        self.details.grid_remove()
        w.grid(row=1, column=0, sticky="nsew")
        actions = ttk.Frame(w, style="Surface.TFrame")
        actions.pack(side="bottom", fill="x", pady=(px(8), 0))
        choose = ttk.Button(actions, text=t("Choose tile folder…"), style="Primary.TButton", command=on_choose)
        choose.pack(side="left", expand=True)
        if not first_run:
            line = ttk.Label(w, text="Choose the folder with your tiles to see the slides it contains.",
                             style="Surface.Muted.TLabel", justify="center")
            line.pack(expand=True)
            w.bind("<Configure>", lambda e: line.configure(wraplength=max(px(100), e.width - px(32))))
            return

        def dismiss() -> None:
            on_dismiss()
            self.show_welcome(False, on_choose, on_dismiss)
        ttk.Button(actions, text=t("Don’t show this again"), style="Link.TButton", command=dismiss).pack(side="right")
        view = ttk.Frame(w, style="Surface.TFrame")
        view.pack(fill="both", expand=True)
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)
        canvas = tk.Canvas(view, background=T["surface"], highlightthickness=1, highlightbackground=T["surface"],
                           highlightcolor=T["focus"], yscrollincrement=px(20), width=1, height=1)
        canvas.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(view, orient="vertical", command=canvas.yview)
        bar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=bar.set)
        inner = ttk.Frame(canvas, style="Surface.TFrame")
        item = canvas.create_window(0, 0, window=inner, anchor="n")
        head = ttk.Frame(inner, style="Surface.TFrame")
        head.pack(pady=(0, px(10)))
        ttk.Label(head, image=self.ctx.icons.logo(px(48)), style="Surface.TLabel").pack(side="left", padx=(0, px(12)))
        title = ttk.Label(head, text="Welcome to your FAVOURITE Stitcher!", style="Welcome.TLabel", justify="center")
        title.pack(side="left")
        intro = ttk.Label(inner, text=t(_INTRO), style="Surface.TLabel", wraplength=px(600), justify="center")
        intro.pack()
        wrapped = [intro]
        steps = ttk.Frame(inner, style="Surface.TFrame")
        steps.pack(pady=px(14))
        for i, text in enumerate(("Choose the folder with your tiles", "Check the list", "Stitch"), 1):
            ttk.Label(steps, text=str(i), style="StepNum.TLabel", width=2).pack(side="left", padx=(px(18) * (i > 1), px(6)))
            ttk.Label(steps, text=text, style="Surface.TLabel").pack(side="left")
        for text, style in (("Tip: network drives stitch at network speed; big slides use RAM or temporary disk "
                             "space.", "Surface.Muted.TLabel"),
                            ("Made by Jacky @ Ramachandran Lab · logic adapted from Pete Bankhead (QuPath founder)",
                             "Surface.Small.TLabel")):
            label = ttk.Label(inner, text=t(text), style=style, wraplength=px(600), justify="center")
            label.pack(pady=(0, px(2)))
            wrapped.append(label)

        def fit(_e=None) -> None:
            width, height = canvas.winfo_width(), canvas.winfo_height()
            content = max(px(100), width - px(32))
            canvas.itemconfigure(item, width=content)
            for label in wrapped:
                label.configure(wraplength=min(px(600), content))
            title.configure(wraplength=max(px(100), content - px(60)))
            needed = inner.winfo_reqheight() + px(8)
            canvas.coords(item, width // 2, max(px(4), (height - needed) // 2))
            canvas.configure(scrollregion=(0, 0, width, max(height, needed)), takefocus=needed > height)
            bar.grid() if needed > height else bar.grid_remove()

        def scroll(n: int, unit: str = "units") -> str:
            canvas.yview_scroll(n, unit)
            return "break"

        canvas.bind("<Configure>", fit)
        inner.bind("<Configure>", fit)
        for key, n, unit in (("Up", -1, "units"), ("Down", 1, "units"),
                             ("Prior", -1, "pages"), ("Next", 1, "pages")):
            canvas.bind(f"<{key}>", lambda _e, n=n, unit=unit: scroll(n, unit))
        pending = [canvas, inner]
        while pending:
            widget = pending.pop()
            widget.bind("<MouseWheel>", lambda e: scroll(-1 if e.delta > 0 else 1))
            widget.bind("<Button-4>", lambda _e: scroll(-1))
            widget.bind("<Button-5>", lambda _e: scroll(1))
            pending.extend(widget.winfo_children())

    @ui_thread
    def reveal(self, row_ids: Sequence[str], section: str) -> None:
        """Open a section and select its requested rows, with keyboard focus on the first one still present."""
        self._set_open(section, True)
        rows = [self._iid[rid] for rid in row_ids if rid in self._iid]
        self.tree.selection_set(rows)
        if rows:
            self.tree.focus(rows[0])
            self.tree.see(rows[0])
        self.tree.focus_set()

    @ui_thread
    def focused_row(self) -> Optional[str]:
        return self._rid.get(self.tree.focus())

    def destroy(self) -> None:
        if self._relayout_id is not None:                   # its callback goes with this widget
            self.after_cancel(self._relayout_id)
        super().destroy()

    # ---- extras for the main window (§2.1.4) and tests
    @ui_thread
    def set_compact(self, on: bool) -> None:
        """Compact mode hides the details strip."""
        self._compact = bool(on)
        self._show_details()

    def min_height(self, rows: int = 4) -> int:
        """Height that shows the toolbar, the headings, `rows` rows and the details strip (the pane's minsize)."""
        th = self.ctx.theme
        heading = th.fonts["YfsSmallBold"].metrics("linespace") + th.px(8) + 4
        strip = 0 if self._compact else self.details.winfo_reqheight() + th.px(6)
        return self._bar.winfo_reqheight() + th.px(6 + 16) + 2 + heading + rows * th.rowheight + strip

    @ui_thread
    def rename(self, row_id: Optional[str]) -> None:
        """§2.3.5 inline editor over the Output file cell: Enter or FocusOut commits, Esc cancels."""
        plan, t = self.plan, self.tree
        if plan is None or row_id not in self._iid:
            return
        r = plan.row(row_id)
        if r.section == "note" or started(r):
            return
        self._close_editor()
        self._reveal(self._iid[row_id])
        t.update_idletasks()
        px = self.ctx.theme.px
        frame = ttk.Frame(t, style="Surface.TFrame")
        entry = ttk.Entry(frame)
        entry.insert(0, r.output_stem)
        suffix = ttk.Label(frame, text=OME_SUFFIX, style="Surface.Muted.TLabel")
        suffix.pack(side="right", padx=(px(2), px(2)))      # packed first: the suffix keeps its width (MW-25)
        entry.pack(side="left", fill="both", expand=True)
        error = ttk.Label(t, style="Surface.Error.TLabel")
        self._edit = {"row": row_id, "frame": frame, "entry": entry, "suffix": suffix, "error": error}
        self._write([self._iid[row_id]])
        entry.bind("<Return>", lambda e: self._commit_edit() or "break")
        entry.bind("<KP_Enter>", lambda e: self._commit_edit() or "break")
        entry.bind("<Escape>", lambda e: self._close_editor() or "break")
        entry.bind("<FocusOut>", lambda e: self._commit_edit())
        self._place_editor()
        entry.select_range(0, "end")
        entry.focus_set()

    # ---- rows and sections
    def _display(self, r: PlanRow) -> tuple:
        """(text, (left image, right image, dimmed slot or -1), values, tags) of one row, with its full texts."""
        t, c = self.ctx.theme.text, r.cand
        if r.section == "note":
            return t(_note_label(r)), (None, status_icon(r), -1), ("",) * 5 + (t(row_text(r)),), ("note",)
        stopped = self._stopped(r)
        icon = "skip" if stopped else status_icon(r)
        size = (human_bytes(r.existing.size) if r.section == "existing" and r.existing else
                human_bytes(r.est.out_typical, True) if r.est else "")
        values = (f"{len(c.tile_names):,}" if c.tile_names else "", size, "", t(parent_rel(c) or "(top)"),
                  t(r.existing.name if r.existing is not None and
                    norm_key(r.existing.name) == norm_key(r.output_stem + OME_SUFFIX) else r.output_stem + OME_SUFFIX),
                  t(self._what(r)))
        left = ("cb_on" if r.checked else "cb_off") if r.checkable else None
        return t(display_name(c, True)), (left, icon, 0 if self._fixed(r) else -1), values, \
            ("dim",) if stopped else row_tags(r, icon)

    def _what(self, r: PlanRow) -> str:
        """The *What will happen* text; once a stop is asked for, no waiting slide starts (MW-19)."""
        return "Won't be stitched (stopping)" if self._stopped(r) else row_text(r)

    def _stopped(self, r: PlanRow) -> bool:
        """A slide that was waiting its turn when a stop was asked for (one unticked before says so already)."""
        return r.status == "waiting" and r.checked and self._stopping()

    def _heads(self, groups: Dict[str, List[PlanRow]], pending: Set[str]) -> Dict[str, tuple]:
        """Per section: (visible, open, display tuple). Only *New slides* has a (tri-state) box, dimmed while it can't
        tick anything (during a run, MW-2); its count is of the slides Start would run (MW-7). The hint goes to the
        What column: a Treeview cell cannot span columns, and the Slide column is narrow. The count of slides done in
        this plan goes there too: the label keeps its width through a run, so the Slide column, never narrower than
        the labels (MW-18), doesn't move when Stitch is pressed."""
        t, out = self.ctx.theme.text, {}
        new, old, other = groups["new"], groups["existing"], groups["excluded"]
        ticks = [r.checked for r in new if r.checkable]
        box = None if not ticks else "cb_on" if all(ticks) else "cb_mixed" if any(ticks) else "cb_off"
        done = sum(map(stitched, new))
        widest = [_new_label(len(new), len(new), 0)]       # the widest it can read: steady while slides finish
        defaults = {"new": True, "existing": len(old) <= 5 or not new, "excluded": False}
        for sec, rows in groups.items():
            is_open, hint = self._open.get(sec, defaults[sec]), ""
            if sec == "new":
                rest = len(new) - done
                text = _new_label(sum(r.row_id in pending for r in new), rest, done) if new else "New slides  (0)"
                hint = "Nothing new to stitch" if not new else f"{done} done" if done and rest else ""
            elif sec == "existing":
                text, hint = f"Already stitched  ({len(old)})", f"Skipped unless you tick {'it' if len(old) == 1 else 'them'}"
                if not is_open and old:
                    hint += "  (" + ", ".join(display_name(r.cand) for r in old[:3]) + (", …)" if len(old) > 3 else ")")
            else:
                text = f"Not included  ({len(other)})"
            widest.append(text)
            visible = bool(rows) or (sec == "new" and bool(old or other))
            images = ("chev_down" if is_open else "chev_right", box if sec == "new" else None,
                      1 if sec == "new" and not self._box_rows(new) else -1)
            out[sec] = (visible, is_open, (t(text), images, ("",) * 5 + (t(hint),), ("section",)))
        bold = self._rulers["YfsBold"]
        self._label_w = max(bold.measure(t(s)) for s in widest) + self._pair_w + self._pad
        return out

    def _help(self, rows: Sequence[PlanRow]) -> str:
        """The toolbar hint (MW-16): what ticking can do now; none when it can do nothing (no slides, or a stop was
        asked for)."""
        if not self._slides or self._stopping():
            return ""
        if self._running:
            return "Untick a waiting slide to skip it." if self._waiting else ""
        if self._state == "FINISHED":
            unticked = any(r.outcome is not None and r.outcome.status in _FAILED and not r.checked for r in rows)
            return "Tick failed slides to try them again." if unticked else ""
        return "Untick anything you don’t want."

    def _stopping(self) -> bool:
        return self._running and self._state == "STOPPING"

    def _fixed(self, r: PlanRow) -> bool:
        """A tick that can't change now, shown dimmed: during a run, every one but a waiting slide's (and every one
        once a stop is asked for); otherwise a slide stitched in this plan (MW-16)."""
        if self._running:
            return self._stopping() or not (r.status == "waiting" and r.checked)
        return stitched(r)

    def _box_rows(self, new: Sequence[PlanRow]) -> List[PlanRow]:
        """The rows the *New slides* box ticks or unticks: none during a run (MW-2)."""
        return [] if self._running else [r for r in new if r.checkable and not self._fixed(r)]

    def _image(self, left: Optional[str], right: Optional[str], dim: int) -> tk.PhotoImage:
        """icons.pair(left, right) with the image in slot `dim` (0 or 1, -1: none) dimmed: a tick box that can't
        change now. Made once per combination, by copying the dimmed icon over the pair without it."""
        icons = self.ctx.icons
        name = (left, right)[dim] if dim >= 0 else None
        if name is None:
            return icons.pair(left, right)
        img = self._dimmed.get((left, right, dim))
        if img is None:
            base = icons.pair(None, right) if dim == 0 else icons.pair(left, None)
            img = self._dimmed[(left, right, dim)] = tk.PhotoImage(master=self, width=base.width(),
                                                                   height=base.height())
            img.tk.call(img, "copy", base, "-compositingrule", "set")
            img.tk.call(img, "copy", icons.get(name, dim=True), "-to", dim * (icons.size + icons.gap), 0,
                        "-compositingrule", "set")
        return img

    # ---- column widths and shortened cells (MW-10, MW-17, MW-18)
    def _ruler(self, tags: Sequence[str]) -> _Ruler:
        return self._rulers["YfsBold" if tags and tags[0] in ("section", "running") else "YfsBody"]

    def _grow(self, d: tuple, run: bool) -> None:
        """Widen the columns' wishes to the texts of a slide or note row, so that no column moves from Stitch to the
        end of the run: a slide's name, folder and output count as wide as they are in bold (as while it runs), and
        its What text only outside a run (`run`: waiting, running or done); a run's texts fit the room planned."""
        text, _, values, tags = d
        need, plain = self._need, self._rulers["YfsBody"]
        ruler = plain if tags == ("note",) else self._rulers["YfsBold"]
        need["#0"] = max(need["#0"], ruler.measure(text) + self._indent + self._pair_w + self._pad)
        for col in ("folder", "output"):
            need[col] = max(need[col], ruler.measure(values[COLUMNS.index(col)]) + self._pad)
        if not run:
            need["what"] = max(need["what"], plain.measure(values[COLUMNS.index("what")]) + self._pad)

    def _layout(self) -> bool:
        """Column widths. Tiles, Size and the gap after them are fixed; Slide, Folder and Output are as wide as their
        longest text, capped, and Slide is never narrower than the section labels; What gets the rest. When room is
        short What keeps its longest text, or at least 35 % of the room, and the others give way in proportion down
        to their minimum; when What has more than it needs, the capped columns get the spare. True when a width
        changed."""
        t, px = self.tree, self.ctx.theme.px
        cols = tuple(c for c in COLUMNS if (c != "size" or not self._narrow) and (c != "folder" or self._multi))
        if cols != self._cols:
            t.configure(displaycolumns=cols)
            self._cols = cols
        fixed = {c: w for c, w in self._fixed_w.items() if c in cols}
        floor = {"#0": max(px(160), self._label_w), "folder": px(80), "output": self._out_base}
        cap = {"#0": self._slide_cap, "folder": px(160), "output": px(200 if self._narrow else 260)}
        grow = [c for c in ("#0", "output", "folder") if c == "#0" or c in cols]
        width = {c: max(floor[c], min(cap[c], self._need[c])) for c in grow}
        tree_w = t.winfo_width()
        room = tree_w - _FIELD - sum(fixed.values()) if tree_w > 1 else sum(width.values()) + self._what_base
        what = max(px(140), self._need["what"])
        short = sum(width.values()) + min(what, max(px(140), room * 35 // 100)) - room
        slack = {c: width[c] - floor[c] for c in grow}
        if short > 0 and any(slack.values()):               # too little room: the others give way in proportion
            total = sum(slack.values())
            for c in grow:
                width[c] -= min(slack[c], -(-short * slack[c] // total))
        spare = room - sum(width.values()) - what
        for c in grow:                                      # room to spare: shortened texts get it first
            if spare > 0 and self._need[c] > width[c]:
                take = min(spare, self._need[c] - width[c])
                width[c], spare = width[c] + take, spare - take
        width.update(fixed, what=max(px(40), room - sum(width.values())))
        changed = [c for c, w in width.items() if self._width.get(c) != w]
        for c in changed:
            t.column(c, width=width[c])
            self._width[c] = width[c]
        return bool(changed)

    def _fit(self, iid: str, d: tuple) -> tuple:
        """d with each text shortened with an ellipsis to its column: Folder and Output keep the end that tells
        copies apart, the others are cut at the end. Hidden columns keep their text. The What cell of the row being
        renamed is blank: the editor covers its start, and its end would read as part of the name."""
        text, images, values, tags = d
        if self._edit is not None and iid == self._iid.get(str(self._edit["row"])):
            values = values[:-1] + ("",)
        ruler, ell, room = self._ruler(tags), self.ctx.theme.G["ell"], self._width
        indent = 0 if iid.startswith("sec:") else self._indent
        text = fit_text(ruler, text, room["#0"] - indent - self._pair_w - self._pad, ell)
        values = tuple(v if not v or c not in self._cols else
                       _FITS.get(c, fit_text)(ruler, v, room[c] - self._pad, ell) for c, v in zip(COLUMNS, values))
        return text, images, values, tags

    def _write(self, iids: Iterable[str]) -> None:
        """Draw items from their full texts, fitted to the columns; unchanged items are skipped."""
        t = self.tree
        for iid in iids:
            shown = self._fit(iid, self._full[iid])
            if self._shown.get(iid) != shown:
                text, images, values, tags = shown
                t.item(iid, text=text, image=self._image(*images), values=values, tags=tags)
                self._shown[iid] = shown

    def _relayout(self) -> None:
        self._relayout_id = None
        if self._layout():
            self._write(list(self._full))
            self._place_editor()

    def _tree_configured(self, e: tk.Event) -> None:
        if e.width != self._tree_w:
            self._tree_w = e.width
            if self._relayout_id is None:
                self._relayout_id = self.after_idle(self._relayout)
        self._place_editor()

    def _cell_text(self, iid: str, col: str) -> str:
        """The full text of a cell that shows it shortened, else ''."""
        full, shown = self._full.get(iid), self._shown.get(iid)
        if full is None or shown is None:
            return ""
        if col == "#0":
            return full[0] if full[0] != shown[0] else ""
        k = COLUMNS.index(self.tree.column(col, "id"))
        return full[2][k] if full[2][k] != shown[2][k] else ""

    def _hover(self, e: tk.Event) -> Optional[str]:
        """The pointer on a cell: its full text as a tooltip when the cell shows it shortened (MW-10). Over a column
        separator the resize cursor isn't offered: the widths follow the window (_layout)."""
        t, tip = self.tree, self._tip
        region = t.identify_region(e.x, e.y)
        cell = (t.identify_row(e.y), t.identify_column(e.x)) if region in ("tree", "cell") else None
        if cell != self._tip_cell:
            self._tip_cell = cell
            tip.hide()
            tip.full = self._cell_text(*cell) if cell and cell[0] else ""
            bbox = t.bbox(*cell) if tip.full else None
            if bbox:
                tip.at = (t.winfo_rootx() + bbox[0], t.winfo_rooty() + bbox[1] + bbox[3] + self.ctx.theme.px(2))
                tip._schedule()
        return "break" if region == "separator" else None

    def _left(self, _e: tk.Event) -> None:
        self._tip_cell, self._tip.full = None, ""

    # ---- sections and rows
    def _section_of(self, iid: str) -> Optional[str]:
        return iid[4:] if iid.startswith("sec:") else None

    def _set_open(self, sec: str, is_open: bool) -> None:
        self._open[sec] = is_open
        self.refresh([])

    def _reveal(self, iid: str) -> None:
        """Scroll row `iid` into view, opening its section as a click does: see() alone opens it behind the table's
        back, so the chevron would still say closed and the next refresh would close it again, editor and all."""
        sec = self._section_of(self.tree.parent(iid))
        if sec and not self.tree.item(f"sec:{sec}", "open"):
            self._set_open(sec, True)
        self.tree.see(iid)

    def _remember_open(self, is_open: bool) -> None:
        sec = self._section_of(self.tree.focus())
        if sec:                                             # Left/Right keys open and close sections too
            self._open[sec] = is_open
            self.after_idle(lambda: self.refresh([]))

    def _row(self, iid: str) -> Optional[PlanRow]:
        return self.plan.row(self._rid[iid]) if self.plan is not None and iid in self._rid else None

    def _selected(self, notes: bool = False) -> List[str]:
        """The selected rows in open sections: a row out of sight is never changed (a Shift+click range across a
        closed section selects its rows, and a row can move into a closed section while selected)."""
        t = self.tree
        rows = [self._row(i) for i in t.selection() if i in self._rid and t.item(t.parent(i), "open")]
        return [r.row_id for r in rows if r is not None and (notes or r.section != "note")]

    # ---- user changes
    def _apply(self, changed: List[str]) -> None:
        if changed:
            self.refresh(changed)
            self._on_changed()

    def _toggle(self, row_ids: List[str], value: Optional[bool] = None) -> None:
        """Tick or untick rows; a dimmed tick box doesn't change."""
        if self.plan is not None:
            rows = [i for i in row_ids if not self._fixed(self.plan.row(i))]
            if rows:
                self._apply(self.plan.toggle(rows, value))

    def _toggle_section(self, sec: str) -> None:
        """The *New slides* box ticks or unticks its slides; while it is dimmed it opens or closes the section, like
        the rest of a section row (MW-2: during a run it never unticks the queue)."""
        rows = self._box_rows(self.plan.rows("new")) if sec == "new" and self.plan is not None else []
        if rows:
            self._toggle([r.row_id for r in rows], not all(r.checked for r in rows))
        elif sec:
            self._set_open(sec, not self.tree.item(f"sec:{sec}", "open"))

    def _select_mode(self, mode: str) -> None:
        if self.plan is not None:
            self._apply(self.plan.select(mode))

    # ---- pointer
    def _slot(self, iid: str, x: int, y: int) -> Optional[int]:
        """0 or 1 when (x, y) is on the left or right half of a row's composite image (§2.3.2), else None."""
        bbox = self.tree.bbox(iid, "#0")
        if not bbox or self.tree.identify_element(x, y) != "image":
            return None
        icons = self.ctx.icons
        return 0 if x - bbox[0] < icons.size + icons.gap / 2 else 1

    def _hit(self, e: tk.Event) -> Tuple[str, Optional[int]]:
        t = self.tree
        iid = t.identify_row(e.y)
        if not iid or t.identify_region(e.x, e.y) not in ("tree", "cell"):
            return "", None
        return iid, self._slot(iid, e.x, e.y)

    def _click(self, e: tk.Event) -> Optional[str]:
        self._tip.hide()
        if self.tree.identify_region(e.x, e.y) == "separator":
            return "break"                                  # the widths follow the window (_layout)
        iid, slot = self._hit(e)
        sec = self._section_of(iid)
        if sec:                                             # the selection follows, as with the arrow keys: Space
            self.tree.focus(iid)                            # then acts on this section, not on rows it hides (MW-2)
            self.tree.selection_set(iid)
            self.tree.focus_set()
            self._toggle_section(sec) if slot == 1 else self._set_open(sec, not self.tree.item(iid, "open"))
            return "break"
        r = self._row(iid)
        if r is not None and slot is not None and r.checkable and r.section != "note" and not self._fixed(r):
            self.tree.focus(iid)                            # the tick box or the status icon: toggle the tick; the
            self.tree.focus_set()                           # keys go on from this row
            self._toggle([r.row_id])
            return "break"
        return None                                         # (a dimmed box selects the row, like its name)

    def _double(self, e: tk.Event) -> Optional[str]:
        iid, slot = self._hit(e)
        if not iid:
            return None
        sec = self._section_of(iid)
        if sec:                                             # the first click already opened or closed it
            return self._click(e) if sec == "new" and slot == 1 and self.plan is not None and \
                self._box_rows(self.plan.rows("new")) else "break"
        if slot is not None and self._click(e):             # quick clicks on a tick box each toggle it
            return "break"
        r = self._row(iid)
        col = self.tree.identify_column(e.x)
        if r is not None and r.section != "note":
            if col != "#0" and self.tree.column(col, "id") == "output":
                self.rename(r.row_id)
            else:
                self._on_details(r.row_id)
        return "break"

    def _context(self, e: tk.Event) -> str:
        self._popup(self.tree.identify_row(e.y), e.x_root, e.y_root)
        return "break"

    def _popup(self, iid: str, x: int, y: int) -> None:
        """The context menu of row `iid` at (x, y) on the screen; the row joins the selection and takes the focus."""
        self._tip.hide()
        if iid in self._rid:
            if iid not in self.tree.selection():
                self.tree.selection_set(iid)
            self.tree.focus(iid)
            if self.fill_menu(self._rid[iid]):
                self.menu.tk_popup(x, y)

    # ---- keys
    def _key_space(self, _e: tk.Event) -> str:
        """The selected slides in sight, else the focused section (§2.3.2)."""
        rows = self._selected()
        if rows:
            self._toggle(rows)
        else:
            self._toggle_section(self._section_of(self.tree.focus()) or "")
        return "break"

    def _key_delete(self, _e: tk.Event) -> str:
        self._toggle(self._selected(), False)
        return "break"

    def _key_menu(self, _e: tk.Event) -> str:
        """Shift+F10 or the menu key: the focused row's context menu, under its name, so that Try again, Finalize
        and the searches don't need a mouse."""
        t, iid = self.tree, self.tree.focus()
        if iid in self._rid:
            self._reveal(iid)
            t.update_idletasks()
            bbox = t.bbox(iid, "#0")
            if bbox:
                self._popup(iid, t.winfo_rootx() + bbox[0] + self._pair_w, t.winfo_rooty() + bbox[1] + bbox[3])
        return "break"

    def _key_enter(self, _e: tk.Event) -> str:
        iid = self.tree.focus()
        sec, r = self._section_of(iid), self._row(iid)
        if sec:
            self._set_open(sec, not self.tree.item(iid, "open"))
        elif r is not None and r.section != "note":
            self._on_details(r.row_id)
        return "break"

    def _focus_in(self, _e: tk.Event) -> None:
        """Tk's arrow keys move from the focused item and do nothing without one: tabbing in focuses the first."""
        t = self.tree
        if not t.focus() and t.get_children(""):
            t.focus(t.get_children("")[0])

    def _select_all(self, _e: tk.Event) -> str:
        t = self.tree
        t.selection_set([i for s in t.get_children("") if t.item(s, "open") for i in t.get_children(s)])
        return "break"

    def _selection_changed(self, _e: Optional[tk.Event] = None) -> None:
        self._show_details()
        self._on_focus(self.focused_row())

    # ---- menus
    def _fill_select(self) -> None:
        """§2.3.6 Select menu, built when it opens; during a run only *Untick all waiting* is enabled."""
        m, plan, t = self.select_menu, self.plan, self.ctx.theme.text
        m.delete(0, "end")
        rows = plan.rows() if plan is not None else []
        dups = any(f.code == "duplicate" for r in rows for f in r.flags)
        failed = any(r.outcome is not None and r.outcome.status in _FAILED for r in rows)
        for label, mode in _SELECT:
            if (mode == "untick_smaller_copies" and not dups) or (mode == "failed" and not failed):
                continue
            m.add_command(label=t(label), command=lambda mo=mode: self._select_mode(mo),
                          state="disabled" if self._running or plan is None else "normal")
        if self._running:
            waiting = any(r.status == "waiting" and r.checked for r in rows) and not self._stopping()
            m.add_separator()
            m.add_command(label="Untick all waiting", command=lambda: self._select_mode("untick_waiting"),
                          state="normal" if waiting else "disabled")

    def fill_menu(self, row_id: str) -> int:
        """§2.3.6 context menu for `row_id` (actions that fit the row); returns the number of entries. Fix channel
        names, Try again and Finalize act on every selected row they fit, when `row_id` is one of them (FS-7)."""
        m, plan, t = self.menu, self.plan, self.ctx.theme.text
        m.delete(0, "end")
        if plan is None:
            return 0
        r, sel = plan.row(row_id), [plan.row(i) for i in self._selected()]
        sel = sel if any(x.row_id == row_id for x in sel) else [r]

        def add(label: str, fn: Callable[[], None], accel: str = "", enabled: bool = True) -> None:
            m.add_command(label=t(label), command=fn, accelerator=accel, state="normal" if enabled else "disabled")

        def each(action: str, rows: List[PlanRow]) -> Callable[[], None]:
            return lambda: [self._on_note_action(action, x.row_id) for x in rows]
        folder = r.cand.folder
        if r.section != "note" or (r.row_id.count("|") > 1 and not r.row_id.endswith("|*")):
            add("Show in folder" if r.note is not None and r.note.kind == "unmatched" else "Open folder",
                lambda: osutil.open_path(folder, on_error=lambda msg: log.warning("Couldn't open %s: %s", folder, msg)))
            add("Copy folder path", lambda: (self.clipboard_clear(), self.clipboard_append(folder)))
        if r.section == "note":
            action = _NOTE_ACTION.get(_kind(r.note)) if r.note else None
            if action is not None:                          # a search can't start during a run (MW-21)
                add(action[0], lambda: self._on_note_action(action[1], row_id), enabled=not self._running)
            return _entries(m)
        add("Slide details…", lambda: self._on_details(row_id), "Enter")
        add("Rename output…", lambda: self.after_idle(self.rename, row_id), "F2", not started(r))
        fix = [x.row_id for x in sel if x.header and x.header.channel_problem and not started(x)]
        if r.header is not None and r.header.channel_problem:     # R26: only rows with a channel problem
            add("Fix channel names…", lambda: self._on_fix(fix or [row_id]), enabled=not started(r))
        retry = [x for x in sel if x.status == "header_error" and x.header_retryable]
        if r.status == "header_error" and r.header_retryable:     # not for a bad tile: reading it again won't help
            add(_many("Try again", len(retry)), each("retry_header", retry), enabled=not self._running)
        final = [x for x in sel if x.status in ("finalize", "ready")]
        if r.status in ("finalize", "ready"):
            add(_many("Finalize", len(final)), each("finalize", final))
        waiting = [x.row_id for x in sel if x.status == "waiting" and x.checked]
        if self._running and not self._stopping() and waiting:
            add("Don’t stitch this slide" if len(waiting) == 1 else f"Don’t stitch these {len(waiting)} slides",
                lambda: self._toggle(waiting, False))
        return _entries(m)

    # ---- inline rename
    def _commit_edit(self) -> None:
        ed = self._edit
        if ed is None or self.plan is None:
            return
        entry, text = ed["entry"], ed["entry"].get()
        error = self.plan.rename_output(str(ed["row"]), text)
        if error is None:
            self._close_editor()
            self.refresh(None)                              # other rows' de-duplicated names may change
            self._on_changed()
            return
        clean = sanitize_stem(text.strip())
        if text.strip() and clean != text.strip() and not error.startswith("This slide"):
            entry.delete(0, "end")                          # show the cleaned name and keep editing
            entry.insert(0, clean)
        ed["error"].configure(text=self.ctx.theme.text(error))
        ed["shows_error"] = True
        self._place_editor()

    def _close_editor(self) -> None:
        ed, self._edit = self._edit, None
        if ed is None:
            return
        for w in (ed["frame"], ed["error"]):
            w.destroy()
        if self.tree.winfo_exists():
            iid = self._iid.get(str(ed["row"]))
            if iid in self._full:                           # its What cell shows again
                self._write([iid])
            self.tree.focus_set()

    def _place_editor(self, _e: Optional[tk.Event] = None) -> None:
        ed = self._edit
        if ed is None:
            return
        if str(ed["row"]) not in self._iid:                 # the row left the plan (e.g. a rescan)
            self._close_editor()
            return
        bbox = self.tree.bbox(self._iid[str(ed["row"])], "output")
        if not bbox:                                        # scrolled out of view: hide until it comes back
            ed["frame"].place_forget()
            ed["error"].place_forget()
            return
        x, y, w, h = bbox                                   # the Entry as wide as the cell, the suffix after it
        px, t = self.ctx.theme.px, self.tree
        right, bottom = t.winfo_width() - _FIELD // 2, t.winfo_height() - _FIELD // 2
        width = max(w, px(160)) + ed["suffix"].winfo_reqwidth() + px(4)
        ed["frame"].place(x=x, y=y, width=min(width, right - x), height=h)
        if ed.get("shows_error"):
            self._place_error(ed["error"], x, y, h, right, bottom)

    def _place_error(self, error: ttk.Label, x: int, y: int, h: int, right: int, bottom: int) -> None:
        """The rename error over the whole row under the field (above it at the bottom of the view), so that no text
        of that row shows beside the message; the message starts under the field as far as it fits, or wraps."""
        px, left = self.ctx.theme.px, _FIELD // 2
        error.configure(wraplength=right - left - px(12), padding=(px(6), px(2)))
        text_w, eh = error.winfo_reqwidth() - px(12), max(error.winfo_reqheight(), h)
        error.configure(padding=(max(px(6), min(x - left, right - left - text_w - px(6))), px(2), px(6), px(2)))
        error.place(x=left, y=y + h if y + h + eh <= bottom else max(0, y - eh), width=right - left, height=eh)
        error.lift()

    def _yscroll(self, first: str, last: str) -> None:
        self._vsb.set(first, last)
        self._place_editor()

    # ---- layout
    def _toolbar(self) -> None:
        """The hint and the Select button follow the state and the plan (MW-16); the hint needs a wide window. Select
        is off while none of its items could work: no slides, or a run with no waiting slide left to untick."""
        if self.plan is None:
            self._help_text, self._slides = "", False
        useful = self._slides and (not self._running or self._waiting and not self._stopping())
        self.select_button.state(["!disabled"] if useful else ["disabled"])
        if self._help_text and self._wide:
            self.help.configure(text=self.ctx.theme.text(self._help_text))
            self.help.grid()
        else:
            self.help.grid_remove()

    def _show_details(self) -> None:
        if self._compact or self._welcome is not None:
            self.details.grid_remove()
            return
        self.details.grid()
        r, th = self._row(self.tree.focus()), self.ctx.theme
        fit_label(self.details, th.text(details_text(r, self._what(r))) if r is not None else "", th.G["ell"])

    def _on_configure(self, _e: tk.Event) -> None:
        width = self.winfo_toplevel().winfo_width()
        if width <= 1:
            return
        px = self.ctx.theme.px
        narrow = width < px(900)
        if narrow != self._narrow:
            self._narrow = narrow
            self._relayout()
        self._wide = width >= px(820)
        self._toolbar()
