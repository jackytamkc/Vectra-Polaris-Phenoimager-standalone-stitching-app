"""Shared widgets: modal helpers, Tooltip, PathField, TileMap and SlideFacts. FINAL_SPEC §2.1.2, §2.3.7, §5.5.13.
Nothing here touches the filesystem: widgets show data and hand user input to callbacks."""
from __future__ import annotations

import os
import statistics
import tkinter as tk
import unicodedata
from tkinter import ttk
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

from yfs.core.names import OME_SUFFIX, human_bytes, norm_key, stitchable_groups
from yfs.ui.theme import TOKENS as T, theme_of

try:                                    # WP2's places (pure string work); a minimal cleaner stands in until it lands
    from yfs.core.places import normalize_user_path
except ImportError:  # pragma: no cover
    normalize_user_path = None

if TYPE_CHECKING:
    from yfs.core.model import DirSummary, OutputIndex, Rollup
    from yfs.core.plan import PlanRow
    from yfs.ui.app import AppContext

Action = Tuple[str, Callable[[], None]]


def safe_grab(win: tk.Toplevel, tries: int = 20) -> None:
    """grab_set() with after(50) retries; never wait_visibility."""
    try:
        win.grab_set()
    except tk.TclError:                  # not viewable yet, or another grab is active
        if tries > 1:                    # scheduled on the root: an after() on a destroyed window would raise
            win._root().after(50, lambda: win.winfo_exists() and safe_grab(win, tries - 1))


def center_over(win: tk.Toplevel, parent: tk.Misc) -> None:
    """Uses parent geometry after update_idletasks(); kept on screen."""
    win.update_idletasks()                                # sizes the window (explicit geometry or natural size)
    w = win.winfo_width() if win.winfo_width() > 1 else win.winfo_reqwidth()
    h = win.winfo_height() if win.winfo_height() > 1 else win.winfo_reqheight()
    if parent.winfo_ismapped():
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
    else:
        x, y = (win.winfo_screenwidth() - w) // 2, (win.winfo_screenheight() - h) // 2
    win.geometry(f"+{max(0, min(x, win.winfo_screenwidth() - w))}+{max(0, min(y, win.winfo_screenheight() - h))}")


class Tooltip:
    """A delayed hint under `widget`, or above it when the screen ends below it; text() is evaluated when it shows
    (empty text: nothing shown). delay_ms: default DELAY_MS."""
    DELAY_MS = 600

    def __init__(self, widget: tk.Misc, text: Callable[[], str], delay_ms: Optional[int] = None) -> None:
        self.widget, self.text, self.delay = widget, text, delay_ms
        self._after: Optional[str] = None
        self._tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._schedule, add="+")
        for seq in ("<Leave>", "<ButtonPress>", "<Destroy>"):
            widget.bind(seq, self.hide, add="+")

    def _schedule(self, _e: Any = None) -> None:
        self.hide()
        self._after = self.widget.after(self.DELAY_MS if self.delay is None else self.delay, self._show)

    def _show(self) -> None:
        self._after = None
        text, theme = self.text(), theme_of(self.widget)
        if not text or not self.widget.winfo_viewable():
            return
        px = theme.px if theme else int
        tip = self._tip = tk.Toplevel(self.widget, background=T["border_strong"])   # 1 px border colour
        tip.wm_overrideredirect(True)
        tk.Label(tip, text=theme.text(text) if theme else text, background=T["surface"], foreground=T["text"],
                 font="YfsSmall", justify="left", wraplength=px(420), padx=px(6), pady=px(3)).pack(padx=1, pady=1)
        tip.update_idletasks()
        x = min(self.widget.winfo_rootx() + px(8), tip.winfo_screenwidth() - tip.winfo_reqwidth() - 2)
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + px(4)
        if y + tip.winfo_reqheight() > tip.winfo_screenheight():    # (a run bar at the bottom of a maximised window)
            y = self.widget.winfo_rooty() - tip.winfo_reqheight() - px(4)
        tip.wm_geometry(f"+{max(0, x)}+{max(0, y)}")

    def hide(self, _e: Any = None) -> None:
        if self._after:
            self.widget.after_cancel(self._after)
            self._after = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def _basic_clean(text: str) -> str:
    """First non-empty line without surrounding quotes (stand-in for places.normalize_user_path)."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line[1:-1].strip() if len(line) > 1 and line[0] == line[-1] and line[0] in "'\"" else line


class PathField(ttk.Frame):
    """Combobox + placeholder label + Browse button + status row (icon, text, action buttons). §2.1.2.
    Commits on Enter, <<ComboboxSelected>>, FocusOut (if changed) or 600 ms after typing an absolute path."""
    DEBOUNCE_MS = 600

    def __init__(self, master: tk.Misc, ctx: "AppContext", *, placeholder: str, recents: Callable[[], List[str]],
                 on_commit: Callable[[str], None], on_browse: Callable[[], None]) -> None:
        super().__init__(master, style="Surface.TFrame")
        self.ctx, self._recents, self._on_commit = ctx, recents, on_commit
        px = ctx.theme.px
        self._var = tk.StringVar(self)
        self._committed = ""                              # text of the last commit or set()
        self._focused = False
        self._timer: Optional[str] = None                 # after() ids: debounce ...
        self._spin: Optional[str] = None                  # ... and spinner animation
        self._spin_frame = 0
        self._actions: List[ttk.Button] = []
        self.columnconfigure(0, weight=1)
        self.combo = ttk.Combobox(self, textvariable=self._var, postcommand=self._fill_recents)
        self.combo.grid(row=0, column=0, sticky="ew")
        self.browse = ttk.Button(self, text="Browse" + ctx.theme.G["ell"], command=on_browse)
        self.browse.grid(row=0, column=1, padx=(px(8), 0))
        self._placeholder = ttk.Label(self.combo, text=ctx.theme.text(placeholder), style="Placeholder.TLabel")
        self._placeholder.bind("<Button-1>", lambda _e: self.combo.focus_set())
        self.status = ttk.Frame(self, style="Surface.TFrame")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(px(4), 0))
        self._icon = ttk.Label(self.status, style="Surface.TLabel")
        self._text = ttk.Label(self.status, style="Surface.Muted.TLabel", wraplength=px(560))
        self.status.bind("<Configure>", lambda _e: self._rewrap())
        enter = lambda _e: self._commit(force=True)       # noqa: E731
        for seq, fn in (("<Return>", enter), ("<KP_Enter>", enter), ("<<ComboboxSelected>>", lambda _e: self._commit()),
                        ("<FocusIn>", self._focus_in), ("<FocusOut>", self._focus_out), ("<KeyRelease>", self._typed),
                        ("<Configure>", self._show_tail)):
            self.combo.bind(seq, fn, add="+")
        self.bind("<Destroy>", lambda e: e.widget is self and self._destroyed(), add="+")
        self._trace = self._var.trace_add("write", lambda *_: self._sync_placeholder())
        self._sync_placeholder()

    # ---- public
    def get(self) -> str:
        return self._var.get().strip()

    def set(self, path: str, commit: bool = False) -> None:
        """Shows the tail (xview_moveto(1.0)) when unfocused."""
        self._cancel("_timer")
        self._var.set(path)
        self._committed = path
        self._show_tail()
        if commit:
            self._on_commit(path)

    def set_status(self, icon: Optional[str], text: str, style: str = "Surface.Muted.TLabel",
                   actions: Sequence[Action] = ()) -> None:
        """icon: an IconSet name, 'spinner' (animated at 120 ms) or None. Frequent updates (search progress) keep
        a running spinner turning and keep the action buttons whose labels did not change, so a click on one is
        never lost to a refresh."""
        px, t = self.ctx.theme.px, self.ctx.theme.text
        if icon != "spinner" or self._spin is None:
            self._cancel("_spin")
            if icon:
                self._icon.configure(image=self.ctx.icons.spinner(0) if icon == "spinner" else self.ctx.icons.get(icon))
            if icon == "spinner":
                self._spin = self.after(120, self._spin_tick)
        self._text.configure(text=t(text), style=style)
        rebuilt = [t(label) for label, _cmd in actions] != [str(b.cget("text")) for b in self._actions]
        if rebuilt:
            for b in self._actions:
                b.destroy()
            self._actions = [ttk.Button(self.status, text=t(label), style="Link.TButton") for label, _cmd in actions]
        for b, (_label, cmd) in zip(self._actions, actions):
            b.configure(command=cmd)
            b.state(["disabled" if self.browse.instate(["disabled"]) else "!disabled"])
        if rebuilt or bool(icon) != bool(self._icon.winfo_manager()) or not self._text.winfo_manager():
            for w in (self._icon, self._text, *self._actions):
                w.pack_forget()
            if icon:
                self._icon.pack(side="left", padx=(0, px(6)))
            self._text.pack(side="left")
            for b in self._actions:
                b.pack(side="left", padx=(px(8), 0))
        self._rewrap()

    def set_enabled(self, on: bool) -> None:
        for w in (self.combo, self.browse, *self._actions):
            w.state(["!disabled"] if on else ["disabled"])

    # ---- internals
    def _fill_recents(self) -> None:
        self.combo.configure(values=list(self._recents())[:8])

    def _clean(self, text: str) -> str:
        mounts = self.ctx.fs.mounts if self.ctx.fs is not None else self.ctx.mounts      # the live table (UD2)
        if normalize_user_path is not None and mounts is not None:
            return normalize_user_path(text, mounts)
        return _basic_clean(text)

    def _commit(self, force: bool = False) -> None:
        self._cancel("_timer")
        text = self._clean(self._var.get())
        if not text:                                      # an emptied field keeps the current folder
            text = self._committed
        if text != self._var.get():
            self._var.set(text)
        if text and (force or text != self._committed):
            self._committed = text
            self._on_commit(text)

    def _typed(self, event: Any) -> None:
        if event.keysym not in ("Return", "KP_Enter", "Escape", "Tab"):
            self._cancel("_timer")
            self._timer = self.after(self.DEBOUNCE_MS, self._debounced)

    def _debounced(self) -> None:
        self._timer = None
        if os.path.isabs(self._clean(self._var.get())):  # string test only
            self._commit()

    def _focus_in(self, _e: Any) -> None:
        self._focused = True
        self._sync_placeholder()

    def _focus_out(self, _e: Any) -> None:
        if str(self.tk.call("focus")).startswith(str(self.combo) + "."):
            return                                        # focus went to our own dropdown list
        self._focused = False
        self._commit()
        self._sync_placeholder()
        self._show_tail()

    def _show_tail(self, _e: Any = None) -> None:
        if not self._focused:
            self.combo.icursor("end")
            self.combo.xview_moveto(1.0)

    def _sync_placeholder(self) -> None:
        if self._var.get() or self._focused:
            self._placeholder.place_forget()
        else:
            px = self.ctx.theme.px
            self._placeholder.place(x=px(7), rely=0.5, anchor="w", relwidth=1.0, width=-px(36))

    def _rewrap(self) -> None:
        width = self.status.winfo_width()
        if width > 1:
            used = sum(w.winfo_reqwidth() for w in (self._icon, *self._actions) if w.winfo_manager())
            used += self.ctx.theme.px(6 + 8 * len(self._actions) + 4)
            self._text.configure(wraplength=max(self.ctx.theme.px(160), width - used))

    def _spin_tick(self) -> None:
        if self.winfo_viewable():                          # animate only while visible
            self._spin_frame = (self._spin_frame + 1) % 8
            self._icon.configure(image=self.ctx.icons.spinner(self._spin_frame))
        self._spin = self.after(120, self._spin_tick)

    def _cancel(self, attr: str) -> None:
        if getattr(self, attr):
            self.after_cancel(getattr(self, attr))
            setattr(self, attr, None)

    def _destroyed(self) -> None:
        self._cancel("_spin")
        self._cancel("_timer")
        self._var.trace_remove("write", self._trace)      # the trace callback would keep this widget alive


def _median_step(values: Sequence[float]) -> Optional[float]:
    u = sorted(set(values))
    steps = [b - a for a, b in zip(u, u[1:]) if b - a > 0]
    return statistics.median(steps) if steps else None


class TileMap(tk.Canvas):
    """Tiles drawn as rectangles at their name-derived centres (no pixels are read)."""

    def __init__(self, master: tk.Misc, *, width: int, height: int, **kw: Any) -> None:
        super().__init__(master, width=width, height=height, highlightthickness=1, highlightbackground=T["border"],
                         **kw)
        self._shown: Optional[Tuple[Any, ...]] = None
        self.bind("<Configure>", lambda _e: self._shown is not None and self._draw(*self._shown))

    def show(self, coords: Sequence[Tuple[float, float]], tile_w_um: Optional[float],
             tile_h_um: Optional[float]) -> None:
        """Rectangles (fill accent_tint, outline accent) scaled to fit with px(8) margin; dashed bbox outline;
        tile size unknown -> median step of sorted unique x/y (or a fixed 1/30 of the span)."""
        self._shown = (tuple(coords), tile_w_um, tile_h_um)
        self._draw(*self._shown)

    def _draw(self, coords: Tuple[Tuple[float, float], ...], tw: Optional[float], th: Optional[float]) -> None:
        self.delete("all")
        if not coords:
            return
        theme = theme_of(self)
        m = theme.px(8) if theme else 8
        xs, ys = [x for x, _ in coords], [y for _, y in coords]
        span = max(max(xs) - min(xs), max(ys) - min(ys)) / 30 or 1.0
        tw = tw or _median_step(xs) or _median_step(ys) or span
        th = th or _median_step(ys) or _median_step(xs) or span
        x0, y0, bw, bh = min(xs) - tw / 2, min(ys) - th / 2, max(xs) - min(xs) + tw, max(ys) - min(ys) + th
        cw = self.winfo_width() if self.winfo_width() > 1 else int(self["width"])
        ch = self.winfo_height() if self.winfo_height() > 1 else int(self["height"])
        s = min((cw - 2 * m) / bw, (ch - 2 * m) / bh)
        ox, oy = (cw - bw * s) / 2, (ch - bh * s) / 2
        for x, y in coords:
            l, t = ox + (x - tw / 2 - x0) * s, oy + (y - th / 2 - y0) * s
            self.create_rectangle(l, t, l + tw * s, t + th * s, fill=T["accent_tint"], outline=T["accent"])
        self.create_rectangle(ox - 2, oy - 2, ox + bw * s + 2, oy + bh * s + 2, outline=T["text_faint"], dash=(3, 3))


def _nk(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _n(count: int, word: str, more: str = "") -> str:
    """'1 tile', '1,204 tiles', '40+ tiles' (more='+')."""
    return f"{count:,}{more} {word}{'' if count == 1 and not more else 's'}"


def _same_dir(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


class SlideFacts(ttk.Frame):
    """TileMap + facts lines. Used by the browser preview (narrow, on the sidebar colour) and SlideDetailsDialog
    (wide=True: map on the left, facts on the right). surface: 'Sidebar' | 'Surface' | 'Bg' (label style prefix).
    on_open(folder) and on_fix_channels([row_id]) add [Open folder] and [Fix channel names...] to show_row().
    The map and its caption are hidden while there are no tiles to draw (BR-25)."""

    def __init__(self, master: tk.Misc, ctx: "AppContext", *, wide: bool = False, surface: str = "Sidebar",
                 on_open: Optional[Callable[[str], None]] = None,
                 on_fix_channels: Optional[Callable[[List[str]], None]] = None) -> None:
        frame_style, self._bg = {"Sidebar": ("Sidebar.TFrame", T["sidebar"]),
                                 "Surface": ("Surface.TFrame", T["surface"])}.get(surface, ("TFrame", T["bg"]))
        super().__init__(master, style=frame_style)
        self.ctx, self._pre, self._on_open, self._on_fix = ctx, surface, on_open, on_fix_channels
        px = ctx.theme.px
        self._wrap = px(330 if wide else 206)
        self.facts = ttk.Frame(self, style=self.cget("style"))
        self.drawing = ttk.Frame(self, style=self.cget("style"))      # the map with its caption underneath
        self.map = TileMap(self.drawing, width=px(260 if wide else 206), height=px(200 if wide else 150))
        self.map.pack(anchor="w")
        self.caption = ttk.Label(self.drawing, style=f"{surface}.Small.TLabel", wraplength=px(260 if wide else 206))
        self.caption.pack(anchor="w", pady=(px(4), 0))
        if wide:
            self.drawing.grid(row=0, column=0, sticky="nw", padx=(0, px(16)))
            self.facts.grid(row=0, column=1, sticky="nw")
        else:
            self.facts.grid(row=0, column=0, sticky="nw")
            self.drawing.grid(row=1, column=0, sticky="nw", pady=(px(8), 0))

    def show_folder(self, summary: Optional["DirSummary"], rollup: Optional["Rollup"],
                    out_index: Optional["OutputIndex"], title: Optional[str] = None) -> None:
        """Facts of a browser target. An out_index of this very folder (output mode, or an input-mode target that is
        the save folder) adds the free space and up to 8 stitched images with their sizes. title: the caller's name
        for the folder (the browser's label, e.g. 'Home'); default: its base name."""
        G = self.ctx.theme.G
        if summary is None:
            self._begin(title)
            self._line(f"Checking what's inside{G['ell']}", "Muted")
            return self._end((), "")
        self._begin(title or os.path.basename(summary.path.rstrip("\\/")) or summary.path)
        if summary.error is not None:
            self._line(summary.error.message, "Error")
        groups = summary.groups
        stitchable = list(stitchable_groups(summary).values())
        if stitchable:
            kind = f"{len(stitchable)} slides in one folder" if len(stitchable) > 1 else "Slide tiles"
            self._line(f"{kind} {G['dash']} {_n(sum(len(g.names) for g in stitchable), 'tile')}")
        elif groups:
            suffixes = ", ".join(sorted({g.suffix for g in groups}))
            self._line(f"Tiles, but not unmixed components ({suffixes})", "Warn")
        if summary.inform_export:
            self._line("inForm batch export", "Muted")
        if rollup is not None and rollup.slide_folders:
            more = "+" if rollup.capped else ""
            self._line(f"{_n(rollup.slide_folders, 'slide folder', more)} inside ({_n(rollup.tiles, 'tile', more)})")
        cut = " (listing cut short)" if summary.truncated else ""
        for count, one, many, tail in (
                (len(summary.outputs), "stitched image", "stitched images", " (.ome.tif)"),
                (len(summary.wsi), "whole-slide image", "whole-slide images", f" from another program ({G['ell']}"
                                                                                f"_stitch.tif)"),
                (summary.other_tiffs, "other TIFF image", "other TIFF images", ""),
                (len(summary.subdirs), "folder", "folders", cut)):
            if count:
                self._line(f"{count:,} {one if count == 1 else many}{tail}", "Muted")
        if summary.raw_scan:
            self._line(f"Raw scans (.qptiff) {G['dash']} unmix in inForm first", "Warn")
        if out_index is not None and _same_dir(out_index.out_dir, summary.path):
            if out_index.free_bytes is not None:
                self._line(f"Free space: {human_bytes(out_index.free_bytes)}")
            infos = sorted(out_index.outputs.values(), key=lambda i: _nk(i.name))
            for info in infos[:8]:
                self._line(f"{info.name} · {human_bytes(info.size)}", "Small")
            if len(infos) > 8:
                self._line(f"{G['ell']}and {len(infos) - 8} more", "Small")
        if not self.facts.winfo_children()[1:]:           # nothing below the title
            self._line("Empty folder" if summary.n_entries == 0 else "No tiles here", "Muted")
        main = stitchable[0] if stitchable else (groups[0] if groups else None)
        self._end(main.coords if main else (), f"Tile layout of {main.prefix} (from file names)" if main else "")

    def show_row(self, row: "PlanRow") -> None:
        """Facts of a plan row (§2.3.7); each part appears once it is known."""
        G, c, h, est = self.ctx.theme.G, row.cand, row.header, row.est
        self._begin(c.prefix if c.multi_slide_folder else c.folder_name)
        self._line(f"Folder: {c.folder_name if c.rel_folder == '.' else c.rel_folder}")
        if self._on_open:
            self._button("Open folder", lambda: self._on_open(c.folder))
        suffix = f"_{c.suffix}" if c.suffix else ""
        self._line("Every TIFF in this folder" if c.kind == "legacy_all" else f"Tiles: {c.prefix}_[x,y]{suffix}.tif",
                   "Muted")
        self._line(_n(len(c.tile_names), "tile") + (f" · {est.coverage:.0%} of the image covered" if est else ""))
        if h is not None:
            self._line(f"Tile {h.tile_w:,} × {h.tile_h:,} px · {h.dtype} · {h.um_per_px_x:.3f} µm/px")
        elif row.header_error:
            self._line(f"Can't read the first tile ({row.header_error})", "Error")
        else:
            self._line(f"Tile details appear once the first tile has been read{G['ell']}", "Muted")
        if est is not None:
            mode = ("fits in memory" if est.memory_mode == "ram"
                    else f"too big for memory: uses {human_bytes(est.raw_bytes, True)} of temporary disk space")
            self._line(f"Image {est.canvas_w:,} × {est.canvas_h:,} px · {human_bytes(est.raw_bytes, True)} · {mode}")
        output = row.output_stem + OME_SUFFIX
        if row.existing is not None and norm_key(row.existing.name) == norm_key(output):
            output = row.existing.name
        self._line(f"Output: {output}")
        for flag in row.flags:
            self._line(flag.text, {"warn": "Warn", "error": "Error"}.get(flag.severity, "Info"))
        if h is not None:
            names = list(row.channel_override or h.channels_extracted)[:h.n_channels]
            names += [f"Channel {i}" for i in range(len(names), h.n_channels)]
            colors = list(h.channel_colors) + [None] * h.n_channels
            for i, name in enumerate(names):
                self._channel(i, name, colors[i])
            if h.channel_problem and self._on_fix:
                self._button(f"Fix channel names{G['ell']}", lambda: self._on_fix([row.row_id]))
        tile_um = (h.tile_w * h.um_per_px_x, h.tile_h * h.um_per_px_y) if h else (None, None)
        self._end(c.coords, "Tile layout from file names (dashed: the stitched image)", *tile_um)

    # ---- building blocks
    def _begin(self, title: Optional[str]) -> None:
        for w in self.facts.winfo_children():
            w.destroy()
        head = ttk.Label(self.facts, text=self.ctx.theme.text(title or ""), style=f"{self._pre}.TLabel", font="YfsBold",
                         wraplength=self._wrap)
        head.pack(anchor="w", pady=(0, self.ctx.theme.px(4)) if title else 0)

    def _line(self, text: str, kind: str = "") -> None:
        style = f"{self._pre}.{kind}.TLabel" if kind else f"{self._pre}.TLabel"
        ttk.Label(self.facts, text=self.ctx.theme.text(text), style=style, wraplength=self._wrap,
                  justify="left").pack(anchor="w")

    def _channel(self, index: int, name: str, color: Optional[str]) -> None:
        px = self.ctx.theme.px
        holder = ttk.Frame(self.facts, style=self.cget("style"))
        dot = tk.Canvas(holder, width=px(10), height=px(10), background=self._bg, highlightthickness=0)
        dot.create_oval(1, 1, px(10) - 1, px(10) - 1, fill=color or T["text_faint"],
                        outline=T["border_strong"] if color else "")       # a white channel stays visible
        dot.pack(side="left", padx=(px(2), px(6)))
        ttk.Label(holder, text=self.ctx.theme.text(f"{index}  {name}"), style=f"{self._pre}.TLabel").pack(side="left")
        holder.pack(anchor="w")

    def _button(self, label: str, cmd: Callable[[], None]) -> None:
        ttk.Button(self.facts, text=self.ctx.theme.text(label), command=cmd,
                   style="Link.TButton" if self._pre == "Surface" else "LinkBg.TButton").pack(anchor="w")

    def _end(self, coords: Sequence[Tuple[float, float]], caption: str, tile_w_um: Optional[float] = None,
             tile_h_um: Optional[float] = None) -> None:
        self.map.show(coords, tile_w_um, tile_h_um)
        self.caption.configure(text=self.ctx.theme.text(caption) if coords else "")
        if coords:
            self.drawing.grid()
        else:
            self.drawing.grid_remove()
