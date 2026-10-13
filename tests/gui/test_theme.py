"""Visual system (FINAL_SPEC §4, acceptance D1): styles, font system, fonts, glyphs, scale and explicit metrics."""
from __future__ import annotations

import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from types import SimpleNamespace

import pytest

import yfs.ui.theme as theme_mod
from yfs.ui.icons import RECIPES, Canvas16
from yfs.ui.theme import (LOG_TAGS, TOKENS, TREE_TAGS, auto_scale, detect_xft, display_text, glyphs, install_theme,
                          tag_tree, theme_of, ui_thread)

KINDS = ("Info", "Warn", "Ok", "Error")
STYLES = (["Surface.TFrame", "Header.TFrame", "Sidebar.TFrame", "Card.TFrame"]
          + [f"{k}.{w}" for k in KINDS for w in ("TFrame", "Banner.TLabel", "Banner.TButton")]
          + [f"{p}{s}.TLabel" for p in ("Surface", "Bg", "Sidebar")
             for s in ("", ".Muted", ".Small", ".Ok", ".Warn", ".Error", ".Info")]
          + ["Title.TLabel", "Tagline.TLabel", "Step.TLabel", "StepNum.TLabel", "Section.TLabel", "Welcome.TLabel",
             "Placeholder.TLabel", "Footer.TLabel", "Pill.Warn.TLabel", "Pill.Error.TLabel",
             "TButton", "Accent.TButton", "Primary.TButton", "Flat.TButton", "FlatBg.TButton", "Crumb.TButton",
             "CrumbCurrent.TButton", "Link.TButton", "LinkBg.TButton", "Danger.TButton", "TMenubutton",
             "Crumb.TMenubutton", "TEntry", "TCombobox", "TSpinbox", "TCheckbutton", "TRadiobutton",
             "Surface.TCheckbutton", "Surface.TRadiobutton", "Treeview", "Treeview.Heading", "Plan.Treeview",
             "Listing.Treeview", "Sidebar.Treeview", "Big.Horizontal.TProgressbar", "Slim.Horizontal.TProgressbar",
             "Thin.Horizontal.TProgressbar", "TScrollbar", "TPanedwindow", "Sash", "TSeparator"])
WIDGET_FOR = {"TFrame": ttk.Frame, "TLabel": ttk.Label, "TButton": ttk.Button, "TMenubutton": ttk.Menubutton,
              "TEntry": ttk.Entry, "TCombobox": ttk.Combobox, "TSpinbox": ttk.Spinbox, "TCheckbutton": ttk.Checkbutton,
              "TRadiobutton": ttk.Radiobutton, "Treeview": ttk.Treeview, "TProgressbar": ttk.Progressbar,
              "TScrollbar": ttk.Scrollbar, "TPanedwindow": ttk.Panedwindow, "TSeparator": ttk.Separator}


def test_every_named_style_exists_and_instantiates(tk_root):
    st = ttk.Style(tk_root)
    assert st.theme_use() == "clam"
    for name in STYLES:
        assert st.configure(name), f"style {name} has no options"
        cls = name.rsplit(".", 1)[-1]
        if cls in WIDGET_FOR and name != "Treeview.Heading":
            WIDGET_FOR[cls](tk_root, style=name).destroy()
    for tree in ("Plan", "Listing", "Sidebar"):
        assert "Treeitem.indicator" not in str(st.layout(f"{tree}.Treeview.Item"))
        assert "Treeitem.image" in str(st.layout(f"{tree}.Treeview.Item"))


def test_no_clam_colour_leaks_through_any_state(tk_root):
    """Every colour a named style can take (in the states the UI uses) is a token or a literal of §4.6.
    clam's own maps would otherwise show #dcdad5/#eeebe7 backgrounds and white text in a focused read-only combobox."""
    st = ttk.Style(tk_root)
    allowed = {v.upper() for v in TOKENS.values()} | {"#FFFFFF", "#F1F3F5", "#123A63", "#EDF0F3", "#E3E7EC", "#CBD2DA",
                                                       "#F3F7F3"}
    options = ("background", "foreground", "fieldbackground", "bordercolor", "lightcolor", "darkcolor", "arrowcolor",
               "selectbackground", "selectforeground", "indicatorbackground", "indicatorforeground", "troughcolor")
    states = ([], ["active"], ["pressed"], ["pressed", "active"], ["focus"], ["disabled"], ["readonly"],
              ["readonly", "focus"], ["selected"], ["!focus"], ["disabled", "selected"])
    leaks = {(name, opt, " ".join(state), value) for name in STYLES + ["TLabel", "TFrame"] for opt in options
             for state in states for value in [str(st.lookup(name, opt, state))]
             if value.startswith("#") and value.upper() not in allowed}
    assert not leaks, sorted(leaks)[:10]
    assert str(st.lookup("TCombobox", "foreground", ["readonly", "focus"])).upper() == TOKENS["text"]


def test_xft_detection_matches_the_tk_build(tk_root, theme):
    heuristic = any(f != f.lower() for f in tkfont.families(tk_root))   # core-X family names are all lower-case
    try:
        pkg = tk_root.tk.call("::tk::pkgconfig", "get", "fontsystem") == "xft"
    except tk.TclError:
        pkg = None                                                      # conda's noxft Tk
    assert detect_xft(tk_root) is (heuristic if pkg is None else pkg)
    assert theme.xft is detect_xft(tk_root)
    if pkg is not None:
        assert pkg is heuristic
    assert theme.G == glyphs(theme.xft)


def test_fonts_follow_the_spec(tk_root, theme):
    fams = {f.lower() for f in tkfont.families(tk_root)}
    if theme.ws == "x11":
        cands = (["noto sans", "dejavu sans", "liberation sans", "ubuntu", "cantarell"] if theme.xft
                 else ["helvetica", "nimbus sans l"])
        expected = next((c for c in cands if c in fams), None)
        if expected:
            assert theme.fonts["YfsBody"].cget("family").lower() == expected
    base = 13 if theme.ws == "aqua" else 10
    want = {"YfsBody": (0, "normal"), "YfsBold": (0, "bold"), "YfsSmall": (-1, "normal"), "YfsSmallBold": (-1, "bold"),
            "YfsH2": (1, "bold"), "YfsTitle": (5, "bold"), "YfsButton": (2, "bold"), "YfsWelcome": (4, "bold"),
            "YfsMono": (-1, "normal"), "YfsMonoBold": (-1, "bold")}
    assert set(theme.fonts) == set(want)
    for name, (delta, weight) in want.items():
        assert (theme.fonts[name].cget("size"), theme.fonts[name].cget("weight")) == (base + delta, weight), name
        assert name in tkfont.names(tk_root)
    body, mono = theme.fonts["YfsBody"].cget("family"), theme.fonts["YfsMono"].cget("family")
    assert body != mono and theme.fonts["YfsMonoBold"].cget("family") == mono
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkCaptionFont", "TkTooltipFont", "TkIconFont",
                 "TkSmallCaptionFont"):
        f = tkfont.Font(root=tk_root, name=name, exists=True)
        assert (f.cget("family"), f.cget("size"), f.cget("weight")) == (body, base, "normal"), name
    head = tkfont.Font(root=tk_root, name="TkHeadingFont", exists=True)
    assert (head.cget("size"), head.cget("weight")) == (base - 1, "bold")
    assert tkfont.Font(root=tk_root, name="TkFixedFont", exists=True).cget("family") == mono
    assert min(f.cget("size") for f in theme.fonts.values()) >= 9


def test_rowheight_and_explicit_metrics(tk_root, theme):
    st, px, icon = ttk.Style(tk_root), theme.px, theme.icon_px
    assert theme.rowheight >= icon + px(8)
    assert theme.rowheight >= theme.fonts["YfsBody"].metrics("linespace") + px(10)
    assert int(st.lookup("Treeview", "rowheight")) == theme.rowheight
    assert int(st.lookup("Plan.Treeview", "indent")) == icon + max(2, icon // 4)
    assert int(st.lookup("Listing.Treeview", "indent")) == 0
    for name, size in (("TScrollbar", 12), ("TCombobox", 12), ("TMenubutton", 4), ("Big.Horizontal.TProgressbar", 10)):
        assert int(st.lookup(name, "arrowsize")) == px(size), name
    assert int(st.lookup("Slim.Horizontal.TProgressbar", "arrowsize")) == px(4) + 2     # see the next test
    assert int(st.lookup("Thin.Horizontal.TProgressbar", "arrowsize")) == px(2) + 4
    assert int(st.lookup("TCheckbutton", "indicatorsize")) == px(13)


@pytest.mark.parametrize("tk_root,icon,pbar", [(1.0, 16, 10), (1.25, 20, 12), (1.5, 24, 15), (2.0, 32, 20)],
                         indirect=["tk_root"])
def test_scale_drives_everything(tk_root, theme, icon, pbar):
    assert theme.icon_px == icon
    assert float(tk_root.tk.call("tk", "scaling")) == pytest.approx(96 * theme.scale / 72, rel=0.01)  # Tk rounds
    assert theme.px(10) == pbar and theme.px(0.1) == 1
    bar = ttk.Progressbar(tk_root, style="Big.Horizontal.TProgressbar", length=100)
    check = ttk.Checkbutton(tk_root, text="x", variable=tk.BooleanVar(tk_root))
    bar.pack(), check.pack()
    tk_root.update_idletasks()
    assert pbar <= bar.winfo_reqheight() <= pbar + 6                    # clam: thickness == arrowsize (+ border)
    assert check.winfo_reqheight() >= theme.px(13)


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_progress_bars_really_draw_their_bar(tk_root, theme, pump):
    """clam draws no bar below arrowsize 5: Slim (px(4)) and Thin (px(2)) bars must still show that many rows."""
    grab = pytest.importorskip("PIL.ImageGrab")
    bars = {}
    for name, holder in (("Big", "TFrame"), ("Slim", "TFrame"), ("Thin", "Surface.TFrame")):
        frame = ttk.Frame(tk_root, style=holder, padding=4)
        frame.pack(fill="x")
        bars[name] = ttk.Progressbar(frame, style=f"{name}.Horizontal.TProgressbar", value=60, length=200)
        bars[name].pack()
    pump(tk_root, 150)
    img = grab.grab(xdisplay=tk_root.winfo_screen() if theme.ws == "x11" else None).convert("RGB")
    for name, rows, color in (("Big", theme.px(10) - 2, "accent"), ("Slim", theme.px(4), "focus"),
                              ("Thin", theme.px(2), "focus")):
        bar = bars[name]
        rgb = tuple(int(TOKENS[color][i:i + 2], 16) for i in (1, 3, 5))
        ys = range(bar.winfo_rooty(), bar.winfo_rooty() + bar.winfo_height())
        assert sum(1 for y in ys if img.getpixel((bar.winfo_rootx() + 60, y)) == rgb) == rows, name   # 60 % filled
        assert all(img.getpixel((bar.winfo_rootx() + 190, y)) != rgb for y in ys), name


def test_each_root_starts_from_the_display_dpi(request, display_scaling, monkeypatch):
    """`tk scaling` rewrites the screen size kept in the process's one X connection, so a root used to inherit the
    scale of the root before it: test_icons failed right after this file (24 px icons at 'auto')."""
    monkeypatch.setattr(theme_mod, "linux_xft_dpi", lambda: None)        # auto scale from the screen alone
    own = auto_scale(SimpleNamespace(winfo_fpixels=lambda _inch: display_scaling * 72), "x11")
    other = tk.Tk()
    try:
        install_theme(other, 2.0 if own != 2.0 else 1.0)                # anything but the display's own scale
        other.update_idletasks()
    finally:
        other.destroy()                                                 # its DPI stays in the X connection
    root = request.getfixturevalue("tk_root")
    assert theme_of(root).scale == (own if theme_of(root).ws != "aqua" else 1.0)


def _rgb(token: str) -> tuple:
    return tuple(int(TOKENS[token][i:i + 2], 16) for i in (1, 3, 5))


def _screen(tk_root, grab):
    return grab.grab(xdisplay=tk_root.winfo_screen() if theme_of(tk_root).ws == "x11" else None).convert("RGB")


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_check_boxes_are_drawn_with_the_tick_box_icons(tk_root, icons, pump):
    """BR-24: a ticked box is the plan's white tick on green, not clam's cross; unticked, mixed and disabled boxes use
    the matching icons. Counted on screen: Tk draws an icon's opaque pixels exactly."""
    grab = pytest.importorskip("PIL.ImageGrab")
    frame = ttk.Frame(tk_root, style="Surface.TFrame", padding=10)
    frame.pack()
    boxes, keep = {}, []
    for name, value, states in (("on", True, ()), ("off", False, ()), ("mixed", False, ["alternate"]),
                                ("dim_on", True, ["disabled"]), ("dim_off", False, ["disabled"])):
        keep.append(tk.BooleanVar(tk_root, value))
        boxes[name] = ttk.Checkbutton(frame, text="Save a summary table (CSV)", variable=keep[-1],
                                      style="Surface.TCheckbutton")
        boxes[name].state(states)
        boxes[name].pack(anchor="w", pady=4)
    pump(tk_root, 150)
    img = _screen(tk_root, grab)

    def count(box, color):
        x, y, w, h = box.winfo_rootx(), box.winfo_rooty(), box.winfo_width(), box.winfo_height()
        return sum(img.getpixel((x + i, y + j)) == color for i in range(w) for j in range(h))

    def opaque(recipe, color):
        c = Canvas16(icons.size)
        RECIPES[recipe](c)
        a = c.rgba()
        return int(((a[..., 3] == 255) & (a[..., :3] == color).all(axis=-1)).sum())

    accent, border = _rgb("accent"), _rgb("border_strong")
    assert "Yfs.Checkbutton.indicator" in str(ttk.Style(tk_root).layout("TCheckbutton"))
    assert count(boxes["on"], accent) == opaque("cb_on", accent) > icons.size ** 2 // 3
    assert count(boxes["mixed"], accent) == opaque("cb_mixed", accent) > count(boxes["on"], accent)
    assert count(boxes["off"], accent) == 0 and count(boxes["off"], border) >= opaque("cb_off", border) > 0
    for name in ("dim_on", "dim_off"):                                 # dimmed: no full-strength colour at all
        assert count(boxes[name], accent) == count(boxes[name], border) == 0
    x, y = boxes["dim_on"].winfo_rootx(), boxes["dim_on"].winfo_rooty()
    pale = [img.getpixel((x + i, y + j)) for i in range(icons.size) for j in range(boxes["dim_on"].winfo_height())]
    assert any(g > r + 20 and g < 255 for r, g, _b in pale)            # a pale green tick box


@pytest.mark.parametrize("tk_root", [1.0, 1.5], indirect=True)
def test_browser_lists_show_where_the_keyboard_is(tk_root, pump):
    """BR-21: the listing and the places sidebar get a blue border while they have the keyboard focus."""
    grab = pytest.importorskip("PIL.ImageGrab")
    entry = ttk.Entry(tk_root)
    entry.pack()
    focus = _rgb("focus")
    for style in ("Listing.Treeview", "Sidebar.Treeview"):
        tree = ttk.Treeview(tk_root, style=style, show="tree", height=3)
        tree.insert("", "end", text="ProjectC")
        tree.pack(padx=10, pady=10)
        for owner, framed in ((entry, False), (tree, True)):
            owner.focus_force()
            pump(tk_root, 120)
            img = _screen(tk_root, grab)
            x, y, w, h = tree.winfo_rootx(), tree.winfo_rooty(), tree.winfo_width(), tree.winfo_height()
            edges = [(x + w // 2, y), (x, y + h // 2), (x + w - 1, y + h // 2), (x + w // 2, y + h - 1)]
            assert all((img.getpixel(p) == focus) is framed for p in edges), (style, framed)


def test_fonts_rows_and_bars_grow_with_scale(xdisplay, display_scaling):
    seen = []
    for scale in (1.0, 1.5, 2.0):
        root = tk.Tk()
        try:
            th = install_theme(root, scale)
            bar = ttk.Progressbar(root, style="Big.Horizontal.TProgressbar")
            bar.pack()
            root.update_idletasks()
            seen.append((th.fonts["YfsBody"].metrics("linespace"), th.rowheight, th.icon_px, bar.winfo_reqheight()))
        finally:
            root.update_idletasks()                     # run the pending <<ThemeChanged>> before destroying
            root.tk.call("tk", "scaling", display_scaling)
            root.destroy()
    for small, big in zip(seen, seen[1:]):
        assert all(b > s for s, b in zip(small, big)), seen


@pytest.mark.parametrize("dpi,expected", [(72, 1.0), (75, 1.0), (96, 1.0), (110, 1.25), (120, 1.25), (144, 1.5),
                                          (168, 1.75), (192, 2.0), (400, 3.0)])
def test_auto_scale_rounds_to_quarters_and_clamps(tk_root, monkeypatch, dpi, expected):
    monkeypatch.setattr(theme_mod, "linux_xft_dpi", lambda: dpi)
    assert auto_scale(tk_root, "x11") == expected
    assert auto_scale(tk_root, "aqua") == 1.0


def test_auto_scale_falls_back_to_the_screen_dpi(tk_root, monkeypatch):
    monkeypatch.setattr(theme_mod, "linux_xft_dpi", lambda: None)
    expected = min(3.0, max(1.0, round(tk_root.winfo_fpixels("1i") / 96 * 4) / 4))
    assert auto_scale(tk_root, "x11") == expected
    assert auto_scale(tk_root, "win32") == expected


def test_install_theme_uses_xft_dpi_when_no_override(xdisplay, display_scaling, monkeypatch):
    monkeypatch.setattr(theme_mod, "linux_xft_dpi", lambda: 144.0)
    root = tk.Tk()
    try:
        th = install_theme(root, None)
        if th.ws == "x11":
            assert th.scale == 1.5 and th.icon_px == 24
    finally:
        root.update_idletasks()
        root.tk.call("tk", "scaling", display_scaling)
        root.destroy()


def test_glyphs_and_display_text():
    assert glyphs(True) == {"ell": "\u2026", "sep": "\u203a", "arrow": "\u2192", "dash": "\u2014",
                            "lq": "\u2018", "rq": "\u2019", "ldq": "\u201c", "rdq": "\u201d"}
    assert glyphs(False) == {"ell": "...", "sep": ">", "arrow": "->", "dash": "-", "lq": "'", "rq": "'",
                             "ldq": '"', "rdq": '"'}
    warning, party, han = "\u26a0\ufe0f", "\U0001f389", "\u4e2d"              # emoji + VS16, non-BMP, CJK
    typo = "\u2026 \u2018a\u2019 \u201cb\u201d \u2014 \u2013 \u2192 \u2248 \u2713 \u2717"
    latin1 = "\u00b5m \u00d7 \u00b7 \u00a3"
    s = f"{warning} Searching{typo} {party} {latin1} {han}"
    assert display_text(s, True) == f"\u26a0 Searching{typo}  {latin1} {han}"   # Xft: only VS16 and non-BMP go
    out = display_text(s, False)
    assert out == f"? Searching... 'a' \"b\" - - -> ~ v x  {latin1} ?"
    assert all(ord(ch) <= 0xFF for ch in out)


def test_theme_text_and_px(theme):
    assert theme.text("a…") == ("a…" if theme.xft else "a...")
    assert theme.px(0) == 1 and theme.px(12) == max(1, round(12 * theme.scale))
    assert theme_of(theme.root) is theme


def test_plain_tk_widgets_take_token_colours(tk_root):
    text, lb, menu, canvas = tk.Text(tk_root), tk.Listbox(tk_root), tk.Menu(tk_root), tk.Canvas(tk_root)
    opt = lambda w, name: str(w.cget(name)).upper()     # noqa: E731
    assert opt(text, "background") == TOKENS["surface"] and opt(text, "selectbackground") == TOKENS["select_bg"]
    assert opt(lb, "selectforeground") == TOKENS["select_fg"] and opt(lb, "font") == "YFSBODY"
    assert int(menu.cget("tearoff")) == 0 and opt(menu, "activebackground") == TOKENS["select_bg"]
    assert int(canvas.cget("highlightthickness")) == 0 and opt(canvas, "background") == TOKENS["surface"]
    assert opt(tk_root, "background") == TOKENS["bg"]


def test_tree_and_log_tag_tables(tk_root):
    for kind in TREE_TAGS:
        tree = ttk.Treeview(tk_root, style=f"{kind}.Treeview")
        tag_tree(tree, kind)
        for tag, opts in TREE_TAGS[kind].items():
            for key, value in opts.items():
                assert str(tree.tag_configure(tag, key)) == value
    assert set(LOG_TAGS) == {"time", "slide", "info", "ok", "warn", "error", "debug"}


def test_explicit_metrics_are_reapplied_on_theme_changed(tk_root, theme, pump):
    st = ttk.Style(tk_root)
    pump(tk_root, 100)                                  # mapped: broadcasts before the first map are not delivered
    st.configure("Treeview", rowheight=5)               # something resets a metric ...
    pump(tk_root, 100)                                  # ... the idle <<ThemeChanged>> broadcast restores it
    assert int(st.lookup("Treeview", "rowheight")) == theme.rowheight
    pump(tk_root, 100)                                  # and settles (no endless re-styling)
    assert int(st.lookup("Treeview", "rowheight")) == theme.rowheight


def test_ui_thread_asserts_off_the_main_thread_only_in_debug(monkeypatch):
    monkeypatch.delenv("YFS_DEBUG", raising=False)
    plain = lambda: 1                                   # noqa: E731
    assert ui_thread(plain) is plain
    monkeypatch.setenv("YFS_DEBUG", "1")
    checked = ui_thread(lambda: 2)
    assert checked() == 2
    errors = []

    def worker():
        try:
            checked()
        except AssertionError as exc:
            errors.append(str(exc))
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors and "off the Tk thread" in errors[0]
