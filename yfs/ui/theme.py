"""Visual system: colour tokens, fonts, glyphs, interface scale and the named ttk styles. FINAL_SPEC §4."""
from __future__ import annotations

import os
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from functools import wraps
from tkinter import ttk
from typing import Any, Callable, Dict, Optional, Tuple

try:                                    # WP1's osutil (xrdb with a cleaned environment)
    from yfs.core.osutil import linux_xft_dpi
except ImportError:  # pragma: no cover - only while osutil.py is being written (X11 then uses winfo_fpixels)
    linux_xft_dpi = None

TOKENS: Dict[str, str] = dict(
    bg="#F3F5F7", surface="#FFFFFF", surface_alt="#F7F8FA", sidebar="#EEF1F4", section="#EEF1F4",
    border="#D6DBE2", border_strong="#B6BFCA", trough="#E3E7EC",
    text="#1D2530", text_muted="#556170", text_faint="#8792A0",
    accent="#2E7D32", accent_hover="#276C2B", accent_pressed="#1B5E20", accent_disabled="#A9C8AB",
    accent_tint="#E8F3E9",
    select_bg="#DCEBFF", select_fg="#0B3D91", focus="#1F6FEB", running_bg="#EEF4FD",
    ok="#1E6B30", ok_bg="#E7F3EA", ok_line="#A9D3B3", warn="#7A4F00", warn_bg="#FFF4D6", warn_icon="#E0A100",
    error="#B3261E", error_bg="#FDECEA", error_line="#F2B8B5",
    info="#0B5CAD", info_bg="#E8F1FB", info_line="#B9D3F2",
    folder="#8DA0B4", folder_tab="#7A8EA3", icon_ink="#4B5868", place_ink="#6E8197",
    btn="#EEF0F3", btn_hover="#E6EAEE", btn_pressed="#DDE2E8", btn_light="#FAFBFC", btn_dark="#E1E5EA",
    flat_hover="#E6EBF0", flat_pressed="#D9E0E7")
LOGO_COLORS = ("#3F6FE0", "#22B05C", "#E2B308", "#E5484D", "#F07F1E", "#9B5DE5")   # logo tiles only
T = TOKENS

# Treeview tag options (§4.6 table) and log-drawer tags (§4.9), shared by the widgets that use them.
TREE_TAGS: Dict[str, Dict[str, Dict[str, str]]] = {
    "Plan": {"section": dict(background=T["section"], font="YfsBold"),
             "running": dict(background=T["running_bg"], font="YfsBold"), "dim": dict(foreground=T["text_muted"]),
             "warn": dict(foreground=T["warn"]), "error": dict(foreground=T["error"]),
             "note": dict(foreground=T["text_muted"])},
    "Listing": {"slide": dict(font="YfsBold"), "dim": dict(foreground=T["text_muted"]),
                "hidden": dict(foreground=T["text_faint"])},
    "Sidebar": {"section": dict(foreground=T["text_muted"], font="YfsSmallBold"),
                "dim": dict(foreground=T["text_faint"])}}
LOG_TAGS: Dict[str, Dict[str, str]] = {
    "time": dict(foreground=T["text_faint"]), "slide": dict(font="YfsMonoBold"), "info": dict(foreground=T["text"]),
    "ok": dict(foreground=T["ok"]), "warn": dict(foreground=T["warn"], background=T["warn_bg"]),
    "error": dict(foreground=T["error"], background=T["error_bg"], font="YfsMonoBold"),
    "debug": dict(foreground=T["text_faint"])}

_GLYPHS = {"ell": ("…", "..."), "sep": ("›", ">"), "arrow": ("→", "->"), "dash": ("—", "-"),
           "lq": ("‘", "'"), "rq": ("’", "'"), "ldq": ("“", '"'), "rdq": ("”", '"')}
_TO_ASCII = str.maketrans({"\u2026": "...", "\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2018": "'", "\u2019": "'",
                           "\u201c": '"', "\u201d": '"', "\u2248": "~", "\u2713": "v", "\u2717": "x"})
_FONTS = (("YfsBody", 0, "normal"), ("YfsBold", 0, "bold"), ("YfsSmall", -1, "normal"), ("YfsSmallBold", -1, "bold"),
          ("YfsH2", 1, "bold"), ("YfsTitle", 5, "bold"), ("YfsButton", 2, "bold"), ("YfsWelcome", 4, "bold"),
          ("YfsMono", -1, "normal"), ("YfsMonoBold", -1, "bold"))
_TK_BODY_FONTS = ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkCaptionFont", "TkTooltipFont", "TkIconFont",
                  "TkSmallCaptionFont")


def glyphs(xft: bool) -> Dict[str, str]:
    """§4.3 glyph table: Unicode where the font system can draw it, ASCII on core-font Tk."""
    return {k: v[0] if xft else v[1] for k, v in _GLYPHS.items()}


def display_text(s: str, xft: bool) -> str:
    """§4.3 display-only text: drop U+FE0F and non-BMP; core fonts also map to ASCII and '?' above U+00FF."""
    s = "".join(ch for ch in s if ch != "\ufe0f" and ord(ch) <= 0xFFFF)
    if xft:
        return s
    return "".join(ch if ord(ch) <= 0xFF else "?" for ch in s.translate(_TO_ASCII))


@dataclass
class Theme:
    root: tk.Tk; scale: float; xft: bool; ws: str; icon_px: int; rowheight: int
    fonts: Dict[str, tkfont.Font]; G: Dict[str, str]

    def px(self, n: float) -> int:
        """Logical px -> device px (§4.4)."""
        return max(1, int(round(n * self.scale)))

    def text(self, s: str) -> str:
        """§4.3 display-safe text."""
        return display_text(s, self.xft)


def theme_of(widget: tk.Misc) -> Optional[Theme]:
    """The Theme installed on the widget's Tk root (for widgets that are not handed an AppContext)."""
    return getattr(widget._root(), "_yfs_theme", None)


def detect_xft(root: tk.Tk) -> bool:
    """True when Tk can draw Unicode text: always off X11; on X11 only with the Xft font system (§4.2)."""
    if root.tk.call("tk", "windowingsystem") != "x11":
        return True
    try:
        return root.tk.call("::tk::pkgconfig", "get", "fontsystem") == "xft"
    except tk.TclError:                                  # conda's noxft Tk has no ::tk::pkgconfig
        return any(f != f.lower() for f in tkfont.families(root))   # core-X family names are all lower-case


def auto_scale(root: tk.Tk, ws: str) -> float:
    """§4.4 automatic scale: macOS 1.0; else dpi / 96 rounded to 0.25 and clamped to 1.0-3.0 (X2Go reports 75 dpi)."""
    if ws == "aqua":
        return 1.0
    dpi = linux_xft_dpi() if ws == "x11" and linux_xft_dpi is not None else None
    dpi = dpi or root.winfo_fpixels("1i")
    return min(3.0, max(1.0, round(dpi / 96.0 * 4) / 4))


def _pick(candidates: Tuple[str, ...], families: Tuple[str, ...]) -> Optional[str]:
    low = {f.lower(): f for f in families}
    return next((low[c.lower()] for c in candidates if c.lower() in low), None)


def _font_families(root: tk.Tk, ws: str, xft: bool) -> Tuple[str, str, int]:
    """(body, mono, base points) per the §4.2 table."""
    fams = tuple(tkfont.families(root))
    body = tkfont.Font(root=root, name="TkDefaultFont", exists=True).actual("family")
    mono = tkfont.Font(root=root, name="TkFixedFont", exists=True).actual("family")
    if ws == "win32":
        return _pick(("Segoe UI",), fams) or body, _pick(("Consolas",), fams) or mono, 10
    if ws == "aqua":
        return body, _pick(("Menlo",), fams) or mono, 13
    if xft:
        return (_pick(("Noto Sans", "DejaVu Sans", "Liberation Sans", "Ubuntu", "Cantarell"), fams) or body,
                _pick(("DejaVu Sans Mono", "Noto Sans Mono", "Liberation Mono", "Ubuntu Mono"), fams) or mono, 10)
    return _pick(("helvetica", "nimbus sans l"), fams) or body, _pick(("courier", "nimbus mono l"), fams) or mono, 10


def _named_font(root: tk.Tk, name: str, family: str, size: int, weight: str) -> tkfont.Font:
    if name not in tkfont.names(root):
        root.tk.call("font", "create", name)             # owned by Tcl, so it outlives Python references
    font = tkfont.Font(root=root, name=name, exists=True)
    font.configure(family=family, size=size, weight=weight)
    return font


def _install_fonts(root: tk.Tk, ws: str, xft: bool) -> Dict[str, tkfont.Font]:
    """§4.2 named fonts, in points (they follow tk scaling, which must already be set)."""
    body, mono, base = _font_families(root, ws, xft)
    fonts = {name: _named_font(root, name, mono if name.startswith("YfsMono") else body, base + d, w)
             for name, d, w in _FONTS}
    for name in _TK_BODY_FONTS:
        _named_font(root, name, body, base, "normal")
    _named_font(root, "TkHeadingFont", body, base - 1, "bold")
    _named_font(root, "TkFixedFont", mono, base - 1, "normal")
    return fonts


def install_theme(root: tk.Tk, user_scale: Optional[float]) -> Theme:
    """§4.2-§4.7 on a fresh root: scale, fonts, clam styles, plain-Tk options. Main thread, right after tk.Tk()."""
    ws = str(root.tk.call("tk", "windowingsystem"))
    scale = max(0.5, min(4.0, float(user_scale))) if user_scale else auto_scale(root, ws)
    if ws != "aqua":
        root.tk.call("tk", "scaling", 96.0 * scale / 72.0)
    elif user_scale:                                     # aqua works in points: 1.0 is the native size
        root.tk.call("tk", "scaling", scale)
    xft = detect_xft(root)
    fonts = _install_fonts(root, ws, xft)
    icon_px = 16 if scale < 1.25 else 20 if scale < 1.5 else 24 if scale < 1.75 else 32
    theme = Theme(root, scale, xft, ws, icon_px, 0, fonts, glyphs(xft))
    theme.rowheight = max(fonts["YfsBody"].metrics("linespace") + theme.px(10), icon_px + theme.px(8))
    _styles(root, theme)
    _plain_tk_options(root)
    root._yfs_theme = theme

    def reapply(event: Any) -> None:                     # every style change broadcasts <<ThemeChanged>>
        style = ttk.Style(root)
        if event.widget is root and str(style.lookup("Treeview", "rowheight")) != str(theme.rowheight):
            _styles(root, theme)                         # only when something reset the explicit metrics
    root.bind("<<ThemeChanged>>", reapply, add="+")
    return theme


def install_indicators(root: tk.Misc, icons: Any) -> None:
    """Draw ttk check boxes with the IconSet's tick boxes (cb_on/cb_off/cb_mixed) instead of clam's cross (BR-24).
    Main thread, once, right after the IconSet exists (app._run, the GUI test fixtures, tools/shots.py); the IconSet's
    cache keeps the images alive. Surface.TCheckbutton inherits the layout."""
    st = ttk.Style(root)
    if "Yfs.Checkbutton.indicator" in st.element_names():
        return
    on, off, mixed = (icons.get(n) for n in ("cb_on", "cb_off", "cb_mixed"))
    dim = {n: icons.get(n, dim=True) for n in ("cb_on", "cb_off", "cb_mixed")}
    st.element_create("Yfs.Checkbutton.indicator", "image", off,            # the first matching state wins
                      ("disabled", "alternate", dim["cb_mixed"]), ("disabled", "selected", dim["cb_on"]),
                      ("disabled", dim["cb_off"]), ("alternate", mixed), ("selected", on),
                      sticky="w", width=icons.size + theme_of(root).px(6))   # the width makes the gap to the label
    label = [("Checkbutton.label", {"sticky": "nswe"})]
    st.layout("TCheckbutton", [("Checkbutton.padding", {"sticky": "nswe", "children": [
        ("Yfs.Checkbutton.indicator", {"side": "left", "sticky": ""}),
        ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": label})]})])


def tag_tree(tree: ttk.Treeview, kind: str) -> None:
    """Configure the §4.6 tags of a 'Plan', 'Listing' or 'Sidebar' tree."""
    for tag, opts in TREE_TAGS[kind].items():
        tree.tag_configure(tag, **opts)


def _styles(root: tk.Tk, theme: Theme) -> None:
    """The normative §4.6 style block."""
    px, rowh, icon_px = theme.px, theme.rowheight, theme.icon_px
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=T["bg"], foreground=T["text"], bordercolor=T["border"], lightcolor=T["surface"],
                 darkcolor=T["border"], troughcolor=T["trough"], focuscolor=T["focus"], selectbackground=T["select_bg"],
                 selectforeground=T["select_fg"], fieldbackground=T["surface"], insertcolor=T["text"], font="YfsBody",
                 arrowcolor=T["icon_ink"], arrowsize=px(12))
    st.map(".", foreground=[("disabled", T["text_faint"])])
    # ---- frames
    st.configure("Surface.TFrame", background=T["surface"])
    st.configure("Header.TFrame", background=T["surface"])
    st.configure("Sidebar.TFrame", background=T["sidebar"])
    st.configure("Card.TFrame", background=T["surface"], relief="solid", borderwidth=1, bordercolor=T["border"])
    for k, bg, line in (("Info", "info_bg", "info_line"), ("Warn", "warn_bg", "warn_icon"), ("Ok", "ok_bg", "ok_line"),
                        ("Error", "error_bg", "error_line")):
        st.configure(f"{k}.TFrame", background=T[bg], relief="solid", borderwidth=1, bordercolor=T[line])
        st.configure(f"{k}.Banner.TLabel", background=T[bg], foreground=T[k.lower()] if k != "Info" else "#123A63")
        st.configure(f"{k}.Banner.TButton", background=T[bg], foreground=T["info"], bordercolor=T[bg],
                     lightcolor=T[bg], darkcolor=T[bg], padding=(px(6), px(2)))          # link-like action in a banner
        st.map(f"{k}.Banner.TButton", foreground=[("active", T["focus"])], background=[("active", T[bg])])
    # ---- labels (Surface.* on white cards, Bg.* on the window background)
    for pre, bgk in (("Surface", "surface"), ("Bg", "bg"), ("Sidebar", "sidebar")):
        st.configure(f"{pre}.TLabel", background=T[bgk])
        st.configure(f"{pre}.Muted.TLabel", background=T[bgk], foreground=T["text_muted"])
        st.configure(f"{pre}.Small.TLabel", background=T[bgk], foreground=T["text_muted"], font="YfsSmall")
        for kind in ("Ok", "Warn", "Error", "Info"):
            st.configure(f"{pre}.{kind}.TLabel", background=T[bgk], foreground=T[kind.lower()])
    st.configure("Title.TLabel", background=T["surface"], font="YfsTitle")
    st.configure("Tagline.TLabel", background=T["surface"], foreground=T["text_muted"], font="YfsSmall")
    st.configure("Step.TLabel", background=T["surface"], font="YfsH2")
    st.configure("StepNum.TLabel", background=T["text_muted"], foreground="#FFFFFF", font="YfsBold", anchor="center",
                 padding=(px(6), px(1)))
    st.configure("Section.TLabel", background=T["sidebar"], foreground=T["text_muted"], font="YfsSmallBold")
    st.configure("Welcome.TLabel", background=T["surface"], font="YfsWelcome")
    st.configure("Placeholder.TLabel", background=T["surface"], foreground=T["text_faint"])
    st.configure("Footer.TLabel", background=T["bg"], foreground=T["text_muted"], font="YfsSmall")
    st.configure("Pill.Warn.TLabel", background=T["warn_bg"], foreground=T["warn"], font="YfsSmallBold", padding=(px(6), 0))
    st.configure("Pill.Error.TLabel", background=T["error_bg"], foreground=T["error"], font="YfsSmallBold",
                 padding=(px(6), 0))
    # ---- buttons
    st.configure("TButton", padding=(px(10), px(4)), background=T["btn"], bordercolor=T["border_strong"],
                 lightcolor=T["btn_light"], darkcolor=T["btn_dark"], width=0)
    st.map("TButton", background=[("disabled", "#F1F3F5"), ("pressed", T["btn_pressed"]), ("active", T["btn_hover"])],
           bordercolor=[("focus", T["focus"])])
    for name, font, pad in (("Accent.TButton", "YfsButton", (px(18), px(7))),     # Stitch only
                            ("Primary.TButton", "YfsBold", (px(12), px(4)))):     # dialog/browser primary actions
        st.configure(name, font=font, padding=pad, background=T["accent"], foreground="#FFFFFF",
                     bordercolor=T["accent_pressed"], lightcolor=T["accent"], darkcolor=T["accent"], focuscolor="#FFFFFF")
        states = [("disabled", T["accent_disabled"]), ("pressed", T["accent_pressed"]), ("active", T["accent_hover"])]
        st.map(name, background=states, lightcolor=states, darkcolor=states,
               foreground=[("disabled", "#F3F7F3")], bordercolor=[("disabled", T["accent_disabled"])])
    for name, bg in (("Flat.TButton", T["surface"]), ("FlatBg.TButton", T["bg"]), ("Crumb.TButton", T["surface"])):
        st.configure(name, padding=(px(6), px(3)), background=bg, bordercolor=bg, lightcolor=bg, darkcolor=bg,
                     relief="flat", width=0)
        st.map(name, background=[("pressed", T["flat_pressed"]), ("active", T["flat_hover"])],
               bordercolor=[("active", T["flat_hover"])], lightcolor=[("active", T["flat_hover"])],
               darkcolor=[("active", T["flat_hover"])])
    st.configure("CrumbCurrent.TButton", font="YfsBold", padding=(px(6), px(3)), background=T["surface"],
                 bordercolor=T["surface"], lightcolor=T["surface"], darkcolor=T["surface"], width=0)
    st.map("CrumbCurrent.TButton", background=[("active", T["flat_hover"])])
    for name, bg in (("Link.TButton", T["surface"]), ("LinkBg.TButton", T["bg"])):
        st.configure(name, padding=(px(2), px(1)), background=bg, foreground=T["info"], bordercolor=bg,
                     lightcolor=bg, darkcolor=bg, width=0)
        st.map(name, foreground=[("disabled", T["text_faint"]), ("active", T["focus"])], background=[("active", bg)])
    st.configure("Danger.TButton", foreground=T["error"])
    st.configure("TMenubutton", padding=(px(10), px(4)), arrowsize=px(4), arrowpadding=px(6), background=T["btn"],
                 bordercolor=T["border_strong"], lightcolor=T["btn_light"], darkcolor=T["btn_dark"])
    st.map("TMenubutton", background=[("pressed", T["btn_pressed"]), ("active", T["btn_hover"])])
    st.configure("Crumb.TMenubutton", padding=(px(6), px(3)), arrowsize=0, arrowpadding=0, background=T["surface"],
                 bordercolor=T["surface"], lightcolor=T["surface"], darkcolor=T["surface"])
    st.map("Crumb.TMenubutton", background=[("active", T["flat_hover"])])
    # ---- entries, checks
    st.configure("TEntry", padding=(px(6), px(4)), bordercolor=T["border_strong"], lightcolor=T["surface"],
                 darkcolor=T["surface"])
    st.map("TEntry", bordercolor=[("focus", T["focus"])], lightcolor=[("focus", T["focus"])])
    st.configure("TCombobox", padding=(px(6), px(4)), arrowsize=px(12), bordercolor=T["border_strong"],
                 lightcolor=T["surface"], darkcolor=T["surface"])
    st.map("TCombobox", bordercolor=[("focus", T["focus"])], fieldbackground=[("readonly", T["surface"])])
    st.configure("TSpinbox", padding=(px(6), px(4)), arrowsize=px(10))
    for w in ("TCheckbutton", "TRadiobutton"):
        st.configure(w, indicatorsize=px(13), indicatormargin=(0, 0, px(6), 0), indicatorbackground=T["surface"],
                     indicatorforeground=T["accent"], upperbordercolor=T["border_strong"],
                     lowerbordercolor=T["border_strong"])
        st.map(w, indicatorbackground=[("pressed", T["accent_tint"])])
        st.configure(f"Surface.{w}", background=T["surface"])
    # ---- trees (item layouts without Treeitem.indicator; verified)
    st.configure("Treeview", rowheight=rowh, background=T["surface"], fieldbackground=T["surface"], foreground=T["text"],
                 bordercolor=T["border"], lightcolor=T["surface"], darkcolor=T["surface"], indent=px(16), font="YfsBody")
    st.map("Treeview", background=[("selected", T["select_bg"])], foreground=[("selected", T["select_fg"])])
    #   ^ replaces the default map, which also neutralises the Tk 8.6.9 tag-colour bug
    st.configure("Treeview.Heading", font="YfsSmallBold", background=T["surface_alt"], foreground=T["text_muted"],
                 padding=(px(6), px(4)), relief="flat", bordercolor=T["border"], lightcolor=T["surface_alt"],
                 darkcolor=T["border"])
    st.map("Treeview.Heading", background=[("active", "#EDF0F3")])
    noind = [("Treeitem.padding", {"sticky": "nswe", "children": [
        ("Treeitem.image", {"side": "left", "sticky": ""}),
        ("Treeitem.focus", {"side": "left", "sticky": "", "children": [("Treeitem.text", {"side": "left", "sticky": ""})]})]})]
    for s in ("Plan", "Listing", "Sidebar"):
        st.layout(f"{s}.Treeview.Item", noind)
    st.configure("Plan.Treeview", indent=icon_px + max(2, icon_px // 4))      # tick boxes align under the section's
    st.configure("Listing.Treeview", indent=0)
    st.configure("Sidebar.Treeview", background=T["sidebar"], fieldbackground=T["sidebar"], bordercolor=T["sidebar"],
                 lightcolor=T["sidebar"], darkcolor=T["sidebar"], indent=0)
    focus = [("focus", T["focus"])]
    for s in ("Listing", "Sidebar"):                    # the browser's lists show where the keyboard is (BR-21)
        st.map(f"{s}.Treeview", bordercolor=focus, lightcolor=focus, darkcolor=focus)
    # ---- progress bars (clam: thickness == arrowsize), scrollbars, panes
    # clam draws no bar at all below arrowsize 5 and shows arrowsize-2 coloured rows (edges + fill), or arrowsize-4
    # when the edges take the trough colour (measured on both Tk builds): Slim gets px(4) rows, Thin px(2).
    for name, size, color, trough, edge in (("Big", px(10), T["accent"], T["trough"], T["accent"]),
                                            ("Slim", px(4) + 2, T["focus"], T["trough"], T["focus"]),
                                            ("Thin", px(2) + 4, T["focus"], T["surface"], T["surface"])):
        st.configure(f"{name}.Horizontal.TProgressbar", arrowsize=size, background=color, troughcolor=trough,
                     bordercolor=trough, lightcolor=edge, darkcolor=edge)
    st.configure("TScrollbar", arrowsize=px(12), background="#E3E7EC", troughcolor=T["surface_alt"],
                 bordercolor=T["surface_alt"], lightcolor="#E3E7EC", darkcolor="#E3E7EC", gripcount=0,
                 arrowcolor=T["text_muted"])
    st.map("TScrollbar", background=[("active", "#CBD2DA"), ("pressed", T["border_strong"])])
    st.configure("TPanedwindow", background=T["bg"])
    st.configure("Sash", sashthickness=px(6))
    st.configure("TSeparator", background=T["border"])
    # ---- clam's own state maps would leak its beige palette (#dcdad5/#eeebe7/#bab5ab) and turn a focused
    #      read-only combobox's text white on white; replace them with tokens (verified with style.lookup)
    st.map(".", background=[], selectbackground=[("!focus", T["border"])], selectforeground=[("!focus", T["text"])])
    st.map("TCombobox", background=[("pressed", T["btn_pressed"]), ("active", T["btn_hover"])],
           foreground=[("disabled", T["text_faint"])], arrowcolor=[("disabled", T["text_faint"])])
    st.map("TEntry", background=[], darkcolor=[("focus", T["focus"])])
    st.map("TSpinbox", background=[], arrowcolor=[("disabled", T["text_faint"])])
    for name, bg in (("TButton", T["btn_pressed"]), ("Link.TButton", T["surface"]), ("LinkBg.TButton", T["bg"]),
                     *((f"{k}.Banner.TButton", T[f"{k.lower()}_bg"]) for k in ("Info", "Warn", "Ok", "Error"))):
        st.map(name, lightcolor=[("pressed", bg)], darkcolor=[("pressed", bg)])


def _plain_tk_options(root: tk.Tk) -> None:
    """§4.7: colours of plain Tk widgets, set in the option database before any widget exists."""
    root.configure(background=T["bg"])
    for k, v in (("*Text.background", T["surface"]), ("*Text.foreground", T["text"]), ("*Text.selectBackground",
                 T["select_bg"]), ("*Text.selectForeground", T["select_fg"]), ("*Text.insertBackground", T["text"]),
                 ("*Text.borderWidth", 0), ("*Listbox.background", T["surface"]), ("*Listbox.foreground", T["text"]),
                 ("*Listbox.selectBackground", T["select_bg"]), ("*Listbox.selectForeground", T["select_fg"]),
                 ("*Listbox.font", "YfsBody"), ("*Listbox.borderWidth", 1), ("*Menu.background", T["surface"]),
                 ("*Menu.foreground", T["text"]), ("*Menu.activeBackground", T["select_bg"]),
                 ("*Menu.activeForeground", T["select_fg"]), ("*Menu.font", "YfsBody"), ("*Menu.tearOff", 0),
                 ("*TCombobox*Listbox.font", "YfsBody"), ("*Canvas.background", T["surface"]),
                 ("*Canvas.highlightThickness", 0)):
        root.option_add(k, v)


def ui_thread(fn: Callable) -> Callable:
    """With YFS_DEBUG=1, assert that a UI method runs on the main thread (§5.2); a no-op otherwise.
    Lives here (re-exported by app.py) so every UI module can use it without importing app.py."""
    if os.environ.get("YFS_DEBUG") != "1":
        return fn

    @wraps(fn)
    def checked(*args: Any, **kwargs: Any) -> Any:
        assert threading.current_thread() is threading.main_thread(), f"{fn.__qualname__} called off the Tk thread"
        return fn(*args, **kwargs)
    return checked
