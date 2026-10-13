from __future__ import annotations

from dataclasses import replace

import pytest

from yfs.core.model import Entry
from yfs.core.names import candidates_from_summary, summarize_listing
from yfs.ui.widgets import SlideFacts


@pytest.mark.parametrize("duplicates", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_suffix_preview_counts_each_prefix_once(ui_ctx, duplicates, reverse):
    names = [f"BatchA_[{i * 100},0]_component_data.tif" for i in range(2)]
    names += [f"BatchB_[{i * 100},0].tif" for i in range(3)]
    if duplicates:
        names += [f"BatchA_[{i * 100},0].tif" for i in range(4)]
    summary = summarize_listing("/data/inform_data", [Entry(name, False, False) for name in names])
    if reverse:
        summary = replace(summary, groups=tuple(reversed(summary.groups)))
    facts = SlideFacts(ui_ctx.root, ui_ctx)
    facts.pack()
    facts.show_folder(summary, None, None)
    ui_ctx.root.update_idletasks()

    shown = [str(w.cget("text")) for w in facts.facts.winfo_children()]
    assert ui_ctx.theme.text("2 slides in one folder — 5 tiles") in shown
    first = "BatchB" if reverse else "BatchA"
    assert str(facts.caption.cget("text")) == f"Tile layout of {first} (from file names)"
    rectangles = [i for i in facts.map.find_all() if facts.map.type(i) == "rectangle"]
    assert len(rectangles) == (3 if reverse else 2) + 1    # first slide plus its canvas outline
    candidates = candidates_from_summary(summary, root="/data", depth=1, dir_mtime=None)
    assert len(candidates) == 2 and sum(len(c.tile_names) for c in candidates) == 5
