from __future__ import annotations

import os

import pytest

from yfs.core.model import Entry
from yfs.core.names import candidates_from_summary, summarize_listing
from yfs.core.plan import Plan
from yfs.ui.dialogs import ChannelNamesDialog, SlideDetailsDialog

from test_dialogs import buttons, ctx, find, header, texts, tile_names  # noqa: F401


PANELS = (("DAPI", "CD3", "CD4", "CD8", "AF"), ("DAPI", "CD20", "CD68", "FOXP3", "AF"))


def add_slide(plan, folder, n=12, names=None):
    summary = summarize_listing(os.path.join(plan.source, folder),
                                [Entry(name, False, False) for name in tile_names("Tiles", n)])
    cand = candidates_from_summary(summary, root=plan.source, depth=2, dir_mtime=None)[0]
    plan.add_candidate(cand)
    plan.apply_header(cand.key, header(None if names else "duplicate", names or ("DAPI",) * 5), None, 64 << 30)
    return cand.key


@pytest.mark.parametrize("source_index", [0, 1])
def test_channel_copy_identifies_twin_folders_and_saves_the_selected_panel(ctx, tk_root, source_index):
    plan = Plan("/data/ProjectC", "/data/out")
    source_ids = [add_slide(plan, folder + "/Scan1", names=names)
                  for folder, names in zip(("batchA", "batchB"), PANELS)]
    target = add_slide(plan, "NeedsNames")
    done = []
    ChannelNamesDialog.open(tk_root, ctx, plan, [target], done.append)
    dlg = find(tk_root, ChannelNamesDialog)
    labels = tuple(dlg.copy.cget("values"))
    assert len(set(labels)) == 2
    for folder in ("batchA", "batchB"):
        assert sum(folder in label for label in labels) == 1
    source = plan.row(source_ids[source_index])
    index = next(i for i, label in enumerate(labels) if source.cand.rel_folder.split("/")[0] in label)
    dlg.copy.current(index)
    dlg.copy.event_generate("<<ComboboxSelected>>")
    assert tuple(v.get() for v in dlg.vars) == PANELS[source_index]
    buttons(dlg)["Save names"].invoke()
    assert done == [True] and plan.row(target).channel_override == PANELS[source_index]


def test_channel_copy_adds_relative_folders_when_plan_labels_collide(ctx, tk_root):
    plan = Plan("/data/ProjectC", "/data/out")
    a = add_slide(plan, "batchA/Scan1", names=PANELS[0])
    add_slide(plan, "batchB/Scan1", names=PANELS[1])
    other = add_slide(plan, "Control", names=PANELS[1])
    assert plan.rename_output(other, "Reference") is None
    assert plan.rename_output(a, "Control") is None
    assert plan.label(plan.row(a)) == plan.label(plan.row(other)) == "Control"
    target = add_slide(plan, "NeedsNames")
    ChannelNamesDialog.open(tk_root, ctx, plan, [target], lambda _done: None)
    dlg = find(tk_root, ChannelNamesDialog)
    labels = tuple(dlg.copy.cget("values"))
    assert len(labels) == len(set(labels)) == 3
    assert any("Control" in label and "batchA/Scan1" in label for label in labels)
    buttons(dlg)["Cancel"].invoke()


def test_slide_details_rebinds_rescan_rename_and_save_folder_rebuild(ctx, tk_root):
    old = Plan("/data/ProjectC", "/data/out")
    row_id = add_slide(old, "Slide1", n=12)
    dlg = SlideDetailsDialog.show(tk_root, ctx, old, row_id, lambda _ids: None)
    old_canvas = next(t for t in texts(dlg) if t.startswith("Image "))
    assert any(t.startswith("12 tiles") for t in texts(dlg))

    current = Plan(old.source, old.out_dir)
    assert add_slide(current, "Slide1", n=13) == row_id
    SlideDetailsDialog.refresh_rows([row_id], plan=current)
    assert dlg.plan is current
    assert any(t.startswith("13 tiles") for t in texts(dlg))
    assert next(t for t in texts(dlg) if t.startswith("Image ")) != old_canvas
    assert current.rename_output(row_id, "NewSlideName") is None
    SlideDetailsDialog.refresh_rows(plan=current)             # inline edit without scan events
    assert "Output: NewSlideName.ome.tif" in texts(dlg)

    rebuilt = current.rebuild("/data/elsewhere")
    rebuilt.set_channels([row_id], PANELS[0])
    SlideDetailsDialog.refresh_rows([], plan=rebuilt)         # replacement is independent of changed row IDs
    assert dlg.plan is rebuilt and "2  CD4" in texts(dlg)
    assert "Output: NewSlideName.ome.tif" in texts(dlg)
    buttons(dlg)["Close"].invoke()


@pytest.mark.parametrize("cleared", [False, True])
def test_slide_details_closes_and_retires_actions_for_an_absent_row(ctx, tk_root, monkeypatch, cleared):
    plan = Plan("/data/ProjectC", "/data/out")
    row_id = add_slide(plan, "Slide1")
    fixes = []
    dlg = SlideDetailsDialog.show(tk_root, ctx, plan, row_id, fixes.append)
    queued_fix = dlg.facts._on_fix
    queued_open = dlg.facts._on_open
    opened = []
    monkeypatch.setattr("yfs.ui.dialogs.osutil.open_path", opened.append)
    replacement = None if cleared else Plan("/data/OtherProject", "/data/out")
    SlideDetailsDialog.refresh_rows([], plan=replacement)
    assert not dlg.winfo_exists() and SlideDetailsDialog.instance is None
    assert dlg.plan is None and dlg.on_fix is None
    queued_fix([row_id])
    queued_open(plan.row(row_id).cand.folder)
    assert fixes == opened == []
    assert dlg.facts._on_open is None and dlg.facts._on_fix is None
