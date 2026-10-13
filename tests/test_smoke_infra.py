"""Sanity checks for the shared test infrastructure itself."""
from yfs.core import events, model


def test_synth_tree_has_expected_layout(synth_tree):
    assert (synth_tree / "ProjectA" / "SlideA1_Scan1").is_dir()
    assert len(list((synth_tree / "ProjectA" / "SlideA1_Scan1").glob("*_component_data.tif"))) == 9
    assert (synth_tree / "inform_export").is_dir()


def test_collector_records_from_threads(collector):
    import threading
    ts = [threading.Thread(target=collector, args=(events.LogLine(0.0, 20, "", str(i), "info"),)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(collector.wait_type(events.LogLine, count=20)) == 20


def test_model_is_frozen():
    e = model.Entry("a", False, False)
    try:
        e.name = "b"
    except Exception:
        return
    raise AssertionError("model dataclasses must be frozen")
