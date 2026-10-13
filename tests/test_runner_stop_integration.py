from __future__ import annotations

import threading

import pytest

from tests.fakes import FakeEngine
from tests.test_runner import make_plan
from yfs.core.events import RowFinished, RowStarted
from yfs.core.runner import BatchRunner, RunOptions


@pytest.mark.parametrize('mode', ['now', 'after_current'])
def test_stop_acceptance_is_bound_to_the_published_job(tmp_path, mode):
    plan = make_plan(tmp_path / 'src', tmp_path / 'out')
    seen, decisions = [], []
    first, second, third = [r.row_id for r in plan.rows()]

    def post(event):
        if isinstance(event, RowStarted):
            seen.append(event.row_id)
            if event.row_id == second:
                decisions.append(batch.request_stop(mode, expected_row=first))
                assert batch.stopping is None and not batch._cancel.is_set()
                decisions.append(batch.request_stop(mode, expected_row=second))
        elif isinstance(event, RowFinished) and event.row_id == first:
            assert batch.current_row is None
            assert not batch.request_stop(mode, expected_row=first)

    batch = BatchRunner(1, plan, FakeEngine.configure(), post, RunOptions(str(tmp_path / 'out'), summary_csv=False))
    batch.start()
    batch.join(3)
    assert not batch.is_alive()
    assert decisions == [False, True] and seen == [first, second]
    assert plan.row(third).status == 'new'
    assert plan.row(second).status == ('cancelled' if mode == 'now' else 'done')
    assert not batch.request_stop(mode, expected_row=None)


def test_stop_and_claim_share_one_transition_lock(tmp_path, monkeypatch):
    plan = make_plan(tmp_path / 'src', tmp_path / 'out')
    entered, release, requested = threading.Event(), threading.Event(), threading.Event()
    retire = threading.Event()
    claim = plan.claim_next
    answers = []

    def delayed_claim(*args):
        entered.set()
        assert release.wait(2)
        return claim(*args)

    monkeypatch.setattr(plan, 'claim_next', delayed_claim)
    def post(event):
        if isinstance(event, RowStarted) and event.row_id == plan.rows()[0].row_id:
            assert retire.wait(2)

    batch = BatchRunner(1, plan, FakeEngine.configure(), post,
                        RunOptions(str(tmp_path / 'out'), summary_csv=False))

    def stop():
        requested.set()
        answers.append(batch.request_stop('now', expected_row=None))

    batch.start()
    assert entered.wait(2)
    requester = threading.Thread(target=stop)
    requester.start()
    assert requested.wait(1)
    release.set()
    requester.join(2)
    retire.set()
    batch.join(3)
    assert not batch.is_alive() and not requester.is_alive()
    assert answers == [False] and batch.stopping is None
    assert all(row.status == 'done' for row in plan.rows())
