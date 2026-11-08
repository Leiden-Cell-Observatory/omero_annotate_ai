"""Claims: two computers annotating one OMERO training set (fakes only).

Both "computers" are the same OMERO user; they differ in their host ID, which
the tests switch with :func:`on`.
"""

from datetime import UTC, datetime, timedelta

import pytest
import tifffile

from omero_annotate_ai import claims
from omero_annotate_ai.events import ClaimsHeld, ClaimsReleased, UnitConflict
from tests.fakes import Crash, annotate, create_set, omero_set, reopen
from tests.test_session import edited, masks_for


def edited_masks(batch):
    """A reviewer's corrections of every item's saved mask."""
    batch.mask_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for position, item in enumerate(batch.items):
        path = batch.mask_dir / f"seg_{position:05}.tif"
        tifffile.imwrite(path, edited(item.pixels, item.initial_mask))
        result[item.annotation_id] = path
    return result


@pytest.fixture
def on(monkeypatch):
    """``on("a")`` makes the following calls run on computer ``a``."""
    host = ["a"]
    monkeypatch.setattr(claims, "host_id", lambda: host[0])

    def switch(name):
        host[0] = name

    return switch


def two_computers(tmp_path, on, n_images=3, events_b=None):
    """The same set, created on computer a and opened on computer b."""
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    on("a")
    a = create_set(config, source, omero, tmp_path / "a")
    on("b")
    b = reopen(a, source, omero, events_b)
    b.home = tmp_path / "b"
    return a, b, source, omero


def table(omero, ts):
    return omero.tables[ts.table_id].set_index("annotation_id")


@pytest.mark.unit
class TestIsLive:
    def test_fresh_claim_is_live(self):
        assert claims.is_live("1@u/h", claims.now(), hours=1)

    def test_old_unreadable_or_empty_claims_are_free(self):
        old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert not claims.is_live("1@u/h", old, hours=1)
        assert not claims.is_live("1@u/h", "garbage", hours=1)
        assert not claims.is_live("", claims.now(), hours=1)


@pytest.mark.unit
class TestTwoComputers:
    def test_claimed_units_are_not_handed_out_twice(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on)
        on("a")
        session_a = a.start_session(batch_size=2)
        batch_a = session_a.next_batch()
        on("b")
        session_b = b.start_session(batch_size=2)
        batch_b = session_b.next_batch()

        ids_a = {i.annotation_id for i in batch_a.items}
        ids_b = {i.annotation_id for i in batch_b.items}
        assert len(ids_a) == 2 and len(ids_b) == 1 and not ids_a & ids_b
        assert session_b.next_batch() is None  # the rest is claimed by a

        on("a")
        session_a.commit(batch_a, masks_for(batch_a))
        on("b")
        session_b.commit(batch_b, masks_for(batch_b))

        rows = table(omero, a)
        assert rows["processed"].all()
        assert (rows["claimed_by"] == "").all()  # committing releases the claim
        assert all(n == (1, 1) for n in omero.uploads_per_tag().values())
        assert session_a.session != session_b.session

    def test_units_saved_elsewhere_are_taken_over_not_shown(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on, n_images=2)
        on("a")
        session_a = a.start_session()
        batch = session_a.next_batch()
        session_a.commit(batch, masks_for(batch, {0}))
        session_a.release(batch)

        on("b")
        batch_b = b.start_session().next_batch()

        assert [i.annotation_id for i in batch_b.items] == [
            batch.items[1].annotation_id
        ]
        done = next(
            r for r in b.rows if r.annotation_id == batch.items[0].annotation_id
        )
        assert done.processed and done.label_id is not None

    def test_release_and_finish_hand_the_claims_back(self, tmp_path, on):
        a, _, _, omero = two_computers(tmp_path, on)
        on("a")
        session = a.start_session(batch_size=1)
        first = session.next_batch()
        session.release(first)
        session.next_batch()  # left open
        assert (table(omero, a)["claimed_by"] != "").sum() == 1

        session.finish()

        assert (table(omero, a)["claimed_by"] == "").all()

    def test_expired_claim_is_taken_over_and_late_commit_is_a_conflict(
        self, tmp_path, on
    ):
        events = []
        a, b, _, omero = two_computers(tmp_path, on, n_images=1)
        on("a")
        session_a = a.start_session()
        batch_a = session_a.next_batch()
        unit = batch_a.items[0].annotation_id
        long_ago = (datetime.now(UTC) - timedelta(hours=7)).isoformat()
        omero.tables[a.table_id].loc[:, "claimed_at"] = long_ago  # a went quiet

        on("b")
        b.events = events.append
        session_b = b.start_session()
        batch_b = session_b.next_batch()
        assert batch_b is not None  # the expired claim is free
        session_b.commit(batch_b, masks_for(batch_b))
        label_b = table(omero, a).loc[unit, "label_id"]

        on("a")
        a.events = events.append
        report = session_a.commit(batch_a, masks_for(batch_a))

        assert report.conflicts == (unit,) and not report.committed
        assert table(omero, a).loc[unit, "label_id"] == label_b
        assert omero.uploads_per_tag() == {a.upload_tag(unit): (1, 1)}
        conflict = next(e for e in events if isinstance(e, UnitConflict))
        assert conflict.annotation_id == unit
        assert (a.bundle_dir / "annotation_output" / "conflicts").is_dir()
        assert next(r for r in a.rows if r.annotation_id == unit).label_id == label_b

    def test_crashed_session_claims_are_released_on_the_same_computer(
        self, tmp_path, on
    ):
        events_b = []
        a, _, source, omero = two_computers(tmp_path, on, n_images=2)
        on("a")
        session = a.start_session(batch_size=1)
        session.next_batch()
        omero.crash = {"label": omero.calls.get("label", 0) + 1}
        batch = session.next_batch()
        with pytest.raises(Crash):
            session.commit(batch, masks_for(batch))
        omero.crash = {}

        on("b")  # another computer: the claims are live, the units skipped
        b = reopen(a, source, omero, events_b)
        held = next(e for e in events_b if isinstance(e, ClaimsHeld))
        assert held.units == 2 and held.holders == ("tester/a",)
        assert b.start_session().next_batch() is None
        assert b.config.sessions[0].status == "open"  # a's session may still run

        on("a")  # the same computer releases its leftovers when opening
        events_a = []
        a2 = reopen(a, source, omero, events_a)
        assert any(isinstance(e, ClaimsReleased) and e.units == 2 for e in events_a)
        assert (table(omero, a)["claimed_by"] == "").all()
        assert a2.config.sessions[0].status == "interrupted"

    def test_sessions_of_both_computers_are_kept_on_omero(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on)
        on("a")
        session_a = a.start_session(batch_size=1)
        on("b")
        session_b = b.start_session(batch_size=1)
        on("a")
        session_a.finish("from a")
        on("b")
        session_b.finish("from b")

        on_omero = omero.read_config(None, a.table_id)
        assert {s.note for s in on_omero.sessions} == {"from a", "from b"}
        assert len({s.id for s in on_omero.sessions}) == 2

    def test_reviews_claim_too_and_a_newer_review_wins(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on, n_images=2)
        on("a")
        annotate(a)
        session_a = a.start_session(review=True, batch_size=1)
        batch_a = session_a.next_batch()

        on("b")
        session_b = b.start_session(review=True)
        batch_b = session_b.next_batch()  # b's copy is refreshed first
        assert [i.annotation_id for i in batch_b.items] != [
            i.annotation_id for i in batch_a.items
        ]
        assert len(batch_b.items) == 1  # a holds the other one
        item = batch_b.items[0]
        assert item.initial_mask is not None  # downloaded: b never annotated it

        report = session_b.commit(batch_b, edited_masks(batch_b))
        assert report.committed == (item.annotation_id,)
        assert all(n == (1, 1) for n in omero.uploads_per_tag().values())
