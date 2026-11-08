"""Claims: two computers annotating one OMERO training set (fakes only).

Both "computers" are the same OMERO user; they differ in their host ID, which
the tests switch with :func:`on`.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import tifffile

from omero_annotate_ai import claims
from omero_annotate_ai.events import ClaimsHeld, ClaimsReleased, UnitConflict
from omero_annotate_ai.sources.uri import omero_uri
from tests.fakes import Crash, annotate, create_set, image_mask, omero_set, reopen
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

    def test_a_batch_that_fails_to_load_frees_its_claims(
        self, tmp_path, on, monkeypatch
    ):
        a, _, _, omero = two_computers(tmp_path, on)
        on("a")
        session = a.start_session()

        def broken(*args, **kwargs):
            raise OSError("image unreadable")

        monkeypatch.setattr(a, "prepare_item", broken)
        with pytest.raises(OSError):
            session.next_batch()

        assert (table(omero, a)["claimed_by"] == "").all()

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


@pytest.mark.unit
class TestConcurrentWrites:
    """Two computers writing at the same moment (the fakes' ``hooks``)."""

    def test_claim_writes_never_write_back_a_concurrent_commit(self, tmp_path, on):
        a, b, source, omero = two_computers(tmp_path, on)
        on("a")
        a.start_session(batch_size=1).next_batch()  # then a's kernel dies
        on("b")
        session_b = b.start_session(batch_size=1)
        batch_b = session_b.next_batch()
        unit_b = batch_b.items[0].annotation_id

        def b_commits():  # between a's read and a's write of the claim cells
            on("b")
            session_b.commit(batch_b, masks_for(batch_b))
            on("a")

        on("a")
        omero.hooks["update"] = b_commits
        reopen(a, source, omero)  # releases a's leftover claim (reads b's row too)

        row = table(omero, a).loc[unit_b]
        assert bool(row["processed"]) and row["label_id"] is not None
        assert row["claimed_by"] == ""
        assert (table(omero, a)["claimed_by"] == "").all()

    def test_a_claim_taken_meanwhile_is_not_overwritten(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on, n_images=1)
        (unit,) = [r.annotation_id for r in a.rows]

        def b_claims():
            on("b")
            claims.claim(b, "9@tester/b", [unit], 1)
            on("a")

        on("a")
        session_a = a.start_session()
        omero.hooks["slice"] = b_claims
        assert session_a.next_batch() is None
        assert table(omero, a).loc[unit, "claimed_by"] == "9@tester/b"

    def test_a_note_on_another_computers_session_is_kept(self, tmp_path, on):
        a, b, source, omero = two_computers(tmp_path, on)
        on("a")
        session_a = a.start_session(batch_size=1)
        session_a.finish("from a")
        on("b")
        b.set_session_note(session_a.session, "edited on b")

        on_omero = omero.read_config(None, a.table_id)
        assert [s.note for s in on_omero.sessions] == ["edited on b"]

    def test_sessions_started_at_once_get_different_numbers(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on)
        started = {}

        def a_starts():  # after b read the sessions on OMERO, before b published
            on("a")
            started["a"] = a.start_session()
            on("b")

        on("b")
        omero.hooks["read_config"] = a_starts
        session_b = b.start_session()

        on_omero = omero.read_config(None, a.table_id)
        assert session_b.session != started["a"].session
        assert sorted((s.id, s.host) for s in on_omero.sessions) == sorted(
            [(started["a"].session, "a"), (session_b.session, "b")]
        )

    def test_counts_raised_elsewhere_are_not_reverted(self, tmp_path, on):
        config, source, omero = omero_set(tmp_path, n_images=6)
        config.training.train_n = 3
        on("a")
        a = create_set(config, source, omero, tmp_path / "a")
        on("b")
        b = reopen(a, source, omero)
        b.home = tmp_path / "b"
        for ts in (a, b):
            ts.get_image_uris = lambda: [omero_uri(i) for i in sorted(source.arrays)]
            ts.get_images_by_ids = lambda wanted: [source.get_image(u) for u in wanted]
        assert b.extend(train_n=5) == 2

        on("a")
        annotate(a, limit=1)  # publishes a's config, which still says 3
        assert omero.read_config(None, a.table_id).training.train_n == 5
        assert a.extend(train_n=6) == 1  # plans against b's rows and counts

        uris = list(omero.tables[a.table_id]["image_uri"])
        assert len(uris) == 6 == len(set(uris))

    def test_failed_and_conflicting_units_are_unclaimed(self, tmp_path, on):
        a, _, _, omero = two_computers(tmp_path, on, n_images=2)
        on("a")
        omero.fail = {"label": omero.calls.get("label", 0) + 1}
        session = a.start_session()
        batch = session.next_batch()
        report = session.commit(batch, masks_for(batch))
        omero.fail = {}

        assert len(report.failed) == 1
        assert (table(omero, a)["claimed_by"] == "").all()
        again = a.start_session().next_batch()  # the same process, no reopening
        assert [i.annotation_id for i in again.items] == list(report.failed)

    def test_collect_skips_units_saved_elsewhere(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on, n_images=2)
        first, second = (r.annotation_id for r in a.rows)
        on("b")
        session_b = b.start_session(batch_size=1)
        batch_b = session_b.next_batch()
        assert batch_b.items[0].annotation_id == first
        session_b.commit(batch_b, masks_for(batch_b))
        label_b = table(omero, a).loc[first, "label_id"]

        on("a")  # a's rows are stale: it still thinks both are pending
        a.prepare_external()
        for unit in (first, second):
            tifffile.imwrite(
                a.bundle_dir / "external_output" / f"{unit}_mask.tif",
                np.ones((16, 16), np.uint16),
            )
        assert a.collect_masks() == 1

        assert table(omero, a).loc[first, "label_id"] == label_b
        assert omero.uploads_per_tag()[a.upload_tag(first)] == (1, 1)
        assert bool(table(omero, a).loc[second, "processed"])

    def _crash_after_uploads(self, tmp_path, on):
        """a uploads a unit's label and ROI, then dies before its table write."""
        a, b, _, omero = two_computers(tmp_path, on, n_images=1)
        (unit,) = [r.annotation_id for r in a.rows]
        on("a")
        omero.crash = {"table": omero.calls.get("table", 0) + 1}
        with pytest.raises(Crash):
            annotate(a)
        omero.crash = {}
        long_ago = (datetime.now(UTC) - timedelta(hours=7)).isoformat()
        omero.tables[a.table_id].loc[:, "claimed_at"] = long_ago
        on("b")
        session = b.start_session()
        return b, session, session.next_batch(), unit, omero

    def test_tagged_upload_of_a_crashed_session_is_adopted(self, tmp_path, on):
        b, session, batch, unit, omero = self._crash_after_uploads(tmp_path, on)
        label_a = next(iter(omero.labels))
        masks = batch.mask_dir
        masks.mkdir(parents=True, exist_ok=True)
        path = masks / "seg_00000.tif"
        tifffile.imwrite(path, image_mask(batch.items[0].pixels))  # same as a drew

        report = session.commit(batch, {unit: path})

        assert report.committed == (unit,)
        assert omero.uploads_per_tag() == {b.upload_tag(unit): (1, 1)}
        assert table(omero, b).loc[unit, "label_id"] == label_a

    def test_different_tagged_upload_is_reported_not_duplicated(self, tmp_path, on):
        b, session, batch, unit, omero = self._crash_after_uploads(tmp_path, on)

        path = batch.mask_dir / "seg_00000.tif"
        batch.mask_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, np.full((16, 16), 7, np.uint16))  # not what a drew
        report = session.commit(batch, {unit: path})

        assert report.failed == (unit,)
        assert omero.uploads_per_tag() == {b.upload_tag(unit): (1, 1)}
        row = next(r for r in b.rows if r.annotation_id == unit)
        assert row.error.startswith("needs review") and not row.processed


@pytest.mark.unit
class TestOwnClaimsNeverBlock:
    """Claims never block the same user on the same computer."""

    def _dropped(self, tmp_path, on):
        """a's session claims every unit, then dies (napari crash, lost connection)
        before committing anything; nothing releases the claims."""
        a, _, source, omero = two_computers(tmp_path, on, n_images=2)
        on("a")
        a.start_session().next_batch()
        assert (table(omero, a)["claimed_by"] != "").all()
        return a, source, omero

    def test_crash_mid_upload_then_restart_finishes_the_set(self, tmp_path, on):
        a, _, source, omero = two_computers(tmp_path, on, n_images=2)
        on("a")
        omero.crash = {"label": omero.calls.get("label", 0) + 1}  # connection gone
        with pytest.raises(Crash):
            annotate(a)
        omero.crash = {}

        restarted = reopen(a, source, omero)  # repairs the accepted unit, frees both
        annotate(restarted)

        rows = table(omero, a)
        assert rows["processed"].all() and (rows["claimed_by"] == "").all()
        assert all(n == (1, 1) for n in omero.uploads_per_tag().values())

    def test_restart_releases_and_takes_own_claims(self, tmp_path, on):
        a, source, omero = self._dropped(tmp_path, on)

        restarted = reopen(a, source, omero)  # a new process on the same computer

        assert (table(omero, a)["claimed_by"] == "").all()
        assert annotate(restarted) == 2

    def test_same_object_takes_own_claims_without_reopening(self, tmp_path, on):
        a, _, omero = self._dropped(tmp_path, on)

        batch = a.start_session().next_batch()  # a new session, same TrainingSet

        assert len(batch.items) == 2
        assert annotate(a) == 2

    def test_review_takes_own_claims_too(self, tmp_path, on):
        a, _, _, _ = two_computers(tmp_path, on, n_images=2)
        on("a")
        annotate(a)
        a.start_session(review=True).next_batch()  # then an exception: no finish

        batch = a.start_session(review=True).next_batch()

        assert len(batch.items) == 2

    def test_other_computers_claims_still_block(self, tmp_path, on):
        a, b, _, omero = two_computers(tmp_path, on, n_images=2)
        on("a")
        a.start_session().next_batch()
        on("b")
        assert b.start_session().next_batch() is None
