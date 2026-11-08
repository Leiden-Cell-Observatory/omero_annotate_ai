"""Resume repairs: the decision table, and a crash at every write step (fakes only)."""

import numpy as np
import pandas as pd
import pytest
import tifffile

from omero_annotate_ai import provenance
from omero_annotate_ai import session as session_module
from omero_annotate_ai.annotators import ScriptedAnnotator
from omero_annotate_ai.bundle import file_digest
from omero_annotate_ai.errors import TableWriteError
from omero_annotate_ai.reconcile import Record, decide
from tests.fakes import (
    Crash,
    Upload,
    annotate,
    create_set,
    image_mask,
    omero_set,
    reopen,
)
from tests.test_session import edited, masks_for

DONE = Record(processed=True, label_id=5, roi_id=6)
ACCEPTED = Record(processed=True, label_id=None, roi_id=None)
PENDING = Record(processed=False, label_id=None, roi_id=None)
UPLOADED = Record(processed=True, label_id=5, roi_id=None)
EXCLUDED = Record(processed=True, label_id=None, roi_id=None, excluded=True, session=2)
# The same session (2) annotated the unit after excluding it; or another one (1).
ACCEPTED_AFTER = Record(processed=True, label_id=None, roi_id=None, session=2)
UPLOADED_AFTER = Record(processed=True, label_id=5, roi_id=None, session=2)
UPLOADED_BEFORE = Record(processed=True, label_id=5, roi_id=None, session=1)


@pytest.mark.unit
class TestDecide:
    @pytest.mark.parametrize(
        ("remote", "local", "mask", "local_known", "step"),
        [
            (DONE, DONE, True, True, "mirror"),
            (DONE, None, False, False, "mirror"),
            (PENDING, UPLOADED, True, True, "verify"),
            (PENDING, ACCEPTED, True, True, "find_tagged"),
            (PENDING, PENDING, True, True, "find_tagged"),  # crash before accept
            (PENDING, None, False, False, "find_tagged"),  # local copy lost
            (PENDING, ACCEPTED, False, True, "reset"),
            (PENDING, PENDING, False, True, "nothing"),
            (None, PENDING, False, True, "nothing"),
            (EXCLUDED, EXCLUDED, False, True, "excluded"),
            (EXCLUDED, None, False, False, "excluded"),
            (EXCLUDED, EXCLUDED, True, True, "find_tagged"),  # crash before accept
            (EXCLUDED, ACCEPTED_AFTER, True, True, "find_tagged"),
            (EXCLUDED, UPLOADED_AFTER, True, True, "verify"),
            (EXCLUDED, UPLOADED_BEFORE, True, True, "excluded"),
            (PENDING, EXCLUDED, False, True, "exclude"),  # table write failed
            (PENDING, EXCLUDED, True, True, "find_tagged"),  # and then annotated
            (None, EXCLUDED, False, True, "exclude"),
            (DONE, EXCLUDED, False, True, "mirror"),
        ],
    )
    def test_steps(self, remote, local, mask, local_known, step):
        assert decide(remote, local, mask, local_known) == step


def run_first(tmp_path, n_images=3, crash=None, **kwargs):
    """A new set on the fakes; ``crash`` fires during its first annotation."""
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    ts = create_set(config, source, omero, tmp_path / "home", **kwargs)
    if crash:
        omero.crash = crash
    return ts, source, omero


def resume(ts, source, omero, events=None, **kwargs):
    """Open the set again (it is repaired) and annotate what is left."""
    reopened = reopen(ts, source, omero, events=events, **kwargs)
    annotate(reopened)
    return reopened


def assert_consistent(omero, pipeline, table_id, n_units):
    """No duplicate uploads, every unit published, table rows == local CSV rows."""
    assert all(n == (1, 1) for n in omero.uploads_per_tag().values())
    assert len(omero.uploads_per_tag()) == n_units
    table = omero.tables[table_id].set_index("annotation_id").sort_index()
    local = pd.read_csv(pipeline.bundle_dir / "tracking_table.csv").set_index(
        "annotation_id"
    )
    local = local.sort_index()
    assert table["processed"].all()
    for column in ("processed", "label_id", "roi_id", "mask_digest"):
        assert list(table[column]) == list(local[column]), column


@pytest.mark.unit
class TestCrashMatrix:
    """Kill the process after each write step of the second unit, then resume."""

    @pytest.mark.parametrize(
        "crash",
        [
            {"label": 2},  # mask accepted locally, nothing uploaded
            {"label_after": 2},  # label posted, its ID never saved
            {"roi": 2},  # label saved, ROI not posted
            {"roi_after": 2},  # ROI posted, its ID never saved
            {"table": 1},  # every upload done, table never updated
        ],
    )
    def test_resume_repairs_without_duplicates(self, tmp_path, crash):
        first, source, omero = run_first(tmp_path, crash=crash)
        with pytest.raises(Crash):
            annotate(first)
        table_id = first.table_id
        omero.crash = {}

        pipeline = resume(first, source, omero)

        assert_consistent(omero, pipeline, table_id, 3)

    def test_crash_between_mask_move_and_accept(self, tmp_path, monkeypatch):
        first, source, omero = run_first(tmp_path)
        real = session_module.file_digest
        calls = {"n": 0}

        def dying_digest(path):
            calls["n"] += 1
            if calls["n"] == 2:
                raise Crash("after step 1")
            return real(path)

        monkeypatch.setattr(session_module, "file_digest", dying_digest)
        with pytest.raises(Crash):
            annotate(first)
        monkeypatch.setattr(session_module, "file_digest", real)

        pipeline = resume(first, source, omero)

        assert_consistent(omero, pipeline, first.table_id, 3)

    def test_reconcile_twice_changes_nothing(self, tmp_path):
        first, source, omero = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            annotate(first)
        omero.crash = {}
        pipeline = resume(first, source, omero)
        uploads = dict(omero.uploads_per_tag())

        assert pipeline.reconcile() is False
        assert omero.uploads_per_tag() == uploads


@pytest.mark.unit
class TestRepairs:
    def _finished(self, tmp_path, n_images=2):
        first, source, omero = run_first(tmp_path, n_images=n_images)
        annotate(first)
        return first, source, omero, first.table_id

    def test_edited_local_mask_is_reported_and_kept_off_omero(self, tmp_path):
        first, source, omero, table_id = self._finished(tmp_path)
        unit = first.rows[0]
        mask = first.bundle_dir / "annotation_output" / f"{unit.annotation_id}_mask.tif"
        tifffile.imwrite(mask, np.full((16, 16), 9, np.uint16))
        events = []

        resume(first, source, omero, events)

        divergent = [e for e in events if getattr(e, "action", "") == "divergent"]
        assert [e.annotation_id for e in divergent] == [unit.annotation_id]
        assert len(omero.labels) == 2
        # The fakes' masks are uniform planes, which all share one ISCC code: the
        # message then rests on the digest alone (no near-duplicate claim).
        assert "ISCC" not in divergent[0].detail

    def test_upload_that_differs_from_the_local_mask_is_not_adopted(self, tmp_path):
        first, source, omero = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            annotate(first)
        omero.crash = {}
        unit = first.rows[0]
        label_id = next(
            i for i, u in omero.labels.items() if u.tag.endswith(unit.annotation_id)
        )
        omero.labels[label_id].pixels = np.zeros((16, 16), np.uint16)
        events = []

        resume(first, source, omero, events)

        row = omero.tables[first.table_id].set_index("annotation_id")
        assert not row.loc[unit.annotation_id, "processed"]
        assert any(getattr(e, "action", "") == "divergent" for e in events)
        assert len(omero.labels) == 3  # not annotated and uploaded a second time

    def test_deleting_the_local_mask_accepts_the_upload(self, tmp_path):
        first, source, omero = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            annotate(first)
        omero.crash = {}
        unit = first.rows[0]
        label_id = next(
            i for i, u in omero.labels.items() if u.tag.endswith(unit.annotation_id)
        )
        omero.labels[label_id].pixels = np.zeros((16, 16), np.uint16)
        resume(first, source, omero)  # flags it for review
        (
            first.bundle_dir / "annotation_output" / f"{unit.annotation_id}_mask.tif"
        ).unlink()

        pipeline = resume(first, source, omero)

        row = omero.tables[first.table_id].set_index("annotation_id")
        assert row.loc[unit.annotation_id, "processed"]
        assert row.loc[unit.annotation_id, "label_id"] == label_id
        assert_consistent(omero, pipeline, first.table_id, 3)

    @pytest.mark.parametrize("prune", [False, True])
    def test_duplicate_uploads_are_reported_or_pruned(self, tmp_path, prune):
        first, source, omero = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            annotate(first)
        omero.crash = {}
        (first.bundle_dir / "tracking_table.csv").unlink()  # force a tag lookup
        unit = first.rows[0]
        label_id = next(
            i for i, u in omero.labels.items() if u.tag.endswith(unit.annotation_id)
        )
        copy_id = max(omero.labels) + 1
        omero.labels[copy_id] = omero.labels[label_id]
        events = []
        reopen(first, source, omero, events=events, prune_duplicates=prune)

        action = "pruned" if prune else "duplicate"
        assert [e.action for e in events if getattr(e, "action", "") == action] == [
            action
        ]
        assert (copy_id in omero.labels) is not prune
        row = omero.tables[first.table_id].set_index("annotation_id")
        assert row.loc[unit.annotation_id, "label_id"] == label_id

    def test_rows_only_the_local_copy_has_are_added_back(self, tmp_path):
        first, source, omero, table_id = self._finished(tmp_path, n_images=2)
        table = omero.tables[table_id]
        dropped = table.iloc[-1]["annotation_id"]
        omero.tables[table_id] = table.iloc[:-1]

        resume(first, source, omero)

        assert dropped in set(omero.tables[table_id]["annotation_id"])


@pytest.mark.unit
class TestExcludedUnits:
    def _excluded(self, tmp_path, n_images=2):
        """A set whose first unit is excluded, the others annotated."""
        first, source, omero = run_first(tmp_path, n_images=n_images)
        session = first.start_session()
        batch = session.next_batch()
        bad = batch.items[0].annotation_id
        session.exclude(batch, [bad])
        session.commit(batch, masks_for(batch, set(range(1, n_images))))
        session.finish()
        return first, source, omero, bad

    def test_excluded_unit_is_left_alone(self, tmp_path):
        first, source, omero, bad = self._excluded(tmp_path)
        before = omero.tables[first.table_id].copy()
        events = []

        reopened = reopen(first, source, omero, events=events)

        assert not [e for e in events if hasattr(e, "action")]
        assert next(r for r in reopened.rows if r.annotation_id == bad).excluded
        pd.testing.assert_frame_equal(omero.tables[first.table_id], before)
        assert reopened.reconcile() is False

    def test_tagged_upload_of_another_session_is_reported_not_adopted(self, tmp_path):
        first, source, omero, bad = self._excluded(tmp_path)
        unit = next(r for r in first.rows if r.annotation_id == bad)
        assert unit.image_id is not None
        orphan = max(omero.labels) + 1
        omero.labels[orphan] = Upload(
            unit.image_id, first.upload_tag(bad), np.ones((16, 16), np.uint16)
        )
        (first.bundle_dir / "tracking_table.csv").unlink()  # force a tag lookup
        events = []

        reopened = reopen(first, source, omero, events=events)

        reported = [e for e in events if getattr(e, "action", "") == "excluded"]
        assert [e.annotation_id for e in reported] == [bad]
        assert f"label {orphan}" in reported[0].detail
        row = omero.tables[first.table_id].set_index("annotation_id").loc[bad]
        assert row["annotation_type"] == "excluded" and row["label_id"] == 0
        assert next(r for r in reopened.rows if r.annotation_id == bad).excluded

    def test_exclusion_the_table_missed_is_written_on_resume(self, tmp_path):
        first, source, omero = run_first(tmp_path, n_images=2)
        session = first.start_session()
        batch = session.next_batch()
        bad = batch.items[0].annotation_id
        omero.fail = {"table": omero.calls.get("table", 0) + 1}
        with pytest.raises(TableWriteError):
            session.exclude(batch, [bad])
        omero.fail = {}

        reopened = resume(first, source, omero)

        row = omero.tables[first.table_id].set_index("annotation_id").loc[bad]
        assert row["processed"] and row["annotation_type"] == "excluded"
        assert row["claimed_by"] == ""
        assert next(r for r in reopened.rows if r.annotation_id == bad).excluded
        assert first.upload_tag(bad) not in omero.uploads_per_tag()

    @pytest.mark.parametrize(
        "crash", ["label", "label_after", "roi", "roi_after", "table"]
    )
    def test_crash_in_a_commit_after_exclude(self, tmp_path, crash):
        first, source, omero = run_first(tmp_path, n_images=2)
        session = first.start_session()
        batch = session.next_batch()
        unit = batch.items[0].annotation_id
        session.exclude(batch, [unit])
        omero.crash = {crash: omero.calls.get(crash, 0) + 1}
        with pytest.raises(Crash):
            session.commit(batch, masks_for(batch, {0}))
        omero.crash = {}

        pipeline = resume(first, source, omero)

        row = next(r for r in pipeline.rows if r.annotation_id == unit)
        assert not row.excluded and row.annotation_type == "segmentation_mask"
        assert_consistent(omero, pipeline, first.table_id, 2)
        table = omero.tables[first.table_id].set_index("annotation_id")
        assert table.loc[unit, "annotation_type"] == "segmentation_mask"


@pytest.mark.unit
@pytest.mark.skipif(not provenance.available(), reason="needs the provenance extra")
class TestTableWithoutIsccColumns:
    """A table made by 0.5.0 has no ISCC columns; the codes live in the local CSV."""

    def test_opening_and_refreshing_keep_the_local_codes(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=2)
        omero.missing_columns = ("input_iscc", "mask_iscc")
        ts = create_set(config, source, omero, tmp_path / "home")
        annotate(ts)
        codes = {r.annotation_id: (r.input_iscc, r.mask_iscc) for r in ts.rows}
        assert all(a and b for a, b in codes.values())
        assert "mask_iscc" not in omero.tables[ts.table_id].columns

        reopened = reopen(ts, source, omero)
        reopened.refresh()

        again = reopened.read_local_table() or {}
        for annotation_id, (input_iscc, mask_iscc) in codes.items():
            assert again[annotation_id].input_iscc == input_iscc
            assert again[annotation_id].mask_iscc == mask_iscc


class TestReviewCrashes:
    """Kill the process at each step of replacing a reviewed mask, then resume."""

    def _reviewed_with_crash(self, tmp_path, crash):

        first, source, omero = run_first(tmp_path, n_images=1)
        annotate(first)
        omero.crash = {name: omero.calls.get(name, 0) + 1 for name in crash}
        with pytest.raises(Crash):
            first.review(ScriptedAnnotator(image_mask, review_fn=edited))
        omero.crash = {}
        return first, source, omero

    @pytest.mark.parametrize(
        "crash",
        [
            ["label_after"],  # new label posted, its ID never saved
            ["roi"],  # new label saved, ROI not posted
            ["roi_after"],  # new ROI posted, its ID never saved
            ["delete"],  # new uploads saved, old ones not deleted
            ["table"],  # old ones deleted, table never updated
        ],
    )
    def test_resume_keeps_the_newer_upload_only(self, tmp_path, crash):
        first, source, omero = self._reviewed_with_crash(tmp_path, crash)

        reopened = reopen(first, source, omero)

        row = reopened.rows[0]
        assert omero.uploads_per_tag() == {first.upload_tag(row.annotation_id): (1, 1)}
        assert omero.labels[row.label_id].pixels[0, 0] == 0  # the reviewed mask
        table = omero.tables[first.table_id].set_index("annotation_id")
        mask = first.bundle_dir / "annotation_output" / f"{row.annotation_id}_mask.tif"
        assert table.loc[row.annotation_id, "mask_digest"] == file_digest(mask)
        assert table.loc[row.annotation_id, "label_id"] == row.label_id
        assert table.loc[row.annotation_id, "roi_id"] == row.roi_id

    def test_crash_before_the_upload_keeps_omero_and_reports(self, tmp_path):
        first, source, omero = self._reviewed_with_crash(tmp_path, ["label"])
        events = []

        reopened = reopen(first, source, omero, events=events)

        row = reopened.rows[0]
        assert omero.uploads_per_tag() == {first.upload_tag(row.annotation_id): (1, 1)}
        assert omero.labels[row.label_id].pixels[0, 0] != 0  # OMERO's is kept
        assert any(getattr(e, "action", "") == "divergent" for e in events)
