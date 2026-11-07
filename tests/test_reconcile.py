"""Resume repairs: the decision table, and a crash at every write step (fakes only)."""

import numpy as np
import pandas as pd
import pytest
import tifffile

from omero_annotate_ai import session as session_module
from omero_annotate_ai.reconcile import Record, decide, reconcile
from tests.fakes import Crash
from tests.test_pipeline import omero_pipeline, omero_set, resume_of, source_images

DONE = Record(processed=True, label_id=5, roi_id=6)
ACCEPTED = Record(processed=True, label_id=None, roi_id=None)
PENDING = Record(processed=False, label_id=None, roi_id=None)
UPLOADED = Record(processed=True, label_id=5, roi_id=None)


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
        ],
    )
    def test_steps(self, remote, local, mask, local_known, step):
        assert decide(remote, local, mask, local_known) == step


def run_first(tmp_path, n_images=3, crash=None, **kwargs):
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    if crash:
        omero.crash = crash
    pipeline = omero_pipeline(config, source, omero, **kwargs)
    return config, source, omero, pipeline


def resume(config, source, omero, table_id, events=None, **kwargs):
    pipeline = omero_pipeline(
        resume_of(config, table_id),
        source,
        omero,
        events=None if events is None else events.append,
        **kwargs,
    )
    pipeline.run_full_micro_sam_workflow(list(source_images(source)))
    return pipeline


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
        config, source, omero, first = run_first(tmp_path, crash=crash)
        with pytest.raises(Crash):
            first.run_full_micro_sam_workflow(list(source_images(source)))
        table_id = config.omero.table_id
        omero.crash = {}

        pipeline = resume(config, source, omero, table_id)

        assert_consistent(omero, pipeline, table_id, 3)

    def test_crash_between_mask_move_and_accept(self, tmp_path, monkeypatch):
        config, source, omero, first = run_first(tmp_path)
        real = session_module.file_digest
        calls = {"n": 0}

        def dying_digest(path):
            calls["n"] += 1
            if calls["n"] == 2:
                raise Crash("after step 1")
            return real(path)

        monkeypatch.setattr(session_module, "file_digest", dying_digest)
        with pytest.raises(Crash):
            first.run_full_micro_sam_workflow(list(source_images(source)))
        monkeypatch.setattr(session_module, "file_digest", real)

        pipeline = resume(config, source, omero, config.omero.table_id)

        assert_consistent(omero, pipeline, config.omero.table_id, 3)

    def test_reconcile_twice_changes_nothing(self, tmp_path):
        config, source, omero, first = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            first.run_full_micro_sam_workflow(list(source_images(source)))
        omero.crash = {}
        pipeline = resume(config, source, omero, config.omero.table_id)
        uploads = dict(omero.uploads_per_tag())

        assert reconcile(pipeline) is False
        assert omero.uploads_per_tag() == uploads


@pytest.mark.unit
class TestRepairs:
    def _finished(self, tmp_path, n_images=2):
        config, source, omero, first = run_first(tmp_path, n_images=n_images)
        table_id, _ = first.run_full_micro_sam_workflow(list(source_images(source)))
        return config, source, omero, first, table_id

    def test_edited_local_mask_is_reported_and_kept_off_omero(self, tmp_path):
        config, source, omero, first, table_id = self._finished(tmp_path)
        unit = config.annotations[0]
        mask = first.bundle_dir / "annotation_output" / f"{unit.annotation_id}_mask.tif"
        tifffile.imwrite(mask, np.full((16, 16), 9, np.uint16))
        events = []

        resume(config, source, omero, table_id, events)

        divergent = [e for e in events if getattr(e, "action", "") == "divergent"]
        assert [e.annotation_id for e in divergent] == [unit.annotation_id]
        assert len(omero.labels) == 2

    def test_upload_that_differs_from_the_local_mask_is_not_adopted(self, tmp_path):
        config, source, omero, first = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            first.run_full_micro_sam_workflow(list(source_images(source)))
        omero.crash = {}
        unit = config.annotations[0]
        label_id = next(
            i for i, u in omero.labels.items() if u.tag.endswith(unit.annotation_id)
        )
        omero.labels[label_id].pixels = np.zeros((16, 16), np.uint16)
        events = []

        resume(config, source, omero, config.omero.table_id, events)

        row = omero.tables[config.omero.table_id].set_index("annotation_id")
        assert not row.loc[unit.annotation_id, "processed"]
        assert any(getattr(e, "action", "") == "divergent" for e in events)
        assert len(omero.labels) == 3  # not annotated and uploaded a second time

    def test_deleting_the_local_mask_accepts_the_upload(self, tmp_path):
        config, source, omero, first = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            first.run_full_micro_sam_workflow(list(source_images(source)))
        omero.crash = {}
        unit = config.annotations[0]
        label_id = next(
            i for i, u in omero.labels.items() if u.tag.endswith(unit.annotation_id)
        )
        omero.labels[label_id].pixels = np.zeros((16, 16), np.uint16)
        resume(config, source, omero, config.omero.table_id)  # flags it for review
        (
            first.bundle_dir / "annotation_output" / f"{unit.annotation_id}_mask.tif"
        ).unlink()

        pipeline = resume(config, source, omero, config.omero.table_id)

        row = omero.tables[config.omero.table_id].set_index("annotation_id")
        assert row.loc[unit.annotation_id, "processed"]
        assert row.loc[unit.annotation_id, "label_id"] == label_id
        assert_consistent(omero, pipeline, config.omero.table_id, 3)

    @pytest.mark.parametrize("prune", [False, True])
    def test_duplicate_uploads_are_reported_or_pruned(self, tmp_path, prune):
        config, source, omero, first = run_first(tmp_path, crash={"table": 1})
        with pytest.raises(Crash):
            first.run_full_micro_sam_workflow(list(source_images(source)))
        omero.crash = {}
        (first.bundle_dir / "tracking_table.csv").unlink()  # force a tag lookup
        unit = config.annotations[0]
        label_id = next(
            i for i, u in omero.labels.items() if u.tag.endswith(unit.annotation_id)
        )
        copy_id = max(omero.labels) + 1
        omero.labels[copy_id] = omero.labels[label_id]
        events = []
        pipeline = omero_pipeline(
            resume_of(config, config.omero.table_id),
            source,
            omero,
            events=events.append,
        )
        pipeline.set_table_id(config.omero.table_id)
        pipeline._load_table(config.omero.table_id)

        reconcile(pipeline, prune_duplicates=prune)

        action = "pruned" if prune else "duplicate"
        assert [e.action for e in events if getattr(e, "action", "") == action] == [
            action
        ]
        assert (copy_id in omero.labels) is not prune
        row = omero.tables[config.omero.table_id].set_index("annotation_id")
        assert row.loc[unit.annotation_id, "label_id"] == label_id

    def test_rows_only_the_local_copy_has_are_added_back(self, tmp_path):
        config, source, omero, first, table_id = self._finished(tmp_path, n_images=2)
        table = omero.tables[table_id]
        dropped = table.iloc[-1]["annotation_id"]
        omero.tables[table_id] = table.iloc[:-1]

        resume(config, source, omero, table_id)

        assert dropped in set(omero.tables[table_id]["annotation_id"])
