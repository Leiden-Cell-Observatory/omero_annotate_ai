"""The step API (AnnotationSession) and the write order of one unit (fakes only)."""

import numpy as np
import pandas as pd
import pytest
import tifffile
import yaml

from omero_annotate_ai import provenance
from omero_annotate_ai.annotators import ScriptedAnnotator
from omero_annotate_ai.bundle import file_digest, mask_path
from omero_annotate_ai.errors import Cancelled, TableWriteError
from omero_annotate_ai.events import (
    BatchReleased,
    RunCancelled,
    SessionFinished,
    SessionStarted,
    UnitCommitted,
    UnitExcluded,
    UploadFailed,
)
from tests.fakes import annotate, create_set, image_mask, omero_set


def ready_pipeline(tmp_path, n_images=3, events=None, **kwargs):
    """An OMERO set with its schema and table, ready for a session."""
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    pipeline = create_set(config, source, omero, tmp_path / "home", events, **kwargs)
    return pipeline, omero


def masks_for(batch, positions=None):
    """Draw masks like micro-SAM for the given batch positions (default: all)."""
    batch.mask_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for position, item in enumerate(batch.items):
        if positions is not None and position not in positions:
            continue
        path = batch.mask_dir / f"seg_{position:05}.tif"
        tifffile.imwrite(path, np.full(item.pixels.shape, 1, np.uint16))
        result[item.annotation_id] = path
    return result


@pytest.mark.unit
class TestAnnotationSession:
    def test_partial_commits_save_each_unit_once(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path)
        session = pipeline.start_session()
        batch = session.next_batch()

        first = session.commit(batch, masks_for(batch, {0}))
        assert batch.mask_dir.exists()  # two units still open
        again = session.commit(batch, masks_for(batch, {0, 1, 2}))

        assert first.committed == (batch.items[0].annotation_id,)
        assert len(again.committed) == 2  # unit 0 is not saved twice
        assert len(omero.labels) == 3
        assert not batch.mask_dir.exists()  # removed once every unit is in
        assert session.next_batch() is None

    def test_release_keeps_units_pending_for_the_next_session(self, tmp_path):
        events = []
        pipeline, _ = ready_pipeline(tmp_path, events=events)
        session = pipeline.start_session()
        batch = session.next_batch()
        session.commit(batch, masks_for(batch, {0}))

        session.release(batch)

        assert not batch.mask_dir.exists()
        released = [e for e in events if isinstance(e, BatchReleased)]
        assert len(released[0].annotation_ids) == 2
        assert session.next_batch() is None  # not handed out twice in one session
        assert sum(not a.processed for a in pipeline.rows) == 2
        assert pipeline.start_session().next_batch() is not None

    def test_batch_size_splits_the_pending_units(self, tmp_path):
        pipeline, _ = ready_pipeline(tmp_path)
        session = pipeline.start_session(batch_size=2)

        sizes = []
        while (batch := session.next_batch()) is not None:
            sizes.append(len(batch.items))
            session.commit(batch, masks_for(batch))

        assert sizes == [2, 1]

    def test_unit_is_saved_locally_before_upload(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path)
        omero.fail = {"label": 1}
        session = pipeline.start_session()
        batch = session.next_batch()
        unit_id = batch.items[0].annotation_id

        session.commit(batch, masks_for(batch, {0}))

        mask = pipeline.bundle_dir / "annotation_output" / f"{unit_id}_mask.tif"
        local = pd.read_csv(pipeline.bundle_dir / "tracking_table.csv")
        row = local.set_index("annotation_id").loc[unit_id]
        assert mask.exists() and row["mask_digest"] == file_digest(mask)
        assert row["input_digest"]  # recorded when the batch was prepared
        if provenance.available():  # the ISCC codes come with the provenance extra
            assert row["mask_iscc"] == provenance.file_code(mask, mask=True)
            assert str(row["input_iscc"]).startswith("ISCC:")

    def test_failed_upload_is_recorded_and_the_session_goes_on(self, tmp_path):
        events = []
        pipeline, omero = ready_pipeline(tmp_path, events=events)
        omero.fail = {"label": 2}
        session = pipeline.start_session()
        batch = session.next_batch()

        report = session.commit(batch, masks_for(batch))

        assert len(report.committed) == 2 and len(report.failed) == 1
        failed = next(a for a in pipeline.rows if a.annotation_id in report.failed)
        assert not failed.processed and "simulated label failure" in failed.error
        assert [type(e) for e in events].count(UploadFailed) == 1
        local = pd.read_csv(pipeline.bundle_dir / "tracking_table.csv")
        assert (
            local["error"].fillna("").str.contains("simulated label failure").sum() == 1
        )
        assert "error" not in omero.tables[pipeline.table_id].columns

    def test_cancel_stops_between_units(self, tmp_path):
        events = []
        asked = {"n": 0}

        def cancel():
            asked["n"] += 1
            return asked["n"] > 1

        pipeline, _ = ready_pipeline(tmp_path, events=events, cancel=cancel)
        session = pipeline.start_session()

        with pytest.raises(Cancelled):
            session.next_batch()
        assert any(isinstance(e, RunCancelled) for e in events)

    def test_config_published_per_commit_or_at_finish(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path)
        session = pipeline.start_session(batch_size=1)
        published = len(omero.configs)  # at creation, and the session's start
        for _ in range(2):
            batch = session.next_batch()
            session.commit(batch, masks_for(batch), publish_config=False)
        assert len(omero.configs) == published

        session.finish("done for today")

        assert omero.configs[published:] == [pipeline.table_id]
        assert pipeline.config.sessions[-1].note == "done for today"
        assert pipeline.config.sessions[-1].status == "finished"

    def test_events_frame_the_session(self, tmp_path):
        events = []
        pipeline, _ = ready_pipeline(tmp_path, events=events)
        session = pipeline.start_session()
        batch = session.next_batch()
        session.commit(batch, masks_for(batch))
        session.finish("nuclei only")

        assert isinstance(events[0], SessionStarted)
        assert events[-1] == SessionFinished(session.session, 3, "nuclei only")
        assert [type(e) for e in events].count(UnitCommitted) == 3

    def test_read_only_set_keeps_masks_locally(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=2)
        config.storage = "local"
        pipeline = create_set(config, source, omero, tmp_path / "home")

        annotate(pipeline)

        assert omero.labels == {} and omero.tables == {}
        masks = list((pipeline.bundle_dir / "annotation_output").glob("*_mask.tif"))
        assert len(masks) == 2
        assert all(a.processed and a.mask_digest for a in pipeline.rows)


@pytest.mark.unit
class TestRevise:
    """Going back to a unit saved earlier in the same session."""

    def _saved(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path, n_images=2)
        session = pipeline.start_session()
        batch = session.next_batch()
        session.commit(batch, masks_for(batch, {0}))
        return pipeline, omero, session, batch, batch.items[0].annotation_id

    def test_a_changed_mask_replaces_the_upload(self, tmp_path):
        pipeline, omero, session, _, unit = self._saved(tmp_path)
        old_label = next(r.label_id for r in pipeline.rows if r.annotation_id == unit)
        saved = session.saved_unit(unit)[1]
        changed = tmp_path / "changed.tif"
        tifffile.imwrite(changed, np.where(saved > 0, 2, 0).astype(np.uint32))

        assert session.revise(unit, changed) == "revised"

        row = next(r for r in pipeline.rows if r.annotation_id == unit)
        assert row.label_id != old_label and old_label not in omero.labels
        assert session.saved_unit(unit)[1].max() == 2
        table = omero.tables[pipeline.table_id].set_index("annotation_id")
        assert table.loc[unit, "label_id"] == row.label_id
        assert session.annotated == 1  # still one unit annotated

    def test_an_unchanged_mask_keeps_the_upload(self, tmp_path):
        pipeline, omero, session, _, unit = self._saved(tmp_path)
        before = dict(omero.labels)
        same = tmp_path / "same.tif"
        tifffile.imwrite(same, session.saved_unit(unit)[1].astype(np.uint32))

        assert session.revise(unit, same) == "unchanged"
        assert omero.labels.keys() == before.keys()

    def test_a_unit_changed_elsewhere_since_is_not_overwritten(self, tmp_path):
        pipeline, omero, session, _, unit = self._saved(tmp_path)
        table = omero.tables[pipeline.table_id]
        theirs = 999_999
        table.loc[table["annotation_id"] == unit, "label_id"] = theirs
        mask = tmp_path / "mine.tif"
        tifffile.imwrite(mask, np.full((4, 4), 5, np.uint32))

        assert session.revise(unit, mask) == "conflict"
        row = next(r for r in pipeline.rows if r.annotation_id == unit)
        assert row.label_id == theirs  # theirs kept, ours aside as a conflict file
        assert not mask.exists()

    def test_a_mask_from_another_filesystem_is_moved_into_place(
        self, tmp_path, monkeypatch
    ):
        import errno
        import os

        pipeline, _, session, _, unit = self._saved(tmp_path)
        saved = session.saved_unit(unit)[1]
        changed = tmp_path / "changed.tif"
        tifffile.imwrite(changed, np.where(saved > 0, 2, 0).astype(np.uint32))
        for name in ("rename", "replace"):  # a rename cannot cross filesystems
            original = getattr(os, name)

            def no_cross_device(src, dst, *args, _original=original, **kwargs):
                if str(src) == str(changed):
                    raise OSError(errno.EXDEV, "Invalid cross-device link")
                return _original(src, dst, *args, **kwargs)

            monkeypatch.setattr(os, name, no_cross_device)
        assert session.revise(unit, changed) == "revised"
        assert session.saved_unit(unit)[1].max() == 2

    def test_local_set_and_counts(self, tmp_path):
        ts, _, _ = annotated_set(tmp_path, n_images=1, storage="local")
        session = ts.start_session(review=True)
        batch = session.next_batch()
        unit = batch.items[0].annotation_id
        same = batch.mask_dir / "same.tif"
        tifffile.imwrite(same, batch.items[0].initial_mask)
        session.commit(batch, {unit: same})  # reviewed: unchanged
        assert session.annotated == 0
        changed = tmp_path / "changed.tif"
        tifffile.imwrite(changed, np.zeros_like(batch.items[0].initial_mask))

        assert session.revise(unit, changed) == "revised"  # corrected with Prev
        assert session.annotated == 1
        assert not session.saved_unit(unit)[1].any()

    def test_only_units_saved_by_this_session(self, tmp_path):
        _, _, session, batch, _ = self._saved(tmp_path)
        with pytest.raises(ValueError, match="not saved in this session"):
            session.revise(batch.items[1].annotation_id, tmp_path / "x.tif")


def local_rows(ts):
    """The set's tracking_table.csv, by annotation ID."""
    return pd.read_csv(ts.bundle_dir / "tracking_table.csv").set_index("annotation_id")


@pytest.mark.unit
class TestExclude:
    def test_excluded_unit_is_done_without_uploads(self, tmp_path):
        events = []
        pipeline, omero = ready_pipeline(tmp_path, events=events)
        session = pipeline.start_session()
        batch = session.next_batch()
        bad = batch.items[0].annotation_id

        assert session.exclude(batch, [bad], publish_config=False) == (bad,)

        table = omero.tables[pipeline.table_id].set_index("annotation_id")
        for rows in (table, local_rows(pipeline)):
            row = rows.loc[bad]
            assert row["processed"] and row["annotation_type"] == "excluded"
            assert (row["label_id"], row["roi_id"]) == (0, 0)
            assert row["session"] == session.session
            assert pd.isna(row["claimed_by"]) or row["claimed_by"] == ""
        others = table.drop(index=bad)
        assert (others["claimed_by"] == session.token).all()  # still held
        assert UnitExcluded(bad) in events
        assert pipeline.upload_tag(bad) not in omero.uploads_per_tag()

        assert session.exclude(batch, [bad]) == ()  # idempotent
        session.commit(batch, masks_for(batch, {1, 2}))
        assert session.next_batch() is None
        session.finish()
        assert pipeline.config.sessions[-1].n_annotated == 2  # excluded not counted
        assert pipeline.start_session().next_batch() is None

    def test_config_keeps_the_schema_older_versions_read(self, tmp_path):
        # 0.5.0-0.5.2 refuse unknown keys (extra="forbid"): nothing new may appear.
        released = {
            "id", "kind", "started", "ended", "status", "user", "host",
            "software_version", "model", "batch_size", "n_annotated", "note",
        }  # fmt: skip
        pipeline, omero = ready_pipeline(tmp_path, n_images=2)
        session = pipeline.start_session()
        batch = session.next_batch()
        session.exclude(batch, [batch.items[0].annotation_id])
        session.finish("one bad image")

        published = yaml.safe_load(omero.config_text[pipeline.table_id])
        for record in published["sessions"]:
            assert set(record) <= released

    @pytest.mark.parametrize("delete_fails", [False, True])
    def test_upload_of_an_earlier_failed_commit_is_deleted(
        self, tmp_path, delete_fails
    ):
        events = []
        pipeline, omero = ready_pipeline(tmp_path, n_images=1, events=events)
        omero.fail = {"roi": omero.calls.get("roi", 0) + 1}  # label posts, ROI fails
        first = pipeline.start_session()
        first.commit(batch := first.next_batch(), masks_for(batch))
        first.finish()
        (row,) = pipeline.rows
        assert row.label_id is not None and not row.processed
        assert mask_path(pipeline.bundle_dir, row.annotation_id).exists()
        if delete_fails:
            omero.fail = {"delete": omero.calls.get("delete", 0) + 1}
        session = pipeline.start_session()
        batch = session.next_batch()

        assert session.exclude(batch, [row.annotation_id]) == (row.annotation_id,)

        assert row.excluded and row.label_id is None
        assert not mask_path(pipeline.bundle_dir, row.annotation_id).exists()
        reported = [e for e in events if getattr(e, "action", "") == "excluded"]
        if delete_fails:
            assert len(omero.labels) == 1 and len(reported) == 1
        else:
            assert omero.uploads_per_tag() == {} and not reported

    def test_unit_saved_elsewhere_meanwhile_is_taken_over(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path, n_images=2)
        session = pipeline.start_session()
        batch = session.next_batch()
        unit = batch.items[0].annotation_id
        table = omero.tables[pipeline.table_id]
        at = table.index[table["annotation_id"] == unit][0]
        table.loc[at, ["processed", "label_id", "session"]] = [True, 999, 77]

        assert session.exclude(batch, [unit]) == ()

        row = next(r for r in pipeline.rows if r.annotation_id == unit)
        assert (row.excluded, row.label_id, row.session) == (False, 999, 77)
        assert omero.tables[pipeline.table_id].loc[at, "annotation_type"] != "excluded"

    def test_local_set(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=2)
        config.storage = "local"
        ts = create_set(config, source, omero, tmp_path / "home")
        session = ts.start_session()
        batch = session.next_batch()
        bad = batch.items[1].annotation_id

        session.exclude(batch, [bad, "not-in-batch"])

        row = local_rows(ts).loc[bad]
        assert row["processed"] and row["annotation_type"] == "excluded"
        assert not mask_path(ts.bundle_dir, bad).exists()
        session.release(batch)
        assert [r.annotation_id for r in ts.rows if not r.processed] == [
            batch.items[0].annotation_id
        ]
        assert ts.start_session().next_batch().items[0].annotation_id != bad

    def test_commit_after_exclude_once_the_batch_is_closed(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path, n_images=2)
        session = pipeline.start_session()
        batch = session.next_batch()
        first, last = (i.annotation_id for i in batch.items)
        session.commit(batch, masks_for(batch, {1}))
        session.exclude(batch, [first])
        assert not batch.mask_dir.exists()  # nothing waits: the batch is closed

        redo = pipeline.bundle_dir / "redo.tif"
        tifffile.imwrite(redo, np.ones(batch.items[0].pixels.shape, np.uint16))
        report = session.commit(batch, {first: redo})

        assert report.committed == (first,)
        assert not redo.exists() and mask_path(pipeline.bundle_dir, first).exists()
        assert {pipeline.upload_tag(first), pipeline.upload_tag(last)} == set(
            omero.uploads_per_tag()
        )

    def test_commit_after_exclude_makes_a_normal_annotation(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path, n_images=1)
        session = pipeline.start_session()
        batch = session.next_batch()
        unit = batch.items[0].annotation_id
        session.exclude(batch, [unit])

        report = session.commit(batch, masks_for(batch))

        assert report.committed == (unit,)
        (row,) = pipeline.rows
        assert not row.excluded and row.annotation_type == "segmentation_mask"
        assert omero.uploads_per_tag() == {pipeline.upload_tag(unit): (1, 1)}
        table = omero.tables[pipeline.table_id].set_index("annotation_id")
        assert table.loc[unit, "label_id"] == row.label_id
        assert table.loc[unit, "annotation_type"] == "segmentation_mask"
        session.finish()
        assert pipeline.config.sessions[-1].n_annotated == 1

    def test_failed_table_write_keeps_the_exclusion_locally(self, tmp_path):
        pipeline, omero = ready_pipeline(tmp_path, n_images=2)
        session = pipeline.start_session()
        batch = session.next_batch()
        bad = batch.items[0].annotation_id
        omero.fail = {"table": omero.calls.get("table", 0) + 1}

        with pytest.raises(TableWriteError):
            session.exclude(batch, [bad])

        assert local_rows(pipeline).loc[bad, "annotation_type"] == "excluded"

    def test_review_does_not_offer_excluded_units(self, tmp_path):
        pipeline, _ = ready_pipeline(tmp_path)
        session = pipeline.start_session()
        batch = session.next_batch()
        bad = batch.items[0].annotation_id
        session.exclude(batch, [bad])
        session.commit(batch, masks_for(batch, {1, 2}))
        session.finish()

        review = pipeline.start_session(review=True)
        offered = review.next_batch()

        assert bad not in [i.annotation_id for i in offered.items]
        assert len(offered.items) == 2
        with pytest.raises(ValueError, match="review"):
            review.exclude(offered, [offered.items[0].annotation_id])


def edited(pixels, mask):
    """A reviewer's correction: one pixel less of the object."""
    out = np.array(mask, copy=True)
    out[..., 0, 0] = 0
    return out


def annotated_set(tmp_path, n_images=2, storage="omero"):
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    config.storage = storage
    ts = create_set(config, source, omero, tmp_path / "home")
    annotate(ts)
    return ts, source, omero


@pytest.mark.unit
class TestReview:
    def test_annotated_units_come_with_their_saved_masks(self, tmp_path):
        ts, _, _ = annotated_set(tmp_path)
        session = ts.start_session(review=True)
        batch = session.next_batch()

        assert len(batch.items) == 2
        for item in batch.items:
            saved = tifffile.imread(mask_path(ts.bundle_dir, item.annotation_id))
            assert np.array_equal(item.initial_mask, saved)

    def test_unchanged_masks_change_nothing(self, tmp_path):
        ts, _, omero = annotated_set(tmp_path)
        before = dict(omero.labels)

        revised = ts.review(ScriptedAnnotator(image_mask), note="all fine")

        assert revised == 0
        assert omero.labels.keys() == before.keys()
        record = ts.config.sessions[-1]
        assert (record.kind, record.n_annotated, record.note) == (
            "review",
            0,
            "all fine",
        )
        assert (omero.tables[ts.table_id]["claimed_by"] == "").all()

    def test_same_labels_in_a_wider_type_are_unchanged(self, tmp_path):
        # micro-SAM writes masks as uint32; the saved ones here are uint16.
        ts, _, omero = annotated_set(tmp_path)
        before = dict(omero.labels)

        def widened(pixels, initial):
            return np.asarray(initial).astype(np.uint32)

        revised = ts.review(ScriptedAnnotator(image_mask, review_fn=widened))

        assert revised == 0
        assert omero.labels.keys() == before.keys()

    def test_changed_mask_replaces_label_and_roi(self, tmp_path):
        ts, _, omero = annotated_set(tmp_path)
        old = {r.annotation_id: (r.label_id, r.roi_id) for r in ts.rows}

        revised = ts.review(ScriptedAnnotator(image_mask, review_fn=edited))

        assert revised == 2
        assert all(n == (1, 1) for n in omero.uploads_per_tag().values())
        table = omero.tables[ts.table_id].set_index("annotation_id")
        for row in ts.rows:
            assert (row.label_id, row.roi_id) != old[row.annotation_id]
            assert old[row.annotation_id][0] not in omero.labels  # deleted
            mask = mask_path(ts.bundle_dir, row.annotation_id)
            assert tifffile.imread(mask)[0, 0] == 0
            assert table.loc[row.annotation_id, "mask_digest"] == file_digest(mask)
            assert table.loc[row.annotation_id, "label_id"] == row.label_id
            assert omero.labels[row.label_id].pixels[0, 0] == 0

    def test_failed_review_upload_leaves_the_unit_as_it_was(self, tmp_path):
        ts, _, omero = annotated_set(tmp_path, n_images=1)
        (row,) = ts.rows
        before = (row.label_id, row.roi_id, row.mask_digest)
        mask = mask_path(ts.bundle_dir, row.annotation_id)
        saved = tifffile.imread(mask)
        omero.fail = {"roi": omero.calls.get("roi", 0) + 1}  # label posts, ROI fails

        assert ts.review(ScriptedAnnotator(image_mask, review_fn=edited)) == 0

        assert (row.label_id, row.roi_id, row.mask_digest) == before
        assert np.array_equal(tifffile.imread(mask), saved)
        assert omero.uploads_per_tag() == {ts.upload_tag(row.annotation_id): (1, 1)}
        kept = list((ts.bundle_dir / "annotation_output" / "conflicts").iterdir())
        assert len(kept) == 1 and tifffile.imread(kept[0])[0, 0] == 0
        assert not list(mask.parent.glob("*.previous.tif"))

    def test_ids_limit_the_review(self, tmp_path):
        ts, _, _ = annotated_set(tmp_path, n_images=3)
        chosen = ts.rows[1].annotation_id

        batch = ts.start_session(review=True, ids=[chosen]).next_batch()

        assert [i.annotation_id for i in batch.items] == [chosen]

    def test_mask_is_downloaded_when_the_folder_lacks_it(self, tmp_path):
        ts, _, omero = annotated_set(tmp_path, n_images=1)
        row = ts.rows[0]
        mask_path(ts.bundle_dir, row.annotation_id).unlink()

        batch = ts.start_session(review=True).next_batch()

        assert np.array_equal(
            batch.items[0].initial_mask, omero.labels[row.label_id].pixels
        )
        assert mask_path(ts.bundle_dir, row.annotation_id).exists()

    def test_failed_upload_keeps_the_old_uploads(self, tmp_path):
        ts, _, omero = annotated_set(tmp_path, n_images=1)
        old = (ts.rows[0].label_id, ts.rows[0].roi_id)
        omero.fail = {"label": omero.calls["label"] + 1}

        revised = ts.review(ScriptedAnnotator(image_mask, review_fn=edited))

        assert revised == 0
        assert old[0] in omero.labels and old[1] in omero.rois
        assert "simulated label failure" in ts.rows[0].error

    def test_local_set_review_replaces_the_mask_only(self, tmp_path):
        ts, _, omero = annotated_set(tmp_path, storage="local")

        revised = ts.review(ScriptedAnnotator(image_mask, review_fn=edited))

        assert revised == 2 and omero.labels == {}
        for row in ts.rows:
            mask = mask_path(ts.bundle_dir, row.annotation_id)
            assert row.mask_digest == file_digest(mask)
