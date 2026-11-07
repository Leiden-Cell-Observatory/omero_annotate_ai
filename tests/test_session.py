"""The step API (AnnotationSession) and the write order of one unit (fakes only)."""

import numpy as np
import pandas as pd
import pytest
import tifffile

from omero_annotate_ai.bundle import file_digest
from omero_annotate_ai.errors import Cancelled
from omero_annotate_ai.events import (
    BatchReleased,
    RunCancelled,
    SessionFinished,
    SessionStarted,
    UnitCommitted,
    UploadFailed,
)
from tests.test_pipeline import omero_pipeline, omero_set, source_images


def ready_pipeline(tmp_path, n_images=3, events=None, **kwargs):
    """An OMERO set with its schema and table, ready for a session."""
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    pipeline = omero_pipeline(
        config,
        source,
        omero,
        events=None if events is None else events.append,
        **kwargs,
    )
    images = list(source_images(source))
    pipeline.initialize_workflow(images)
    pipeline.define_annotation_schema(images)
    pipeline.create_tracking_table()
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
        assert sum(not a.processed for a in pipeline.config.annotations) == 2
        assert pipeline.start_session().next_batch() is not None

    def test_batch_size_splits_the_pending_units(self, tmp_path):
        pipeline, _ = ready_pipeline(tmp_path)
        pipeline.config.workflow.batch_size = 2
        session = pipeline.start_session()

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

    def test_failed_upload_is_recorded_and_the_session_goes_on(self, tmp_path):
        events = []
        pipeline, omero = ready_pipeline(tmp_path, events=events)
        omero.fail = {"label": 2}
        session = pipeline.start_session()
        batch = session.next_batch()

        report = session.commit(batch, masks_for(batch))

        assert len(report.committed) == 2 and len(report.failed) == 1
        failed = next(
            a for a in pipeline.config.annotations if a.annotation_id in report.failed
        )
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
        pipeline.config.workflow.batch_size = 1
        session = pipeline.start_session()
        for _ in range(2):
            batch = session.next_batch()
            session.commit(batch, masks_for(batch), publish_config=False)
        assert omero.configs == []

        session.finish("done for today")

        assert omero.configs == [pipeline.table_id]

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
        config.workflow.read_only_mode = True
        pipeline = omero_pipeline(config, source, omero)

        pipeline.run_full_micro_sam_workflow(list(source_images(source)))

        assert omero.labels == {} and omero.tables == {}
        masks = list((pipeline.bundle_dir / "annotation_output").glob("*_mask.tif"))
        assert len(masks) == 2
        assert all(a.processed and a.mask_digest for a in config.annotations)
