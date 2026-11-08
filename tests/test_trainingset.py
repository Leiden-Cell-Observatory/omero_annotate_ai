"""TrainingSet: create, open, extend, sessions and notes, external tools (fakes only)."""

import numpy as np
import pandas as pd
import pytest
import tifffile

from omero_annotate_ai.core.annotation_config import AnnotationConfig
from omero_annotate_ai.errors import ConfigError
from omero_annotate_ai.trainingset import TrainingSet, summarize
from tests.fakes import annotate, create_set, omero_set, reopen

pytestmark = pytest.mark.unit


def new(tmp_path, n_images=3, **kwargs):
    config, source, omero = omero_set(tmp_path, n_images=n_images)
    return create_set(config, source, omero, tmp_path / "home", **kwargs), source, omero


class TestCreate:
    def test_omero_set_gets_table_folder_and_published_config(self, tmp_path):
        ts, _, omero = new(tmp_path)

        assert ts.table_id in omero.tables
        assert ts.bundle_dir == (
            tmp_path / "home" / "fake.omero-4064" / f"{ts.table_id}_fake_set"
        )
        saved = AnnotationConfig.from_yaml(ts.bundle_dir / "config.yaml")
        assert saved.identity.table_id == ts.table_id
        assert saved.training.seed is not None
        assert omero.configs == [ts.table_id]
        assert len(pd.read_csv(ts.bundle_dir / "tracking_table.csv")) == 3

    def test_the_callers_config_is_not_changed(self, tmp_path):
        config, source, omero = omero_set(tmp_path)
        ts = create_set(config, source, omero, tmp_path / "home")
        assert config.identity.table_id is None
        changed = ts.config
        changed.name = "other"
        assert ts.name == "fake_set"

    def test_a_config_of_an_existing_set_is_refused(self, tmp_path):
        ts, source, omero = new(tmp_path)
        with pytest.raises(ConfigError, match="open"):
            create_set(ts.config, source, omero, tmp_path / "home")

    def test_local_set_keeps_everything_in_its_folder(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=2)
        config.storage = "local"
        ts = create_set(config, source, omero, tmp_path / "home")

        assert ts.table_id is None and omero.tables == {}
        assert ts.bundle_dir.parent == tmp_path / "home" / "local"
        assert annotate(ts) == 2
        again = TrainingSet.open(ts.bundle_dir, image_source=source)
        assert all(r.processed for r in again.rows)


class TestOpen:
    def test_by_table_id_uses_the_local_copy(self, tmp_path):
        ts, source, omero = new(tmp_path)
        again = reopen(ts, source, omero)
        assert again.bundle_dir == ts.bundle_dir
        assert [r.annotation_id for r in again.rows] == [
            r.annotation_id for r in ts.rows
        ]

    def test_by_table_id_without_a_local_copy_uses_the_omero_config(self, tmp_path):
        ts, source, omero = new(tmp_path)
        elsewhere = TrainingSet.open(
            ts.table_id,
            omero.conn(),
            home=tmp_path / "other_home",
            image_source=source,
            omero_writes=omero,
        )
        assert elsewhere.name == ts.name
        assert elsewhere.bundle_dir.is_relative_to(tmp_path / "other_home")

    def test_a_session_left_open_becomes_interrupted(self, tmp_path):
        ts, source, omero = new(tmp_path)
        session = ts.start_session()
        batch = session.next_batch()
        path = batch.mask_dir / "seg_00000.tif"
        tifffile.imwrite(path, np.ones(batch.items[0].pixels.shape, np.uint16))
        session.commit(batch, {batch.items[0].annotation_id: path})
        # the kernel dies here: finish() is never called

        again = reopen(ts, source, omero)

        record = again.config.sessions[-1]
        assert (record.status, record.n_annotated) == ("interrupted", 1)


class TestExtendAndNotes:
    def test_extend_adds_images_to_the_same_table(self, tmp_path):
        config, source, omero = omero_set(tmp_path, n_images=4)
        config.training.train_n = 2
        ts = create_set(config, source, omero, tmp_path / "home")
        table_id = ts.table_id
        uris = [f"omero:image:{i}" for i in (1, 2, 3, 4)]
        ts.get_image_uris = lambda: uris
        ts.get_images_by_ids = lambda wanted: [
            source.get_image(int(str(u).rsplit(":", 1)[-1])) for u in wanted
        ]

        assert ts.extend(train_n=3) == 1
        assert ts.table_id == table_id
        assert len(omero.tables[table_id]) == 3

    def test_counts_cannot_be_lowered(self, tmp_path):
        ts, _, _ = new(tmp_path)
        with pytest.raises(ConfigError, match="raised"):
            ts.extend(train_n=1)

    @pytest.mark.parametrize("storage", ["omero", "local"])
    def test_sessions_are_numbered_from_one(self, tmp_path, storage):
        config, source, omero = omero_set(tmp_path, n_images=2)
        config.storage = storage
        ts = create_set(config, source, omero, tmp_path / "home")
        session = ts.start_session()
        batch = session.next_batch()
        path = batch.mask_dir / "seg_00000.tif"
        tifffile.imwrite(path, np.ones(batch.items[0].pixels.shape, np.uint16))
        session.commit(batch, {batch.items[0].annotation_id: path})
        session.finish()
        annotate(ts)

        assert [s.id for s in ts.config.sessions] == [1, 2]
        assert {r.session for r in ts.rows} == {1, 2}

    def test_session_note_is_kept_locally_and_on_omero(self, tmp_path):
        ts, _, omero = new(tmp_path)
        annotate(ts, note="nuclei only")
        session_id = ts.config.sessions[-1].id
        ts.set_session_note(session_id, "nuclei only, small objects skipped")

        assert "small objects skipped" in omero.config_text[ts.table_id]
        record = summarize(ts.bundle_dir).sessions[-1]
        assert record.note == "nuclei only, small objects skipped"
        assert record.user == "tester"


class TestExternalTool:
    def test_units_stay_pending_until_their_masks_are_collected(self, tmp_path):
        ts, _, omero = new(tmp_path, n_images=2)
        assert ts.prepare_external() == 2
        assert not any(r.processed for r in ts.rows)
        assert len(list((ts.bundle_dir / "annotation_input").glob("*.tif"))) == 2

        first = ts.rows[0].annotation_id
        draft = ts.bundle_dir / "external_output" / f"{first}_mask.tif"
        tifffile.imwrite(draft, np.ones((16, 16), np.uint16))
        assert ts.collect_masks() == 1

        assert [r.processed for r in ts.rows] == [True, False]
        # Collected masks move to annotation_output/, which holds accepted masks only.
        assert not draft.exists()
        assert (ts.bundle_dir / "annotation_output" / f"{first}_mask.tif").exists()
        assert omero.uploads_per_tag()


class TestSummaryAndFacade:
    def test_summary_counts_per_split(self, tmp_path):
        ts, _, _ = new(tmp_path)
        annotate(ts, limit=2)
        assert summarize(ts.bundle_dir).progress["training"] == (2, 3)

    def test_create_pipeline_is_deprecated(self, tmp_path):
        from omero_annotate_ai.core.annotation_pipeline import create_pipeline

        config, _, omero = omero_set(tmp_path)
        with pytest.deprecated_call():
            create_pipeline(config, omero.conn())

    def test_pipeline_runs_the_workflow_on_its_set(self, tmp_path):
        from omero_annotate_ai.annotators import ScriptedAnnotator
        from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline
        from tests.fakes import image_mask

        config, source, omero = omero_set(tmp_path, n_images=2)
        pipeline = AnnotationPipeline(
            config, omero.conn(), annotator=ScriptedAnnotator(image_mask)
        )
        pipeline._training_set = create_set(config, source, omero, tmp_path / "home")
        table_id, done = pipeline.run_full_micro_sam_workflow()
        assert table_id == pipeline.training_set.table_id
        assert done.sessions[-1].n_annotated == 2
