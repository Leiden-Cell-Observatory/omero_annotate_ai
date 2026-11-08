"""Functional (acceptance) tests: the real annotation workflow against the Docker OMERO.

Everything runs through the public API — ``create_pipeline`` and its workflows,
``collect_annotations_from_disk``, ``prepare_training_data_from_table``,
``setup_training``/``run_training``, ``delete_training_set``, the workflow widget and
a YAML file loaded with ``load_config_from_yaml`` — on the seeded test
server (tests/omero_seed.py). Only the interactive step is replaced: the pipeline gets
a :class:`~omero_annotate_ai.annotators.ScriptedAnnotator` instead of micro-SAM in
napari. It writes masks the way micro-SAM does (``seg_{i:05}.tif`` in the output
folder, honouring ``skip_segmented``) and can "close napari" early. Failures are
injected through the pipeline's ``omero_writes`` object.

The masks encode which image they were drawn on: seeded image ``k`` has every plane's
minimum at ``1000 * k + ...``, so its mask label is ``k + 1``. A label file or ROI on
the wrong image is therefore detectable after upload.

Behaviour that is known to be wrong today is marked ``xfail(strict=True)`` with the
PR that fixes it, so the suite turns red the moment such a test starts passing.

Run: ``pixi run -e dev pytest tests/test_functional_workflow.py -m docker``
(add ``-m "docker and slow"`` for the micro-SAM training run).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from omero_annotate_ai.annotators import ScriptedAnnotator
from omero_annotate_ai.core.annotation_pipeline import OmeroWrites

pytestmark = [pytest.mark.docker, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Scripted annotator (replaces napari + micro-SAM)
# ---------------------------------------------------------------------------


def mask_value_for(image: np.ndarray) -> int:
    """Label for a seeded image: its index + 1 (plane minimum is 1000 * index + ...)."""
    return int(np.asarray(image).min()) // 1000 + 1


def mask_for(image: np.ndarray, value: int | None = None) -> np.ndarray:
    """One rectangular object in the middle of the image (every z for volumes)."""
    image = np.asarray(image)
    value = mask_value_for(image) if value is None else value
    mask = np.zeros(image.shape, dtype=np.uint16)
    h, w = image.shape[-2:]
    mask[..., h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = value
    return mask


class Annotators:
    """Hands out scripted annotators; :func:`run_microsam` uses the latest one."""

    def __init__(self) -> None:
        self.current = ScriptedAnnotator(mask_for)

    def __call__(
        self, limit: int | None = None, value: int | None = None
    ) -> ScriptedAnnotator:
        """A fresh annotator; ``limit`` closes "napari" early, ``value`` fixes the label."""

        def mask_fn(image: np.ndarray) -> np.ndarray:
            return mask_for(image, value)

        self.current = ScriptedAnnotator(mask_fn, limit=limit)
        return self.current


@pytest.fixture
def annotator() -> Annotators:
    return Annotators()


class SpyWrites(OmeroWrites):
    """Records the table each table write goes to."""

    def __init__(self) -> None:
        self.table_ids: list[int] = []

    def write_rows(self, conn, table_id, rows):
        self.table_ids.append(table_id)
        return super().write_rows(conn, table_id, rows)


class FlakyWrites(OmeroWrites):
    """Fails the n-th label upload and/or the first table write."""

    def __init__(self, fail_upload: int | None = None, fail_table_once: bool = False):
        self.fail_upload = fail_upload
        self.fail_table_once = fail_table_once
        self.uploads = 0

    def upload_label(self, conn, image_id, label_img, **fields):
        self.uploads += 1
        if self.uploads == self.fail_upload:
            raise ConnectionError("simulated upload failure")
        return super().upload_label(conn, image_id, label_img, **fields)

    def write_rows(self, conn, table_id, rows):
        if self.fail_table_once:
            self.fail_table_once = False
            raise ConnectionError("simulated table update failure")
        return super().write_rows(conn, table_id, rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    seed,
    name: str,
    *,
    container="dataset",
    train_n=2,
    validate_n=1,
    test_n=1,
    batch_size=0,
    **sc,
):
    from omero_annotate_ai.core.annotation_config import create_default_config

    config = create_default_config()
    config.name = f"{seed.tag}_{name}"
    config.omero.container_type = container
    config.omero.container_id = (
        seed.dataset_id if container == "dataset" else seed.plate_id
    )
    config.training.train_n = train_n
    config.training.validate_n = validate_n
    config.training.test_n = test_n
    config.workflow.batch_size = batch_size
    config.spatial_coverage.channels = [0]
    for key, value in sc.items():
        setattr(config.spatial_coverage, key, value)
    return config


def make_pipeline(
    config,
    conn,
    annotators: Annotators,
    writes: OmeroWrites | None = None,
    events: list | None = None,
):
    from omero_annotate_ai.core.annotation_pipeline import AnnotationPipeline

    return AnnotationPipeline(
        config,
        conn,
        annotator=annotators.current,
        omero_writes=writes,
        events=None if events is None else events.append,
    )


def run_microsam(
    config,
    conn,
    annotators: Annotators,
    writes: OmeroWrites | None = None,
    events: list | None = None,
):
    pipeline = make_pipeline(config, conn, annotators, writes, events)
    table_id, config = pipeline.run_full_micro_sam_workflow()
    return pipeline, table_id, config


def resume_config(config, table_id: int, **training):
    """The same set again, opened by its table ID (rows come from the table)."""
    resumed = config.model_copy(deep=True)
    resumed.identity.table_id = table_id
    for key, value in training.items():
        setattr(resumed.training, key, value)
    return resumed


def table_rows(conn, table_id: int):
    from omero_annotate_ai.omero.omero_functions import read_tracking_table

    return read_tracking_table(conn, table_id)


def tagged_rois(conn, table_id: int, image_ids) -> dict[str, list[int]]:
    """annotation_id -> ROI IDs whose description carries this table's tag."""
    from omero_annotate_ai.omero.tracking_table import tag_prefix

    prefix = tag_prefix(table_id)
    found: dict[str, list[int]] = {}
    service = conn.getRoiService()
    for image_id in {int(i) for i in image_ids}:
        for roi in service.findByImage(image_id, None, conn.SERVICE_OPTS).rois:
            desc = roi.getDescription().getValue() if roi.getDescription() else ""
            if desc.startswith(prefix):
                annotation_id = desc[len(prefix) :].split(" ")[0].split("|")[0].strip()
                found.setdefault(annotation_id, []).append(roi.getId().getValue())
    return found


def download_label(conn, label_id: int) -> np.ndarray:
    import ezomero

    with tempfile.TemporaryDirectory() as tmp:
        return tifffile.imread(ezomero.get_file_annotation(conn, int(label_id), tmp))


def seed_index(seed, image_id: int) -> int:
    return seed.image_ids.index(int(image_id))


def config_annotations(conn, table_id: int):
    from omero_annotate_ai.omero.omero_functions import _table_config_annotations

    return _table_config_annotations(conn, table_id)


def assert_labels_on_right_images(conn, seed, rows):
    """Every processed row's label file and ROI belong to that row's image."""
    for _, row in rows.iterrows():
        assert bool(row["processed"]), row["annotation_id"]
        image_id = int(row["image_id"])
        label = download_label(conn, row["label_id"])
        values = set(np.unique(label)) - {0}
        assert values == {seed_index(seed, image_id) + 1}, (
            f"{row['annotation_id']}: label values {values} on image index "
            f"{seed_index(seed, image_id)}"
        )
        roi = conn.getObject("Roi", int(row["roi_id"]))
        assert roi is not None and roi.getImage().getId() == image_id


# ---------------------------------------------------------------------------
# 1. A new training set
# ---------------------------------------------------------------------------


class TestNewTrainingSet:
    def test_single_batch_dataset_2d(self, omero_conn, omero_seed, annotator, tmp_path):
        config = make_config(omero_seed, "new2d")
        pipeline, table_id, config = run_microsam(config, omero_conn, annotator)

        rows = table_rows(omero_conn, table_id)
        assert len(rows) == 4
        assert sorted(rows["category"]) == [
            "test",
            "training",
            "training",
            "validation",
        ]
        assert_labels_on_right_images(omero_conn, omero_seed, rows)

        # Uploads are tagged with the table, exactly one ROI per annotation.
        rois = tagged_rois(omero_conn, table_id, rows["image_id"])
        assert {k: len(v) for k, v in rois.items()} == {
            a: 1 for a in rows["annotation_id"]
        }
        # Exactly one config on OMERO for the set.
        assert len(config_annotations(omero_conn, table_id)) == 1
        # Label files are lossless: the local mask and the upload are identical.
        image = pipeline.image_source.get_image(
            f"omero:image:{int(rows['image_id'][0])}"
        )
        plane = pipeline.image_source.read_plane(image, t=0, c=0, z=0)
        np.testing.assert_array_equal(
            download_label(omero_conn, rows["label_id"][0]), mask_for(plane)
        )

    def test_table_id_stable_across_batches(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        spy = SpyWrites()
        config = make_config(omero_seed, "batches", batch_size=2)
        _, table_id, config = run_microsam(config, omero_conn, annotator, spy)

        assert spy.table_ids == [table_id, table_id]
        assert omero_conn.getObject("FileAnnotation", table_id) is not None
        rows = table_rows(omero_conn, table_id)
        assert rows["processed"].all() and len(rows) == 4
        assert len(config_annotations(omero_conn, table_id)) == 1

    def test_multi_batch_masks_on_right_images(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        config = make_config(omero_seed, "batchmasks", batch_size=2)
        _, table_id, _ = run_microsam(config, omero_conn, annotator)
        assert_labels_on_right_images(
            omero_conn, omero_seed, table_rows(omero_conn, table_id)
        )


# ---------------------------------------------------------------------------
# 2. Plate, patches, 3D
# ---------------------------------------------------------------------------


class TestSources:
    def test_plate(self, omero_conn, omero_seed, annotator, tmp_path):
        if omero_seed.plate_id is None:
            pytest.skip("Plate could not be imported on this server")
        annotator(value=7)  # fake plate pixels carry no image index
        config = make_config(
            omero_seed,
            "plate",
            container="plate",
            train_n=2,
            validate_n=1,
            test_n=0,
        )
        _, table_id, _ = run_microsam(config, omero_conn, annotator)

        rows = table_rows(omero_conn, table_id)
        assert len(rows) == 3 and rows["processed"].all()
        assert set(rows["image_id"].astype(int)) <= set(omero_seed.plate_image_ids)
        for _, row in rows.iterrows():
            roi = omero_conn.getObject("Roi", int(row["roi_id"]))
            assert roi.getImage().getId() == int(row["image_id"])
            assert set(np.unique(download_label(omero_conn, row["label_id"]))) == {0, 7}

    def test_patches(self, omero_conn, omero_seed, annotator, tmp_path):
        config = make_config(
            omero_seed,
            "patches",
            train_n=2,
            validate_n=1,
            test_n=0,
            use_patches=True,
            patch_size=[32, 32],
            patches_per_image=2,
            random_patches=False,
        )
        scripted = annotator()
        _, table_id, _ = run_microsam(config, omero_conn, annotator)

        rows = table_rows(omero_conn, table_id)
        assert len(rows) == 6 and rows["is_patch"].all()
        assert all(shape == (32, 32) for shape in scripted.calls[0].shapes)
        assert_labels_on_right_images(omero_conn, omero_seed, rows)

    def test_volumetric(self, omero_conn, omero_seed, annotator, tmp_path):
        config = make_config(
            omero_seed,
            "vol3d",
            train_n=1,
            validate_n=1,
            test_n=0,
            three_d=True,
            z_range_start=0,
            z_range_end=2,
            z_slice_mode="all",
        )
        scripted = annotator()
        _, table_id, _ = run_microsam(config, omero_conn, annotator)

        rows = table_rows(omero_conn, table_id)
        assert len(rows) == 2 and rows["is_volumetric"].all()
        assert all(len(shape) == 3 for shape in scripted.calls[0].shapes)
        assert_labels_on_right_images(omero_conn, omero_seed, rows)
        assert download_label(omero_conn, rows["label_id"][0]).ndim == 3


# ---------------------------------------------------------------------------
# 3-4. Resume
# ---------------------------------------------------------------------------


class TestResume:
    def _early_close_then_resume(self, conn, seed, annotator, tmp_path, name):
        annotator(limit=1)  # user closes napari after one image
        config = make_config(seed, name)
        _, table_id, config = run_microsam(config, conn, annotator)
        first = table_rows(conn, table_id)
        done_first = first[first["processed"]]

        annotator()
        _, table_id2, _ = run_microsam(resume_config(config, table_id), conn, annotator)
        return table_id, table_id2, done_first, table_rows(conn, table_id)

    def test_resume_after_early_close_completes(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        table_id, table_id2, done_first, rows = self._early_close_then_resume(
            omero_conn, omero_seed, annotator, tmp_path, "early"
        )

        assert len(done_first) == 1
        assert table_id2 == table_id
        assert len(rows) == 4 and rows["processed"].all()
        # The row done in session 1 is untouched.
        before = done_first.iloc[0]
        after = rows[rows["annotation_id"] == before["annotation_id"]].iloc[0]
        assert (after["roi_id"], after["label_id"]) == (
            before["roi_id"],
            before["label_id"],
        )
        # No duplicate uploads.
        rois = tagged_rois(omero_conn, table_id, rows["image_id"])
        assert all(len(v) == 1 for v in rois.values()) and len(rois) == 4

    def test_resume_after_early_close_masks_on_right_images(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        _, _, _, rows = self._early_close_then_resume(
            omero_conn, omero_seed, annotator, tmp_path, "earlymasks"
        )
        assert_labels_on_right_images(omero_conn, omero_seed, rows)

    def test_resume_with_raised_counts_extends_same_table(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        config = make_config(omero_seed, "extend", train_n=2, validate_n=1, test_n=0)
        _, table_id, config = run_microsam(config, omero_conn, annotator)
        assert len(table_rows(omero_conn, table_id)) == 3

        _, table_id2, _ = run_microsam(
            resume_config(config, table_id, train_n=3), omero_conn, annotator
        )

        rows = table_rows(omero_conn, table_id)
        assert table_id2 == table_id
        assert len(rows) == 4 and rows["processed"].all()
        assert sorted(rows["category"]) == [
            "training",
            "training",
            "training",
            "validation",
        ]
        assert len(set(rows["image_id"])) == 4
        assert len(config_annotations(omero_conn, table_id)) == 1


# ---------------------------------------------------------------------------
# 5. Crashes
# ---------------------------------------------------------------------------


class TestCrashes:
    def test_failed_upload_leaves_row_unprocessed(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        config = make_config(omero_seed, "uploadfail")
        events: list = []
        pipeline, table_id, _ = run_microsam(
            config, omero_conn, annotator, FlakyWrites(fail_upload=2), events
        )

        rows = table_rows(omero_conn, table_id)
        broken = rows[rows["roi_id"].isna() | (rows["roi_id"] <= 0)]
        assert len(broken) == 1
        assert not broken["processed"].any()
        local = pd.read_csv(pipeline.bundle_dir / "tracking_table.csv")
        failed = local[local["annotation_id"] == broken.iloc[0]["annotation_id"]]
        assert "simulated upload failure" in failed.iloc[0]["error"]
        assert [type(e).__name__ for e in events].count("UploadFailed") == 1

    def test_failed_table_update_stops_run_and_uploads_stay_traceable(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        from omero_annotate_ai.errors import TableWriteError
        from omero_annotate_ai.omero.omero_functions import delete_training_set

        config = make_config(omero_seed, "tablefail")
        pipeline = make_pipeline(
            config, omero_conn, annotator, FlakyWrites(fail_table_once=True)
        )
        with pytest.raises(TableWriteError) as raised:
            pipeline.run_full_micro_sam_workflow()
        assert isinstance(raised.value.__cause__, ConnectionError)

        table_id = pipeline.table_id
        rows = table_rows(omero_conn, table_id)
        assert not rows["processed"].any()  # the update never reached the table
        rois = tagged_rois(omero_conn, table_id, omero_seed.image_ids)
        assert len(rois) == 4  # ...but the uploads carry the table's tag
        report = delete_training_set(omero_conn, table_id, dry_run=True)
        assert sorted(report["rois"]) == sorted(r for v in rois.values() for r in v)

    def test_resume_after_failed_table_update_has_no_duplicates(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        from omero_annotate_ai.errors import TableWriteError

        config = make_config(omero_seed, "tablefailresume")
        pipeline = make_pipeline(
            config, omero_conn, annotator, FlakyWrites(fail_table_once=True)
        )
        with pytest.raises(TableWriteError):
            pipeline.run_full_micro_sam_workflow()
        table_id = pipeline.table_id

        run_microsam(resume_config(config, table_id), omero_conn, annotator)
        rows = table_rows(omero_conn, table_id)
        rois = tagged_rois(omero_conn, table_id, rows["image_id"])
        assert rows["processed"].all()
        assert all(len(v) == 1 for v in rois.values())

    def test_resume_adopts_uploads_the_local_copy_lost(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        """Table update failed and the local CSV is gone: the tags find the uploads."""
        from omero_annotate_ai.errors import TableWriteError

        config = make_config(omero_seed, "orphans")
        pipeline = make_pipeline(
            config, omero_conn, annotator, FlakyWrites(fail_table_once=True)
        )
        with pytest.raises(TableWriteError):
            pipeline.run_full_micro_sam_workflow()
        table_id = pipeline.table_id
        (pipeline.bundle_dir / "tracking_table.csv").unlink()
        before = tagged_rois(omero_conn, table_id, omero_seed.image_ids)

        events: list = []
        annotator(limit=0)  # nothing left to annotate by hand
        run_microsam(
            resume_config(config, table_id), omero_conn, annotator, events=events
        )

        rows = table_rows(omero_conn, table_id)
        assert rows["processed"].all()
        assert tagged_rois(omero_conn, table_id, rows["image_id"]) == before
        adopted = [e for e in events if type(e).__name__ == "ReconcileAction"]
        assert {e.action for e in adopted} == {"adopted"} and len(adopted) == 4
        assert_labels_on_right_images(omero_conn, omero_seed, rows)

    def test_edited_local_mask_is_reported_not_uploaded(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        config = make_config(omero_seed, "divergent")
        pipeline, table_id, config = run_microsam(config, omero_conn, annotator)
        rows = table_rows(omero_conn, table_id)
        edited = rows.iloc[0]["annotation_id"]
        mask = pipeline.bundle_dir / "annotation_output" / f"{edited}_mask.tif"
        tifffile.imwrite(mask, tifffile.imread(mask) * 0 + 7)

        events: list = []
        run_microsam(
            resume_config(config, table_id), omero_conn, annotator, events=events
        )

        divergent = [
            e
            for e in events
            if type(e).__name__ == "ReconcileAction" and e.action == "divergent"
        ]
        assert [e.annotation_id for e in divergent] == [edited]
        after = table_rows(omero_conn, table_id)
        assert (
            after.set_index("annotation_id").loc[edited, "label_id"]
            == (rows.set_index("annotation_id").loc[edited, "label_id"])
        )


# ---------------------------------------------------------------------------
# 6-7. Training data and training
# ---------------------------------------------------------------------------


@pytest.fixture
def finished_set(omero_conn, omero_seed, annotator, tmp_path):
    """A completed 2/1/1 training set and its config."""
    config = make_config(omero_seed, "export")
    _, table_id, config = run_microsam(config, omero_conn, annotator)
    return table_id, config


class TestTrainingData:
    def _export(self, conn, table_id, config, out):
        from omero_annotate_ai.processing.training_functions import (
            prepare_training_data_from_table,
        )

        return prepare_training_data_from_table(
            conn,
            table_id,
            out,
            normalization=config.normalization,
            label_channel=0,
            training_channels=[0],
        )

    def test_layout_dtype_pairing_and_normalisation(
        self, omero_conn, finished_set, tmp_path
    ):
        table_id, config = finished_set
        result = self._export(omero_conn, table_id, config, tmp_path / "train")

        train_inputs = sorted(Path(result["train_input"]).glob("*.tif"))
        val_inputs = sorted(Path(result["val_input"]).glob("*.tif"))
        assert len(train_inputs) == 2 and len(val_inputs) == 1
        for image_path in train_inputs + val_inputs:
            image = tifffile.imread(image_path)
            assert image.dtype == np.uint16  # native bit depth
            label_dir = Path(str(image_path.parent).replace("_input", "_label"))
            label = tifffile.imread(label_dir / image_path.name)
            assert set(np.unique(label)) - {0} == {mask_value_for(image)}
        assert (tmp_path / "train" / "normalization.json").exists()

    def test_test_split_is_exported(self, omero_conn, finished_set, tmp_path):
        table_id, config = finished_set
        result = self._export(omero_conn, table_id, config, tmp_path / "train")
        assert "test_input" in result
        assert len(list(Path(result["test_input"]).glob("*.tif"))) == 1

    def _training_config(self, conn, table_id, config, out):
        from omero_annotate_ai.processing.training_utils import setup_training

        result = self._export(conn, table_id, config, out)
        # Two iterations keep a real training run short enough for CPU.
        return setup_training(
            result,
            model_name="functional_test",
            model_type="vit_b_lm",
            epochs=1,
            n_iterations=2,
            batch_size=1,
            patch_shape=(64, 64),
            n_objects_per_batch=1,
        )

    @pytest.mark.slow
    def test_training_loaders_feed_microsam_ready_batches(
        self, omero_conn, finished_set, tmp_path, monkeypatch
    ):
        """The real run_training up to micro-SAM's train_sam, which is replaced by a check.

        Covers export -> normalization.json -> raw transform -> torch_em loaders: the
        batches micro-SAM would train on must be in [0, 255] and carry labels.
        """
        pytest.importorskip("micro_sam")
        import micro_sam.training as sam_training

        from omero_annotate_ai.processing.training_utils import run_training

        table_id, config = finished_set
        training = self._training_config(
            omero_conn, table_id, config, tmp_path / "train"
        )
        assert training["normalization"]["applied_at_export"] is False

        seen = {}

        def check_train_sam(train_loader, val_loader, **kwargs):
            for name, loader in (("train", train_loader), ("val", val_loader)):
                x, y = next(iter(loader))
                seen[name] = (float(x.min()), float(x.max()), int(y.max()))
            raise RuntimeError("stop after checking the loaders")

        monkeypatch.setattr(sam_training, "train_sam", check_train_sam)
        with pytest.raises(RuntimeError, match="stop after checking"):
            run_training(training)

        for name in ("train", "val"):
            low, high, label_max = seen[name]
            assert low >= 0.0 and high <= 255.0 and high > 1.0, (name, low, high)
            assert label_max > 0, name

    @pytest.mark.slow
    def test_microsam_training_runs(self, omero_conn, finished_set, tmp_path):
        cache = Path(
            os.environ.get("MICROSAM_CACHEDIR", Path.home() / ".cache" / "micro_sam")
        )
        if not (cache / "models" / "vit_b_lm").exists():
            pytest.skip(
                "micro-SAM vit_b_lm weights not cached (training would download them)"
            )
        from omero_annotate_ai.processing.training_utils import run_training

        table_id, config = finished_set
        outcome = run_training(
            self._training_config(omero_conn, table_id, config, tmp_path / "train")
        )
        assert Path(outcome["final_model_path"]).exists() or outcome.get(
            "checkpoint_folder"
        )


# ---------------------------------------------------------------------------
# 8. Cellpose, read-only, OME-Zarr
# ---------------------------------------------------------------------------


class TestOtherPaths:
    def test_cellpose_prepare_then_collect(self, omero_conn, omero_seed, tmp_path):
        from omero_annotate_ai.core.annotation_pipeline import create_pipeline

        config = make_config(omero_seed, "cellpose", train_n=2, validate_n=1, test_n=0)
        config.ai_model.framework = "cellpose"
        pipeline = create_pipeline(config, omero_conn)
        table_id, config = pipeline.run_cp_workflow()

        # The user segments the exported images with Cellpose: masks from the images.
        bundle = pipeline.bundle_dir
        inputs = sorted((bundle / "annotation_input").glob("*.tif"))
        assert len(inputs) == 3
        (bundle / "annotation_output").mkdir(exist_ok=True)
        for path in inputs:
            tifffile.imwrite(
                bundle / "annotation_output" / f"{path.stem}_mask.tif",
                mask_for(tifffile.imread(path)),
            )
        n, _ = pipeline.collect_annotations_from_disk()

        assert n == 3
        rows = table_rows(omero_conn, table_id)
        assert_labels_on_right_images(omero_conn, omero_seed, rows)

    def test_read_only_writes_nothing_to_omero(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        before = {
            i: len(omero_conn.getRoiService().findByImage(i, None).rois)
            for i in omero_seed.image_ids
        }
        config = make_config(omero_seed, "readonly", test_n=0)
        config.storage = "local"
        pipeline, table_id, config = run_microsam(config, omero_conn, annotator)

        assert table_id == -1
        bundle = pipeline.bundle_dir
        assert bundle.parent.name == "local"
        masks = sorted((bundle / "annotation_output").glob("*_mask.tif"))
        inputs = sorted((bundle / "annotation_input").glob("*.tif"))
        assert len(masks) == 3 and len(inputs) == 3
        for mask_path in masks:
            image = tifffile.imread(
                bundle / "annotation_input" / mask_path.name.replace("_mask", "")
            )
            assert set(np.unique(tifffile.imread(mask_path))) - {0} == {
                mask_value_for(image)
            }
        after = {
            i: len(omero_conn.getRoiService().findByImage(i, None).rois)
            for i in omero_seed.image_ids
        }
        assert after == before


class TestZarr:
    """OME-Zarr route (no OMERO needed): read-only, masks and inputs written locally."""

    def test_zarr_microsam_end_to_end(self, ngff_stores, annotator, tmp_path):
        from omero_annotate_ai.core.annotation_config import AnnotationConfig
        from omero_annotate_ai.core.annotation_pipeline import create_pipeline

        # Built through the model so the zarr validator applies (zarr is read-only).
        config = AnnotationConfig(
            name="zarr_functional",
            source="zarr",
            storage="local",
            zarr={"uris": [ngff_stores["img04"], ngff_stores["img05"]]},
        )
        config.training.train_n, config.training.validate_n, config.training.test_n = (
            1,
            1,
            0,
        )
        scripted = annotator(value=3)
        pipeline = create_pipeline(config, annotator=scripted, home=tmp_path)
        table_id, config = pipeline.run_full_micro_sam_workflow()

        assert table_id == -1
        assert all(a.processed for a in pipeline.training_set.units())
        assert len(scripted.calls) == 1 and scripted.calls[0].n_images == 2
        bundle = pipeline.bundle_dir
        assert bundle.parent == tmp_path / "local"
        masks = sorted((bundle / "annotation_output").glob("*_mask.tif"))
        assert len(masks) == 2
        assert (bundle / "tracking_table.csv").exists()
        assert (bundle / "config.yaml").exists()
        assert not (bundle / ".scratch").exists() or not any(
            (bundle / ".scratch").iterdir()
        )


# ---------------------------------------------------------------------------
# 9. Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_dry_run_then_delete_leaves_nothing(
        self, omero_conn, omero_seed, annotator, tmp_path
    ):
        from omero_annotate_ai.omero.omero_functions import delete_training_set

        config = make_config(omero_seed, "delete", train_n=1, validate_n=1, test_n=0)
        _, table_id, _ = run_microsam(config, omero_conn, annotator)
        rows = table_rows(omero_conn, table_id)
        roi_ids = [int(i) for i in rows["roi_id"]]
        label_ids = [int(i) for i in rows["label_id"]]
        config_ids = [a["id"] for a in config_annotations(omero_conn, table_id)]
        assert len(config_ids) == 1

        report = delete_training_set(omero_conn, table_id)  # dry run by default
        assert report["deleted"] is False
        assert sorted(report["rois"]) == sorted(roi_ids)
        assert sorted(report["labels"]) == sorted(label_ids)
        assert report["configs"] == config_ids
        assert omero_conn.getObject("FileAnnotation", table_id) is not None

        report = delete_training_set(omero_conn, table_id, dry_run=False)
        assert report["deleted"] is True
        assert omero_conn.getObject("FileAnnotation", table_id) is None
        assert all(omero_conn.getObject("Roi", i) is None for i in roi_ids)
        assert all(
            omero_conn.getObject("FileAnnotation", i) is None
            for i in label_ids + config_ids
        )
        assert tagged_rois(omero_conn, table_id, rows["image_id"]) == {}


# ---------------------------------------------------------------------------
# Two computers annotating one set (claims)
# ---------------------------------------------------------------------------


@pytest.fixture
def on(monkeypatch):
    """``on("a")`` makes the following calls run on computer ``a`` (its host ID)."""
    from omero_annotate_ai import claims

    host = ["a"]
    monkeypatch.setattr(claims, "host_id", lambda: host[0])
    return lambda name: host.__setitem__(0, name)


def scripted_masks(batch) -> dict[str, Path]:
    """Masks for every unit of a batch, written like micro-SAM writes them."""
    batch.mask_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for position, item in enumerate(batch.items):
        path = batch.mask_dir / f"seg_{position:05}.tif"
        tifffile.imwrite(path, mask_for(item.pixels))
        result[item.annotation_id] = path
    return result


def age_claims(conn, table_id: int, hours: float) -> None:
    """Make every claim in the table ``hours`` old (a computer that went quiet)."""
    from datetime import UTC, datetime, timedelta

    from omero_annotate_ai.omero.tracking_table import update_cells

    stamp = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    keys = list(table_rows(conn, table_id)["annotation_id"])
    update_cells(
        conn,
        table_id,
        keys,
        lambda row: {"claimed_at": stamp} if row["claimed_by"] else None,
    )


def open_set(conn, table_id: int, home: Path, events: list | None = None):
    from omero_annotate_ai.trainingset import TrainingSet

    return TrainingSet.open(
        table_id, conn, home=home, events=None if events is None else events.append
    )


class TestTwoComputers:
    def _create_and_open(self, conn, seed, on, tmp_path, name, **kwargs):
        from omero_annotate_ai.trainingset import TrainingSet

        on("a")
        a = TrainingSet.create(
            make_config(seed, name, **kwargs), conn, home=tmp_path / "a"
        )
        on("b")
        return a, open_set(conn, a.table_id, tmp_path / "b")

    def test_interleaved_sessions_share_the_units(
        self, omero_conn, omero_seed, on, tmp_path
    ):
        a, b = self._create_and_open(omero_conn, omero_seed, on, tmp_path, "two")
        on("a")
        session_a = a.start_session(batch_size=2)
        batch_a = session_a.next_batch()
        on("b")
        session_b = b.start_session(batch_size=4)
        batch_b = session_b.next_batch()

        ids_a = {i.annotation_id for i in batch_a.items}
        ids_b = {i.annotation_id for i in batch_b.items}
        assert len(ids_a) == len(ids_b) == 2 and not ids_a & ids_b

        session_b.commit(batch_b, scripted_masks(batch_b))
        session_b.finish("from b")
        on("a")
        session_a.commit(batch_a, scripted_masks(batch_a))
        session_a.finish("from a")

        rows = table_rows(omero_conn, a.table_id)
        assert len(rows) == 4 and rows["processed"].all()
        assert (rows["claimed_by"] == "").all()
        assert_labels_on_right_images(omero_conn, omero_seed, rows)
        rois = tagged_rois(omero_conn, a.table_id, rows["image_id"])
        assert len(rois) == 4 and all(len(v) == 1 for v in rois.values())
        elsewhere = open_set(omero_conn, a.table_id, tmp_path / "c")
        assert {s.note for s in elsewhere.config.sessions} == {"from a", "from b"}

    def test_expired_claim_is_taken_over_and_late_commit_conflicts(
        self, omero_conn, omero_seed, on, tmp_path
    ):
        a, b = self._create_and_open(
            omero_conn, omero_seed, on, tmp_path, "expire", validate_n=0, test_n=0
        )
        on("a")
        session_a = a.start_session(batch_size=1)
        batch_a = session_a.next_batch()
        unit = batch_a.items[0].annotation_id
        age_claims(omero_conn, a.table_id, hours=7)  # a went quiet

        on("b")
        session_b = b.start_session(batch_size=2)
        batch_b = session_b.next_batch()
        assert unit in {i.annotation_id for i in batch_b.items}
        session_b.commit(batch_b, scripted_masks(batch_b))

        on("a")
        report = session_a.commit(batch_a, scripted_masks(batch_a))

        assert report.conflicts == (unit,) and not report.committed
        rows = table_rows(omero_conn, a.table_id)
        rois = tagged_rois(omero_conn, a.table_id, rows["image_id"])
        assert all(len(v) == 1 for v in rois.values())
        assert any((a.bundle_dir / "annotation_output" / "conflicts").iterdir())

    def test_crash_leaves_a_claim_that_expires(
        self, omero_conn, omero_seed, on, tmp_path
    ):
        from omero_annotate_ai.events import ClaimsHeld, ClaimsReleased

        a, _ = self._create_and_open(
            omero_conn, omero_seed, on, tmp_path, "crash", validate_n=0, test_n=0
        )
        on("a")
        a.start_session(batch_size=1).next_batch()  # then the kernel dies

        on("b")
        events: list = []
        b = open_set(omero_conn, a.table_id, tmp_path / "b", events)
        assert any(isinstance(e, ClaimsHeld) and e.units == 1 for e in events)
        assert len(b.start_session().next_batch().items) == 1  # only the free one

        on("a")  # the crashed computer releases its leftovers when it opens the set
        events = []
        a2 = open_set(omero_conn, a.table_id, tmp_path / "a", events)
        assert any(isinstance(e, ClaimsReleased) and e.units == 1 for e in events)
        a2.start_session().next_batch()  # claims it again, then dies again

        age_claims(omero_conn, a.table_id, hours=7)
        on("b")
        again = open_set(omero_conn, a.table_id, tmp_path / "b")
        assert len(again.start_session().next_batch().items) == 2  # expired: free


# ---------------------------------------------------------------------------
# Reviewing annotated units
# ---------------------------------------------------------------------------


def corrected(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A reviewer's correction: the object loses its first row."""
    out = np.array(mask, copy=True)
    rows = np.nonzero(out.any(axis=-1))[0]
    out[..., rows[0], :] = 0
    return out


class DyingDelete(OmeroWrites):
    """The process dies when a review deletes an old upload (then works again)."""

    def __init__(self) -> None:
        self.dead = False

    def delete_objects(self, conn, kind, ids):
        if not self.dead:
            self.dead = True
            raise KeyboardInterrupt("killed before the old upload was deleted")
        super().delete_objects(conn, kind, ids)


def assert_one_upload_matching_the_local_mask(conn, ts, row) -> None:
    """The unit has one tagged label file and ROI, and the label is the local mask."""
    from omero_annotate_ai.bundle import mask_path
    from omero_annotate_ai.omero.omero_functions import find_tagged

    labels, rois = find_tagged(
        conn, int(row["image_id"]), ts.upload_tag(row["annotation_id"])
    )
    assert labels == [row["label_id"]] and rois == [row["roi_id"]]
    local = tifffile.imread(mask_path(ts.bundle_dir, row["annotation_id"]))
    assert np.array_equal(download_label(conn, row["label_id"]), local)


class TestReview:
    def _annotated(self, conn, seed, tmp_path, name, writes=None):
        from omero_annotate_ai.trainingset import TrainingSet

        ts = TrainingSet.create(
            make_config(seed, name, validate_n=0, test_n=0),
            conn,
            home=tmp_path,
            omero_writes=writes,
        )
        ts.annotate(ScriptedAnnotator(mask_for))
        return ts

    def test_review_replaces_changed_masks_only(self, omero_conn, omero_seed, tmp_path):
        ts = self._annotated(omero_conn, omero_seed, tmp_path, "review")
        before = table_rows(omero_conn, ts.table_id).set_index("annotation_id")

        assert ts.review(ScriptedAnnotator(mask_for)) == 0  # looked, changed nothing
        assert (
            table_rows(omero_conn, ts.table_id)
            .set_index("annotation_id")["label_id"]
            .equals(before["label_id"])
        )
        assert ts.review(ScriptedAnnotator(mask_for, review_fn=corrected)) == 2

        rows = table_rows(omero_conn, ts.table_id)
        for _, row in rows.iterrows():
            old = before.loc[row["annotation_id"]]
            assert row["label_id"] > old["label_id"] and row["roi_id"] != old["roi_id"]
            assert omero_conn.getObject("FileAnnotation", int(old["label_id"])) is None
            assert omero_conn.getObject("Roi", int(old["roi_id"])) is None
            assert row["mask_digest"] != old["mask_digest"]
            assert_one_upload_matching_the_local_mask(omero_conn, ts, row)
        assert [s.kind for s in ts.config.sessions] == ["annotate", "review", "review"]

    def test_crash_between_upload_and_delete_keeps_the_newer_upload(
        self, omero_conn, omero_seed, tmp_path
    ):
        from omero_annotate_ai.trainingset import TrainingSet

        ts = self._annotated(
            omero_conn, omero_seed, tmp_path, "reviewcrash", DyingDelete()
        )
        old = table_rows(omero_conn, ts.table_id).set_index("annotation_id")
        with pytest.raises(KeyboardInterrupt):
            ts.review(ScriptedAnnotator(mask_for, review_fn=corrected))

        TrainingSet.open(ts.table_id, omero_conn, home=tmp_path)

        rows = table_rows(omero_conn, ts.table_id)
        replaced = [
            row
            for _, row in rows.iterrows()
            if row["label_id"] != old.loc[row["annotation_id"], "label_id"]
        ]
        assert len(replaced) == 1  # the unit the crash interrupted, finished on open
        gone = old.loc[replaced[0]["annotation_id"]]
        assert omero_conn.getObject("FileAnnotation", int(gone["label_id"])) is None
        assert omero_conn.getObject("Roi", int(gone["roi_id"])) is None
        for _, row in rows.iterrows():
            assert_one_upload_matching_the_local_mask(omero_conn, ts, row)


# ---------------------------------------------------------------------------
# 10. Front ends: the Jupyter widget and a YAML file
# ---------------------------------------------------------------------------


class TestFrontEndRoutes:
    """The two ways users start a set: filling in the widget, or loading a YAML."""

    @staticmethod
    def _widget(conn, seed, home):
        from omero_annotate_ai.widgets.workflow_widget import create_workflow_widget

        widget = create_workflow_widget(connection=conn, home=home)
        widget.container_type.value = "dataset"
        widget.container.value = seed.dataset_id
        return widget

    def test_widget_create_annotate_note_reopen_extend(
        self, omero_conn, omero_seed, tmp_path
    ):
        home = tmp_path / "home"
        widget = self._widget(omero_conn, omero_seed, home)
        widget.name.value = f"{omero_seed.tag}_widget"
        widget.channel.value = 1
        widget.timepoints.mode.value = "random"
        widget.timepoints.n_random.value = 1
        widget.z_slices.listed.value = "1"
        for category, n in (("training", 2), ("validation", 1), ("test", 0)):
            widget.counts[category].value = n
        widget.seed.value = "42"
        widget.storage.value = "omero"
        widget.batch_size.value = 2
        widget.go.click()

        assert "✅" in widget.status.value, widget.status.value
        ts = widget.training_set()
        table_id = ts.table_id
        coverage = ts.config.spatial_coverage
        assert (coverage.channels, coverage.timepoint_mode, coverage.n_timepoints) == (
            [1],
            "random",
            1,
        )
        assert ts.bundle_dir.parent.parent == home

        assert widget.annotate(ScriptedAnnotator(mask_for), note="session 1") == 3
        rows = table_rows(omero_conn, table_id)
        assert len(rows) == 3 and (rows["z_slice"] == 1).all()
        assert_labels_on_right_images(omero_conn, omero_seed, rows)
        assert len(config_annotations(omero_conn, table_id)) == 1
        assert widget.note.value == "session 1"

        widget.note.value = "checked"
        widget.save_note.click()  # replaces the note, locally and on OMERO

        reopened = self._widget(omero_conn, omero_seed, home)
        reopened.mode.value = "open"
        assert table_id in [value for _, value in reopened.sets.options]
        reopened.sets.value = table_id
        assert "2 / 2 annotated" in reopened.summary.value
        assert "checked" in reopened.summary.value
        assert "session 1" not in reopened.summary.value

        reopened.add["training"].value = 1
        reopened.go.click()
        assert "1 units to annotate" in reopened.status.value, reopened.status.value
        assert reopened.training_set().table_id == table_id
        assert reopened.annotate(ScriptedAnnotator(mask_for)) == 1
        rows = table_rows(omero_conn, table_id)
        assert len(rows) == 4 and rows["processed"].all()
        assert len(tagged_rois(omero_conn, table_id, rows["image_id"])) == 4

    def test_yaml_create_annotate_reopen(self, omero_conn, omero_seed, tmp_path):
        """As in docs/configuration.md and notebooks/marimo/annotate-from-yaml.py."""
        from omero_annotate_ai import TrainingSet, load_config_from_yaml
        from omero_annotate_ai.core.annotation_pipeline import OmeroWrites

        home = tmp_path / "home"
        path = tmp_path / "set.yaml"
        path.write_text(
            f"name: {omero_seed.tag}_yaml\n"
            "omero:\n"
            "  container_type: dataset\n"
            f"  container_id: {omero_seed.dataset_id}\n"
            "spatial_coverage:\n"
            "  channels: [0]\n"
            "  use_patches: true\n"
            "  patch_size: [16, 16]\n"
            "  patches_per_image: 2\n"
            "training:\n"
            "  train_n: 2\n"
            "  validate_n: 1\n"
            "  test_n: 0\n"
            "  seed: 7\n"
            "ai_model:\n"
            "  pretrained_from: vit_b_lm\n"
            "workflow:\n"
            "  batch_size: 4\n"
            "storage: omero\n"
        )
        text = path.read_text()

        ts = TrainingSet.create(load_config_from_yaml(str(path)), omero_conn, home=home)
        assert ts.annotate(ScriptedAnnotator(mask_for), note="from yaml") == 6
        assert path.read_text() == text  # the user's file is not changed
        rows = table_rows(omero_conn, ts.table_id)
        assert len(rows) == 6 and rows["is_patch"].all()
        assert_labels_on_right_images(omero_conn, omero_seed, rows)

        # The config on OMERO round-trips: same definition, now with the set's identity.
        remote = OmeroWrites().read_config(omero_conn, ts.table_id)
        local = ts.config
        assert remote.identity.table_id == ts.table_id
        for part in ("omero", "spatial_coverage", "training", "ai_model"):
            assert getattr(remote, part) == getattr(local, part), part
        assert [s.note for s in remote.sessions] == ["from yaml"]

        # Reopen by table ID, and like the notebook does from the set's folder.
        by_id = TrainingSet.open(ts.table_id, omero_conn, home=home)
        folder_config = load_config_from_yaml(str(ts.bundle_dir / "config.yaml"))
        by_folder = TrainingSet.open(ts.bundle_dir, omero_conn, home=home)
        assert folder_config.identity.table_id == ts.table_id
        for opened in (by_id, by_folder):
            assert opened.table_id == ts.table_id
            assert all(unit.processed for unit in opened.units())
