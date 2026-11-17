"""A training set: the images to annotate, their units, folder and OMERO table.

:class:`TrainingSet` is the core of omero-annotate-ai. Every front-end (the
notebook pipeline, the widgets, the napari plugin, scripts) creates or opens one and
annotates it through :class:`~omero_annotate_ai.session.AnnotationSession`.
"""

from __future__ import annotations

import getpass
import importlib.metadata
import logging
import secrets
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import tifffile

from . import claims
from .annotators import Annotator, Batch, BatchItem, MicroSamAnnotator
from .bundle import (
    atomic_write,
    file_digest,
    home_dir,
    local_bundle,
    mask_path,
    new_local_id,
    omero_bundle,
    pixel_digest,
    server_slug,
)
from .core.annotation_config import AnnotationConfig, SessionRecord
from .crate import write_ro_crate
from .errors import (
    ConfigError,
    OmeroAnnotateAIError,
    SelectionError,
    TableReadError,
    TableWriteError,
)
from .events import ClaimsHeld, ClaimsReleased, EventSink, TableWritten, log_events
from .omero import omero_functions, tracking_table
from .omero._untyped import ezomero
from .omero.omero_functions import create_tracking_table as create_omero_tracking_table
from .omero.omero_functions import replace_table_config
from .omero.simple_connection import ensure_connected
from .omero.tracking_table import is_supported, tag_for
from .planning import (
    CoverageSpec,
    ImageIdentity,
    ImageInfo,
    SplitSpec,
    plan_extension,
    plan_split,
    rng_for,
    units_for_image,
)
from .processing.image_functions import generate_patch_coordinates
from .provenance import image_code
from .reconcile import reconcile
from .rows import Category, Row, keep_local_codes, rows_from_frame, rows_to_frame
from .session import AnnotationSession, commit_unit
from .sources import ImageMeta, ImageSource, OmeroImageSource, make_image_source
from .sources.uri import omero_uri, parse_omero_uri
from .sources.wells import match_well_filter
from .sources.zarr_discovery import ZarrImageRef, discover_zarr_images

logger = logging.getLogger(__name__)

# Where external tools (e.g. Cellpose) write masks for collect_masks.
EXTERNAL_OUTPUT = "external_output"


def _well_name(row: int, column: int) -> str:
    """Well name from 0-based plate indices: (1, 1) -> B2, (27, 11) -> AB12."""
    letters = ""
    row += 1
    while row:
        row, rest = divmod(row - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return f"{letters}{column + 1}"


class OmeroWrites:
    """Every OMERO call a training set makes to track and upload, as one object.

    Pass a subclass to :meth:`TrainingSet.create` / :meth:`TrainingSet.open` to
    observe these calls or make them fail (for example in tests). The methods look
    the functions up when called.
    """

    def create_table(
        self,
        conn: Any,
        rows: Sequence[Row],
        title: str,
        container_type: str,
        container_ids: list[int],
    ) -> int:
        """Create the tracking table holding ``rows``; returns its ID."""
        return create_omero_tracking_table(
            conn, rows, title, container_type, container_ids
        )

    def read_table(self, conn: Any, table_id: int) -> Any:
        """All rows of the tracking table (a DataFrame)."""
        return omero_functions.read_tracking_table(conn, table_id)

    def write_rows(self, conn: Any, table_id: int, rows: Sequence[Row]) -> dict:
        """Write rows to the tracking table in place."""
        return omero_functions.write_rows_to_table(conn, table_id, rows)

    def read_rows(self, conn: Any, table_id: int, keys: Sequence[str]) -> Any:
        """The current table rows for ``keys`` (a DataFrame)."""
        return tracking_table.read_rows(conn, table_id, keys)

    def update_cells(
        self,
        conn: Any,
        table_id: int,
        keys: Sequence[str],
        change: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> list[str]:
        """Change some cells of the rows for ``keys`` in place (claims)."""
        return tracking_table.update_cells(conn, table_id, keys, change)

    def read_config(self, conn: Any, table_id: int) -> AnnotationConfig | None:
        """The set's one config YAML on OMERO, or None if it has none."""
        return omero_functions.find_annotation_config_for_table(conn, table_id)

    def upload_label(
        self, conn: Any, image_id: int | None, label_img: np.ndarray, **fields: Any
    ) -> int:
        """Attach a mask as a label file; returns the FileAnnotation ID."""
        if image_id is None:
            raise ValueError("Only OMERO images can carry uploads")
        return omero_functions.upload_label_file(conn, image_id, label_img, **fields)

    def upload_roi(
        self, conn: Any, image_id: int | None, label_img: np.ndarray, **fields: Any
    ) -> int | None:
        """Post the mask's ROI; returns its ID, or None for an empty mask."""
        if image_id is None:
            raise ValueError("Only OMERO images can carry uploads")
        return omero_functions.upload_label_rois(conn, image_id, label_img, **fields)

    def find_tagged(
        self, conn: Any, image_id: int, tag: str
    ) -> tuple[list[int], list[int]]:
        """``(label_ids, roi_ids)`` on the image carrying exactly ``tag``."""
        return omero_functions.find_tagged(conn, image_id, tag)

    def download_label(self, conn: Any, label_id: int) -> np.ndarray | None:
        """An uploaded label file's pixels, or None if it is gone."""
        return omero_functions.download_label(conn, label_id)

    def object_exists(self, conn: Any, kind: str, object_id: int) -> bool:
        """Whether an OMERO object (e.g. ``"Roi"``) still exists."""
        return conn.getObject(kind, int(object_id)) is not None

    def delete_objects(self, conn: Any, kind: str, ids: Sequence[int]) -> None:
        """Delete OMERO objects (e.g. duplicate uploads)."""
        conn.deleteObjects(kind, [int(i) for i in ids], wait=True)

    def replace_config(
        self,
        conn: Any,
        table_id: int,
        file_path: str,
        container_type: str,
        container_id: int,
    ) -> int:
        """Replace the set's one config YAML on OMERO."""
        return replace_table_config(
            conn, table_id, file_path, container_type, container_id
        )


class TrainingSet:
    """A training set: its definition (the config), its rows, folder and OMERO table.

    Make one with :meth:`create` (selects images, plans the units, creates the
    tracking table and the folder) or open one with :meth:`open` (by table ID or
    folder; repairs what an interrupted session left first). Annotate with
    :meth:`annotate`, or step by step through :meth:`start_session`.

    The connection is used from one thread at a time.
    """

    def __init__(
        self,
        config: AnnotationConfig,
        conn: Any = None,
        *,
        rows: Sequence[Row] = (),
        image_source: ImageSource | None = None,
        omero_writes: OmeroWrites | None = None,
        home: str | Path | None = None,
        events: EventSink | None = None,
        cancel: Callable[[], bool] | None = None,
        bundle: Path | None = None,
    ) -> None:
        """Use :meth:`create` or :meth:`open`; this only wires the parts together."""
        self._config = config
        self.rows: list[Row] = list(rows)
        self.home = home_dir(home)
        self.conn: Any = conn
        self.omero_writes = omero_writes if omero_writes is not None else OmeroWrites()
        self.events: EventSink = events if events is not None else log_events
        self.cancel = cancel
        self._bundle = bundle
        self._batches_run = 0
        # Notes set here on sessions of other computers, kept over their copy on
        # OMERO until published (see _merge_remote_config).
        self._note_edits: dict[int, str] = {}
        # A session being started: renumbered if another computer took its number.
        self._starting: SessionRecord | None = None
        _ = self._seed  # generate and store the seed before anything is planned
        self.image_source = (
            image_source
            if image_source is not None
            else make_image_source(config, conn)
        )
        self._zarr_refs: dict[str, ZarrImageRef] = {}
        # OMERO plate images: URI -> well, field and well key-value pairs
        self._omero_wells: dict[str, dict] = {}
        # OMERO plate images: URI -> plate id, so equal well names on two plates differ
        self._omero_plates: dict[str, int] = {}
        self._validate_setup()

    # --- What the set is -------------------------------------------------------

    @property
    def config(self) -> AnnotationConfig:
        """A copy of the set's config; changing it does not change the set."""
        return self._config.model_copy(deep=True)

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def table_id(self) -> int | None:
        """The OMERO tracking table's ID; None for local sets."""
        return self._config.identity.table_id

    @property
    def is_local(self) -> bool:
        """Whether the set is kept only in its folder (``storage: local``)."""
        return self._config.storage == "local"

    @property
    def annotation_type(self) -> str:
        return self._config.annotation_methodology.annotation_type

    @property
    def _seed(self) -> int:
        """The training set's seed; generated and stored in the config when unset."""
        seed = self._config.training.seed
        if seed is None:
            seed = secrets.randbelow(2**31)
            self._config.training.seed = seed
        return seed

    def _server(self) -> str:
        host = getattr(self.conn, "host", None) or "omero"
        return server_slug(str(host), getattr(self.conn, "port", None))

    @property
    def bundle_dir(self) -> Path:
        """This training set's folder under :attr:`home` (see :mod:`..bundle`)."""
        if self._bundle is None:
            identity, name = self._config.identity, self.name or "training_set"
            if self.is_local:
                if identity.local_uuid is None:
                    identity.local_uuid = new_local_id()
                self._bundle = local_bundle(self.home, identity.local_uuid, name)
            elif identity.table_id is None:
                raise RuntimeError(
                    "The training set's folder is named after its OMERO table, "
                    "which does not exist yet"
                )
            else:
                self._bundle = omero_bundle(
                    self.home, self._server(), identity.table_id, name
                )
        return self._bundle

    def _config_path(self) -> Path:
        return self.bundle_dir / "config.yaml"

    def save_config(self) -> None:
        """Write the config into the set's folder, atomically."""
        yaml_text = self._config.to_yaml()
        atomic_write(
            self._config_path(), lambda tmp: tmp.write_text(yaml_text, encoding="utf-8")
        )

    def _validate_setup(self):
        """Validate the pipeline setup."""
        # Without a connection nothing can be written back; fail now, not after annotating.
        if self.conn is None and not self.is_local:
            raise ValueError(
                "An OMERO connection is required unless the set is stored locally (storage: local)"
            )
        # An OMERO source needs its own connection to read images.
        if (
            isinstance(self.image_source, OmeroImageSource)
            and self.image_source.conn is None
        ):
            raise ValueError("OMERO connection is required to read images")

    def _get_input_folders(self) -> dict:
        """Return input folder paths for the current workflow configuration.

        Separate channels (label_channel != training_channels):
          - 'annotation_input': the channel you annotate (e.g. fluorescence)
          - 'model_input': the channel the model consumes (e.g. brightfield)
        Single channel:
          - 'annotation_input': serves both roles
        """
        folders = {"annotation_input": self.bundle_dir / "annotation_input"}
        if self._config.spatial_coverage.uses_separate_channels():
            folders["model_input"] = self.bundle_dir / "model_input"
        return folders

    def _get_table_title(self) -> str:
        """Generate table title based on configuration."""
        if self._config.name:
            return self._config.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        container_ids = self._config.omero.get_all_container_ids()
        if len(container_ids) > 1:
            return f"{self._config.omero.container_type}_multi_{len(container_ids)}_{timestamp}"
        container_id = container_ids[0] if container_ids else 0
        return f"{self._config.omero.container_type}_{container_id}_{timestamp}"

    def _identity_fields(self, image_meta: ImageMeta) -> dict:
        """Row fields that say which image a unit belongs to."""
        uri = image_meta.image_uri
        fields = {
            "image_uri": uri,
            "image_id": parse_omero_uri(uri),
            "image_name": image_meta.name,
            "level": self._config.zarr.level if self._config.source == "zarr" else 0,
        }
        ref = self._zarr_refs.get(uri)
        if ref is not None:
            fields.update(
                image_name=ref.name or image_meta.name,
                well=ref.well,
                field=ref.field,
                well_metadata=dict(ref.well_metadata),
            )
        well = self._omero_wells.get(uri)
        if well is not None:
            fields.update(well, well_metadata=dict(well["well_metadata"]))
        return fields

    def _well_info(self, uri: str) -> tuple[str, str, dict[str, str]] | None:
        """(plate, well, well key-value pairs) of a plate image, else None."""
        ref = self._zarr_refs.get(uri)
        if ref is not None and ref.well:
            return ref.plate, ref.well, dict(ref.well_metadata)
        well = self._omero_wells.get(uri)
        if well is not None:
            plate = str(self._omero_plates.get(uri, ""))
            return plate, well["well"], dict(well["well_metadata"])
        return None

    def _image_info(self, uri: str) -> ImageInfo:
        """What split planning needs to know about an image."""
        info = self._well_info(uri)
        if info is None:
            return ImageInfo(uri=uri)
        plate, well, well_metadata = info
        return ImageInfo(uri=uri, plate=plate, well=well, well_metadata=well_metadata)

    def _split_spec(self) -> SplitSpec:
        training = self._config.training
        return SplitSpec(
            split_unit=training.split_unit,
            stratify_by=tuple(training.stratify_by),
            segment_all=training.segment_all,
            train_n=training.train_n,
            validate_n=training.validate_n,
            test_n=training.test_n,
            validation_fraction=training.validation_fraction,
            test_fraction=training.test_fraction,
        )

    def _plan_split(self, uris: list[str]) -> dict[str, Category]:
        """Assign images to splits, keeping every split unit within one split.

        See :func:`~omero_annotate_ai.planning.plan_split`. Returns
        ``{uri: category}`` for the selected images, training first.
        """
        plan = plan_split(
            [self._image_info(uri) for uri in uris],
            self._split_spec(),
            rng_for(self._seed, "split"),
        )
        message = plan.shortfall_message()
        if message is not None:
            warnings.warn(message, UserWarning, stacklevel=2)
        return dict(plan.assignment)

    def _coverage(self) -> CoverageSpec:
        """What to annotate in each image, from ``config.spatial_coverage``."""
        sc = self._config.spatial_coverage

        def patches(
            shape: tuple[int, int], rng
        ) -> tuple[list[tuple[int, int]], tuple[int, int]]:
            origins, (height, width) = generate_patch_coordinates(
                image_shape=shape,
                patch_size=sc.patch_size,
                n_patches=sc.patches_per_image,
                random_patch=sc.random_patches,
                rng=rng,
            )
            return origins, (height, width)

        return CoverageSpec(
            channel=sc.get_label_channel(),
            timepoint_mode=sc.timepoint_mode,
            timepoints=tuple(sc.timepoints),
            n_timepoints=sc.n_timepoints,
            z_slice_mode=sc.z_slice_mode,
            z_slices=tuple(sc.z_slices),
            n_slices=sc.n_slices,
            three_d=sc.three_d,
            patches=patches if sc.use_patches else None,
        )

    def _identity(self, image_meta: ImageMeta) -> ImageIdentity:
        fields = self._identity_fields(image_meta)
        return ImageIdentity(
            image_uri=fields["image_uri"],
            image_name=fields["image_name"],
            image_id=fields["image_id"],
            level=fields["level"],
            well=fields.get("well", ""),
            field=fields.get("field", ""),
            well_metadata=fields.get("well_metadata", {}),
        )

    def _units_for_image(self, image: Any, category: Category) -> list[Row]:
        """Processing units (timepoints, z-slices, patches) for one image."""
        meta = self.image_source.metadata(image)
        units = units_for_image(
            meta, self._identity(meta), self._coverage(), category, self._seed
        )
        return [Row.from_unit(unit) for unit in units]

    def prepare_item(self, annotation: Row, *, with_mask: bool = False) -> BatchItem:
        """Load a unit for annotation and save its input image(s) into the bundle.

        Records the digest of the pixels shown to the annotator. ``with_mask`` adds
        the unit's saved mask (a review).
        """
        meta = self._meta_for(annotation)
        image_obj = self.image_source.get_image(annotation.image_uri)
        pixels = self._load_image_data(image_obj, meta)
        self._save_inputs(image_obj, meta, annotation.annotation_id, pixels)
        annotation.input_digest = pixel_digest(pixels)
        annotation.input_iscc = image_code(pixels)
        return BatchItem(
            annotation_id=annotation.annotation_id,
            pixels=pixels,
            context=self._context_for(image_obj, meta),
            unit=annotation.unit(),
            initial_mask=self.saved_mask(annotation) if with_mask else None,
        )

    def saved_mask(self, annotation: Row) -> np.ndarray | None:
        """The unit's mask: the folder's copy if it matches the recorded digest,
        else the label file on OMERO (saved into the folder if it has none)."""
        path = mask_path(self.bundle_dir, annotation.annotation_id)
        if path.exists() and file_digest(path) == annotation.mask_digest:
            return np.asarray(tifffile.imread(path))
        if not self.is_local and annotation.label_id is not None:
            pixels = self.omero_writes.download_label(self.conn, annotation.label_id)
            if pixels is not None:
                if not path.exists():
                    atomic_write(path, lambda tmp: tifffile.imwrite(tmp, pixels))
                return pixels
        return np.asarray(tifffile.imread(path)) if path.exists() else None

    def make_batch(
        self, index: int, items: Sequence[BatchItem], scratch: Path
    ) -> Batch:
        """A batch for the annotator, writing into its own ``scratch`` folder."""
        specs = self._config.spatial_coverage.get_context_channel_specs()
        return Batch(
            index=index,
            items=tuple(items),
            mask_dir=scratch,
            embedding_dir=scratch / "embeddings",
            model_type=self._config.ai_model.pretrained_from or "vit_b_lm",
            is_volumetric=self._config.spatial_coverage.three_d,
            context_colormaps={name: cmap for _ch, name, cmap in specs},
        )

    def upload_fields(self, annotation: Row) -> dict[str, Any]:
        """Keyword arguments for :meth:`OmeroWrites.upload_label` / ``upload_roi``."""
        return {
            "tag": self.upload_tag(annotation.annotation_id),
            "trainingset_name": self._config.name,
            "trainingset_description": f"Training set: {self._config.name}",
            "timepoint": annotation.timepoint,
            "z": annotation.z_start if annotation.is_volumetric else annotation.z_slice,
            "channel": annotation.channel,
            "patch_offset": (
                (annotation.patch_x, annotation.patch_y)
                if annotation.is_patch
                else None
            ),
        }

    def _load_image_data(self, image_obj, metadata: dict) -> np.ndarray:
        """Load the plane or volume an annotation unit refers to, cropped to its patch."""
        timepoint = metadata["timepoint"]
        channel = metadata.get(
            "channel_override", self._config.spatial_coverage.get_label_channel()
        )

        region = None
        if self._config.spatial_coverage.use_patches and "patch_x" in metadata:
            # Use the actual patch dimensions from metadata instead of config
            region = (
                metadata["patch_x"],
                metadata["patch_y"],
                metadata.get(
                    "patch_width", self._config.spatial_coverage.patch_size[1]
                ),
                metadata.get(
                    "patch_height", self._config.spatial_coverage.patch_size[0]
                ),
            )

        if metadata.get("is_volumetric", False):
            image_data = self.image_source.read_volume(
                image_obj,
                t=timepoint,
                c=channel,
                z_start=metadata.get("z_start", 0),
                z_end=metadata.get("z_end", 0),
                region=region,
            )
        else:
            image_data = self.image_source.read_plane(
                image_obj, t=timepoint, c=channel, z=metadata["z_slice"], region=region
            )

        if not image_data.any():
            image_uri = self.image_source.metadata(image_obj).image_uri
            logger.info(
                f"Warning: Image data appears to be all zeros for image {image_uri}"
            )
        return image_data

    def _load_and_stack_channels(
        self, image_obj, meta: dict, channel_indices: list[int]
    ) -> np.ndarray:
        """Load multiple channels and stack them channels-last: (Y,X,C) or (Z,Y,X,C)."""
        arrays = [
            self._load_image_data(image_obj, {**meta, "channel_override": ch})
            for ch in channel_indices
        ]
        return np.stack(arrays, axis=-1)

    def _context_for(self, image_obj, meta: dict) -> dict[str, np.ndarray]:
        """Context channels shown next to the annotated one: ``{name: pixels}``."""
        if not self._config.spatial_coverage.uses_context_channels():
            return {}
        return {
            name: self._load_image_data(image_obj, {**meta, "channel_override": ch})
            for ch, name, _cmap in self._config.spatial_coverage.get_context_channel_specs()
        }

    def _save_inputs(
        self, image_obj, meta: dict, annotation_id: str, pixels: np.ndarray
    ) -> None:
        """Save the annotated pixels to ``annotation_input/{id}.tif``.

        When the model reads another channel than the one annotated, that channel
        goes to ``model_input/{id}.tif``, where reorganize_local_data_for_training()
        reads it from.
        """
        folders = self._get_input_folders()
        self._save_image_to_disk(
            pixels, folders["annotation_input"] / f"{annotation_id}.tif"
        )
        if "model_input" in folders:
            training_channels = self._config.spatial_coverage.get_training_channels()
            train_meta = {**meta, "channel_override": training_channels[0]}
            self._save_training_image(
                image_obj, train_meta, annotation_id, folders["model_input"]
            )

    def _save_image_to_disk(
        self, image_data: np.ndarray, file_path: Path, axes: str | None = None
    ) -> bool:
        """Write ``image_data`` as a TIFF, atomically; ``axes`` e.g. ``'YXC'``."""
        import tifffile

        metadata: dict[str, Any] | None = {"axes": axes} if axes else None
        try:
            atomic_write(
                file_path,
                lambda tmp: tifffile.imwrite(tmp, image_data, metadata=metadata),
            )
        except OSError as e:
            logger.debug(f"Error saving image to {file_path}: {e}")
            return False
        logger.debug(f"Saved {file_path} - shape: {image_data.shape}")
        return True

    def _save_training_image(
        self, image_obj, meta: dict, annotation_id: str, output_folder: Path
    ) -> Path | None:
        """Save an image for training/annotation purposes.

        Args:
            image_obj: OMERO image object
            meta: Metadata dict with timepoint, z_slice, channel, etc.
            annotation_id: Unique identifier for the annotation
            output_folder: Directory to save the image

        Returns:
            Path to saved file, or None on failure
        """
        try:
            if not image_obj:
                logger.debug(f"No image object for annotation {annotation_id}")
                return None

            image_data = self._load_image_data(image_obj, meta)
            if image_data is None:
                logger.debug(f"No image data for annotation {annotation_id}")
                return None

            # Validate image data
            if np.all(image_data == 0):
                logger.debug(
                    f"Warning: Image {annotation_id} appears to be all black (all zeros)"
                )

            output_folder.mkdir(parents=True, exist_ok=True)
            image_file = output_folder / f"{annotation_id}.tif"

            if self._save_image_to_disk(image_data, image_file):
                return image_file
            return None

        except Exception as e:  # noqa: BLE001 - a bad image is skipped, not fatal
            logger.warning("Could not save the image of %s: %s", annotation_id, e)
            return None

    def _get_well_map_annotations(self, well_id: int) -> dict:
        """Get map annotations from a well as a dictionary.

        Args:
            well_id: OMERO well ID

        Returns:
            Dictionary with all key-value pairs from well map annotations
        """
        well_obj = self.conn.getObject("Well", well_id)
        if not well_obj:
            return {}
        return self._well_kv(well_obj)

    @staticmethod
    def _well_kv(well_obj) -> dict:
        """Key-value pairs of all map annotations on an OMERO well."""
        kv_pairs = {}
        for ann in well_obj.listAnnotations():
            if (
                hasattr(ann, "OMERO_TYPE")
                and ann.OMERO_TYPE.__name__ == "MapAnnotationI"
            ):
                # Get key-value pairs from this map annotation
                for kv in ann.getMapValue():
                    kv_pairs[kv.name] = kv.value

        return kv_pairs

    def _check_well_filter(
        self, well_kv_pairs: dict, filters: dict[str, list[str]]
    ) -> bool:
        """Check if well key-value pairs match filter criteria.

        Uses AND logic: ALL filter conditions must be met for a match.

        Args:
            well_kv_pairs: Key-value pairs from well map annotations
            filters: Filter criteria from config (key -> list of acceptable values)

        Returns:
            True if ALL filter conditions are met
        """
        return match_well_filter(well_kv_pairs, filters)

    def _get_image_ids_from_plate(self, container_id: int) -> list[int]:
        """Get image IDs from a single plate, applying well filters if configured.

        Args:
            container_id: The plate ID to get images from.

        Returns:
            List of image IDs from the plate (filtered if well_filters configured).
        """
        filters = self._config.omero.well_filters
        if filters:
            logger.info(f"Applying well filters: {filters}")
            logger.info(f"Filter mode: {self._config.omero.well_filter_mode}")

        well_ids = ezomero.get_well_ids(self.conn, plate=container_id)
        logger.info(f"Found {len(well_ids)} wells in plate {container_id}")

        image_ids = []
        matched_wells = 0
        for well_id in well_ids:
            well_obj = self.conn.getObject("Well", well_id)
            if well_obj is None:
                continue
            well_kv_pairs = self._well_kv(well_obj)
            if filters:
                matches_filter = self._check_well_filter(well_kv_pairs, filters)
                if matches_filter != (self._config.omero.well_filter_mode == "include"):
                    logger.debug(f"  Well {well_id}: filtered out")
                    continue
            matched_wells += 1
            well_name = _well_name(well_obj.row, well_obj.column)
            # The field is the WellSample index, as BlitzGateway's Well.getImage(index) uses it.
            for field, sample in enumerate(well_obj.listChildren()):
                image_id = sample.getImage().getId()
                image_ids.append(image_id)
                self._omero_wells[omero_uri(image_id)] = {
                    "well": well_name,
                    "field": str(field),
                    "well_metadata": dict(well_kv_pairs),
                }
                self._omero_plates[omero_uri(image_id)] = container_id

        if filters:
            logger.info(
                f"After well filtering: {matched_wells}/{len(well_ids)} wells matched, "
                f"{len(image_ids)} images selected"
            )
        return image_ids

    def _get_image_ids_from_single_container(
        self, container_type: str, container_id: int
    ) -> list[int]:
        """Get image IDs from a single OMERO container.

        Args:
            container_type: Type of container (dataset, project, plate, screen, image).
            container_id: The container ID.

        Returns:
            List of image IDs from the container.
        """
        if container_type == "dataset":
            return list(ezomero.get_image_ids(self.conn, dataset=container_id))
        if container_type == "project":
            dataset_ids = ezomero.get_dataset_ids(self.conn, project=container_id)
            image_ids = []
            for ds_id in dataset_ids:
                image_ids.extend(ezomero.get_image_ids(self.conn, dataset=ds_id))
            return image_ids
        if container_type == "plate":
            return self._get_image_ids_from_plate(container_id)
        if container_type == "screen":
            plate_ids = ezomero.get_plate_ids(self.conn, screen=container_id)
            image_ids = []
            for plate_id in plate_ids:
                image_ids.extend(ezomero.get_image_ids(self.conn, plate=plate_id))
            return image_ids
        if container_type == "image":
            return [container_id]
        raise ValueError(f"Unsupported container type: {container_type}")

    def get_image_uris(self) -> list[str]:
        """URIs of the images the config selects (OMERO container or OME-Zarr URIs)."""
        if self._config.source == "zarr":
            refs = discover_zarr_images(self._config.zarr, output_dir=self.bundle_dir)
            self._zarr_refs = {ref.uri: ref for ref in refs}
            logger.info(f"Found {len(refs)} OME-Zarr images")
            return list(self._zarr_refs)
        return [omero_uri(i) for i in self.get_image_ids_from_container()]

    def get_image_ids_from_container(self) -> list[int]:
        """Get image IDs from the configured OMERO container(s).

        Supports multiple containers of the same type via container_ids.
        For plate containers, applies well filtering if well_filters are configured.

        Returns:
            List of unique image IDs from all configured containers.
        """
        container_type = self._config.omero.container_type
        container_ids = self._config.omero.get_all_container_ids()

        if not container_ids:
            raise ValueError("No container IDs configured")

        all_image_ids = []

        for container_id in container_ids:
            logger.info(f"Loading image IDs from {container_type} {container_id}")
            image_ids = self._get_image_ids_from_single_container(
                container_type, container_id
            )
            logger.info(
                f"  Found {len(image_ids)} image IDs from container {container_id}"
            )
            all_image_ids.extend(image_ids)

        # Remove duplicates while preserving order
        unique_image_ids = list(dict.fromkeys(all_image_ids))

        if len(container_ids) > 1:
            logger.info(
                f"Total: {len(unique_image_ids)} unique image IDs from "
                f"{len(container_ids)} containers"
            )
        else:
            logger.info(f"Found {len(unique_image_ids)} image IDs")

        return unique_image_ids

    def get_images_by_ids(self, image_ids: Sequence[int | str]) -> list[Any]:
        """Get image handles for the given image IDs, skipping ones that do not exist."""
        images = []
        for img_id in image_ids:
            try:
                images.append(self.image_source.get_image(img_id))
            except LookupError:
                logger.debug(f"Image {img_id} not found; skipped")
        logger.info(f"Loaded {len(images)} images (by ID)")
        return images

    def get_images_from_container(self) -> list[Any]:
        """Get OMERO image objects from the configured container.

        Returns:
            List of OMERO image objects
        """
        image_ids = self.get_image_ids_from_container()
        return self.get_images_by_ids(image_ids)

    def save_local_table(self) -> None:
        """Write the folder's ``tracking_table.csv`` (every mode), atomically.

        Unlike the OMERO table it also keeps each unit's last upload error.
        """
        if not self.rows:
            return
        frame = rows_to_frame(self.rows, local=True)
        table_file = self.bundle_dir / "tracking_table.csv"
        atomic_write(table_file, lambda tmp: frame.to_csv(tmp, index=False))

    def read_local_table(self) -> dict[str, Row] | None:
        """The folder's ``tracking_table.csv`` as rows by ID; None if there is none."""
        path = self.bundle_dir / "tracking_table.csv"
        if not path.exists():
            return None
        rows = rows_from_frame(pd.read_csv(path, keep_default_na=False))
        return {r.annotation_id: r for r in rows}

    def upload_tag(self, annotation_id: str) -> str:
        """Tag for an upload, tying it to this training set's table."""
        if self.table_id is None:
            raise RuntimeError(
                "The tracking table must exist before anything is uploaded"
            )
        return tag_for(self.table_id, annotation_id)

    def write_table(self, rows: Sequence[Row]) -> int:
        """Write ``rows`` to the tracking table, in place (it keeps its ID).

        Only the rows this run changed are written, so rows other computers
        annotate meanwhile are never overwritten with a stale copy.

        Raises:
            TableWriteError: The update failed (chained to the OMERO error).
        """
        table_id = self.table_id
        assert table_id is not None, "local sets have no tracking table"
        try:
            result = self.omero_writes.write_rows(self.conn, table_id, rows)
        except OmeroAnnotateAIError:
            raise
        except Exception as error:
            raise TableWriteError(
                f"Could not update tracking table {table_id}; the uploads are "
                "saved locally and tagged - resume the set to repair the table"
            ) from error
        self.events(TableWritten(table_id, result["updated"] + result["added"]))
        return table_id

    def _read_table_rows(self, table_id: int) -> list[Row]:
        """The tracking table's rows.

        Raises:
            TableReadError: The table cannot be read or is empty. Opening stops here
                instead of carrying on as if the table were empty.
        """
        try:
            df = self.omero_writes.read_table(self.conn, table_id)
        except OmeroAnnotateAIError:
            raise
        except Exception as error:
            raise TableReadError(f"Could not read tracking table {table_id}") from error
        if df is None or df.empty:
            raise TableReadError(f"Tracking table {table_id} has no rows")
        return rows_from_frame(df)

    def update_cells(
        self,
        keys: Sequence[str],
        change: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> list[str]:
        """Change cells of the table rows for ``keys``, mirrored into ``rows``."""
        assert self.table_id is not None, "local sets have no tracking table"
        changes: dict[str, dict[str, Any]] = {}

        def record(row: dict[str, Any]) -> dict[str, Any] | None:
            cells = change(row)
            if cells:
                changes[str(row["annotation_id"])] = cells
            return cells

        changed = self.omero_writes.update_cells(self.conn, self.table_id, keys, record)
        by_id = {r.annotation_id: r for r in self.rows}
        for annotation_id in changed:
            for name, value in changes[annotation_id].items():
                setattr(by_id[annotation_id], name, value)
        if changed:
            self.save_local_table()
        return changed

    def current_rows(self, keys: Sequence[str]) -> dict[str, Row]:
        """The tracking table's rows for ``keys`` as they are now, by ID."""
        assert self.table_id is not None, "local sets have no tracking table"
        frame = self.omero_writes.read_rows(self.conn, self.table_id, keys)
        return {r.annotation_id: r for r in rows_from_frame(frame)}

    def ensure_connected(self) -> None:
        """Before remote work: restore a dropped OMERO connection.

        See :func:`omero_annotate_ai.omero.simple_connection.ensure_connected`.
        """
        if self.conn is not None:
            ensure_connected(self.conn)

    def refresh(self) -> None:
        """Take over what other computers saved meanwhile (OMERO sets).

        Units another session annotated (or reviewed: a newer label file) get the
        table's row; rows another computer added are appended; claims are updated.
        """
        if self.is_local or self.table_id is None:
            return
        self.ensure_connected()
        remote = {r.annotation_id: r for r in self._read_table_rows(self.table_id)}
        for position, row in enumerate(self.rows):
            theirs = remote.pop(row.annotation_id, None)
            if theirs is None:
                continue
            newer = (theirs.label_id or 0) > (row.label_id or 0)  # OMERO IDs only grow
            if theirs.processed and (not row.processed or newer):
                keep_local_codes(theirs, row)
                self.rows[position] = theirs
            else:
                row.claimed_by, row.claimed_at = theirs.claimed_by, theirs.claimed_at
        self.rows.extend(remote.values())
        self.save_local_table()

    def publish_config(self) -> int | None:
        """Replace the set's one config YAML on OMERO with the local one.

        What other computers published meanwhile is merged in first (see
        :meth:`_merge_remote_config`).
        """
        if self.table_id is None:
            return None
        self._merge_remote_config()
        self.save_config()
        file_id = self.omero_writes.replace_config(
            self.conn,
            self.table_id,
            str(self._config_path()),
            self._config.omero.container_type,
            self._config.omero.get_primary_container_id(),
        )
        self._note_edits.clear()
        return file_id

    # --- Creating and opening ----------------------------------------------------

    @classmethod
    def create(
        cls,
        config: AnnotationConfig,
        conn: Any = None,
        *,
        home: str | Path | None = None,
        image_source: ImageSource | None = None,
        omero_writes: OmeroWrites | None = None,
        events: EventSink | None = None,
        cancel: Callable[[], bool] | None = None,
        images: Sequence[Any] | None = None,
    ) -> TrainingSet:
        """A new training set: select the images, plan the units, start tracking.

        ``images`` (image handles of the image source) limits the selection to
        those images; by default the config's container or OME-Zarr URIs are used.

        For ``storage: omero`` the tracking table is created (and linked to the
        containers) and the config published next to it; ``storage: local`` sets
        get a local ID. Either way the set's folder holds ``config.yaml`` and
        ``tracking_table.csv``. ``config`` itself is not changed.

        Raises:
            ConfigError: ``config`` already belongs to a set (use :meth:`open`).
            SelectionError: Too few images for the requested counts.
        """
        if conn is not None:
            ensure_connected(conn)
        identity = config.identity
        if identity.table_id is not None or identity.local_uuid is not None:
            raise ConfigError(
                "This config already belongs to a training set; open it with "
                "TrainingSet.open()"
            )
        ts = cls(
            config.model_copy(deep=True),
            conn,
            image_source=image_source,
            omero_writes=omero_writes,
            home=home,
            events=events,
            cancel=cancel,
        )
        ts.rows = ts._plan_rows(images)
        if not ts.is_local:
            ts._config.identity.server = ts._server()
            ts._config.identity.table_id = ts.omero_writes.create_table(
                conn,
                ts.rows,
                ts._get_table_title(),
                ts._config.omero.container_type,
                ts._config.omero.get_all_container_ids(),
            )
        ts.save_local_table()
        ts.save_config()
        ts.publish_config()
        return ts

    @classmethod
    def open(
        cls,
        ref: int | str | Path,
        conn: Any = None,
        *,
        home: str | Path | None = None,
        image_source: ImageSource | None = None,
        omero_writes: OmeroWrites | None = None,
        events: EventSink | None = None,
        cancel: Callable[[], bool] | None = None,
        prune_duplicates: bool = False,
    ) -> TrainingSet:
        """An existing training set, by tracking-table ID or by its folder.

        For OMERO sets the rows come from the tracking table and the set is
        repaired (:meth:`reconcile`) before it is returned. Sessions an earlier
        run left open are closed as ``interrupted``.

        Raises:
            ConfigError: The set has no config (on OMERO or in its folder).
            TableReadError: The tracking table cannot be read.
        """
        if conn is not None:
            ensure_connected(conn)
        writes = omero_writes if omero_writes is not None else OmeroWrites()
        root = home_dir(home)
        bundle: Path | None = None
        if isinstance(ref, int):
            if conn is None:
                raise ConfigError(
                    "Opening a training set by table ID needs a connection"
                )
            server = server_slug(
                str(getattr(conn, "host", None) or "omero"), getattr(conn, "port", None)
            )
            local = sorted((root / server).glob(f"{ref}_*/config.yaml"))
            if local:
                bundle = local[0].parent
                config = AnnotationConfig.from_yaml(local[0])
            else:
                found = writes.read_config(conn, ref)
                if found is None:
                    raise ConfigError(f"Training set {ref} has no config on OMERO")
                config = found
        else:
            bundle = Path(ref).expanduser()
            config_file = bundle / "config.yaml"
            if not config_file.exists():
                raise ConfigError(f"No training set in {bundle} (no config.yaml)")
            config = AnnotationConfig.from_yaml(config_file)
        ts = cls(
            config,
            conn,
            image_source=image_source,
            omero_writes=writes,
            home=root,
            events=events,
            cancel=cancel,
            bundle=bundle,
        )
        if ts.is_local:
            ts.rows = list((ts.read_local_table() or {}).values())
        else:
            if ts.table_id is None:
                raise ConfigError("This OMERO training set's config has no table ID")
            ts.rows = ts._read_table_rows(ts.table_id)
            local_rows = ts.read_local_table()  # before releasing claims rewrites it
            ts._release_own_claims()
            reconcile(ts, local_rows, prune_duplicates=prune_duplicates, cancel=cancel)
            ts._report_claims()
        ts._close_open_sessions()
        return ts

    def _plan_rows(self, images: Sequence[Any] | None = None) -> list[Row]:
        """Select the images and plan their units (a new set)."""
        given = {}
        if images is not None:
            given = {self.image_source.metadata(i).image_uri: i for i in images}
        uris = list(given) if images is not None else self.get_image_uris()
        if not uris:
            where = (
                f"OME-Zarr URIs {self._config.zarr.uris}"
                if self._config.source == "zarr"
                else f"{self._config.omero.container_type} "
                f"{self._config.omero.get_all_container_ids()}"
            )
            raise SelectionError(f"No images found in {where}")
        plan = self._plan_split(uris)
        rows: list[Row] = []
        chosen = (
            [given[uri] for uri in plan]
            if given
            else self.get_images_by_ids(list(plan))
        )
        for image in chosen:
            uri = self.image_source.metadata(image).image_uri
            rows.extend(self._units_for_image(image, plan[uri]))
        logger.info("Planned %d units in %d images", len(rows), len(plan))
        return rows

    # --- Using the set -----------------------------------------------------------

    def units(self) -> tuple[Row, ...]:
        """Copies of the set's rows, one per annotation unit."""
        return tuple(r.model_copy() for r in self.rows)

    def extend(
        self,
        *,
        train_n: int | None = None,
        validate_n: int | None = None,
        test_n: int | None = None,
    ) -> int:
        """Raise the image counts and add images to reach them.

        Existing rows are never removed. Returns the number of images added.

        Raises:
            ConfigError: A count is lower than the set already has.
        """
        self.ensure_connected()
        if not self.is_local:
            self.refresh()  # rows other computers added count towards the targets
            self._merge_remote_config()
        training = self._config.training
        for key, value in (
            ("train_n", train_n),
            ("validate_n", validate_n),
            ("test_n", test_n),
        ):
            if value is None:
                continue
            if value < getattr(training, key):
                raise ConfigError(
                    f"{key} can only be raised (it is {getattr(training, key)})"
                )
            setattr(training, key, value)
        before = len(self.rows)
        added = self._extend_rows()
        if added:
            self.save_local_table()
            if not self.is_local:
                self.write_table(self.rows[before:])
        self.save_config()
        self.publish_config()
        return added

    def _extend_rows(self) -> int:
        """Add images until the counts are met (see :func:`..planning.plan_extension`)."""
        training = self._config.training
        if training.segment_all:
            return 0
        existing = self.rows
        plan = plan_extension(
            [(a.image_uri, a.category) for a in existing],
            {
                "training": training.train_n,
                "validation": training.validate_n,
                "test": training.test_n,
            },
            self.get_image_uris,
            rng_for(self._seed, "extend", len(existing)),
        )
        for category, (got, asked) in plan.shortfalls.items():
            logger.warning(
                "Only %d unused images left for %s (asked for %d more)",
                got,
                category,
                asked,
            )
        added = 0
        for category, uris in plan.picks.items():
            for image in self.get_images_by_ids(list(uris)):
                self.rows.extend(self._units_for_image(image, category))
                added += 1
        return added

    def start_session(
        self,
        *,
        model: str | None = None,
        batch_size: int | None = None,
        review: bool = False,
        ids: Sequence[str] | None = None,
    ) -> AnnotationSession:
        """A new session over the pending units, or a review of annotated ones.

        See :mod:`..session`; ``ids`` limits a session to those units.
        """
        self.ensure_connected()
        if not self.rows:
            raise ValueError("The training set has no units")
        return AnnotationSession(
            self,
            batch_size=batch_size or self._config.workflow.batch_size or None,
            model=model,
            cancel=self.cancel,
            review=review,
            ids=ids,
        )

    def annotate(
        self,
        annotator: Annotator | None = None,
        *,
        batch_size: int | None = None,
        model: str | None = None,
        note: str | None = None,
    ) -> int:
        """Annotate the pending units, batch by batch; returns how many were saved.

        ``annotator`` defaults to micro-SAM in napari. ``note`` is stored with the
        session (see :meth:`set_session_note` to add one afterwards).
        """
        session = self.start_session(model=model, batch_size=batch_size)
        return self._run(session, annotator, note)

    def review(
        self,
        annotator: Annotator | None = None,
        *,
        ids: Sequence[str] | None = None,
        batch_size: int | None = None,
        model: str | None = None,
        note: str | None = None,
    ) -> int:
        """Correct annotated units (all, or ``ids``); returns how many changed.

        Each unit opens with its saved mask (downloaded from OMERO if the folder
        lacks it). A changed mask replaces the upload; an unchanged one is left
        alone. The session is recorded as a review, with ``note``.
        """
        session = self.start_session(
            model=model, batch_size=batch_size, review=True, ids=ids
        )
        return self._run(session, annotator, note)

    def _run(
        self,
        session: AnnotationSession,
        annotator: Annotator | None,
        note: str | None,
    ) -> int:
        annotator = annotator if annotator is not None else MicroSamAnnotator()
        try:
            while (batch := session.next_batch()) is not None:
                session.commit(batch, annotator.annotate(batch))
                session.release(batch)
        except BaseException:
            session.finish(status="interrupted")
            raise
        return session.finish(note)

    def _merge_remote_config(self) -> AnnotationConfig | None:
        """Take what other computers published from the config on OMERO.

        - Sessions: this computer's records are kept, other computers' records are
          taken from OMERO - except notes set here (:meth:`set_session_note`). A
          session being started whose number another computer took meanwhile is
          renumbered (no rows carry it yet).
        - Image counts only grow (:meth:`extend`), so the larger count wins.

        Everything else in a set's config is fixed once it is created.
        """
        if self.table_id is None:
            return None
        remote = self.omero_writes.read_config(self.conn, self.table_id)
        if remote is None:
            return None
        host = claims.host_id()
        merged = {s.id: s for s in self._config.sessions}
        for theirs in remote.sessions:
            mine = merged.get(theirs.id)
            if mine is not None and mine is self._starting and theirs.host != host:
                merged[theirs.id] = theirs
                taken = [*merged, *(t.id for t in remote.sessions)]
                taken += [r.session for r in self.rows]
                mine.id = 1 + max(taken)
                merged[mine.id] = mine
            elif mine is None or mine.host not in (None, host):
                merged[theirs.id] = theirs  # another computer's session: theirs wins
        for session_id, note in self._note_edits.items():
            if session_id in merged:
                merged[session_id].note = note
        self._config.sessions = sorted(merged.values(), key=lambda s: s.id)
        training = self._config.training
        for key in ("train_n", "validate_n", "test_n"):
            setattr(
                training,
                key,
                max(getattr(training, key), getattr(remote.training, key)),
            )
        return remote

    def begin_session(
        self,
        model: str | None,
        batch_size: int | None,
        kind: Literal["annotate", "review"] = "annotate",
    ) -> int:
        """Record a new session in the config; returns its number.

        For OMERO sets the number follows the sessions on OMERO and is published at
        once, then checked: if another computer published the same number at the
        same time, this session takes the next free one. (Two computers publishing
        within the same instant can still overwrite each other's new record; see
        :mod:`omero_annotate_ai.claims`.)
        """
        self._merge_remote_config()
        known = [s.id for s in self._config.sessions] + [r.session for r in self.rows]
        record = SessionRecord(
            id=1 + max([0, *known]),  # unannotated rows carry session -1
            kind=kind,
            started=datetime.now(),
            user=self._user(),
            host=claims.host_id(),
            software_version=_software_version(),
            model=model or self._config.ai_model.pretrained_from,
            batch_size=batch_size,
        )
        self._config.sessions.append(record)
        if self.table_id is None:
            if self._can_save():
                self.save_config()
            return record.id
        self._starting = record
        try:
            for _ in range(5):
                self.publish_config()
                remote = self.omero_writes.read_config(self.conn, self.table_id)
                owner = next(
                    (
                        s.host
                        for s in (remote.sessions if remote else [])
                        if s.id == record.id
                    ),
                    record.host,
                )
                if owner == record.host:
                    break
        finally:
            self._starting = None
        return record.id

    def end_session(
        self,
        session_id: int,
        annotated: int,
        note: str | None,
        status: Literal["finished", "interrupted"] = "finished",
    ) -> None:
        """Close a session's record and publish the config."""
        record = self._session(session_id)
        record.ended = datetime.now()
        record.status = status
        record.n_annotated = annotated
        if note is not None:
            record.note = note
        self.save_config()
        self.publish_config()

    def set_session_note(self, session_id: int, note: str) -> None:
        """Set (or replace) a session's note, locally and on OMERO.

        Works for sessions of other computers too: the note set here wins over
        their copy on OMERO.

        Raises:
            KeyError: The set has no such session.
        """
        if not self.is_local:  # a local set's note and masks need no OMERO
            self.ensure_connected()
        if self.table_id is not None:
            self._merge_remote_config()
        self._session(session_id).note = note
        self._note_edits[session_id] = note
        self.save_config()
        self.publish_config()

    def _session(self, session_id: int) -> SessionRecord:
        for record in self._config.sessions:
            if record.id == session_id:
                return record
        raise KeyError(f"No session {session_id} in this training set")

    def _close_open_sessions(self) -> None:
        """Sessions a crashed run on this computer left open become ``interrupted``.

        Open sessions of other computers may still be running and stay open.
        """
        changed = False
        host = claims.host_id()
        for record in self._config.sessions:
            if record.status == "open" and record.host in (None, host):
                record.status = "interrupted"
                record.n_annotated = sum(
                    r.processed and not r.excluded and r.session == record.id
                    for r in self.rows
                )
                changed = True
        if changed:
            self.save_config()

    def _release_own_claims(self) -> None:
        """Release the claims an earlier run of this user on this computer left."""
        held = [r.annotation_id for r in self.rows if r.claimed_by]
        if held:
            released = claims.release(
                self, claims.owner(self._user()), held, suffix=True
            )
            if released:
                self.events(ClaimsReleased(len(released)))

    def _report_claims(self) -> None:
        """Report units other sessions are annotating now (they are skipped)."""
        hours = self._config.workflow.claim_hours
        live = [
            r.claimed_by.partition("@")[2]
            for r in self.rows
            if not r.processed and claims.is_live(r.claimed_by, r.claimed_at, hours)
        ]
        if live:
            self.events(ClaimsHeld(len(live), tuple(sorted(set(live)))))

    def _can_save(self) -> bool:
        return self.is_local or self.table_id is not None or self._bundle is not None

    @property
    def user(self) -> str | None:
        """The OMERO user (or the login name without a connection)."""
        return self._user()

    def _user(self) -> str | None:
        try:
            return str(self.conn.getUser().getName())
        except Exception:  # noqa: BLE001 - no connection or no user: fall back
            return getpass.getuser()

    def reconcile(self, *, prune_duplicates: bool = False) -> bool:
        """Repair what an interrupted session left (see :mod:`..reconcile`)."""
        self.ensure_connected()
        if self.is_local:
            return False
        return reconcile(
            self,
            self.read_local_table(),
            prune_duplicates=prune_duplicates,
            cancel=self.cancel,
        )

    def write_ro_crate(self) -> Path:
        """Describe the set's folder as an RO-Crate; returns the metadata file.

        Writes ``ro-crate-metadata.json`` (see :mod:`omero_annotate_ai.crate`): the
        files with their digests, the source images, and the sessions with their
        notes. Call it before sharing or archiving the folder.
        """
        from . import __version__

        self.save_local_table()
        self.save_config()
        return write_ro_crate(
            self._config, self.rows, self.bundle_dir, software_version=__version__
        )

    def next_batch_index(self) -> int:
        """Number of the next batch handed out (1, 2, ...)."""
        self._batches_run += 1
        return self._batches_run

    # --- External tools (e.g. Cellpose) -------------------------------------------

    def prepare_external(self) -> int:
        """Save the pending units' images for an external tool; returns how many.

        Images go to ``annotation_input/`` (and ``model_input/`` for separate
        channels); the tool writes ``external_output/{annotation_id}_mask.tif``.
        The units stay pending until :meth:`collect_masks` finds their masks:
        ``annotation_output/`` only ever holds accepted masks, so drafts in
        ``external_output/`` are never uploaded by the repair on open.
        """
        self.ensure_connected()
        pending = [r for r in self.rows if not r.processed]
        self._save_images_for_external(pending)
        (self.bundle_dir / EXTERNAL_OUTPUT).mkdir(parents=True, exist_ok=True)
        self.save_local_table()
        self.save_config()
        return len(pending)

    def _save_images_for_external(self, rows: Sequence[Row]) -> None:
        folders = self._get_input_folders()
        for row in rows:
            meta = self._meta_for(row)
            image_obj = self.image_source.get_image(row.image_uri)
            saved = self._save_training_image(
                image_obj, meta, row.annotation_id, folders["annotation_input"]
            )
            if saved is None:
                logger.warning("Could not save the image of %s", row.annotation_id)
            if "model_input" in folders:
                channels = self._config.spatial_coverage.get_training_channels()
                stacked = self._load_and_stack_channels(image_obj, meta, channels)
                folders["model_input"].mkdir(parents=True, exist_ok=True)
                target = folders["model_input"] / f"{row.annotation_id}.tif"
                axes = "ZYXC" if stacked.ndim == 4 else "YXC"
                if not self._save_image_to_disk(stacked, target, axes=axes):
                    logger.warning(
                        "Could not save the model image of %s", row.annotation_id
                    )

    def collect_masks(self, folder: str = EXTERNAL_OUTPUT) -> int:
        """Save the masks an external tool wrote as ``{annotation_id}_mask.tif``.

        ``folder`` is relative to the set's folder and may be a glob pattern. Each
        mask goes through the same write order as a session (one new session). On
        OMERO sets the table is read first and the units claimed: units saved or
        claimed on another computer are skipped. Returns the number of units saved.
        """
        if not self.is_local:  # a local set's note and masks need no OMERO
            self.ensure_connected()
        files: list[Path] = []
        folders = (
            list(self.bundle_dir.glob(folder))
            if "*" in folder
            else [self.bundle_dir / folder]
        )
        for found in folders:
            if found.is_dir():
                files += [*found.glob("*_mask.tif"), *found.glob("*_mask.tiff")]
        if not self.is_local:
            self.refresh()  # units saved on other computers meanwhile are skipped
        by_id = {r.annotation_id: r for r in self.rows}
        todo = {
            f.stem.removesuffix("_mask"): f
            for f in files
            if f.stem.removesuffix("_mask") in by_id
            and not by_id[f.stem.removesuffix("_mask")].processed
        }
        if not todo:
            return 0
        session_id = self.begin_session(None, None)
        token = claims.token(session_id, self._user())
        if not self.is_local:  # like a session: claim first, skip units held elsewhere
            todo = {
                k: todo[k] for k in claims.claim(self, token, list(todo), len(todo))
            }
        committed: list[Row] = []
        failed: list[str] = []
        for annotation_id, path in todo.items():
            row = by_id[annotation_id]
            if commit_unit(self, row, path, session_id):
                committed.append(row)
            else:
                failed.append(annotation_id)
        if not self.is_local:
            if committed:
                self.write_table(committed)
            if failed:
                claims.release(self, token, failed)
        self.end_session(session_id, len(committed), None)
        logger.info("Saved %d annotations", len(committed))
        return len(committed)

    def _meta_for(self, annotation: Row) -> dict:
        """A unit's plane, patch and image as the image loaders expect them."""
        meta = {
            "image_id": annotation.image_id,
            "image_uri": annotation.image_uri,
            "timepoint": annotation.timepoint,
            "z_slice": annotation.z_slice,
            "z_start": annotation.z_start,
            "z_end": annotation.z_end,
            "z_length": annotation.z_length,
            "category": annotation.category,
            "channel": annotation.channel,
            "is_volumetric": annotation.is_volumetric,
        }
        if annotation.is_patch:
            meta.update(
                patch_x=annotation.patch_x,
                patch_y=annotation.patch_y,
                patch_width=annotation.patch_width,
                patch_height=annotation.patch_height,
            )
        return meta


def _software_version() -> str | None:
    try:
        return importlib.metadata.version("omero-annotate-ai")
    except importlib.metadata.PackageNotFoundError:
        return None


# --- Read-only views (never repair, never write) -------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingSetInfo:
    """A training set as an Open dialog lists it."""

    table_id: int
    name: str
    created: str


@dataclass(frozen=True, slots=True)
class TrainingSetSummary:
    """What a set holds: per split ``(done, total)`` units, and its sessions.

    Done counts annotated and excluded units; ``excluded`` holds, per split, how
    many of them were excluded.
    """

    name: str
    table_id: int | None
    storage: str
    progress: dict[str, tuple[int, int]]
    sessions: tuple[SessionRecord, ...]
    excluded: dict[str, int] = dataclass_field(default_factory=dict[str, int])


def list_training_sets(
    conn: Any, container_type: str, container_id: int
) -> list[TrainingSetInfo]:
    """The training sets tracked on a container, newest first (read-only)."""
    ensure_connected(conn)
    found: list[TrainingSetInfo] = []
    for table in omero_functions.list_annotation_tables(
        conn, container_type, container_id
    ):
        file_ann = conn.getObject("FileAnnotation", int(table["id"]))
        if file_ann is not None and is_supported(file_ann):
            found.append(
                TrainingSetInfo(
                    int(table["id"]), str(table["name"]), str(table.get("created", ""))
                )
            )
    return found


def training_set_choices(
    conn: Any = None,
    container_type: str | None = None,
    container_id: int | None = None,
    *,
    home: str | Path | None = None,
    on_error: Callable[[str], None] | None = None,
) -> list[tuple[str, int | Path]]:
    """``(label, ref)`` for a picker: the sets on a container, then the local sets.

    ``ref`` is a table ID (OMERO set) or a folder (local set), as :func:`summarize`,
    :meth:`TrainingSet.open` and the training export take it. Read-only. With
    ``on_error``, an error listing the OMERO sets is reported there and the local sets
    are still returned; without it, it is raised.
    """
    choices: list[tuple[str, int | Path]] = []
    if conn is not None and container_type and container_id is not None:
        try:
            choices += [
                (f"{s.name} (table {s.table_id})", s.table_id)
                for s in list_training_sets(conn, container_type, int(container_id))
            ]
        except Exception as error:
            if on_error is None:
                raise
            on_error(f"Could not list training sets: {error}")
    folders = sorted((home_dir(home) / "local").glob("*/config.yaml"))
    choices += [(f"{f.parent.name} (local)", f.parent) for f in folders]
    return choices


def read_set_config(ref: int | str | Path, conn: Any = None) -> AnnotationConfig:
    """A set's config, by table ID (from OMERO) or folder, without opening it."""
    if isinstance(ref, int):
        if conn is None:
            raise ConfigError("Reading a set by table ID needs a connection")
        config = OmeroWrites().read_config(conn, ref)
        if config is None:
            raise ConfigError(f"Training set {ref} has no config on OMERO")
        return config
    return AnnotationConfig.from_yaml(Path(ref).expanduser() / "config.yaml")


def summarize(
    ref: int | str | Path, conn: Any = None, *, home: str | Path | None = None
) -> TrainingSetSummary:
    """Progress and sessions of a set, by table ID or folder, without opening it.

    Unlike :meth:`TrainingSet.open` nothing is repaired or written.
    """
    if conn is not None:
        ensure_connected(conn)
    config = read_set_config(ref, conn)
    if isinstance(ref, int):
        rows = rows_from_frame(omero_functions.read_tracking_table(conn, ref))
    else:
        folder = Path(ref).expanduser()
        csv = folder / "tracking_table.csv"
        rows = (
            rows_from_frame(pd.read_csv(csv, keep_default_na=False))
            if csv.exists()
            else []
        )
    progress = {
        category: (
            sum(r.processed for r in rows if r.category == category),
            sum(r.category == category for r in rows),
        )
        for category in ("training", "validation", "test")
    }
    excluded = {
        category: sum(r.excluded for r in rows if r.category == category)
        for category in progress
    }
    return TrainingSetSummary(
        name=config.name,
        table_id=config.identity.table_id,
        storage=config.storage,
        progress=progress,
        sessions=tuple(config.sessions),
        excluded=excluded,
    )
