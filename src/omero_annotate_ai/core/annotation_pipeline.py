"""Main pipeline for OMERO annotation workflows."""

# ========================================
# DEBUG CONFIGURATION
# ========================================
# To disable all debug output, change DEBUG_MODE to False in the AnnotationPipeline class
# or set AnnotationPipeline.DEBUG_MODE = False after import
# ========================================

# Standard library imports
import secrets
import shutil
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import ezomero

# Third-party imports
import numpy as np

# Local imports
from ..annotators import (
    Annotator,
    Batch,
    BatchItem,
    MicroSamAnnotator,
    collect_series_masks,
)
from ..omero.omero_functions import (
    create_tracking_table as create_omero_tracking_table,
)
from ..omero.omero_functions import (
    replace_table_config,
    sync_omero_table_to_config,
    upload_rois_and_labels,
    write_config_to_table,
)
from ..omero.tracking_table import tag_for
from ..planning import (
    CoverageSpec,
    ImageIdentity,
    ImageInfo,
    SplitSpec,
    Unit,
    plan_extension,
    plan_split,
    rng_for,
    timepoints_for,
    units_for_image,
    z_slices_for,
)
from ..processing.image_functions import generate_patch_coordinates
from ..sources import ImageMeta, ImageSource, OmeroImageSource, make_image_source
from ..sources.uri import (
    omero_uri,
    parse_omero_uri,
)
from ..sources.wells import match_well_filter
from ..sources.zarr_discovery import ZarrImageRef, discover_zarr_images
from .annotation_config import (
    AnnotationConfig,
    ImageAnnotation,
    validate_annotations_against_config,
)


def _well_name(row: int, column: int) -> str:
    """Well name from 0-based plate indices: (1, 1) -> B2, (27, 11) -> AB12."""
    letters = ""
    row += 1
    while row:
        row, rest = divmod(row - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return f"{letters}{column + 1}"


class OmeroWrites:
    """The OMERO writes the pipeline makes, as one replaceable object.

    Pass a subclass to :func:`create_pipeline` to observe these writes or make them
    fail (for example in tests). The methods look the functions up when called.
    """

    def upload_rois_and_labels(self, **kwargs: Any) -> tuple[int | None, int | None]:
        """Upload one mask as a label file and ROIs; returns ``(label_id, roi_id)``."""
        return upload_rois_and_labels(**kwargs)

    def write_config_to_table(
        self, conn: Any, table_id: int, config: AnnotationConfig
    ) -> dict:
        """Write the config's rows to the tracking table in place."""
        return write_config_to_table(conn, table_id, config)


def _annotation_from_unit(unit: Unit) -> ImageAnnotation:
    """The config's record of a planned unit."""
    x, y, width, height = unit.patch if unit.patch is not None else (0, 0, 0, 0)
    image = unit.identity
    return ImageAnnotation(
        image_uri=image.image_uri,
        image_id=image.image_id,
        image_name=image.image_name,
        level=image.level,
        well=image.well,
        field=image.field,
        well_metadata=dict(image.well_metadata),
        annotation_id=unit.annotation_id,
        category=unit.category,
        timepoint=unit.timepoint,
        z_slice=unit.z_slice,
        z_start=unit.z_start,
        z_end=unit.z_end,
        z_length=unit.z_length,
        channel=unit.channel,
        is_volumetric=unit.is_volumetric,
        is_patch=unit.patch is not None,
        patch_x=x,
        patch_y=y,
        patch_width=width,
        patch_height=height,
    )


class AnnotationPipeline:
    """Main pipeline for running micro-SAM annotation workflows with OMERO."""

    # DEBUG FLAG - Set to False to disable all debug output
    DEBUG_MODE = True

    def __init__(
        self,
        config: AnnotationConfig,
        conn=None,
        config_file_path: str | Path | None = None,
        image_source: ImageSource | None = None,
        annotator: Annotator | None = None,
        omero_writes: OmeroWrites | None = None,
    ):
        """Initialize the pipeline with configuration and OMERO connection.

        Args:
            config: AnnotationConfig object containing all parameters
            conn: OMERO connection object (BlitzGateway)
            config_file_path: Optional path to the source YAML configuration file for persistence
            image_source: Where pixels and image sizes are read from. Defaults to
                the source the config asks for (``make_image_source``).
            annotator: Annotation tool for the micro-SAM workflow. Defaults to
                micro-SAM's image series annotator in napari.
            omero_writes: The OMERO writes to use (replaceable for tests).
        """
        self.config = config
        self.conn = conn
        self.annotator = annotator
        self.omero_writes = omero_writes if omero_writes is not None else OmeroWrites()
        self._batches_run = 0
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
        # Initialize table_id from config if available, otherwise None
        self.table_id = config.omero.table_id
        self.config_file_path = Path(config_file_path) if config_file_path else None
        # If no explicit path, derive from config name and output directory
        if (
            self.config_file_path is None
            and config.name
            and config.output.output_directory
        ):
            self.config_file_path = (
                Path(config.output.output_directory)
                / f"annotation_config_{config.name}.yaml"
            )
        self._validate_setup()

    @property
    def _seed(self) -> int:
        """The training set's seed; generated and stored in the config when unset."""
        seed = self.config.training.seed
        if seed is None:
            seed = secrets.randbelow(2**31)
            self.config.training.seed = seed
        return seed

    def set_table_id(self, table_id: int) -> None:
        """Update the current table ID and sync to config."""
        self.table_id = table_id
        self.config.omero.table_id = table_id

    def get_table_id(self) -> int | None:
        """Get the current table ID."""
        return self.table_id

    def save_config_to_yaml(self, file_path: str | Path | None = None) -> None:
        """Save current config state back to YAML file for persistence.

        Args:
            file_path: Optional path to save to. If None, uses stored config_file_path
        """
        save_path = file_path or self.config_file_path

        if save_path is None:
            print("Warning: No YAML file path provided for saving config state")
            return

        try:
            self.config.save_yaml(save_path)
            print(f"Config state saved to: {save_path}")
        except Exception as e:
            print(f"Error saving config to YAML: {e}")

    def _auto_save_config(self) -> None:
        """Automatically save config to YAML if path is available."""
        if self.config_file_path is not None:
            self.save_config_to_yaml()

    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.DEBUG_MODE:
            print(message)

    def _validate_setup(self):
        """Validate the pipeline setup."""
        # Without a connection nothing can be written back; fail now, not after annotating.
        if self.conn is None and not self.config.workflow.read_only_mode:
            raise ValueError(
                "OMERO connection is required unless read_only_mode is set"
            )
        # An OMERO source needs its own connection to read images.
        if (
            isinstance(self.image_source, OmeroImageSource)
            and self.image_source.conn is None
        ):
            raise ValueError("OMERO connection is required to read images")

    def _get_input_folders(self, output_path: Path) -> dict:
        """Return input folder paths for the current workflow configuration.

        Separate channels (label_channel != training_channels):
          - 'annotation_input': the channel you annotate (e.g. fluorescence)
          - 'model_input': the channel the model consumes (e.g. brightfield)
        Single channel:
          - 'annotation_input': serves both roles
        """
        folders = {"annotation_input": output_path / "annotation_input"}
        if self.config.spatial_coverage.uses_separate_channels():
            folders["model_input"] = output_path / "model_input"
        return folders

    def _setup_directories(self):
        """Create output directories for annotation workflow.

        Separate channels:  annotation_input/  model_input/  annotation_output/
        Single channel:     annotation_input/  annotation_output/
        micro-SAM adds:     sam_embeddings/
        """
        output_path = Path(self.config.output.output_directory)

        dirs = [output_path, output_path / "annotation_output"]
        dirs.extend(self._get_input_folders(output_path).values())
        if self.config.ai_model.framework == "micro_sam":
            dirs.append(output_path / "sam_embeddings")

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        return output_path

    def _cleanup_embeddings(self, output_path: Path):
        """Clean up any existing embeddings from interrupted runs."""
        embed_path = output_path / "sam_embeddings"
        if embed_path.exists():
            for file in embed_path.glob("*"):
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)

    def _get_table_title(self) -> str:
        """Generate table title based on configuration."""
        # Use the existing trainingset_name from config directly
        # (the name from config already includes proper formatting via generate_unique_table_name)
        if self.config.name:
            return self.config.name
        else:
            # Fallback to container-based naming with timestamp
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            container_ids = self.config.omero.get_all_container_ids()
            if len(container_ids) > 1:
                return f"{self.config.omero.container_type}_multi_{len(container_ids)}_{timestamp}"
            else:
                container_id = container_ids[0] if container_ids else 0
                return f"{self.config.omero.container_type}_{container_id}_{timestamp}"

    def _identity_fields(self, image_meta: ImageMeta) -> dict:
        """ImageAnnotation fields that say which image a unit belongs to."""
        uri = image_meta.image_uri
        fields = {
            "image_uri": uri,
            "image_id": parse_omero_uri(uri),
            "image_name": image_meta.name,
            "level": self.config.zarr.level if self.config.source == "zarr" else 0,
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
        training = self.config.training
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

    def _plan_split(self, uris: list[str]) -> dict[str, str]:
        """Assign images to splits, keeping every split unit within one split.

        See :func:`~omero_annotate_ai.trainingset.planning.plan_split`. In count
        mode the fractions in the config are updated to the resulting split.
        Returns ``{uri: category}`` for the selected images, training first.
        """
        training = self.config.training
        plan = plan_split(
            [self._image_info(uri) for uri in uris],
            self._split_spec(),
            rng_for(self._seed, "split"),
        )
        if not training.segment_all and plan.n_units > 0:
            training.train_fraction = round(training.train_n / plan.n_units, 4)
            training.validation_fraction = round(training.validate_n / plan.n_units, 4)
            training.test_fraction = round(training.test_n / plan.n_units, 4)
        message = plan.shortfall_message()
        if message is not None:
            warnings.warn(message, UserWarning, stacklevel=2)
        return dict(plan.assignment)

    def _prepare_processing_units(self, images_list: list[Any]) -> None:
        """Prepare processing units by populating config.annotations."""

        # Clear existing annotations first
        self.config.annotations.clear()

        metas = [self.image_source.metadata(image) for image in images_list]
        plan = self._plan_split([meta.image_uri for meta in metas])
        by_uri = {
            meta.image_uri: (image, meta) for image, meta in zip(images_list, metas)
        }

        # Create ImageAnnotation objects for each processing unit
        for uri, category in plan.items():
            image, _ = by_uri[uri]
            for annotation in self._units_for_image(image, category):
                self.config.add_annotation(annotation)

    def _coverage(self) -> CoverageSpec:
        """What to annotate in each image, from ``config.spatial_coverage``."""
        sc = self.config.spatial_coverage

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

    def _units_for_image(self, image: Any, category: str) -> list[ImageAnnotation]:
        """Processing units (timepoints, z-slices, patches) for one image."""
        meta = self.image_source.metadata(image)
        units = units_for_image(
            meta, self._identity(meta), self._coverage(), category, self._seed
        )
        return [_annotation_from_unit(unit) for unit in units]

    def _upload_annotation_config_to_omero(self) -> int | None:
        """Replace the training set's one config YAML on OMERO with the local one."""
        if self.table_id in (None, -1) or self.config_file_path is None:
            return None
        return replace_table_config(
            self.conn,
            table_id=self.table_id,
            file_path=str(self.config_file_path),
            container_type=self.config.omero.container_type,
            container_id=self.config.omero.get_primary_container_id(),
        )

    def _get_timepoints_for_image(self, image) -> list[int]:
        """Get timepoints to process for an image based on configuration."""
        meta = self.image_source.metadata(image)
        return timepoints_for(meta, self._coverage(), self._seed)

    def _get_z_slices_for_image(self, image) -> list[int]:
        """Get z-slices to process for an image based on configuration."""
        meta = self.image_source.metadata(image)
        return z_slices_for(meta, self._coverage(), self._seed)

    def _generate_patch_coordinates(
        self, image
    ) -> tuple[list[tuple[int, int]], tuple[int, int]]:
        """Patch origins for an image and the patch size actually used."""
        meta = self.image_source.metadata(image)
        sc = self.config.spatial_coverage
        return generate_patch_coordinates(
            image_shape=(meta.size_y, meta.size_x),
            patch_size=sc.patch_size,
            n_patches=sc.patches_per_image,
            random_patch=sc.random_patches,
            rng=rng_for(self._seed, "patches", meta.image_uri),
        )

    def _run_micro_sam_annotation(self, batch_data: list[tuple]) -> dict:
        """Let the annotator annotate a batch; returns its metadata and mask files."""
        images = [
            self._load_image_data(image_obj, meta)
            for image_obj, _annotation_id, meta, _row_idx in batch_data
        ]
        contexts = self._build_context_images(batch_data)
        output_path = Path(self.config.output.output_directory)
        mask_dir = output_path / "annotations"
        mask_dir.mkdir(exist_ok=True)
        specs = self.config.spatial_coverage.get_context_channel_specs()
        self._batches_run += 1
        batch = Batch(
            index=self._batches_run,
            items=tuple(
                BatchItem(annotation_id=annotation_id, pixels=pixels, context=context)
                for (_obj, annotation_id, _meta, _row), pixels, context in zip(
                    batch_data, images, contexts, strict=True
                )
            ),
            mask_dir=mask_dir,
            embedding_dir=output_path / "sam_embeddings",
            model_type=self.config.ai_model.pretrained_from,
            is_volumetric=self.config.spatial_coverage.three_d,
            context_colormaps={name: cmap for _ch, name, cmap in specs},
        )
        annotator = (
            self.annotator if self.annotator is not None else MicroSamAnnotator()
        )
        masks = annotator.annotate(batch)
        return {
            "metadata": [
                (annotation_id, meta.copy(), row_idx)
                for _obj, annotation_id, meta, row_idx in batch_data
            ],
            "masks": masks,
        }

    def _load_image_data(self, image_obj, metadata: dict) -> np.ndarray:
        """Load the plane or volume an annotation unit refers to, cropped to its patch."""
        timepoint = metadata["timepoint"]
        channel = metadata.get(
            "channel_override", self.config.spatial_coverage.get_label_channel()
        )

        region = None
        if self.config.spatial_coverage.use_patches and "patch_x" in metadata:
            # Use the actual patch dimensions from metadata instead of config
            region = (
                metadata["patch_x"],
                metadata["patch_y"],
                metadata.get("patch_width", self.config.spatial_coverage.patch_size[1]),
                metadata.get(
                    "patch_height", self.config.spatial_coverage.patch_size[0]
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
            print(f"Warning: Image data appears to be all zeros for image {image_uri}")
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

    def _build_context_images(self, batch_data: list[tuple]) -> list[dict]:
        """Build context_images parallel list for the context channel hook.

        Returns a list of dicts, one per batch item. Each dict maps
        channel display name -> numpy array. Empty dicts when no context
        channels are configured.
        """
        if not self.config.spatial_coverage.uses_context_channels():
            return [{} for _ in batch_data]

        specs = self.config.spatial_coverage.get_context_channel_specs()
        context_images = []
        for image_obj, _annotation_id, meta, _row_idx in batch_data:
            per_item = {}
            for ch_idx, ch_name, _cmap in specs:
                ctx_meta = {**meta, "channel_override": ch_idx}
                arr = self._load_image_data(image_obj, ctx_meta)
                per_item[ch_name] = arr
            context_images.append(per_item)
        return context_images

    def _process_annotation_results(self, annotation_results: dict) -> None:
        """Process annotation results, upload ROIs to OMERO, and update config annotations."""

        self._debug_print("Processing annotation results...")

        if not annotation_results:
            self._debug_print("No annotation results received")
            return

        # Results carry the annotator's mask files ("masks"), or - from custom
        # annotation functions - the folder they were written to ("annotations_path").
        try:
            metadata = annotation_results["metadata"]
            masks = annotation_results.get("masks")
            if masks is None:
                masks = collect_series_masks(
                    Path(annotation_results["annotations_path"]),
                    [annotation_id for annotation_id, _meta, _row in metadata],
                )
        except KeyError as e:
            self._debug_print(f"Missing required key: {e}")
            return
        self._debug_print(
            f"Found {len(masks)} annotation files for {len(metadata)} metadata entries"
        )

        # Process each annotation metadata entry
        updated_count = 0
        for i, (annotation_id, meta, _row_idx) in enumerate(metadata):
            # Find matching annotation in config by annotation_id or sequence_val
            matching_annotation = None
            for annotation in self.config.annotations:
                # Try both annotation_id and sequence_val for backward compatibility
                if annotation.annotation_id == annotation_id or (
                    hasattr(annotation, "sequence_val")
                    and annotation.sequence_val == annotation_id
                ):
                    matching_annotation = annotation
                    break

            if not matching_annotation:
                self._debug_print(
                    f"No matching annotation for annotation_id: '{annotation_id}'"
                )
                continue

            tiff_file = masks.get(annotation_id)
            if tiff_file is None:
                self._debug_print(f"No annotation file for metadata entry {i + 1}")
                continue

            # Process based on mode
            if self.config.workflow.read_only_mode:
                # Read-only mode: save locally (category tracked in config, not folders)
                matching_annotation.mark_processed()

                # Save mask to output folder
                output_dir = (
                    Path(self.config.output.output_directory) / "annotation_output"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                local_file = output_dir / f"{annotation_id}_mask.tif"

                shutil.move(tiff_file, local_file)

                # Also save the original image to input folder
                self._save_original_image_for_annotation(meta, annotation_id)

                updated_count += 1

            else:
                # Upload mode: Upload ROIs to OMERO
                try:
                    image_id = meta.get("image_id")
                    patch_offset = None

                    if self.config.spatial_coverage.use_patches:
                        patch_offset = (meta.get("patch_x", 0), meta.get("patch_y", 0))

                    # Upload ROIs and labels using the existing function
                    label_id, roi_id = self.omero_writes.upload_rois_and_labels(
                        conn=self.conn,
                        image_id=image_id,
                        annotation_file=str(tiff_file),
                        patch_offset=patch_offset,
                        trainingset_name=self.config.name,
                        trainingset_description=f"Training set: {self.config.name}",
                        z_slice=meta.get("z_slice"),
                        channel=meta.get("channel"),
                        timepoint=meta.get("timepoint"),
                        is_volumetric=meta.get("is_volumetric"),
                        z_start=meta.get("z_start"),
                        tag=self._upload_tag(matching_annotation.annotation_id),
                    )

                    # Update config annotation with OMERO IDs
                    matching_annotation.mark_processed(
                        annotation_type=self.config.annotation_methodology.annotation_type,
                        roi_id=roi_id,
                        label_id=label_id,
                    )

                    updated_count += 1
                    self._debug_print(
                        f"Uploaded ROI for annotation_id '{annotation_id}' (label_id: {label_id}, roi_id: {roi_id})"
                    )

                except Exception as e:
                    self._debug_print(f"Error uploading ROI for '{annotation_id}': {e}")
                    # Still mark as processed but without OMERO IDs
                    matching_annotation.mark_processed()
                    updated_count += 1

        # Show final results
        processed_count = sum(1 for ann in self.config.annotations if ann.processed)
        self._debug_print(f"Updated {updated_count}/{len(metadata)} annotations")
        self._debug_print(
            f"Config now has {processed_count}/{len(self.config.annotations)} processed annotations"
        )

    def _save_original_image_for_annotation(
        self, meta: dict, annotation_id: str
    ) -> None:
        """Save the original image used for annotation.

        The channel being annotated goes to annotation_input/{annotation_id}.tif.
        Category metadata is tracked in config.yaml, not in the folder structure.

        When separate channels are configured (label_channel != training_channels),
        the training channel goes to model_input/{annotation_id}.tif - the folder
        reorganize_local_data_for_training() reads it from.

        Args:
            meta: Metadata dictionary containing image information
            annotation_id: Unique annotation identifier
        """
        image_key = meta.get("image_uri") or meta.get("image_id")
        if not image_key:
            self._debug_print(f"No image found for annotation {annotation_id}")
            return

        image_obj = self.image_source.get_image(image_key)
        output_path = Path(self.config.output.output_directory)
        folders = self._get_input_folders(output_path)

        # The channel being annotated (always)
        saved_path = self._save_training_image(
            image_obj, meta, annotation_id, folders["annotation_input"]
        )
        if saved_path is None:
            self._debug_print(f"Could not save original image for {annotation_id}")

        # The channel the model consumes, when it differs from the annotated one
        if "model_input" in folders:
            training_channels = self.config.spatial_coverage.get_training_channels()
            train_meta = {**meta, "channel_override": training_channels[0]}
            folders["model_input"].mkdir(parents=True, exist_ok=True)
            train_path = self._save_training_image(
                image_obj, train_meta, annotation_id, folders["model_input"]
            )
            if train_path is None:
                self._debug_print(
                    f"Could not save model channel image for {annotation_id}"
                )

    def _save_image_to_disk(
        self, image_data: np.ndarray, file_path: Path, axes: str | None = None
    ) -> bool:
        """Save image data to disk using tifffile (preferred) or imageio fallback.

        Args:
            image_data: NumPy array of image data to save
            file_path: Path where the image should be saved
            axes: Optional axes string for tifffile metadata (e.g. 'YXC', 'ZYXC')

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            import tifffile

            kwargs = {"metadata": {"axes": axes}} if axes else {}
            tifffile.imwrite(file_path, image_data, **kwargs)
            self._debug_print(
                f"Saved (tifffile): {file_path} - shape: {image_data.shape}, range: [{np.min(image_data)}-{np.max(image_data)}]"
            )
            return True
        except ImportError:
            try:
                from imageio import imwrite

                imwrite(file_path, image_data)
                self._debug_print(
                    f"Saved (imageio): {file_path} - shape: {image_data.shape}, range: [{np.min(image_data)}-{np.max(image_data)}]"
                )
                return True
            except ImportError:
                self._debug_print(
                    "Could not save image - tifffile and imageio not available"
                )
                return False
        except Exception as e:
            self._debug_print(f"Error saving image to {file_path}: {e}")
            return False

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
                self._debug_print(f"No image object for annotation {annotation_id}")
                return None

            image_data = self._load_image_data(image_obj, meta)
            if image_data is None:
                self._debug_print(f"No image data for annotation {annotation_id}")
                return None

            # Validate image data
            if np.all(image_data == 0):
                self._debug_print(
                    f"Warning: Image {annotation_id} appears to be all black (all zeros)"
                )

            output_folder.mkdir(parents=True, exist_ok=True)
            image_file = output_folder / f"{annotation_id}.tif"

            if self._save_image_to_disk(image_data, image_file):
                return image_file
            return None

        except Exception as e:
            self._debug_print(f"Error saving training image for {annotation_id}: {e}")
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
        filters = self.config.omero.well_filters
        if filters:
            print(f"Applying well filters: {filters}")
            print(f"Filter mode: {self.config.omero.well_filter_mode}")

        well_ids = ezomero.get_well_ids(self.conn, plate=container_id)
        print(f"Found {len(well_ids)} wells in plate {container_id}")

        image_ids = []
        matched_wells = 0
        for well_id in well_ids:
            well_obj = self.conn.getObject("Well", well_id)
            if well_obj is None:
                continue
            well_kv_pairs = self._well_kv(well_obj)
            if filters:
                matches_filter = self._check_well_filter(well_kv_pairs, filters)
                if matches_filter != (self.config.omero.well_filter_mode == "include"):
                    self._debug_print(f"  Well {well_id}: filtered out")
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
            print(
                f"After well filtering: {matched_wells}/{len(well_ids)} wells matched, {len(image_ids)} images selected"
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
        elif container_type == "project":
            dataset_ids = ezomero.get_dataset_ids(self.conn, project=container_id)
            image_ids = []
            for ds_id in dataset_ids:
                image_ids.extend(ezomero.get_image_ids(self.conn, dataset=ds_id))
            return image_ids
        elif container_type == "plate":
            return self._get_image_ids_from_plate(container_id)
        elif container_type == "screen":
            plate_ids = ezomero.get_plate_ids(self.conn, screen=container_id)
            image_ids = []
            for plate_id in plate_ids:
                image_ids.extend(ezomero.get_image_ids(self.conn, plate=plate_id))
            return image_ids
        elif container_type == "image":
            return [container_id]
        else:
            raise ValueError(f"Unsupported container type: {container_type}")

    def get_image_uris(self) -> list[str]:
        """URIs of the images the config selects (OMERO container or OME-Zarr URIs)."""
        if self.config.source == "zarr":
            refs = discover_zarr_images(
                self.config.zarr, output_dir=Path(self.config.output.output_directory)
            )
            self._zarr_refs = {ref.uri: ref for ref in refs}
            print(f"Found {len(refs)} OME-Zarr images")
            return list(self._zarr_refs)
        return [omero_uri(i) for i in self.get_image_ids_from_container()]

    def get_image_ids_from_container(self) -> list[int]:
        """Get image IDs from the configured OMERO container(s).

        Supports multiple containers of the same type via container_ids.
        For plate containers, applies well filtering if well_filters are configured.

        Returns:
            List of unique image IDs from all configured containers.
        """
        container_type = self.config.omero.container_type
        container_ids = self.config.omero.get_all_container_ids()

        if not container_ids:
            raise ValueError("No container IDs configured")

        all_image_ids = []

        for container_id in container_ids:
            print(f"Loading image IDs from {container_type} {container_id}")
            image_ids = self._get_image_ids_from_single_container(
                container_type, container_id
            )
            print(f"  Found {len(image_ids)} image IDs from container {container_id}")
            all_image_ids.extend(image_ids)

        # Remove duplicates while preserving order
        unique_image_ids = list(dict.fromkeys(all_image_ids))

        if len(container_ids) > 1:
            print(
                f"Total: {len(unique_image_ids)} unique image IDs from {len(container_ids)} containers"
            )
        else:
            print(f"Found {len(unique_image_ids)} image IDs")

        return unique_image_ids

    def get_images_by_ids(self, image_ids: list[int]) -> list[Any]:
        """Get image handles for the given image IDs, skipping ones that do not exist."""
        images = []
        for img_id in image_ids:
            try:
                images.append(self.image_source.get_image(img_id))
            except LookupError:
                self._debug_print(f"Image {img_id} not found; skipped")
        print(f"Loaded {len(images)} images (by ID)")
        return images

    def get_images_from_container(self) -> list[Any]:
        """Get OMERO image objects from the configured container.

        Returns:
            List of OMERO image objects
        """
        image_ids = self.get_image_ids_from_container()
        return self.get_images_by_ids(image_ids)

    def _save_local_progress_from_annotations(self, completed_annotations) -> None:
        """Save progress locally in read-only mode using completed annotations."""
        output_path = Path(self.config.output.output_directory)
        table_file = output_path / "tracking_table.csv"

        # Convert current config state to DataFrame for local storage
        df = self.config.to_dataframe()

        # Mark completed annotations as processed
        for annotation in completed_annotations:
            mask = df["annotation_id"] == annotation.annotation_id
            df.loc[mask, "processed"] = True
            df.loc[mask, "completed_timestamp"] = annotation.annotation_created_at

        df.to_csv(table_file, index=False)
        completed_count = df["processed"].sum()
        total_count = len(df)
        print(f"Local progress saved: {completed_count}/{total_count} units completed")
        print(f" Progress file: {table_file}")

    def _upload_tag(self, annotation_id: str) -> str:
        """Tag for an upload, tying it to this training set's table."""
        if self.table_id in (None, -1):
            raise RuntimeError(
                "The tracking table must exist before anything is uploaded"
            )
        return tag_for(self.table_id, annotation_id)

    def _find_existing_table(self) -> int:
        """The table to resume: ``config.omero.table_id``, which must exist.

        Training sets are identified by their table ID, never by name.

        Raises:
            ValueError: No table ID is set, or the table does not exist.
        """
        table_id = self.config.omero.table_id
        if not table_id:
            raise ValueError(
                "resume_from_table is set but config.omero.table_id is not: "
                "choose the training set (table) to resume"
            )
        if self.conn.getObject("FileAnnotation", table_id) is None:
            raise ValueError(
                f"Tracking table {table_id} not found (or not visible to you)"
            )
        print(f"Resuming table ID {table_id}")
        return table_id

    def _write_omero_table(self) -> int:
        """Write the config's rows to the tracking table, in place.

        The table keeps its ID: changed rows are updated, new rows appended.
        Creates the table first if the run has none yet.

        Returns:
            The table ID.
        """
        if self.table_id in (None, -1):
            return self.create_tracking_table()
        print("Updating OMERO table...")
        result = self.omero_writes.write_config_to_table(
            self.conn, self.table_id, self.config
        )
        print(
            f"OMERO table updated (ID: {self.table_id}; "
            f"{result['updated']} rows updated, {result['added']} added)"
        )
        return self.table_id

    def initialize_workflow(
        self, images_list: list[Any] | None = None
    ) -> tuple[int, list[Any]]:
        """Initialize the workflow: setup directories, load only required images, handle resume logic.

        Args:
            images_list: Optional list of OMERO image objects. If None, loads only required images from container.
        Returns:
            Tuple of (table_id, images_list)
        """
        # Setup directories
        output_path = self._setup_directories()
        self._cleanup_embeddings(output_path)

        # If images_list is not provided, select only required images
        if images_list is None:
            image_ids = self.get_image_uris()
            n_total = len(image_ids)
            if n_total == 0:
                where = (
                    f"OME-Zarr URIs {self.config.zarr.uris}"
                    if self.config.source == "zarr"
                    else f"{self.config.omero.container_type} {self.config.omero.container_id}"
                )
                raise ValueError(f"No images found in {where}")

            # Load only the images of the selected split units; their categories are
            # assigned again in _prepare_processing_units, unit by unit.
            selected_image_ids = list(self._plan_split(image_ids))
            images_list = self.get_images_by_ids(selected_image_ids)
        else:
            print(f"Loaded {len(images_list)} images (provided)")

        # Handle resume logic (one-time sync at workflow start)
        if (
            self.config.workflow.resume_from_table
            and not self.config.workflow.read_only_mode
        ):
            existing_table_id = self._find_existing_table()
            print("Syncing OMERO table state to config...")
            sync_omero_table_to_config(self.conn, existing_table_id, self.config)
            self.set_table_id(existing_table_id)

        # Always return int for table_id (use -1 if None)
        table_id = self.table_id if self.table_id is not None else -1
        return table_id, images_list

    def define_annotation_schema(self, images_list: list[Any]) -> AnnotationConfig:
        """Define the annotation schema by populating config.annotations.

        Args:
            images_list: List of OMERO image objects to process

        Returns:
            Updated AnnotationConfig with populated annotations
        """
        # Preserve existing annotations if already present in config
        if self.config.annotations:
            print(
                f"Using existing {len(self.config.annotations)} annotations from config"
            )
            result = validate_annotations_against_config(self.config)
            for w in result.warnings:
                print(f"Warning [{w.field}]: {w.message}")
            if not result.is_valid:
                error_summary = "; ".join(
                    f"{e.field}: {e.message}" for e in result.errors
                )
                raise ValueError(
                    f"Existing annotations are inconsistent with current config "
                    f"({result.annotation_count} annotations checked). "
                    f"Errors: {error_summary}. "
                    f"Fix the config or clear config.annotations to regenerate."
                )
            if self.config.workflow.resume_from_table:
                added = self._extend_annotations()
                if (
                    added
                    and self.table_id not in (None, -1)
                    and not self.config.workflow.read_only_mode
                ):
                    self._write_omero_table()
            return self.config

        # Generate new annotation schema
        print("Defining annotation schema...")
        self._prepare_processing_units(images_list)
        print(f"Schema defined: {len(self.config.annotations)} annotations")
        return self.config

    def _extend_annotations(self) -> int:
        """Add images when a resumed run asks for more than the table holds.

        See :func:`~omero_annotate_ai.trainingset.planning.plan_extension`. Existing
        rows are never removed, so a lower count leaves the table as it is.

        Returns:
            Number of images added.
        """
        training = self.config.training
        if training.segment_all:
            return 0

        existing = self.config.annotations
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
            print(
                f"Only {got} unused images left for {category} (asked for {asked} more)"
            )

        added = 0
        for category, uris in plan.picks.items():
            for image in self.get_images_by_ids(list(uris)):
                for annotation in self._units_for_image(image, category):
                    self.config.add_annotation(annotation)
                added += 1

        if added:
            print(
                f"Added {added} new images to the table "
                f"(now {len({a.image_uri for a in self.config.annotations})} images)"
            )
        return added

    def create_tracking_table(self) -> int:
        """Create OMERO tracking table from the annotation schema.

        Returns:
            Table ID of created/updated OMERO table
        """
        if self.config.workflow.read_only_mode:
            print("Read-only mode: Skipping OMERO table creation")
            self.table_id = -1  # Mock table ID for read-only mode
            return self.table_id

        if self.table_id is None:
            print("Creating OMERO tracking table...")
            table_id = create_omero_tracking_table(
                self.conn,
                self.config,
                table_title=self._get_table_title(),
                container_type=self.config.omero.container_type,
                container_ids=self.config.omero.get_all_container_ids(),
            )
            self.set_table_id(table_id)

        return self.table_id

    def _convert_annotations_to_processing_units(
        self, annotations: list
    ) -> list[tuple]:
        """Convert annotation objects to processing unit format.

        Args:
            annotations: List of ImageAnnotation objects

        Returns:
            List of processing units in format (image_id, annotation_id, metadata, row_index)
        """
        processing_units = []
        for i, annotation in enumerate(annotations):
            metadata = {
                "image_id": annotation.image_id,
                "image_uri": annotation.image_uri,
                "timepoint": annotation.timepoint,
                "z_slice": annotation.z_slice,
                "z_start": annotation.z_start,
                "z_end": annotation.z_end,
                "z_length": annotation.z_length,
                "patch_x": annotation.patch_x,
                "patch_y": annotation.patch_y,
                "category": annotation.category,
                "channel": annotation.channel,
                "is_volumetric": annotation.is_volumetric,
            }
            processing_units.append(
                (annotation.image_uri, annotation.annotation_id, metadata, i)
            )
        return processing_units

    def _run_annotation_batches(
        self, processing_units: list[tuple], annotation_func
    ) -> int:
        """Run annotation function on batches of processing units.

        Args:
            processing_units: List of processing units to process
            annotation_func: Function to call for annotation (e.g., self._run_micro_sam_annotation)

        Returns:
            Number of successfully processed units
        """
        batch_size = self.config.workflow.batch_size or len(processing_units)
        processed_count = 0

        for i in range(0, len(processing_units), batch_size):
            batch = processing_units[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(processing_units) + batch_size - 1) // batch_size

            print(f"\nProcessing batch {batch_num}/{total_batches}")
            self._debug_print(f"   Batch size: {len(batch)}")

            # Show config state before processing
            processed_before = sum(
                1 for ann in self.config.annotations if ann.processed
            )
            self._debug_print(
                f"   Config state before: {processed_before}/{len(self.config.annotations)} processed"
            )

            try:
                # Load image objects for batch
                batch_with_images = []
                for img_id, seq_val, meta, row_idx in batch:
                    image_obj = self.image_source.get_image(img_id)
                    batch_with_images.append((image_obj, seq_val, meta, row_idx))

                # Run the specific annotation function
                self._debug_print("   Running annotation...")
                annotation_results = annotation_func(batch_with_images)
                self._debug_print(
                    f"   Annotation completed, got results: {type(annotation_results)}"
                )

                # Process results and update config
                self._debug_print("   Processing annotation results...")
                self._process_annotation_results(annotation_results)

                # Show config state after processing
                processed_after = sum(
                    1 for ann in self.config.annotations if ann.processed
                )
                self._debug_print(
                    f"   Config state after: {processed_after}/{len(self.config.annotations)} processed"
                )

                # Update OMERO table and save config
                if not self.config.workflow.read_only_mode:
                    self._debug_print("   Updating OMERO table...")
                    self._write_omero_table()
                    self._debug_print(f"   OMERO table updated (ID: {self.table_id})")
                else:
                    self._save_local_progress_from_annotations(
                        self.config.get_processed()
                    )

                self._debug_print("   Auto-saving config...")
                self._auto_save_config()
                self._debug_print("   Config saved")

                processed_count += len(batch)

            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                traceback.print_exc()
                raise

        return processed_count

    def _finalize_workflow(self, processed_count: int) -> None:
        """Finalize the workflow with cleanup and uploads.

        Args:
            processed_count: Number of units that were processed
        """
        # Final config save
        self._auto_save_config()

        # Upload config to OMERO (only if not in read-only mode)
        if not self.config.workflow.read_only_mode:
            self._upload_annotation_config_to_omero()

        if processed_count > 0:
            print(f"Workflow completed successfully! Processed {processed_count} units")
        else:
            print("Workflow failed - no units were processed")

    def run_microsam_annotation(self) -> tuple[int, AnnotationConfig]:
        """Run micro-SAM annotation workflow on unprocessed annotations.

        Returns:
            Tuple of (table_id, updated_config)
        """
        if not self.config.annotations:
            raise ValueError(
                "No annotation schema defined - run define_annotation_schema() first"
            )

        # Get unprocessed annotations
        unprocessed_annotations = self.config.get_unprocessed()
        if not unprocessed_annotations:
            print("All annotations already processed!")
            if self.table_id is None:
                self.table_id = -1
            return self.table_id, self.config

        print(
            f"Running micro-SAM annotation on {len(unprocessed_annotations)} unprocessed annotations"
        )

        # Convert to processing units and run annotation
        processing_units = self._convert_annotations_to_processing_units(
            unprocessed_annotations
        )
        processed_count = self._run_annotation_batches(
            processing_units, self._run_micro_sam_annotation
        )

        # Finalize workflow
        self._finalize_workflow(processed_count)

        if self.table_id is None:
            self.table_id = -1
        return self.table_id, self.config

    def run_micro_sam_annotation(self) -> tuple[int, AnnotationConfig]:
        """Alias for run_microsam_annotation for backward compatibility."""
        return self.run_microsam_annotation()

    def run_cellpose_preparation(self) -> tuple[int, AnnotationConfig]:
        """Run Cellpose preparation workflow - save images locally for training.

        Returns:
            Tuple of (table_id, updated_config)
        """
        if not self.config.annotations:
            raise ValueError(
                "No annotation schema defined - run define_annotation_schema() first"
            )

        # Get unprocessed annotations
        unprocessed_annotations = self.config.get_unprocessed()
        if not unprocessed_annotations:
            print("All annotations already processed!")
            table_id = self.table_id if self.table_id is not None else -1
            return table_id, self.config

        print(f"Preparing {len(unprocessed_annotations)} images for Cellpose training")

        # Convert to processing units and save images
        processing_units = self._convert_annotations_to_processing_units(
            unprocessed_annotations
        )
        self._save_images_for_cellpose(processing_units)

        # Mark all as processed (since we just saved them)
        for annotation in unprocessed_annotations:
            annotation.mark_processed()

        # Update tracking
        if not self.config.workflow.read_only_mode:
            self._write_omero_table()
        else:
            self._save_local_progress_from_annotations(self.config.get_processed())

        self._auto_save_config()
        print(
            f"Cellpose preparation completed: {len(unprocessed_annotations)} images saved"
        )

        table_id = self.table_id if self.table_id is not None else -1
        return table_id, self.config

    def _save_images_for_cellpose(self, processing_units: list[tuple]) -> None:
        """Save images locally for Cellpose training.

        Single-channel: saves to annotation_input/{annotation_id}.tif
        Separate-channel: saves the annotation channel to annotation_input/{annotation_id}.tif
                          and the model channel to model_input/{annotation_id}.tif
        Category metadata is tracked in config.yaml, not in folder structure.
        """
        output_path = Path(self.config.output.output_directory)
        folders = self._get_input_folders(output_path)

        for img_id, annotation_id, meta, row_idx in processing_units:
            print(f"Processing image {annotation_id} (ID: {img_id})")

            image_obj = self.image_source.get_image(img_id)

            if "model_input" in folders:
                # Separate-channel: the channel you annotate → annotation_input/ (single-channel)
                if (
                    self._save_training_image(
                        image_obj, meta, annotation_id, folders["annotation_input"]
                    )
                    is None
                ):
                    print(
                        f"Warning: Could not save annotation channel image for {annotation_id}"
                    )

                # model_input/ → all training channels stacked channels-last: (Y,X,C) or (Z,Y,X,C)
                training_channels = self.config.spatial_coverage.get_training_channels()
                stacked = self._load_and_stack_channels(
                    image_obj, meta, training_channels
                )
                axes_str = "ZYXC" if stacked.ndim == 4 else "YXC"
                train_file = folders["model_input"] / f"{annotation_id}.tif"
                folders["model_input"].mkdir(parents=True, exist_ok=True)
                if not self._save_image_to_disk(stacked, train_file, axes=axes_str):
                    print(
                        f"Warning: Could not save model channel image for {annotation_id}"
                    )
            else:
                # Single-channel: → annotation_input/ (serves both roles)
                if (
                    self._save_training_image(
                        image_obj, meta, annotation_id, folders["annotation_input"]
                    )
                    is None
                ):
                    print(f"Warning: Could not save image {annotation_id}")

    def _process_annotation_file(
        self, annotation_file: Path, annotation_id: str, matching_annotation
    ) -> bool:
        """Process a single annotation file - either upload to OMERO or save locally.

        This is a generalized method that can be reused for any annotation workflow.

        Args:
            annotation_file: Path to the annotation mask file
            annotation_id: Unique annotation identifier
            matching_annotation: ImageAnnotation object from config

        Returns:
            True if successfully processed, False otherwise
        """
        try:
            if self.config.workflow.read_only_mode:
                # Read-only mode: save locally (category tracked in config, not folders)
                output_dir = (
                    Path(self.config.output.output_directory) / "annotation_output"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                local_file = output_dir / f"{annotation_id}_mask.tif"

                # Copy file to output location if not already there
                if annotation_file != local_file:
                    shutil.copy(annotation_file, local_file)

                # Mark as processed
                matching_annotation.processed = True
                now = datetime.now().isoformat()
                matching_annotation.annotation_updated_at = now
                if matching_annotation.annotation_created_at is None:
                    matching_annotation.annotation_created_at = now

                print(f"Processed locally: {annotation_id}")
                return True

            else:
                # Upload mode: Upload ROIs to OMERO
                image_id = matching_annotation.image_id
                patch_offset = None

                if matching_annotation.is_patch:
                    patch_offset = (
                        matching_annotation.patch_x,
                        matching_annotation.patch_y,
                    )

                # Upload ROIs and labels using the existing function
                label_id, roi_id = upload_rois_and_labels(
                    conn=self.conn,
                    image_id=image_id,
                    annotation_file=str(annotation_file),
                    patch_offset=patch_offset,
                    trainingset_name=self.config.name,
                    trainingset_description=f"Training set: {self.config.name}",
                    z_slice=matching_annotation.z_slice,
                    channel=matching_annotation.channel,
                    timepoint=matching_annotation.timepoint,
                    is_volumetric=matching_annotation.is_volumetric,
                    z_start=matching_annotation.z_start,
                    tag=self._upload_tag(annotation_id),
                )

                # Update config annotation with OMERO IDs
                matching_annotation.processed = True
                matching_annotation.label_id = label_id
                matching_annotation.roi_id = roi_id
                now = datetime.now().isoformat()
                matching_annotation.annotation_updated_at = now
                if matching_annotation.annotation_created_at is None:
                    matching_annotation.annotation_created_at = now
                matching_annotation.annotation_type = (
                    self.config.annotation_methodology.annotation_type
                )

                print(
                    f"Uploaded to OMERO: {annotation_id} (label_id: {label_id}, roi_id: {roi_id})"
                )
                return True

        except Exception as e:
            print(f"Error processing {annotation_id}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def collect_annotations_from_disk(
        self, folder_pattern: str = "annotation_output"
    ) -> tuple[int, AnnotationConfig]:
        """Collect annotation masks from disk and upload to OMERO.

        This method scans output folder(s) for annotation masks, matches them with
        the annotation schema, and processes them (upload to OMERO or local save).
        Can be used for CellPose, manual annotations, or any other external annotation tool.

        Args:
            folder_pattern: Folder name or glob pattern for folders containing annotation masks.
                           Default is "annotation_output" (the standard output folder).

        Returns:
            Tuple of (number_processed, updated_config)
        """
        if not self.config.annotations:
            raise ValueError("No annotation schema defined")

        output_path = Path(self.config.output.output_directory)

        # Find all annotation mask files matching the pattern
        annotation_files = []

        # Check if it's a direct folder name or a glob pattern
        if "*" in folder_pattern:
            # Glob pattern - search multiple folders
            for folder in output_path.glob(folder_pattern):
                if folder.is_dir():
                    for mask_file in folder.glob("*_mask.tif"):
                        annotation_files.append(mask_file)
                    for mask_file in folder.glob("*_mask.tiff"):
                        annotation_files.append(mask_file)
        else:
            # Direct folder name
            folder = output_path / folder_pattern
            if folder.is_dir():
                for mask_file in folder.glob("*_mask.tif"):
                    annotation_files.append(mask_file)
                for mask_file in folder.glob("*_mask.tiff"):
                    annotation_files.append(mask_file)

        if not annotation_files:
            print(f"No annotation masks found in '{folder_pattern}'")
            return 0, self.config

        print(f"Found {len(annotation_files)} annotation mask files")

        # Process each annotation file
        processed_count = 0
        for mask_file in annotation_files:
            # Extract annotation_id from filename
            annotation_id = mask_file.stem.replace("_mask", "")

            # Find matching annotation in config
            matching_annotation = None
            for annotation in self.config.annotations:
                if annotation.annotation_id == annotation_id:
                    matching_annotation = annotation
                    break

            if not matching_annotation:
                print(f"Warning: No matching annotation for {annotation_id}, skipping")
                continue

            if matching_annotation.processed and matching_annotation.roi_id is not None:
                print(
                    f"Skipping {annotation_id}: already uploaded (ROI: {matching_annotation.roi_id})"
                )
                continue

            # Process the annotation file
            if self._process_annotation_file(
                mask_file, annotation_id, matching_annotation
            ):
                processed_count += 1

        # Update tracking and save config
        if processed_count > 0:
            if not self.config.workflow.read_only_mode:
                print("Updating OMERO tracking table...")
                self._write_omero_table()

            print("Saving updated config...")
            self._auto_save_config()

            print(f"Successfully processed {processed_count} annotations")
        else:
            print("No new annotations to process")

        return processed_count, self.config

    def get_annotation_status_from_disk(
        self, folder_name: str = "annotation_output"
    ) -> dict:
        """Check status of annotations available on disk.

        Args:
            folder_name: Name of the output folder to check. Default is "annotation_output".

        Returns:
            Dictionary with status information about available annotations
        """
        output_path = Path(self.config.output.output_directory)
        output_folder = output_path / folder_name

        status = {
            "total_annotations": len(self.config.annotations),
            "processed_annotations": len(self.config.get_processed()),
            "pending_annotations": len(self.config.get_unprocessed()),
            "available_masks": [],
            "missing_masks": [],
        }

        # Check which annotation files exist
        for annotation in self.config.annotations:
            annotation_id = annotation.annotation_id

            # Check in output folder (category is tracked in config, not folder structure)
            mask_file = output_folder / f"{annotation_id}_mask.tif"
            if not mask_file.exists():
                mask_file = output_folder / f"{annotation_id}_mask.tiff"

            if mask_file.exists():
                status["available_masks"].append(
                    {
                        "annotation_id": annotation_id,
                        "image_name": annotation.image_name,
                        "category": annotation.category,
                        "uploaded": annotation.processed
                        and annotation.roi_id is not None,
                        "roi_id": annotation.roi_id,
                        "file_path": str(mask_file),
                    }
                )
            else:
                status["missing_masks"].append(
                    {
                        "annotation_id": annotation_id,
                        "image_name": annotation.image_name,
                        "category": annotation.category,
                    }
                )

        return status

    def run_cp_workflow(
        self, images_list: list[Any] | None = None
    ) -> tuple[int, AnnotationConfig]:
        """Complete Cellpose workflow: setup + schema + table + preparation.

        Args:
            images_list: Optional list of OMERO image objects

        Returns:
            Tuple of (table_id, config)
        """
        # Step 1: Initialize workflow
        table_id, images_list = self.initialize_workflow(images_list)

        # Step 2: Define annotation schema
        self.define_annotation_schema(images_list)

        # Step 3: Create tracking table
        self.create_tracking_table()

        # Step 4: Run Cellpose preparation
        return self.run_cellpose_preparation()

    # Convenience methods for common workflows
    def run_full_microsam_workflow(
        self, images_list: list[Any] | None = None
    ) -> tuple[int, AnnotationConfig]:
        """Complete micro-SAM workflow: setup + schema + table + annotation.

        Args:
            images_list: Optional list of OMERO image objects

        Returns:
            Tuple of (table_id, config)
        """
        # Step 1: Initialize workflow
        table_id, images_list = self.initialize_workflow(images_list)

        # Step 2: Define annotation schema
        self.define_annotation_schema(images_list)

        # Step 3: Create tracking table
        self.create_tracking_table()

        # Step 4: Run micro-SAM annotation
        return self.run_microsam_annotation()

    def run_full_micro_sam_workflow(
        self, images_list: list[Any] | None = None
    ) -> tuple[int, AnnotationConfig]:
        """Alias for run_full_microsam_workflow for backward compatibility."""
        return self.run_full_microsam_workflow(images_list)

    def prepare_training_data_from_local(
        self,
        output_dir: str | Path | None = None,
        file_mode: str = "copy",
        clean_existing: bool = True,
        include_test: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Reorganize locally-stored annotation data into training folder structure.

        This is a convenience method that wraps reorganize_local_data_for_training().
        It uses the pipeline's config and output directory settings.

        Works entirely offline - no OMERO connection required.

        Args:
            output_dir: Target directory for the training layout. Must not be inside the
                annotation directory. Default: the sibling <annotation_dir>_training/.
            file_mode: How to handle files:
                - "copy": Copy files (keeps originals) - default
                - "move": Move files (removes originals)
                - "symlink": Create symbolic links (falls back to copy on Windows)
            clean_existing: Remove existing training folders before reorganization
            include_test: If True, also create test_input/test_label folders
            verbose: Show detailed progress

        Returns:
            Dictionary with paths to created directories and statistics

        Raises:
            ValueError: If config has no annotations or no processed annotations
            FileNotFoundError: If annotation directory doesn't exist
        """
        from ..processing.training_functions import reorganize_local_data_for_training

        # Use config's output directory as the annotation source
        annotation_dir = Path(self.config.output.output_directory)

        # output_dir stays None here on purpose: reorganize_local_data_for_training
        # defaults it to the sibling <annotation_dir>_training/. It must not default to
        # annotation_dir - cleaning the training layout there would delete the source
        # images, so that case now raises.
        return reorganize_local_data_for_training(
            config=self.config,
            annotation_dir=annotation_dir,
            output_dir=output_dir,
            file_mode=file_mode,  # type: ignore
            clean_existing=clean_existing,
            include_test=include_test,
            verbose=verbose,
        )

    def run_custom_annotation(
        self, annotation_func, images_list: list[Any] | None = None
    ) -> tuple[int, AnnotationConfig]:
        """Run a custom annotation function with the standard workflow.

        Args:
            annotation_func: Custom annotation function to use
            images_list: Optional list of OMERO image objects

        Returns:
            Tuple of (table_id, config)
        """
        # Step 1: Initialize workflow
        table_id, images_list = self.initialize_workflow(images_list)

        # Step 2: Define annotation schema
        self.define_annotation_schema(images_list)

        # Step 3: Create tracking table
        self.create_tracking_table()

        # Step 4: Get unprocessed annotations and run custom function
        unprocessed_annotations = self.config.get_unprocessed()
        if not unprocessed_annotations:
            print("All annotations already processed!")
            table_id = self.table_id if self.table_id is not None else -1
            return table_id, self.config

        print(
            f"Running custom annotation on {len(unprocessed_annotations)} unprocessed annotations"
        )

        # Convert to processing units and run annotation
        processing_units = self._convert_annotations_to_processing_units(
            unprocessed_annotations
        )
        processed_count = self._run_annotation_batches(
            processing_units, annotation_func
        )

        # Finalize workflow
        self._finalize_workflow(processed_count)

        table_id = self.table_id if self.table_id is not None else -1
        return table_id, self.config


def create_pipeline(
    config: AnnotationConfig,
    conn=None,
    config_file_path: str | Path | None = None,
    *,
    annotator: Annotator | None = None,
    omero_writes: OmeroWrites | None = None,
) -> AnnotationPipeline:
    """Create a micro-SAM annotation pipeline with the given configuration.

    Args:
        config: AnnotationConfig object
        conn: OMERO connection object
        config_file_path: Optional path to the source YAML configuration file for persistence.
                         If None, uses config.config_file_path if available.
        annotator: Annotation tool; defaults to micro-SAM in napari. Pass e.g. a
            :class:`~omero_annotate_ai.annotators.ScriptedAnnotator` to run without
            a display.
        omero_writes: The OMERO writes to use (replaceable for tests).

    Returns:
        AnnotationPipeline instance
    """
    # Use provided path, or fall back to config's stored path
    actual_config_path = config_file_path or config.config_file_path

    return AnnotationPipeline(
        config,
        conn,
        actual_config_path,
        annotator=annotator,
        omero_writes=omero_writes,
    )
