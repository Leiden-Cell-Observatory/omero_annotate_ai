"""Main pipeline for OMERO annotation workflows."""

import random
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import ezomero

try:
    import napari
    from micro_sam.sam_annotator import image_series_annotator
    MICRO_SAM_AVAILABLE = True
except ImportError:
    MICRO_SAM_AVAILABLE = False
    napari = None
    image_series_annotator = None
except Exception as e:
    MICRO_SAM_AVAILABLE = False
    napari = None
    image_series_annotator = None

from .config import AnnotationConfig
from ..omero.omero_functions import (
    create_roi_namespace_for_table,
    initialize_tracking_table
)
from ..omero.omero_utils import get_dask_image_multiple, get_table_by_name
from ..processing.image_functions import generate_patch_coordinates


class AnnotationPipeline:
    """Main pipeline for running micro-SAM annotation workflows with OMERO."""

    def __init__(self, config: AnnotationConfig, conn=None):
        """Initialize the pipeline with configuration and OMERO connection.

        Args:
            config: AnnotationConfig object containing all parameters
            conn: OMERO connection object (BlitzGateway)
        """
        self.config = config
        self.conn = conn
        self.table_id = None  # Track current table ID
        self._validate_setup()

    def set_table_id(self, table_id: int) -> None:
        """Update the current table ID."""
        self.table_id = table_id

    def get_table_id(self) -> Optional[int]:
        """Get the current table ID."""
        return self.table_id

    def _validate_setup(self):
        """Validate the pipeline setup."""
        if self.conn is None:
            raise ValueError("OMERO connection is required")

        # Validate configuration
        self.config.validate()

    def _setup_directories(self):
        """Create necessary output directories."""
        output_path = Path(self.config.batch_processing.output_folder)

        # Create main directories - everything under the single output folder
        dirs = [
            output_path,
            output_path / "embed",
            output_path / "zarr",
            output_path / "annotations",  # Final annotations go here
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        return output_path

    def _cleanup_embeddings(self, output_path: Path):
        """Clean up any existing embeddings from interrupted runs."""
        embed_path = output_path / "embed"
        if embed_path.exists():
            for file in embed_path.glob("*"):
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)

    def _get_table_title(self) -> str:
        """Generate table title based on configuration."""
        # Use the existing trainingset_name from config
        if self.config.training.trainingset_name:
            return f"micro_sam_training_{self.config.training.trainingset_name}"
        else:
            # Fallback to container-based naming
            return f"micro_sam_training_{self.config.omero.container_type}_{self.config.omero.container_id}"

    def _initialize_tracking_table(self, images_list: List[Any]) -> int:
        """Initialize or resume tracking table for the annotation process."""
        
        import logging
        logger = logging.getLogger(__name__)

        # If table_id is already set, validate and use it
        if self.table_id is not None:
            logger.info(f"Using existing table ID: {self.table_id}")
            # Validate table exists
            try:
                table_obj = self.conn.getObject("FileAnnotation", self.table_id)
                if table_obj is None:
                    raise ValueError(f"Table with ID {self.table_id} not found")
                return self.table_id
            except Exception as e:
                raise ValueError(f"Cannot access table {self.table_id}: {e}")

        # Generate table name
        table_title = self._get_table_title()

        # Check if need to resume from existing table
        if self.config.workflow.resume_from_table:
            print(f"🔍 Looking for existing table: {table_title}")
            existing_table = get_table_by_name(self.conn, table_title)
            if existing_table:
                table_id = existing_table.getId()

                # Check workflow status before resuming
                status_map = self._get_workflow_status_map()
                if status_map:
                    status = status_map.get("workflow_status", "unknown")
                    completed = status_map.get("completed_units", "0")
                    total = status_map.get("total_units", "0")

                    if status == "complete":
                        print(f"✅ Workflow already complete: {completed}/{total} units processed")
                        print(f"📋 Table: {table_title} (ID: {table_id})")
                        self.table_id = table_id
                        return table_id
                    elif status == "incomplete":
                        print(f"🔄 Resuming incomplete workflow: {completed}/{total} units completed")
                        print(f"📋 Table: {table_title} (ID: {table_id})")
                        self.table_id = table_id
                        return table_id
                    else:
                        print(f"📋 Resuming from existing table: {table_title}")
                        self.table_id = table_id
                        return table_id
                else:
                    print(f"📋 Resuming from existing table: {table_title}")
                    self.table_id = table_id
                    return table_id

        # Create new table
        logger.info(f"Creating new tracking table: {table_title}")

        # Store ROI namespace for consistent naming
        roi_namespace = create_roi_namespace_for_table(table_title)
        logger.info(f"ROI namespace: {roi_namespace}")

        # Prepare processing units for table initialization
        processing_units = self._prepare_processing_units(images_list)

        # Initialize table with all planned processing units
        table_id = initialize_tracking_table(
            conn=self.conn,
            table_title=table_title,
            processing_units=processing_units,
            container_type=self.config.omero.container_type,
            container_id=self.config.omero.container_id,
            source_desc=self.config.omero.source_desc,
        )

        # Store configuration in OMERO
        config_dict = self.config.to_dict()

        # Flatten nested config and convert to strings
        flat_config = {}
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_config[f"{key}.{sub_key}"] = str(sub_value)
            else:
                flat_config[key] = str(value)

        # Add metadata
        flat_config.update({
            "config_type": "micro_sam_annotation_settings",
            "created_at": pd.Timestamp.now().isoformat(),
            "config_name": f"micro_sam_config_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        })

        try:
            config_ann_id = ezomero.post_map_annotation(
                self.conn,
                object_type=self.config.omero.container_type.capitalize(),
                object_id=self.config.omero.container_id,
                kv_dict=flat_config,
                ns="openmicroscopy.org/omero/annotation",
            )
            print(f"Stored configuration as annotation ID: {config_ann_id}")
        except Exception as e:
            print(f"Warning: Could not store configuration annotation: {e}")

        # Initialize workflow status map for new table
        self._update_workflow_status_map(table_id)

        # Store table ID internally
        self.table_id = table_id
        return table_id

    def _prepare_processing_units(self, images_list: List[Any]) -> List[Tuple]:
        """Prepare processing units based on configuration."""
        processing_units = []

        # Determine which images to process
        if self.config.training.segment_all:
            selected_images = images_list
            image_categories = ["training"] * len(images_list)
        else:
            # Select subset for training/validation
            n_total = len(images_list)
            n_train = min(self.config.training.train_n, n_total)
            n_val = min(self.config.training.validate_n, n_total - n_train)

            # Randomly select images
            shuffled_indices = list(range(n_total))
            random.shuffle(shuffled_indices)

            train_indices = shuffled_indices[:n_train]
            val_indices = shuffled_indices[n_train : n_train + n_val]

            selected_images = [images_list[i] for i in train_indices + val_indices]
            image_categories = ["training"] * n_train + ["validation"] * n_val

        # Create processing units for each image
        for image, category in zip(selected_images, image_categories):
            image_id = image.getId()

            # Handle timepoints
            timepoints = self._get_timepoints_for_image(image)

            # Handle z-slices
            z_slices = self._get_z_slices_for_image(image)

            # Create units for each timepoint/z-slice combination
            for t in timepoints:
                for z in z_slices:
                    if self.config.patches.use_patches:
                        # Create multiple units for patches
                        patch_coords = self._generate_patch_coordinates(image)
                        for i, (x, y) in enumerate(patch_coords):
                            processing_units.append(
                                (
                                    image_id,
                                    f"{t}_{z}_{i}",
                                    {
                                        "timepoint": t,
                                        "z_slice": z,
                                        "patch_x": x,
                                        "patch_y": y,
                                        "category": category,
                                        "model_type": self.config.micro_sam.model_type,
                                        "channel": self.config.omero.channel,
                                        "three_d": self.config.micro_sam.three_d,
                                    },
                                )
                            )
                    else:
                        # Single unit for full image
                        processing_units.append(
                            (
                                image_id,
                                f"{t}_{z}",
                                {
                                    "timepoint": t,
                                    "z_slice": z,
                                    "category": category,
                                    "model_type": self.config.micro_sam.model_type,
                                    "channel": self.config.omero.channel,
                                    "three_d": self.config.micro_sam.three_d,
                                },
                            )
                        )

        return processing_units

    def _get_timepoints_for_image(self, image) -> List[int]:
        """Get timepoints to process for an image based on configuration."""
        max_t = image.getSizeT()

        if self.config.micro_sam.timepoint_mode == "all":
            return list(range(max_t))
        elif self.config.micro_sam.timepoint_mode == "random":
            n_points = min(len(self.config.micro_sam.timepoints), max_t)
            return random.sample(range(max_t), n_points)
        else:  # specific
            return [t for t in self.config.micro_sam.timepoints if t < max_t]

    def _get_z_slices_for_image(self, image) -> List[int]:
        """Get z-slices to process for an image based on configuration."""
        max_z = image.getSizeZ()

        if self.config.micro_sam.z_slice_mode == "all":
            return list(range(max_z))
        elif self.config.micro_sam.z_slice_mode == "random":
            n_slices = min(len(self.config.micro_sam.z_slices), max_z)
            return random.sample(range(max_z), n_slices)
        else:  # specific
            return [z for z in self.config.micro_sam.z_slices if z < max_z]

    def _generate_patch_coordinates(self, image) -> List[Tuple[int, int]]:
        """Generate patch coordinates for an image."""
        patch_size = self.config.patches.patch_size
        patches_per_image = self.config.patches.patches_per_image
        random_patches = self.config.patches.random_patches

        image_shape = (image.getSizeY(), image.getSizeX())

        return generate_patch_coordinates(
            image_shape=image_shape,
            patch_size=patch_size,
            n_patches=patches_per_image,
            random_patch=random_patches,
        )

    def _run_micro_sam_annotation(self, batch_data: List[Tuple]) -> dict:
        """Run micro-SAM annotation on a batch of data."""
        if not MICRO_SAM_AVAILABLE:
            raise ImportError(
                "micro-sam and napari are required. Install micro-sam via conda: conda install -c conda-forge micro_sam"
            )

        # Configure napari settings to ensure proper blocking behavior
        napari_settings = napari.settings.get_settings()

        # Store original setting to restore later
        original_ipy_interactive = napari_settings.application.ipy_interactive

        # Force blocking behavior - this is crucial for proper event loop handling
        napari_settings.application.ipy_interactive = False
        napari_settings.application.save_window_geometry = False

        try:
            # Prepare image data for annotation
            images = []
            metadata = []

            for image_obj, sequence_val, meta, row_idx in batch_data:
                # Load image data from OMERO
                image_data = self._load_image_data(image_obj, meta)
                images.append(image_data)
                # Include image_id in metadata for later ROI upload
                meta_with_image_id = meta.copy()
                meta_with_image_id["image_id"] = image_obj.getId()
                metadata.append((sequence_val, meta_with_image_id, row_idx))

            # Set up output paths
            output_path = Path(self.config.batch_processing.output_folder)
            embedding_path = output_path / "embed"
            annotations_path = output_path / "annotations"
            annotations_path.mkdir(exist_ok=True)

            # Run micro-SAM annotation
            model_type = self.config.micro_sam.model_type

            # Run image series annotator with explicit napari.run() call
            viewer = image_series_annotator(
                images=images,
                output_folder=str(annotations_path),
                model_type=model_type,
                viewer=None,  # Let it create its own viewer
                return_viewer=True,  # Important: get the viewer back
                embedding_path=str(embedding_path),
                is_volumetric=self.config.micro_sam.three_d,
                skip_segmented=True
            )

            # Explicitly start the event loop - this will block until viewer is closed
            if napari is not None:
                napari.run()
            else:
                raise ImportError("napari is not available. Please install napari to run the annotation viewer.")

            # The following delay helps ensure all files are written before proceeding.
            # If file writing is synchronous, this may be unnecessary; profile or document as needed.
            time.sleep(0.5)

        finally:
            # Always restore the original setting
            napari_settings.application.ipy_interactive = original_ipy_interactive

        return {
            "metadata": metadata,
            "embedding_path": embedding_path,
            "annotations_path": annotations_path,
    }

    def _load_image_data(self, image_obj, metadata: dict) -> np.ndarray:
        """Load image data from OMERO based on metadata."""

        timepoint = metadata["timepoint"]
        z_slice = metadata["z_slice"]
        channel = self.config.omero.channel

        # Load image data using dask for efficiency
        image_data = get_dask_image_multiple(
            conn=self.conn,
            image_list=[image_obj],
            timepoints=[timepoint],
            channels=[channel],
            z_slices=[z_slice],
        )[
            0
        ]  # Get first (and only) image

        # Handle patches if needed
        if self.config.patches.use_patches and "patch_x" in metadata:
            patch_x = metadata["patch_x"]
            patch_y = metadata["patch_y"]
            patch_h, patch_w = self.config.patches.patch_size

            # Extract patch
            image_data = image_data[
                patch_y : patch_y + patch_h, patch_x : patch_x + patch_w
            ]

        return image_data

    def _process_annotation_results(
        self, annotation_results: dict, table_id: int
    ) -> int:
        """Process and upload annotation results to OMERO.

        Returns:
            Updated table ID (may be different if table was recreated)
        """
        try:
            from ..omero.omero_functions import (
                update_tracking_table_rows,
                upload_rois_and_labels,
            )
            from ..processing.file_io_functions import (
                cleanup_local_embeddings,
                zip_directory,
            )
        except ImportError:
            raise ImportError(
                "File I/O and OMERO functions are required. Install with: pip install -e .[omero]"
            )

        metadata = annotation_results["metadata"]
        embedding_path = Path(annotation_results["embedding_path"])
        annotations_path = Path(annotation_results["annotations_path"])

        # Construct TIFF file paths based on micro-SAM naming convention
        # image_series_annotator saves files as seg_00000.tif, seg_00001.tif, etc.
        tiff_files = []
        for i in range(len(metadata)):
            tiff_file = (
                annotations_path / f"seg_{i:05d}.tif"
            )  # TODO make this configurable for future compatibility
            
            # More robust file existence check that handles potential file locking issues
            try:
                if tiff_file.exists() and tiff_file.stat().st_size > 0:
                    tiff_files.append(str(tiff_file))
                else:
                    # File doesn't exist or is empty - annotation failed for this image
                    print(f"⚠️ Warning: Expected annotation file not found or empty: {tiff_file}")
            except (OSError, PermissionError) as e:
                # Handle file access issues (e.g., still locked by micro-SAM)
                print(f"⚠️ Warning: Cannot access annotation file {tiff_file}: {e}")
                # Small additional delay and retry once
                time.sleep(0.2)
                try:
                    if tiff_file.exists() and tiff_file.stat().st_size > 0:
                        tiff_files.append(str(tiff_file))
                    else:
                        print(f"⚠️ Warning: Expected annotation file not found after retry: {tiff_file}")
                except (OSError, PermissionError):
                    print(f"⚠️ Warning: Annotation file still inaccessible after retry: {tiff_file}")

        if not tiff_files:
            raise RuntimeError(
                "No annotation files were created by image_series_annotator"
            )

        print(f"📁 Found {len(tiff_files)} annotation files in {annotations_path}")

        # Collect all row indices and annotation IDs for batch update
        completed_row_indices = []
        annotation_ids = []  # Store (label_id, roi_id) pairs

        # Process each annotation result
        for i, (sequence_val, meta, row_idx) in enumerate(metadata):
            if i < len(tiff_files):
                tiff_path = tiff_files[i]

                # Upload ROIs and labels to OMERO
                if not self.config.workflow.read_only_mode:
                    image_id = meta.get("image_id")
                    patch_offset = None

                    if self.config.patches.use_patches:
                        patch_offset = (meta["patch_x"], meta["patch_y"])

                    label_id, roi_id = upload_rois_and_labels(
                        conn=self.conn,
                        image_id=image_id,
                        annotation_file=tiff_path,
                        patch_offset=patch_offset,
                        trainingset_name=self.config.training.trainingset_name,
                        trainingset_description=f"Training set: {self.config.training.trainingset_name}",
                    )

                    # Store the annotation IDs
                    annotation_ids.append((label_id, roi_id))
                else:
                    # Save locally
                    local_dir = Path(self.config.batch_processing.output_folder)
                    local_file = local_dir / f"annotation_{sequence_val}.tiff"
                    shutil.copy(tiff_path, local_file)
                    annotation_ids.append((None, None))

                # Add to list of completed rows
                completed_row_indices.append(row_idx)

        # Update tracking table with annotation IDs for each completed row
        if completed_row_indices:
            if not self.config.workflow.read_only_mode:
                # Update rows individually since each has different annotation IDs
                for i, row_idx in enumerate(completed_row_indices):
                    label_id, roi_id = (
                        annotation_ids[i] if i < len(annotation_ids) else (None, None)
                    )
                    table_id = update_tracking_table_rows(
                        conn=self.conn,
                        table_id=table_id,
                        row_indices=[row_idx],  # Update one row at a time
                        status="completed",
                        label_id=label_id,
                        roi_id=roi_id,
                        annotation_type=self.config.training.annotation_type,
                        container_type=self.config.omero.container_type,
                        container_id=self.config.omero.container_id,
                    )
            else:
                # In read-only mode, save progress locally
                self._save_local_progress(completed_row_indices, metadata)

        # Create and upload embeddings
        if embedding_path.exists():
            zip_path = embedding_path.parent / f"embeddings_{len(metadata)}.zip"
            zip_directory(embedding_path, zip_path)

            # Upload embeddings to OMERO as file annotation
            # This would require additional OMERO upload functionality

            # Cleanup embeddings
            cleanup_local_embeddings(embedding_path)

        # Cleanup temporary TIFF files (keep annotations folder)
        # Note: We keep the annotations folder and files for debugging/review
        # Users can manually delete if needed

        return table_id

    def create_annotation_table(
        self, images_list: Optional[List[Any]] = None
    ) -> Tuple[int, List[Any]]:
        """Create annotation table and initialize tracking for annotation workflow.

        Args:
            images_list: List of OMERO image objects to process. If None, loads from container.

        Returns:
            Tuple of (table_id, images_list)
        """
        print(f"🚀 Creating annotation table")

        # Load images if not provided
        if images_list is None:
            images_list = self.get_images_from_container()
            if not images_list:
                raise ValueError(
                    f"No images found in {self.config.omero.container_type} {self.config.omero.container_id}"
                )

        print(
            f"📊 Creating table for {len(images_list)} images with model: {self.config.micro_sam.model_type}"
        )

        # Setup directories
        output_path = self._setup_directories()
        self._cleanup_embeddings(output_path)

        # Initialize tracking table (skip in read-only mode)
        if not self.config.workflow.read_only_mode:
            table_id = self._initialize_tracking_table(images_list)
            print(f"📋 Created annotation table with ID: {table_id}")
        else:
            print("📋 Read-only mode: Skipping OMERO table creation")
            table_id = -1  # Mock table ID for read-only mode

        return table_id, images_list

    def run_annotation(
        self, table_id: int, images_list: Optional[List[Any]] = None
    ) -> Tuple[int, List[Any]]:
        """Run annotation workflow using existing table.

        Args:
            table_id: ID of existing annotation table
            images_list: Optional list of images (for read-only mode)

        Returns:
            Tuple of (table_id, processed_images)
        """
        print(f"🚀 Starting annotation processing from table ID: {table_id}")

        # Setup directories
        output_path = self._setup_directories()
        self._cleanup_embeddings(output_path)

        # Get processing units from table
        if not self.config.workflow.read_only_mode:
            # Get unprocessed units from existing table
            try:
                from ..omero.omero_functions import get_unprocessed_units

                processing_units = get_unprocessed_units(self.conn, table_id)
            except ImportError:
                raise ImportError(
                    "Cannot get unprocessed units without OMERO functions. Install with: pip install -e .[omero]"
                )
        else:
            if images_list is None:
                raise ValueError("images_list is required for read-only mode")
            # Create processing units directly from images
            processing_units = self._prepare_processing_units(images_list)
            processing_units = [
                (img_id, seq, meta, i)
                for i, (img_id, seq, meta) in enumerate(processing_units)
            ]

        if not processing_units:
            print("✅ All images already processed!")
            return table_id, images_list or []

        print(f"📋 Found {len(processing_units)} processing units")

        # Process in batches
        batch_size = self.config.batch_processing.batch_size
        processed_count = 0

        # Handle batch_size = 0 (process all images in one batch)
        if batch_size == 0:
            batch_size = len(processing_units)

        for i in range(0, len(processing_units), batch_size):
            batch = processing_units[i : i + batch_size]
            batch_num = i // batch_size + 1

            print(
                f"🔄 Processing batch {batch_num}/{(len(processing_units) + batch_size - 1) // batch_size}"
            )

            try:
                # Load image objects for batch
                batch_with_images = []
                for img_id, seq_val, meta, row_idx in batch:
                    image_obj = self.conn.getObject("Image", img_id)
                    batch_with_images.append((image_obj, seq_val, meta, row_idx))

                # Run annotation
                annotation_results = self._run_micro_sam_annotation(batch_with_images)

                # Process results
                table_id = self._process_annotation_results(
                    annotation_results, table_id
                )

                # Update workflow status map after batch completion
                if not self.config.workflow.read_only_mode:
                    self._update_workflow_status_map(table_id)

                processed_count += len(batch)
                print(
                    f"✅ Completed batch {batch_num} ({processed_count}/{len(processing_units)} total)"
                )

            except Exception as e:
                print(f"❌ Error processing batch {batch_num}: {e}")
                traceback.print_exc()
                # Continue with next batch - don't let one batch failure stop the pipeline
                continue

        if processed_count > 0:
            print(
                f"🎉 Annotation processing completed successfully! Processed {processed_count} units"
            )
        else:
            print(f"❌ Annotation processing failed - no units were processed")

        return table_id, images_list or []

    def _update_workflow_status_map(self, table_id: int) -> None:
        """Update workflow status map annotation after batch completion."""
        try:
            from ..omero.omero_functions import update_workflow_status_map

            update_workflow_status_map(
                conn=self.conn,
                container_type=self.config.omero.container_type,
                container_id=self.config.omero.container_id,
                table_id=table_id,
            )
        except ImportError:
            print("⚠️ Could not update workflow status - OMERO functions not available")
        except Exception as e:
            print(f"⚠️ Could not update workflow status: {e}")

    def _get_workflow_status_map(self) -> Optional[Dict[str, str]]:
        """Get current workflow status map annotation."""
        try:
            from ..omero.omero_functions import get_workflow_status_map

            return get_workflow_status_map(
                conn=self.conn,
                container_type=self.config.omero.container_type,
                container_id=self.config.omero.container_id,
            )
        except ImportError:
            return None
        except Exception as e:
            print(f"⚠️ Could not get workflow status: {e}")
            return None

    def get_images_from_container(self) -> List[Any]:
        """Get images from the configured OMERO container."""

        container_type = self.config.omero.container_type
        container_id = self.config.omero.container_id

        print(f"📁 Loading images from {container_type} {container_id}")

        if container_type == "dataset":
            image_ids = list(ezomero.get_image_ids(self.conn, dataset=container_id))
        elif container_type == "project":
            dataset_ids = ezomero.get_dataset_ids(self.conn, project=container_id)
            image_ids = []
            for ds_id in dataset_ids:
                image_ids.extend(ezomero.get_image_ids(self.conn, dataset=ds_id))
        elif container_type == "plate":
            image_ids = list(ezomero.get_image_ids(self.conn, plate=container_id))
        elif container_type == "screen":
            plate_ids = ezomero.get_plate_ids(self.conn, screen=container_id)
            image_ids = []
            for plate_id in plate_ids:
                image_ids.extend(ezomero.get_image_ids(self.conn, plate=plate_id))
        elif container_type == "image":
            return [self.conn.getObject("Image", container_id)]
        else:
            raise ValueError(f"Unsupported container type: {container_type}")

        # Convert IDs to image objects
        images = [self.conn.getObject("Image", img_id) for img_id in image_ids]
        images = [img for img in images if img is not None]  # Filter out None values

        print(f"📊 Found {len(images)} images")
        return images

    def _save_local_progress(
        self, completed_row_indices: List[int], metadata: List[Tuple]
    ) -> None:
        """Save progress locally in read-only mode by creating/updating a CSV table.

        Args:
            completed_row_indices: List of completed row indices
            metadata: List of metadata tuples (sequence_val, meta_dict, row_idx)
        """
        output_path = Path(self.config.batch_processing.output_folder)
        table_file = output_path / "tracking_table.csv"

        # Create table if it doesn't exist, or load existing
        if table_file.exists():
            df = pd.read_csv(table_file)
        else:
            # Create new DataFrame with all metadata
            rows = []
            for sequence_val, meta, row_idx in metadata:
                image_id = meta.get("image_id", -1)
                rows.append(
                    {
                        "row_index": row_idx,
                        "image_id": image_id,
                        "sequence_val": sequence_val,
                        "timepoint": meta.get("timepoint", -1),
                        "z_slice": meta.get("z_slice", -1),
                        "channel": meta.get("channel", -1),
                        "model_type": meta.get("model_type", "vit_b_lm"),
                        "processed": False,
                        "completed_timestamp": "",
                    }
                )
            df = pd.DataFrame(rows)

        # Mark completed rows
        timestamp = datetime.now().isoformat()
        for row_idx in completed_row_indices:
            mask = df["row_index"] == row_idx
            df.loc[mask, "processed"] = True
            df.loc[mask, "completed_timestamp"] = timestamp

        # Save updated table
        df.to_csv(table_file, index=False)
        completed_count = df["processed"].sum()
        total_count = len(df)

        print(
            f"📋 Local progress saved: {completed_count}/{total_count} units completed"
        )
        print(f"   Progress file: {table_file}")

    def run_full_workflow(self) -> Tuple[int, List[Any]]:
        """Run the complete workflow: create table and process annotations."""
        table_id, images_list = self.create_annotation_table()
        return self.run_annotation(table_id, images_list)


def create_pipeline(
    config: AnnotationConfig, conn=None
) -> AnnotationPipeline:
    """Create a micro-SAM annotation pipeline with the given configuration.

    Args:
        config: AnnotationConfig object
        conn: OMERO connection object

    Returns:
        AnnotationPipeline instance
    """
    return AnnotationPipeline(config, conn)
