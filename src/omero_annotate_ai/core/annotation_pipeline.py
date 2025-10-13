"""Main pipeline for OMERO annotation workflows."""

# ========================================
# DEBUG CONFIGURATION
# ========================================
# To disable all debug output, change DEBUG_MODE to False in the AnnotationPipeline class
# or set AnnotationPipeline.DEBUG_MODE = False after import
# ========================================

# Standard library imports
import logging
import random
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import pandas as pd
import ezomero

# Optional imports (with error handling)
try:
    import napari
    from micro_sam.sam_annotator import image_series_annotator
    MICRO_SAM_AVAILABLE = True
except ImportError:
    MICRO_SAM_AVAILABLE = False
    napari = None
    image_series_annotator = None
except Exception:
    MICRO_SAM_AVAILABLE = False
    napari = None
    image_series_annotator = None

# Local imports
from .annotation_config import AnnotationConfig, ImageAnnotation
from ..omero.omero_functions import (
    create_or_replace_tracking_table,
    sync_omero_table_to_config,
   # update_workflow_status_map,
   # get_workflow_status_map,
    upload_annotation_config_to_omero,
    upload_rois_and_labels
)
from ..omero.omero_utils import get_dask_image_single, get_table_by_name
from ..processing.image_functions import generate_patch_coordinates


class AnnotationPipeline:
    """Main pipeline for running micro-SAM annotation workflows with OMERO."""
    
    # DEBUG FLAG - Set to False to disable all debug output
    DEBUG_MODE = True

    def __init__(self, config: AnnotationConfig, conn=None, config_file_path: Optional[Union[str, Path]] = None):
        """Initialize the pipeline with configuration and OMERO connection.

        Args:
            config: AnnotationConfig object containing all parameters
            conn: OMERO connection object (BlitzGateway)
            config_file_path: Optional path to the source YAML configuration file for persistence
        """
        self.config = config
        self.conn = conn
        self.table_id = None  # Track current table ID
        self.config_file_path = Path(config_file_path) if config_file_path else None
        self._validate_setup()

    def set_table_id(self, table_id: int) -> None:
        """Update the current table ID."""
        self.table_id = table_id

    def get_table_id(self) -> Optional[int]:
        """Get the current table ID."""
        return self.table_id

    def save_config_to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> None:
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
        if self.conn is None:
            raise ValueError("OMERO connection is required")

    def _setup_directories(self):
        """Create necessary output directories."""
        output_path = Path(self.config.output.output_directory)

        if self.config.workflow.read_only_mode:
            # Create organized folder structure for read-only mode
            self._create_organized_folder_structure()
        else:
            # Create main directories - everything under the single output folder
            dirs = [
                output_path,
                output_path / "embed",
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
        # Use the existing trainingset_name from config directly
        # (the name from config already includes proper formatting via generate_unique_table_name)
        if self.config.name:
            return self.config.name
        else:
            # Fallback to container-based naming with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{self.config.omero.container_type}_{self.config.omero.container_id}_{timestamp}"

    def _initialize_tracking_table(self, images_list: List[Any]) -> int:
        """Initialize tracking table for the annotation process using config-first approach.
        
        This method only creates new tables from config state. Resume logic is handled elsewhere.
        """
        logger = logging.getLogger(__name__)

        # Generate table name
        table_title = self._get_table_title()
        logger.info(f"Creating new tracking table: {table_title}")

        # Prepare processing units by populating config.annotations
        self._prepare_processing_units(images_list)

        # Create table from config (all processed=False for new table)
        config_df = self.config.to_dataframe()
        config_df['processed'] = False  # Ensure all start as unprocessed
        
        table_id = create_or_replace_tracking_table(
            conn=self.conn,
            config_df=config_df,
            table_title=table_title,
            container_type=self.config.omero.container_type,
            container_id=self.config.omero.container_id,
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
            container_type_str = str(self.config.omero.container_type).capitalize()
            config_ann_id = ezomero.post_map_annotation(
                self.conn,
                object_type=container_type_str,
                object_id=self.config.omero.container_id,
                kv_dict=flat_config,
                ns="openmicroscopy.org/omero/annotation",
            )
            print(f"Stored configuration as annotation ID: {config_ann_id}")
        except Exception as e:
            print(f"Warning: Could not store configuration annotation: {e}")

        # Store table ID internally
        self.table_id = table_id
        return table_id

    def _prepare_processing_units(self, images_list: List[Any]) -> None:
        """Prepare processing units by populating config.annotations."""

        # Clear existing annotations first
        self.config.annotations.clear()
        n_total = len(images_list)

        # Determine which images to process
        if self.config.training.segment_all:
            # Use all images with fraction-based split
            selected_images = images_list
            n_train = int(n_total * self.config.training.train_fraction)
            n_val = int(n_total * self.config.training.validation_fraction)
            n_test = n_total - n_train - n_val

            # Create categories for all images
            image_categories = (
                ["training"] * n_train +
                ["validation"] * n_val +
                ["test"] * n_test
            )
        else:
            # Select subset with specific counts
            n_train = min(self.config.training.train_n, n_total)
            n_val = min(self.config.training.validate_n, n_total - n_train)
            n_test = min(self.config.training.test_n, n_total - n_train - n_val)

            # Randomly select images
            shuffled_indices = list(range(n_total))
            random.shuffle(shuffled_indices)

            train_indices = shuffled_indices[:n_train]
            val_indices = shuffled_indices[n_train : n_train + n_val]
            test_indices = shuffled_indices[n_train + n_val : n_train + n_val + n_test]

            selected_images = [images_list[i] for i in train_indices + val_indices + test_indices]
            image_categories = (
                ["training"] * n_train +
                ["validation"] * n_val +
                ["test"] * n_test
            )

        # Create ImageAnnotation objects for each processing unit
        for image, category in zip(selected_images, image_categories):
            image_id = image.getId()
            image_name = image.getName()

            # Handle timepoints
            timepoints = self._get_timepoints_for_image(image)

            # Handle z-slices based on processing mode
            z_slices = self._get_z_slices_for_image(image)
            
            # Check if we're doing volumetric 3D processing
            is_volumetric = self.config.spatial_coverage.is_volumetric

            # Create annotations based on processing mode
            for t in timepoints:
                if is_volumetric:
                    # 3D volumetric mode: one annotation per timepoint (includes all z-slices)
                    z_start = min(z_slices) if z_slices else 0
                    z_end = max(z_slices) if z_slices else 0 
                    z_length = len(z_slices) if z_slices else 1
                    
                    if self.config.processing.use_patches:
                        # Create multiple annotations for 3D patches
                        patch_coords, actual_patch_size = self._generate_patch_coordinates(image)
                        patch_h, patch_w = actual_patch_size
                        for i, (x, y) in enumerate(patch_coords):
                            annotation = ImageAnnotation(
                                image_id=image_id,
                                image_name=image_name,
                                annotation_id=f"{image_id}_{t}_3d_{i}",
                                timepoint=t,
                                z_slice=z_start,  # Start slice for compatibility
                                z_start=z_start,
                                z_end=z_end,
                                z_length=z_length,
                                is_patch=True,
                                patch_x=x,
                                patch_y=y,
                                patch_width=patch_w,
                                patch_height=patch_h,
                                category=category,
                                model_type=self.config.ai_model.model_type,
                                channel=self.config.spatial_coverage.primary_channel,
                                is_volumetric=True,
                            )
                            self.config.add_annotation(annotation)
                    else:
                        # Single annotation for full 3D volume
                        annotation = ImageAnnotation(
                            image_id=image_id,
                            image_name=image_name,
                            annotation_id=f"{image_id}_{t}_3d",
                            timepoint=t,
                            z_slice=z_start,  # Start slice for compatibility
                            z_start=z_start,
                            z_end=z_end,
                            z_length=z_length,
                            category=category,
                            model_type=self.config.ai_model.model_type,
                            channel=self.config.spatial_coverage.primary_channel,
                            is_volumetric=True,
                        )
                        self.config.add_annotation(annotation)
                else:
                    # 2D mode: create annotations per z-slice (existing behavior)
                    for z in z_slices:
                        if self.config.processing.use_patches:
                            # Create multiple annotations for 2D patches
                            patch_coords, actual_patch_size = self._generate_patch_coordinates(image)
                            patch_h, patch_w = actual_patch_size
                            for i, (x, y) in enumerate(patch_coords):
                                annotation = ImageAnnotation(
                                    image_id=image_id,
                                    image_name=image_name,
                                    annotation_id=f"{image_id}_{t}_{z}_{i}",
                                    timepoint=t,
                                    z_slice=z,
                                    z_start=z,
                                    z_end=z,
                                    z_length=1,
                                    patch_x=x,
                                    patch_y=y,
                                    is_patch=True,
                                    patch_width=patch_w,
                                    patch_height=patch_h,
                                    category=category,
                                    model_type=self.config.ai_model.model_type,
                                    channel=self.config.spatial_coverage.primary_channel,
                                    is_volumetric=False,
                                )
                                self.config.add_annotation(annotation)
                        else:
                            # Single annotation for 2D slice
                            annotation = ImageAnnotation(
                                image_id=image_id,
                                image_name=image_name,
                                annotation_id=f"{image_id}_{t}_{z}",
                                timepoint=t,
                                z_slice=z,
                                z_start=z,
                                z_end=z,
                                z_length=1,
                                category=category,
                                model_type=self.config.ai_model.model_type,
                                channel=self.config.spatial_coverage.primary_channel,
                                is_volumetric=False,
                            )
                            self.config.add_annotation(annotation)

    def _upload_annotation_config_to_omero(self) -> int: 
        container_type_str = str(self.config.omero.container_type).capitalize()
        file_path = str(self.config_file_path) if isinstance(self.config_file_path, Path) else self.config_file_path
        id = upload_annotation_config_to_omero(
            self.conn,
            object_type=container_type_str,
            object_id=self.config.omero.container_id,
            file_path=file_path
        )
        return id

    def _get_timepoints_for_image(self, image) -> List[int]:
        """Get timepoints to process for an image based on configuration."""
        max_t = image.getSizeT()

        if self.config.spatial_coverage.timepoint_mode == "all":
            return list(range(max_t))
        elif self.config.spatial_coverage.timepoint_mode == "random":
            n_points = min(len(self.config.spatial_coverage.timepoints), max_t)
            return random.sample(range(max_t), n_points)
        else: # specific
            return [t for t in self.config.spatial_coverage.timepoints if t < max_t]

    def _get_z_slices_for_image(self, image) -> List[int]:
        """Get z-slices to process for an image based on configuration."""
        max_z = image.getSizeZ()

        if self.config.spatial_coverage.z_slice_mode == "all":
            return list(range(max_z))
        elif self.config.spatial_coverage.z_slice_mode == "random":
            n_slices = min(len(self.config.spatial_coverage.z_slices), max_z)
            return random.sample(range(max_z), n_slices)
        else: # specific
            return [z for z in self.config.spatial_coverage.z_slices if z < max_z]

    def _generate_patch_coordinates(self, image) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Generate patch coordinates for an image and return actual patch size."""
        patch_size = self.config.processing.patch_size
        patches_per_image = self.config.processing.patches_per_image
        random_patches = self.config.processing.random_patches

        image_shape = (image.getSizeY(), image.getSizeX())

        coordinates, actual_patch_size = generate_patch_coordinates(
            image_shape=image_shape,
            patch_size=patch_size,
            n_patches=patches_per_image,
            random_patch=random_patches,
        )
        
        return coordinates, actual_patch_size

    def _run_micro_sam_annotation(self, batch_data: List[Tuple]) -> dict:
        """Run micro-SAM annotation on a batch of data."""
        # Single comprehensive dependency check for micro-SAM annotation
        if not MICRO_SAM_AVAILABLE or napari is None or image_series_annotator is None:
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

            for image_obj, annotation_id, meta, row_idx in batch_data    :
                # Load image data from OMERO
                image_data = self._load_image_data(image_obj, meta)
                images.append(image_data)
                # Include image_id in metadata for later ROI upload
                meta_with_image_id = meta.copy()
                meta_with_image_id["image_id"] = image_obj.getId()
                metadata.append((annotation_id, meta_with_image_id, row_idx))

            # Set up output paths
            output_path = Path(self.config.output.output_directory)
            embedding_path = output_path / "embed"
            annotations_path = output_path / "annotations"
            annotations_path.mkdir(exist_ok=True)
            # print("DEBUG: Ready to run micro_sam annotations")
            # Run micro-SAM annotation
            model_type = self.config.ai_model.model_type

    
            # Run image series annotator with explicit napari.run() call
            viewer = image_series_annotator(
                images=images,
                output_folder=str(annotations_path),
                model_type=model_type,
                viewer=None,  # Let it create its own viewer
                return_viewer=True,  # Important: get the viewer back
                embedding_path=str(embedding_path),
                is_volumetric=self.config.spatial_coverage.three_d,
                skip_segmented=True,
            )

            # Explicitly start the event loop - this will block until viewer is closed
            napari.run()

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
        channel = self.config.spatial_coverage.primary_channel
        is_volumetric = metadata.get("is_volumetric", False)

        if is_volumetric:
            # Load 3D volume: all z-slices at once
            z_start = metadata.get("z_start", 0)
            z_end = metadata.get("z_end", 0)
            z_length = metadata.get("z_length", 1)

            # Generate z-slice list
            z_slices = list(range(z_start, z_end + 1))

            print(f"Loading 3D volume: z_slices {z_start}-{z_end} (length: {z_length})")

            # Use get_dask_image_single for single image loading
            image_data = get_dask_image_single(
                conn=self.conn,
                image=image_obj,
                timepoints=[timepoint],
                channels=[channel],
                z_slices=z_slices,  # All z-slices for 3D volume
            )

            # Ensure we have proper data
            if image_data is None:
                raise ValueError(f"Failed to load 3D image data for image {image_obj.getId()}")
            
            print(f"Loaded 3D data shape: {image_data.shape}")

        else:
            # Load single z-slice for 2D processing
            z_slice = metadata["z_slice"]
            
            # Use get_dask_image_single for single image loading
            image_data = get_dask_image_single(
                conn=self.conn,
                image=image_obj,
                timepoints=[timepoint],
                channels=[channel],
                z_slices=[z_slice],
            )

            # Ensure we have proper data
            if image_data is None:
                raise ValueError(f"Failed to load 2D image data for image {image_obj.getId()}")
            
            print(f"Loaded 2D data shape: {image_data.shape}")
            
        # Ensure image data is properly converted to numpy array and has correct dtype
        if hasattr(image_data, 'compute'):
            image_data = image_data.compute()
        
        # Convert to numpy array and ensure proper dtype
        image_data = np.asarray(image_data)
        
        # Ensure we have valid pixel values (not all zeros)
        if np.all(image_data == 0):
            print(f"Warning: Image data appears to be all zeros for image {image_obj.getId()}")
            print(f"Image info: {image_obj.getSizeX()}x{image_obj.getSizeY()}, {image_obj.getSizeC()} channels, {image_obj.getSizeZ()} z-slices, {image_obj.getSizeT()} timepoints")
        else:
            print(f"Image data range: {np.min(image_data)} - {np.max(image_data)}, dtype: {image_data.dtype}")

        # Handle patches if needed (works for both 2D and 3D)
        if self.config.processing.use_patches and "patch_x" in metadata:
            patch_x = metadata["patch_x"]
            patch_y = metadata["patch_y"]
            # Use the actual patch dimensions from metadata instead of config
            patch_h = metadata.get("patch_height", self.config.processing.patch_size[0])
            patch_w = metadata.get("patch_width", self.config.processing.patch_size[1])

            if is_volumetric:
                # Extract 3D patch: (z, y, x)
                image_data = image_data[
                    :, patch_y : patch_y + patch_h, patch_x : patch_x + patch_w
                ]
            else:
                # Extract 2D patch: (y, x)
                image_data = image_data[
                    patch_y : patch_y + patch_h, patch_x : patch_x + patch_w
                ]

        return image_data

    def _process_annotation_results(
        self, annotation_results: dict
    ) -> None:
        """Process annotation results, upload ROIs to OMERO, and update config annotations."""
        
        self._debug_print("Processing annotation results...")
        
        if not annotation_results:
            self._debug_print("No annotation results received")
            return

        # Extract metadata and paths from results
        try:
            metadata = annotation_results["metadata"]
            annotations_path = Path(annotation_results["annotations_path"])
        except KeyError as e:
            self._debug_print(f"Missing required key: {e}")
            return

        # Get all TIFF files in the annotations directory
        tiff_files = list(annotations_path.glob("*.tiff")) + list(annotations_path.glob("*.tif"))
        self._debug_print(f"Found {len(tiff_files)} annotation files for {len(metadata)} metadata entries")

        # Process each annotation metadata entry
        updated_count = 0
        for i, (annotation_id, meta, row_idx) in enumerate(metadata):
            # Find matching annotation in config by annotation_id or sequence_val
            matching_annotation = None
            for annotation in self.config.annotations:
                # Try both annotation_id and sequence_val for backward compatibility
                if (annotation.annotation_id == annotation_id or 
                    (hasattr(annotation, 'sequence_val') and annotation.sequence_val == annotation_id)):
                    matching_annotation = annotation
                    break
            
            if not matching_annotation:
                self._debug_print(f"No matching annotation for annotation_id: '{annotation_id}'")
                continue
            
            # Find corresponding TIFF file
            tiff_file = tiff_files[i] if i < len(tiff_files) else None
            if not tiff_file:
                self._debug_print(f"No annotation file for metadata entry {i+1}")
                continue
            
            # Process based on mode
            if self.config.workflow.read_only_mode:
                # Read-only mode: save locally in organized folders
                matching_annotation.processed = True
                matching_annotation.annotation_creation_time = datetime.now().isoformat()
                
                # Determine category and create organized folder structure
                category = meta.get("category", "training")
                output_dir = f"output_{category}"
                
                local_dir = Path(self.config.output.output_directory) / output_dir
                local_dir.mkdir(parents=True, exist_ok=True)
                local_file = local_dir / f"{annotation_id}_mask.tif"
                
                shutil.move(tiff_file, local_file)
                
                # Also save the original image to input folder
                self._save_original_image_for_annotation(meta, annotation_id, category)
                
                updated_count += 1
                
            else:
                # Upload mode: Upload ROIs to OMERO
                try:
                    image_id = meta.get("image_id")
                    patch_offset = None
                    
                    if self.config.processing.use_patches:
                        patch_offset = (meta.get("patch_x", 0), meta.get("patch_y", 0))
                    
                    # Upload ROIs and labels using the existing function
                    label_id, roi_id = upload_rois_and_labels(
                        conn=self.conn,
                        image_id=image_id,
                        annotation_file=str(tiff_file),
                        patch_offset=patch_offset,
                        trainingset_name=self.config.name,
                        trainingset_description=f"Training set: {self.config.name}",
                        z_slice=meta.get("z_slice"),
                        channel=meta.get("channel"),
                        timepoint=meta.get("timepoint"),
                        model_type=meta.get("model_type"),
                        is_volumetric=meta.get("is_volumetric"),
                        z_start=meta.get("z_start")
                    )
                    
                    # Update config annotation with OMERO IDs
                    matching_annotation.processed = True
                    matching_annotation.label_id = label_id
                    matching_annotation.roi_id = roi_id
                    matching_annotation.annotation_creation_time = datetime.now().isoformat()
                    matching_annotation.annotation_type = self.config.annotation_methodology.annotation_type
                    
                    updated_count += 1
                    self._debug_print(f"Uploaded ROI for annotation_id '{annotation_id}' (label_id: {label_id}, roi_id: {roi_id})")
                    
                except Exception as e:
                    self._debug_print(f"Error uploading ROI for '{annotation_id}': {e}")
                    # Still mark as processed but without OMERO IDs
                    matching_annotation.processed = True
                    matching_annotation.annotation_creation_time = datetime.now().isoformat()
                    updated_count += 1
        
        # Show final results
        processed_count = sum(1 for ann in self.config.annotations if ann.processed)
        self._debug_print(f"Updated {updated_count}/{len(metadata)} annotations")
        self._debug_print(f"Config now has {processed_count}/{len(self.config.annotations)} processed annotations")

    def _save_original_image_for_annotation(self, meta: dict, annotation_id: str, category: str) -> None:
        """Save the original image used for annotation to organized input folder.
        
        Args:
            meta: Metadata dictionary containing image information
            annotation_id: Unique annotation identifier
            category: Category (training/validation) for folder organization
        """
        try:
            # Get image object
            image_id = meta.get("image_id")
            if not image_id:
                self._debug_print(f"No image_id found for annotation {annotation_id}")
                return
                
            image_obj = self.conn.getObject("Image", image_id)
            if not image_obj:
                self._debug_print(f"Could not load image {image_id} for annotation {annotation_id}")
                return
            
            # Load the image data using the same method as annotation
            image_data = self._load_image_data(image_obj, meta)
            
            # Create input folder structure
            input_dir = f"input_{category}"
            local_dir = Path(self.config.output.output_directory) / input_dir
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original image
            image_file = local_dir / f"{annotation_id}_image.tif"
            
            # Import imwrite here to handle optional dependency
            try:
                from imageio import imwrite
                imwrite(image_file, image_data)
                self._debug_print(f"Saved original image: {image_file}")
            except ImportError:
                # Fallback to tifffile if available
                try:
                    import tifffile
                    tifffile.imwrite(image_file, image_data)
                    self._debug_print(f"Saved original image: {image_file}")
                except ImportError:
                    self._debug_print("Could not save image - imageio and tifffile not available")
                    
        except Exception as e:
            self._debug_print(f"Error saving original image for {annotation_id}: {e}")

    def _create_organized_folder_structure(self) -> None:
        """Create organized folder structure for read-only mode data export."""
        base_path = Path(self.config.output.output_directory)

        # Always create training folders
        folders = ["input_training", "output_training"]

        # Add validation folders only if validation images exist
        if any(ann.category == "validation" for ann in self.config.annotations):
            folders.extend(["input_validation", "output_validation"])

        # Add test folders only if test images exist
        if any(ann.category == "test" for ann in self.config.annotations):
            folders.extend(["input_test", "output_test"])

        for folder in folders:
            folder_path = base_path / folder
            folder_path.mkdir(parents=True, exist_ok=True)

        # Create a README file explaining the structure
        readme_content = """# Dataset Organization

This folder contains an organized dataset for training image segmentation models.

## Folder Structure:

- `input_training/`: Original images used for training
- `output_training/`: Corresponding annotation masks for training images
- `input_validation/`: Original images used for validation (if validation_n > 0)
- `output_validation/`: Corresponding annotation masks for validation images
- `input_test/`: Original images used for testing (if test_n > 0)
- `output_test/`: Corresponding annotation masks for test images

## Usage with Different Platforms:

### Cellpose Training:
Use `input_training/` and `input_test/` directories with:
```bash
python -m cellpose --train --dir ./input_training --test_dir ./input_test
```

### BiaPy Training:
Organize as:
- train/raw/ ← copy from input_training
- train/label/ ← copy from output_training
- val/raw/ ← copy from input_validation
- val/label/ ← copy from output_validation
- test/raw/ ← copy from input_test
- test/label/ ← copy from output_test

## File Naming:

Each image and its corresponding mask share the same annotation ID:
- Original image: `{annotation_id}_image.tif`
- Annotation mask: `{annotation_id}_mask.tif`

## Configuration:

The `annotation_config.yaml` file contains all the parameters used to generate this dataset.

## Tracking:

The `tracking_table.csv` file contains detailed information about each annotation unit processed.
"""
        
        readme_path = base_path / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        print(f"Created organized folder structure at: {base_path}")
        print("Folders: input_training, output_training, input_validation, output_validation")

    # def _update_workflow_status_map(self, table_id: int) -> None:
    #     """Update workflow status map annotation after batch completion."""
    #     try:
    #         update_workflow_status_map(
    #             conn=self.conn,
    #             container_type=self.config.omero.container_type,
    #             container_id=self.config.omero.container_id,
    #             table_id=table_id,
    #         )
    #     except ImportError:
    #         print("Could not update workflow status - OMERO functions not available")
    #     except Exception as e:
    #         print(f"Could not update workflow status: {e}")

    # def _get_workflow_status_map(self) -> Optional[Dict[str, str]]:
    #     """Get current workflow status map annotation."""
    #     try:
    #         return get_workflow_status_map(
    #             conn=self.conn,
    #             container_type=self.config.omero.container_type,
    #             container_id=self.config.omero.container_id,
    #         )
    #     except ImportError:
    #         return None
    #     except Exception as e:
    #         print(f"Could not get workflow status: {e}")
    #         return None

    def get_image_ids_from_container(self) -> List[int]:
        """Get image IDs from the configured OMERO container."""
        container_type = self.config.omero.container_type
        container_id = self.config.omero.container_id
        print(f"Loading image IDs from {container_type} {container_id}")
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
            return [container_id]
        else:
            raise ValueError(f"Unsupported container type: {container_type}")
        print(f"Found {len(image_ids)} image IDs")
        return image_ids

    def get_images_by_ids(self, image_ids: List[int]) -> List[Any]:
        """Get OMERO image objects for the given image IDs."""
        if self.conn is None:
            raise ValueError("OMERO connection is not set.")
        images = [self.conn.getObject("Image", img_id) for img_id in image_ids]
        images = [img for img in images if img is not None]
        print(f"Loaded {len(images)} images (by ID)")
        return images

    def get_images_from_container(self) -> List[Any]:
        """Get OMERO image objects from the configured container.
        
        Returns:
            List of OMERO image objects
        """
        image_ids = self.get_image_ids_from_container()
        return self.get_images_by_ids(image_ids)

    def _save_local_progress(
        self, completed_row_indices: List[int], metadata: List[Tuple]
    ) -> None:
        """Save progress locally in read-only mode by creating/updating a CSV table."""
        output_path = Path(self.config.output.output_directory)
        table_file = output_path / "tracking_table.csv"
        if table_file.exists():
            df = pd.read_csv(table_file)
        else:
            rows = []
            for annotation_id, meta, row_idx in metadata:
                image_id = meta.get("image_id", -1)
                rows.append(
                    {
                        "row_index": row_idx,
                        "image_id": image_id,
                        "annotation_id": annotation_id,
                        "timepoint": meta.get("timepoint", -1),
                        "z_slice": meta.get("z_slice", -1),
                        "channel": meta.get("channel", -1),
                        "model_type": meta.get("model_type", "vit_b_lm"),
                        "processed": False,
                        "completed_timestamp": "",
                    }
                )
            df = pd.DataFrame(rows)
        timestamp = datetime.now().isoformat()
        for row_idx in completed_row_indices:
            mask = df["row_index"] == row_idx
            df.loc[mask, "processed"] = True
            df.loc[mask, "completed_timestamp"] = timestamp
        df.to_csv(table_file, index=False)
        completed_count = df["processed"].sum()
        total_count = len(df)
        print(f"Local progress saved: {completed_count}/{total_count} units completed")
        print(f" Progress file: {table_file}")

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
            df.loc[mask, "completed_timestamp"] = annotation.annotation_creation_time

        df.to_csv(table_file, index=False)
        completed_count = df["processed"].sum()
        total_count = len(df)
        print(f"Local progress saved: {completed_count}/{total_count} units completed")
        print(f" Progress file: {table_file}")

    def _find_existing_table(self) -> Optional[int]:
        """Find existing table by name for resume functionality.
        
        Returns:
            Table ID if found, None otherwise
        """
        table_title = self._get_table_title()
        print(f"Looking for existing table: {table_title}")
        
        try:
            existing_table = get_table_by_name(self.conn, table_title)
            if existing_table:
                table_id = existing_table.getId()
                print(f"Found existing table: {table_title} (ID: {table_id})")
                return table_id
            else:
                print(f"No existing table found with name: {table_title}")
                return None
        except Exception as e:
            print(f"Error searching for existing table: {e}")
            return None

    def _replace_omero_table_from_config(self) -> int:
        """Replace entire OMERO table with current config state.
        
        Returns:
            Updated table ID
        """
        print("Updating OMERO table...")
        
        config_df = self.config.to_dataframe()
        table_title = self._get_table_title()
        
        table_id = create_or_replace_tracking_table(
            conn=self.conn,
            config_df=config_df,
            table_title=table_title,
            container_type=self.config.omero.container_type,
            container_id=self.config.omero.container_id,
            existing_table_id=self.table_id
        )
        
        # Update internal table ID
        self.table_id = table_id
        print(f"OMERO table updated successfully (ID: {table_id})")
        
        return table_id

    def initialize_workflow(self, images_list: Optional[List[Any]] = None) -> Tuple[int, List[Any]]:
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
            image_ids = self.get_image_ids_from_container()
            n_total = len(image_ids)
            if n_total == 0:
                raise ValueError(
                    f"No images found in {self.config.omero.container_type} {self.config.omero.container_id}"
                )

            # Determine which images to process (mimic _prepare_processing_units logic)
            if self.config.training.segment_all:
                selected_indices = list(range(n_total))
            else:
                n_train = min(self.config.training.train_n, n_total)
                n_val = min(self.config.training.validate_n, n_total - n_train)
                shuffled_indices = list(range(n_total))
                random.shuffle(shuffled_indices)
                selected_indices = shuffled_indices[: n_train + n_val]

            selected_image_ids = [image_ids[i] for i in selected_indices]
            images_list = self.get_images_by_ids(selected_image_ids)
        else:
            print(f"Loaded {len(images_list)} images (provided)")

        # Handle resume logic (one-time sync at workflow start)
        if self.config.workflow.resume_from_table and not self.config.workflow.read_only_mode:
            existing_table_id = self._find_existing_table()
            if existing_table_id:
                print("Syncing OMERO table state to config...")
                sync_omero_table_to_config(self.conn, existing_table_id, self.config)
                self.table_id = existing_table_id

        # Always return int for table_id (use -1 if None)
        table_id = self.table_id if self.table_id is not None else -1
        return table_id, images_list

    def define_annotation_schema(self, images_list: List[Any]) -> AnnotationConfig:
        """Define the annotation schema by populating config.annotations.
        
        Args:
            images_list: List of OMERO image objects to process
            
        Returns:
            Updated AnnotationConfig with populated annotations
        """
        # Prepare processing units (populate config.annotations)
        if not self.config.annotations or not self.config.workflow.resume_from_table:
            print("Defining annotation schema...")
            self._prepare_processing_units(images_list)
        
        print(f"Schema defined: {len(self.config.annotations)} annotations")
        return self.config

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
            config_df = self.config.to_dataframe()
            
            # Ensure all rows start as unprocessed for new table
            if not self.config.workflow.resume_from_table:
                config_df['processed'] = False
                
            table_id = create_or_replace_tracking_table(
                conn=self.conn,
                config_df=config_df,
                table_title=self._get_table_title(),
                container_type=self.config.omero.container_type,
                container_id=self.config.omero.container_id,
            )
            self.table_id = table_id
            print(f"Created OMERO table with ID: {table_id}")
        
        return self.table_id

    def _convert_annotations_to_processing_units(self, annotations: List) -> List[Tuple]:
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
                "timepoint": annotation.timepoint,
                "z_slice": annotation.z_slice,
                "z_start": annotation.z_start,
                "z_end": annotation.z_end,
                "z_length": annotation.z_length,
                "patch_x": annotation.patch_x,
                "patch_y": annotation.patch_y,
                "category": annotation.category,
                "model_type": annotation.model_type,
                "channel": annotation.channel,
                "is_volumetric": annotation.is_volumetric,
            }
            processing_units.append((annotation.image_id, annotation.annotation_id, metadata, i))
        return processing_units

    def _run_annotation_batches(self, processing_units: List[Tuple], annotation_func) -> int:
        """Run annotation function on batches of processing units.
        
        Args:
            processing_units: List of processing units to process
            annotation_func: Function to call for annotation (e.g., self._run_micro_sam_annotation)
            
        Returns:
            Number of successfully processed units
        """
        batch_size = self.config.processing.batch_size or len(processing_units)
        processed_count = 0
        
        for i in range(0, len(processing_units), batch_size):
            batch = processing_units[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(processing_units) + batch_size - 1) // batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches}")
            self._debug_print(f"   Batch size: {len(batch)}")
            
            # Show config state before processing
            processed_before = sum(1 for ann in self.config.annotations if ann.processed)
            self._debug_print(f"   Config state before: {processed_before}/{len(self.config.annotations)} processed")
            
            try:
                # Load image objects for batch
                if self.conn is None:
                    raise ValueError("OMERO connection is not set.")
                batch_with_images = []
                for img_id, seq_val, meta, row_idx in batch:
                    image_obj = self.conn.getObject("Image", img_id)
                    batch_with_images.append((image_obj, seq_val, meta, row_idx))
                
                # Run the specific annotation function
                self._debug_print("   Running annotation...")
                annotation_results = annotation_func(batch_with_images)
                self._debug_print(f"   Annotation completed, got results: {type(annotation_results)}")
                
                # Process results and update config
                self._debug_print("   Processing annotation results...")
                self._process_annotation_results(annotation_results)
                
                # Show config state after processing
                processed_after = sum(1 for ann in self.config.annotations if ann.processed)
                self._debug_print(f"   Config state after: {processed_after}/{len(self.config.annotations)} processed")
                
                # Update OMERO table and save config
                if not self.config.workflow.read_only_mode:
                    self._debug_print("   Updating OMERO table...")
                    self._replace_omero_table_from_config()
                    self._debug_print(f"   OMERO table updated (ID: {self.table_id})")
                
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

    def run_microsam_annotation(self) -> Tuple[int, AnnotationConfig]:
        """Run micro-SAM annotation workflow on unprocessed annotations.
        
        Returns:
            Tuple of (table_id, updated_config)
        """
        if not self.config.annotations:
            raise ValueError("No annotation schema defined - run define_annotation_schema() first")
        
        # Get unprocessed annotations
        unprocessed_annotations = self.config.get_unprocessed()
        if not unprocessed_annotations:
            print("All annotations already processed!")
            if self.table_id is None:
                self.table_id = -1
            return self.table_id, self.config
        
        print(f"Running micro-SAM annotation on {len(unprocessed_annotations)} unprocessed annotations")
        
        # Convert to processing units and run annotation
        processing_units = self._convert_annotations_to_processing_units(unprocessed_annotations)
        processed_count = self._run_annotation_batches(processing_units, self._run_micro_sam_annotation)
        
        # Finalize workflow
        self._finalize_workflow(processed_count)
        
        if self.table_id is None:
            self.table_id = -1
        return self.table_id, self.config

    def run_micro_sam_annotation(self) -> Tuple[int, AnnotationConfig]:
        """Alias for run_microsam_annotation for backward compatibility."""
        return self.run_microsam_annotation()

    def run_cellpose_preparation(self) -> Tuple[int, AnnotationConfig]:
        """Run Cellpose preparation workflow - save images locally for training.
        
        Returns:
            Tuple of (table_id, updated_config)
        """
        if not self.config.annotations:
            raise ValueError("No annotation schema defined - run define_annotation_schema() first")
        
        # Get unprocessed annotations
        unprocessed_annotations = self.config.get_unprocessed()
        if not unprocessed_annotations:
            print("All annotations already processed!")
            table_id = self.table_id if self.table_id is not None else -1
            return table_id, self.config
        
        print(f"Preparing {len(unprocessed_annotations)} images for Cellpose training")
        
        # Convert to processing units and save images
        processing_units = self._convert_annotations_to_processing_units(unprocessed_annotations)
        self._save_images_for_cellpose(processing_units)
        
        # Mark all as processed (since we just saved them)
        for annotation in unprocessed_annotations:
            annotation.processed = True
            annotation.annotation_creation_time = datetime.now().isoformat()
        
        # Update tracking
        if not self.config.workflow.read_only_mode:
            self._replace_omero_table_from_config()
        
        self._auto_save_config()
        print(f"Cellpose preparation completed: {len(unprocessed_annotations)} images saved")
        
        table_id = self.table_id if self.table_id is not None else -1
        return table_id, self.config

    def _save_images_for_cellpose(self, processing_units: List[Tuple]) -> None:
        """Save images locally for Cellpose training in organized folders."""
        if self.conn is None:
            raise ValueError("OMERO connection is not set.")
        output_path = Path(self.config.output.output_directory)
        
        for img_id, annotation_id, meta, row_idx in processing_units:
            try:
                print(f"Processing image {annotation_id} (ID: {img_id})")
                
                image_obj = self.conn.getObject("Image", img_id)
                if not image_obj:
                    print(f"Warning: Could not retrieve image {img_id}")
                    continue
                
                # Load image data using the fixed method
                image_data = self._load_image_data(image_obj, meta)
                
                if image_data is None:
                    print(f"Warning: No image data for {annotation_id}")
                    continue
                
                # Determine category and create organized folder structure
                category = meta.get("category", "training")
                input_dir = f"input_{category}"
                
                # Create input folder
                input_folder = output_path / input_dir
                input_folder.mkdir(parents=True, exist_ok=True)
                
                # Save image with organized naming
                image_file = input_folder / f"{annotation_id}_image.tif"
                
                # Validate image data before saving
                if np.all(image_data == 0):
                    print(f"Warning: Image {annotation_id} appears to be all black (all zeros)")
                
                # Use tifffile preferentially (better for scientific imaging)
                try:
                    import tifffile
                    tifffile.imwrite(image_file, image_data)
                    print(f"Saved (tifffile): {image_file} - shape: {image_data.shape}, range: [{np.min(image_data)}-{np.max(image_data)}]")
                except ImportError:
                    try:
                        from imageio import imwrite
                        imwrite(image_file, image_data)
                        print(f"Saved (imageio): {image_file} - shape: {image_data.shape}, range: [{np.min(image_data)}-{np.max(image_data)}]")
                    except ImportError:
                        print("Warning: Could not save image - imageio and tifffile not available")
                        continue
                
            except Exception as e:
                print(f"Error processing {annotation_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

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
                # Read-only mode: organize locally
                category = matching_annotation.category
                output_dir = f"output_{category}"
                
                local_dir = Path(self.config.output.output_directory) / output_dir
                local_dir.mkdir(parents=True, exist_ok=True)
                local_file = local_dir / f"{annotation_id}_mask.tif"
                
                # Copy or move file to organized location
                if annotation_file != local_file:
                    shutil.copy(annotation_file, local_file)
                
                # Mark as processed
                matching_annotation.processed = True
                matching_annotation.annotation_creation_time = datetime.now().isoformat()
                
                print(f"Processed locally: {annotation_id}")
                return True
                
            else:
                # Upload mode: Upload ROIs to OMERO
                image_id = matching_annotation.image_id
                patch_offset = None
                
                if matching_annotation.is_patch:
                    patch_offset = (matching_annotation.patch_x, matching_annotation.patch_y)
                
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
                    model_type=matching_annotation.model_type,
                    is_volumetric=matching_annotation.is_volumetric,
                    z_start=matching_annotation.z_start
                )
                
                # Update config annotation with OMERO IDs
                matching_annotation.processed = True
                matching_annotation.label_id = label_id
                matching_annotation.roi_id = roi_id
                matching_annotation.annotation_creation_time = datetime.now().isoformat()
                matching_annotation.annotation_type = self.config.annotation_methodology.annotation_type
                
                print(f"Uploaded to OMERO: {annotation_id} (label_id: {label_id}, roi_id: {roi_id})")
                return True
                
        except Exception as e:
            print(f"Error processing {annotation_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def collect_annotations_from_disk(self, folder_pattern: str = "output_*") -> Tuple[int, AnnotationConfig]:
        """Collect annotation masks from disk and upload to OMERO.
        
        This is a generalized method that scans output folders for annotation masks,
        matches them with the annotation schema, and processes them (upload or local save).
        Can be used for CellPose, manual annotations, or any other external annotation tool.
        
        Args:
            folder_pattern: Glob pattern for folders containing annotation masks
            
        Returns:
            Tuple of (number_processed, updated_config)
        """
        if not self.config.annotations:
            raise ValueError("No annotation schema defined")
        
        output_path = Path(self.config.output.output_directory)
        
        # Find all annotation mask files matching the pattern
        annotation_files = []
        for folder in output_path.glob(folder_pattern):
            if folder.is_dir():
                # Look for mask files
                for mask_file in folder.glob("*_mask.tif"):
                    annotation_files.append(mask_file)
                for mask_file in folder.glob("*_mask.tiff"):
                    annotation_files.append(mask_file)
        
        if not annotation_files:
            print(f"No annotation masks found in folders matching '{folder_pattern}'")
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
                print(f"Skipping {annotation_id}: already uploaded (ROI: {matching_annotation.roi_id})")
                continue
            
            # Process the annotation file
            if self._process_annotation_file(mask_file, annotation_id, matching_annotation):
                processed_count += 1
        
        # Update tracking and save config
        if processed_count > 0:
            if not self.config.workflow.read_only_mode:
                print("Updating OMERO tracking table...")
                self._replace_omero_table_from_config()
            
            print("Saving updated config...")
            self._auto_save_config()
            
            print(f"Successfully processed {processed_count} annotations")
        else:
            print("No new annotations to process")
        
        return processed_count, self.config

    def get_annotation_status_from_disk(self, folder_pattern: str = "output_*") -> dict:
        """Check status of annotations available on disk.
        
        Returns:
            Dictionary with status information about available annotations
        """
        output_path = Path(self.config.output.output_directory)
        
        status = {
            "total_annotations": len(self.config.annotations),
            "processed_annotations": len(self.config.get_processed()),
            "pending_annotations": len(self.config.get_unprocessed()),
            "available_masks": [],
            "missing_masks": [],
        }
        
        # Check which annotation files exist
        for annotation in self.config.annotations:
            category = annotation.category
            annotation_id = annotation.annotation_id
            
            # Check in output folders
            mask_file = output_path / f"output_{category}" / f"{annotation_id}_mask.tif"
            if not mask_file.exists():
                mask_file = output_path / f"output_{category}" / f"{annotation_id}_mask.tiff"
            
            if mask_file.exists():
                status["available_masks"].append({
                    "annotation_id": annotation_id,
                    "image_name": annotation.image_name,
                    "category": category,
                    "uploaded": annotation.processed and annotation.roi_id is not None,
                    "roi_id": annotation.roi_id,
                    "file_path": str(mask_file),
                })
            else:
                status["missing_masks"].append({
                    "annotation_id": annotation_id,
                    "image_name": annotation.image_name,
                    "category": category,
                })
        
        return status

    def run_cp_workflow(self, images_list: Optional[List[Any]] = None) -> Tuple[int, AnnotationConfig]:
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
    def run_full_microsam_workflow(self, images_list: Optional[List[Any]] = None) -> Tuple[int, AnnotationConfig]:
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

    def run_full_micro_sam_workflow(self, images_list: Optional[List[Any]] = None) -> Tuple[int, AnnotationConfig]:
        """Alias for run_full_microsam_workflow for backward compatibility."""
        return self.run_full_microsam_workflow(images_list)

    def run_custom_annotation(self, annotation_func, images_list: Optional[List[Any]] = None) -> Tuple[int, AnnotationConfig]:
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
        
        print(f"Running custom annotation on {len(unprocessed_annotations)} unprocessed annotations")
        
        # Convert to processing units and run annotation
        processing_units = self._convert_annotations_to_processing_units(unprocessed_annotations)
        processed_count = self._run_annotation_batches(processing_units, annotation_func)
        
        # Finalize workflow
        self._finalize_workflow(processed_count)
        
        table_id = self.table_id if self.table_id is not None else -1
        return table_id, self.config



def create_pipeline(
    config: AnnotationConfig, conn=None, config_file_path: Optional[Union[str, Path]] = None
) -> AnnotationPipeline:
    """Create a micro-SAM annotation pipeline with the given configuration.

    Args:
        config: AnnotationConfig object
        conn: OMERO connection object
        config_file_path: Optional path to the source YAML configuration file for persistence.
                         If None, uses config.config_file_path if available.

    Returns:
        AnnotationPipeline instance
    """
    # Use provided path, or fall back to config's stored path
    actual_config_path = config_file_path or config.config_file_path
    
    return AnnotationPipeline(config, conn, actual_config_path)
