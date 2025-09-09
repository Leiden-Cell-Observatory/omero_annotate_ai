"""Training convenience functions for clean notebook experiences."""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


def setup_training(
    training_result: Dict[str, Any],
    model_name: str = "",
    # Model parameters
    model_type: str = "vit_b_lm",
    # Training parameters
    epochs: int = 50,
    n_iterations: Optional[int] = None,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    
    # Data parameters
    patch_shape: Union[Tuple[int, int], Tuple[int, int, int]] = (512, 512),
    n_objects_per_batch: int = 25,
    
    # Checkpointing
    save_every: int = 1000,
    validate_every: int = 500,
    
    **kwargs
) -> Dict[str, Any]:
    """
    Setup training configuration from training_result dict.
    
    Args:
        training_result: Dictionary from prepare_training_data_from_table()
        model_name: Name for the training session/model
        model_type: SAM model variant ("vit_b", "vit_l", "vit_h")
        epochs: Number of training epochs (primary training parameter)
        n_iterations: Number of training iterations (calculated from epochs if None)
        batch_size: Training batch size
        learning_rate: Learning rate for training
        patch_shape: Input patch dimensions (height, width) or (slices, height, width)
        n_objects_per_batch: Number of objects per batch for sampling
        save_every: Save checkpoint every N iterations
        validate_every: Run validation every N iterations
        **kwargs: Framework-specific parameters
        
    Returns:
        Dictionary containing all training configuration and paths
        
    Raises:
        ValueError: If training_result is missing required keys
        FileNotFoundError: If training directories don't exist
    """
    # Validate training_result dict
    required_keys = ['training_input', 'training_label', 'val_input', 'val_label']
    missing_keys = [key for key in required_keys if key not in training_result]
    if missing_keys:
        raise ValueError(f"training_result missing required keys: {missing_keys}")
    
    # Validate directories exist
    for key in required_keys:
        path = Path(training_result[key])
        if not path.exists():
            raise FileNotFoundError(f"Training directory does not exist: {path}")
    
    # Generate model name and output paths
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not model_name or model_name.strip() == '':
        model_name = f"micro_sam_training_{timestamp}"
    
    # Determine output directory - use from training_result or create from paths
    if 'output_dir' in training_result:
        output_dir = Path(training_result['output_dir'])
    else:
        # Infer output directory from training paths
        training_path = Path(training_result['training_input'])
        output_dir = training_path.parent
    
    checkpoint_folder = output_dir / "checkpoints"
    
    # Calculate n_iterations if not provided
    if n_iterations is None and epochs > 0:
        # Estimate iterations per epoch based on dataset size
        training_stats = training_result.get('stats', {})
        n_training_images = training_stats.get('n_training_images', 100)
        # Rough estimate: iterations per epoch = dataset_size / batch_size
        iterations_per_epoch = max(1, n_training_images // batch_size)
        n_iterations = epochs * iterations_per_epoch
    
    # Build training configuration
    training_config = {
        # Paths
        'training_input': Path(training_result['training_input']),
        'training_label': Path(training_result['training_label']),
        'val_input': Path(training_result['val_input']),
        'val_label': Path(training_result['val_label']),
        'output_dir': output_dir,
        'checkpoint_folder': checkpoint_folder,
        'model_name': model_name,
        
        # Model parameters
        'model_type': model_type,
        
        # Training parameters
        'epochs': epochs,
        'n_iterations': n_iterations,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        
        # Data parameters
        'patch_shape': patch_shape,
        'n_objects_per_batch': n_objects_per_batch,
        
        # Checkpointing
        'save_every': save_every,
        'validate_every': validate_every,
        
        # Original training result for reference
        'training_result': training_result,
    }
    
    # Add any additional framework-specific parameters
    training_config.update(kwargs)
    
    return training_config


def run_training(
    training_config: Dict[str, Any],
    framework: str = "microsam"
) -> Dict[str, Any]:
    """
    Execute training with framework-specific implementation.
    
    Args:
        training_config: Configuration dictionary from setup_training()
        framework: Training framework to use ("microsam", future: "cellpose", etc.)
        
    Returns:
        Dictionary containing training results and model paths
        
    Raises:
        ValueError: If framework is not supported
        ImportError: If required framework packages are not available
    """
    if framework.lower() == "microsam":
        return _run_microsam_training(training_config)
    else:
        supported_frameworks = ["microsam"]
        raise ValueError(
            f"Unsupported framework: {framework}. "
            f"Supported frameworks: {supported_frameworks}"
        )


def _run_microsam_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal micro-SAM training implementation.
    
    Args:
        config: Training configuration from setup_training()
        
    Returns:
        Dictionary with training results
        
    Raises:
        ImportError: If micro-SAM dependencies are not available
    """
    try:
        import micro_sam.training as sam_training
        from torch_em.data import MinInstanceSampler
        import torch
        import os
    except ImportError as e:
        raise ImportError(
            "micro-SAM dependencies not available. Install with: "
            "conda install -c conda-forge micro-sam"
        ) from e
    
    # Create checkpoint directory
    checkpoint_folder = config['checkpoint_folder']
    checkpoint_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"Starting micro-SAM training...")
    print(f"Model name: {config['model_name']}")
    print(f"Model type: {config['model_type']}")
    print(f"Training configuration:")
    print(f"  Patch shape: {config['patch_shape']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Objects per batch: {config['n_objects_per_batch']}")
    print(f"  Checkpoint folder: {checkpoint_folder}")
    
    # Convert patch_shape to 3D format (C, H, W) if needed
    # micro-SAM expects 3D patch shape: (channels, height, width)
    if len(config['patch_shape']) == 2:
        patch_shape_3d = (1, config['patch_shape'][0], config['patch_shape'][1])
    else:
        patch_shape_3d = config['patch_shape']
    
    print(f"  Using patch shape: {patch_shape_3d}")
    
    # Create sampler with minimum size filter
    sampler = MinInstanceSampler(min_size=25)
    
    # Set up training parameters
    train_instance_segmentation = config.get('train_instance_segmentation', True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training device: {device}")
    
    # Create data loaders with correct API
    train_loader = sam_training.default_sam_loader(
        raw_paths=str(config["training_input"]),
        raw_key="*.tif",
        label_paths=str(config["training_label"]),
        label_key="*.tif",
        with_segmentation_decoder=train_instance_segmentation,
        patch_shape=patch_shape_3d,
        batch_size=config["batch_size"],
        is_seg_dataset=True,
        shuffle=True,
        raw_transform=sam_training.identity,
        sampler=sampler,
    )
    
    val_loader = sam_training.default_sam_loader(
        raw_paths=str(config["val_input"]),
        raw_key="*.tif",
        label_paths=str(config["val_label"]),
        label_key="*.tif",
        with_segmentation_decoder=train_instance_segmentation,
        patch_shape=patch_shape_3d,
        batch_size=config["batch_size"],
        is_seg_dataset=True,
        shuffle=True,
        raw_transform=sam_training.identity,
        sampler=sampler,
    )
    
    print("Data loaders created successfully!")
    
    # Run training with correct function name and parameters
    try:
        sam_training.train_sam(
            name=config['model_name'],
            save_root=str(checkpoint_folder.parent),  # save_root instead of checkpoint_folder
            model_type=config['model_type'],
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=config['epochs'],  # n_epochs instead of n_iterations
            n_objects_per_batch=config['n_objects_per_batch'],
            with_segmentation_decoder=train_instance_segmentation,
            device=device,
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Find checkpoints in the correct location
    model_checkpoint_folder = checkpoint_folder.parent / config['model_name']
    if not model_checkpoint_folder.exists():
        model_checkpoint_folder = checkpoint_folder / config['model_name']
    
    checkpoints = []
    if model_checkpoint_folder.exists():
        checkpoints = list(model_checkpoint_folder.glob("*.pt"))
        checkpoints.sort()
    
    # Find the best checkpoint (micro-SAM typically saves as 'best.pt')
    best_checkpoint = model_checkpoint_folder / "best.pt" if model_checkpoint_folder.exists() else None
    latest_checkpoint = checkpoints[-1] if checkpoints else best_checkpoint
    
    # Export final model path
    final_model_path = config['output_dir'] / f"{config['model_name']}_final.pt"
    
    # Copy best/latest checkpoint to final model location if available
    if latest_checkpoint and latest_checkpoint.exists():
        import shutil
        shutil.copy2(latest_checkpoint, final_model_path)
        print(f"Final model exported to: {final_model_path}")
    
    # Prepare results dictionary
    results = {
        'model_name': config['model_name'],
        'model_type': config['model_type'],
        'checkpoint_folder': model_checkpoint_folder,
        'checkpoints': checkpoints,
        'best_checkpoint': best_checkpoint if best_checkpoint and best_checkpoint.exists() else None,
        'latest_checkpoint': latest_checkpoint,
        'final_model_path': final_model_path if final_model_path.exists() else None,
        'output_dir': config['output_dir'],
        'training_config': config,
        'training_stats': {
            'n_epochs': config['epochs'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
            'n_checkpoints': len(checkpoints),
            'device': device,
            'train_instance_segmentation': train_instance_segmentation,
        }
    }
    
    print(f"\nTraining summary:")
    print(f"  Model name: {config['model_name']}")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Checkpoints saved: {len(checkpoints)}")
    if results['best_checkpoint']:
        print(f"  Best checkpoint: {results['best_checkpoint']}")
    if results['final_model_path']:
        print(f"  Final model: {final_model_path}")
    
    return results