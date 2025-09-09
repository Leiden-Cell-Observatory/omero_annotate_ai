"""Logging utilities for omero_annotate_ai package."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "omero_annotate_ai",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    console_level: Optional[int] = None,
    jupyter_mode: bool = None
) -> logging.Logger:
    """
    Set up a logger for the omero_annotate_ai package.
    
    Args:
        name: Logger name
        level: Default logging level for file output
        log_file: Path to log file. If None, no file logging
        verbose: If True, show DEBUG level in console, otherwise INFO
        console_level: Override console level directly (takes precedence over verbose)
        jupyter_mode: If True, optimize for Jupyter notebook. If None, auto-detect
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set the base level to DEBUG so all messages are captured
    logger.setLevel(logging.DEBUG)
    
    # Auto-detect Jupyter if not specified
    if jupyter_mode is None:
        jupyter_mode = 'ipykernel' in sys.modules
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - be more restrictive in Jupyter
    if jupyter_mode and not verbose:
        # In Jupyter with verbose=False, only show WARNING and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
    else:
        # Normal console behavior
        console_handler = logging.StreamHandler(sys.stdout)
        if console_level is not None:
            console_handler.setLevel(console_level)
        elif verbose:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(logging.INFO)
    
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_training_logger(
    output_dir: Union[str, Path],
    verbose: bool = False,
    timestamp: Optional[str] = None
) -> logging.Logger:
    """
    Create a logger specifically for training data preparation.
    
    Args:
        output_dir: Directory where training data and logs will be saved
        verbose: If True, show debug info in console
        timestamp: Optional timestamp string for log filename
        
    Returns:
        Configured logger with file output in the training directory
    """
    output_dir = Path(output_dir)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = output_dir / f"training_preparation_{timestamp}.log"
    
    logger = setup_logger(
        name="omero_annotate_ai.training",
        level=logging.DEBUG,  # Always log everything to file
        log_file=log_file,
        verbose=verbose,
        jupyter_mode=True  # Assume Jupyter mode for training functions
    )
    
    if verbose:
        logger.info(f"Training preparation log created: {log_file}")
    return logger