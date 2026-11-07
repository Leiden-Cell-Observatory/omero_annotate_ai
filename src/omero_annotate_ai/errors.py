"""Errors omero-annotate-ai raises on purpose.

All derive from :class:`OmeroAnnotateAIError`; those describing bad input also derive
from the matching built-in, so ``except ValueError`` / ``except ImportError`` still work.
"""


class OmeroAnnotateAIError(Exception):
    """Base class of every error omero-annotate-ai raises on purpose."""


class SelectionError(OmeroAnnotateAIError, ValueError):
    """The requested images cannot be selected (too few, unknown stratify key, no wells)."""


class UnsupportedTableError(OmeroAnnotateAIError, ValueError):
    """The tracking table was not written by this version of omero-annotate-ai."""


class TableValueTooLongError(OmeroAnnotateAIError, ValueError):
    """A string value is longer than its tracking-table column allows."""


class AnnotatorUnavailable(OmeroAnnotateAIError, ImportError):
    """The annotation tool (e.g. micro-SAM with napari) is not installed."""


class RemoteError(OmeroAnnotateAIError):
    """Reading from or writing to OMERO failed; the original error is chained."""


class TableReadError(RemoteError):
    """The tracking table could not be read. A resume stops rather than start fresh."""


class TableWriteError(RemoteError):
    """The tracking table could not be updated. Uploads stay tagged; resume repairs."""


class IntegrityError(OmeroAnnotateAIError):
    """Pixels on disk or on OMERO no longer match the digest recorded for them."""


class Cancelled(OmeroAnnotateAIError):
    """The caller's ``cancel`` callback asked a long operation to stop."""
