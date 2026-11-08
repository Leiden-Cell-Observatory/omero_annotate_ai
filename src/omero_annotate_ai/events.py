"""What a run reports while it works, for whoever displays it.

The pipeline calls an :data:`EventSink` with one of the events below at each step.
A notebook widget, the napari plugin or a script can pass their own sink; the default,
:func:`log_events`, writes them to :mod:`logging`. Sinks are called on the thread
doing the work and must not raise.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Stage = Literal["load", "annotate", "upload", "reconcile"]
RepairKind = Literal[
    "adopted", "uploaded", "reset", "divergent", "duplicate", "pruned", "excluded"
]


@dataclass(frozen=True, slots=True)
class SessionStarted:
    session: int
    pending: int


@dataclass(frozen=True, slots=True)
class BatchPrepared:
    index: int
    annotation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnitCommitted:
    """A unit's mask is saved locally and, for OMERO sets, uploaded."""

    annotation_id: str
    label_id: int | None = None
    roi_id: int | None = None


@dataclass(frozen=True, slots=True)
class UnitExcluded:
    """The annotator excluded a unit: it is done, without a mask or uploads."""

    annotation_id: str


@dataclass(frozen=True, slots=True)
class UploadFailed:
    """Uploading a unit failed; it stays unannotated and is retried next time."""

    annotation_id: str
    error: str


@dataclass(frozen=True, slots=True)
class TableWritten:
    table_id: int
    rows: int


@dataclass(frozen=True, slots=True)
class ReconcileAction:
    """What resuming did to a unit whose local and OMERO records disagreed."""

    annotation_id: str
    action: RepairKind
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BatchReleased:
    """Units handed back without a mask; they stay pending."""

    index: int
    annotation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnitConflict:
    """Another session saved this unit first: its upload is kept, not replaced.

    The mask drawn here is kept in the folder as ``mask_file``.
    """

    annotation_id: str
    mask_file: str


@dataclass(frozen=True, slots=True)
class ClaimsReleased:
    """Opening the set released claims an earlier run on this computer left."""

    units: int


@dataclass(frozen=True, slots=True)
class ClaimsHeld:
    """Other sessions (``<user>/<host>``) are annotating these units right now."""

    units: int
    holders: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RunCancelled:
    stage: Stage


@dataclass(frozen=True, slots=True)
class Progress:
    stage: Stage
    done: int
    total: int


@dataclass(frozen=True, slots=True)
class SessionFinished:
    session: int
    annotated: int
    note: str | None = None


Event = (
    SessionStarted
    | BatchPrepared
    | UnitCommitted
    | UnitExcluded
    | UploadFailed
    | TableWritten
    | ReconcileAction
    | BatchReleased
    | UnitConflict
    | ClaimsReleased
    | ClaimsHeld
    | RunCancelled
    | Progress
    | SessionFinished
)
EventSink = Callable[[Event], None]


def log_events(event: Event) -> None:
    """Default sink: problems as warnings, everything else at INFO (progress at DEBUG)."""
    match event:
        case UploadFailed(annotation_id, error):
            logger.warning("Upload failed for %s: %s", annotation_id, error)
        case UnitConflict(annotation_id, mask_file):
            logger.warning(
                "%s was saved by another session first; your mask is kept as %s",
                annotation_id,
                mask_file,
            )
        case ReconcileAction(annotation_id, action, detail) if action in (
            "divergent",
            "duplicate",
            "reset",
            "excluded",
        ):
            logger.warning("%s: %s %s", annotation_id, action, detail)
        case Progress():
            logger.debug("%s", event)
        case _:
            logger.info("%s", event)
