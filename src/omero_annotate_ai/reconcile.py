"""Bring the local copy and OMERO back in step when a training set is resumed.

The write order (see :mod:`omero_annotate_ai.session`) means a crash leaves one of
a few known states. On resume each unit is compared across the OMERO table row, the
local ``tracking_table.csv`` row and the mask file, and :func:`decide` picks a step:

=================  =========================================================
``mirror``         OMERO already has the unit: keep its row; report a local
                   mask whose pixels differ from the digest OMERO recorded.
``verify``         Only the local copy has the upload IDs (the table update
                   failed): check the uploads still exist and match the mask,
                   then write them to the table.
``find_tagged``    A mask exists (or the local copy is gone) but no IDs: look
                   for uploads carrying the unit's tag, adopt them, upload what
                   is missing - never a second copy.
``reset``          The local copy says done but has no mask and no IDs: the
                   unit goes back to pending.
``nothing``        Nothing to repair.
=================  =========================================================

Repairing never overwrites silently: a mask whose pixels differ from the upload is
reported (``divergent``) and left for the user. Running it twice changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tifffile

from .bundle import atomic_write, file_digest, input_path, mask_path, pixel_digest
from .events import Progress, ReconcileAction, RepairKind
from .session import NEEDS_REVIEW, check_cancel, upload_unit

if TYPE_CHECKING:
    from collections.abc import Callable

    from .rows import Row
    from .trainingset import TrainingSet

Step = Literal["mirror", "verify", "find_tagged", "reset", "nothing"]


@dataclass(frozen=True, slots=True)
class Record:
    """What one side (the OMERO table or the local CSV) knows about a unit."""

    processed: bool
    label_id: int | None
    roi_id: int | None


def decide(
    remote: Record | None, local: Record | None, mask_present: bool, local_known: bool
) -> Step:
    """The repair step for one unit (see the module docstring).

    Args:
        remote: The unit's OMERO table row; None if the table lacks it.
        local: The unit's local CSV row; None if the CSV lacks it.
        mask_present: Whether ``annotation_output/{id}_mask.tif`` exists.
        local_known: Whether the bundle has a local CSV at all.
    """
    if remote is not None and remote.processed and remote.label_id is not None:
        return "mirror"
    if local is not None and local.label_id is not None:
        return "verify"
    if mask_present or not local_known:
        return "find_tagged"
    if local is not None and local.processed:
        return "reset"
    return "nothing"


def _record(annotation: Row | None) -> Record | None:
    if annotation is None:
        return None
    return Record(annotation.processed, annotation.label_id, annotation.roi_id)


DIVERGENT = NEEDS_REVIEW + "the local mask differs from the uploaded label file"


@dataclass(slots=True)
class _Reconciler:
    training_set: TrainingSet
    prune_duplicates: bool
    changed: bool = False

    def event(self, annotation_id: str, action: RepairKind, detail: str = "") -> None:
        self.training_set.events(ReconcileAction(annotation_id, action, detail))

    def mirror(self, unit: Row, mask: Path) -> None:
        """OMERO has the unit; only report local files that no longer match it."""
        if unit.mask_digest and mask.exists() and file_digest(mask) != unit.mask_digest:
            self.event(
                unit.annotation_id,
                "divergent",
                "the local mask differs from the one uploaded to OMERO; kept OMERO's",
            )
        image = input_path(self.training_set.bundle_dir, unit.annotation_id)
        if (
            unit.input_digest
            and image.exists()
            and file_digest(image) != unit.input_digest
        ):
            self.event(
                unit.annotation_id,
                "divergent",
                "the local input image differs from the one annotated",
            )

    def verify(self, unit: Row, local: Row, mask: Path) -> None:
        """The local copy has upload IDs the table lacks: check them, then adopt."""
        writes, conn = self.training_set.omero_writes, self.training_set.conn
        assert local.label_id is not None
        pixels = writes.download_label(conn, local.label_id)
        if pixels is None:  # the label file is gone: as if it was never uploaded
            self.find_tagged(unit, mask)
            return
        if not self._match_mask(unit, mask, pixels):
            unit.label_id, unit.roi_id = local.label_id, local.roi_id  # for next time
            return
        unit.label_id = local.label_id
        if local.roi_id is not None and writes.object_exists(conn, "Roi", local.roi_id):
            unit.roi_id = local.roi_id
        elif unit.image_id is not None:  # the ROI may be posted but never recorded
            tag = self.training_set.upload_tag(unit.annotation_id)
            _, rois = writes.find_tagged(conn, unit.image_id, tag)
            unit.roi_id = self._keep_first(unit, "Roi", rois)
        unit.session = local.session
        unit.input_digest = local.input_digest or unit.input_digest
        unit.annotation_created_at = local.annotation_created_at
        self._accept(unit, mask, "adopted", f"label {unit.label_id}")

    def find_tagged(self, unit: Row, mask: Path) -> None:
        """Adopt uploads carrying the unit's tag; upload what is missing."""
        if unit.image_id is None:
            return
        writes, conn = self.training_set.omero_writes, self.training_set.conn
        tag = self.training_set.upload_tag(unit.annotation_id)
        labels, rois = writes.find_tagged(conn, unit.image_id, tag)
        label_id = self._keep_first(unit, "FileAnnotation", labels)
        roi_id = self._keep_first(unit, "Roi", rois)
        pixels = writes.download_label(conn, label_id) if label_id else None
        if pixels is None and not mask.exists():
            return  # nothing annotated, nothing uploaded
        if pixels is not None:
            if not self._match_mask(unit, mask, pixels):
                unit.label_id, unit.roi_id = label_id, roi_id  # for next time
                return
            unit.label_id = label_id
        unit.roi_id = roi_id
        self._accept(unit, mask, "adopted" if pixels is not None else "uploaded")

    def _keep_first(self, unit: Row, kind: str, ids: list[int]) -> int | None:
        """The oldest tagged upload; later copies are reported (deleted if pruning)."""
        if len(ids) > 1:
            if self.prune_duplicates:
                self.training_set.omero_writes.delete_objects(
                    self.training_set.conn, kind, ids[1:]
                )
                self.event(unit.annotation_id, "pruned", f"{kind} {ids[1:]}")
            else:
                self.event(unit.annotation_id, "duplicate", f"{kind} {ids[1:]}")
        return ids[0] if ids else None

    def _match_mask(self, unit: Row, mask: Path, pixels: np.ndarray) -> bool:
        """Write the mask from the upload if missing; report it if it differs."""
        if not mask.exists():
            atomic_write(mask, lambda tmp: tifffile.imwrite(tmp, pixels))
            return True
        if pixel_digest(pixels) != file_digest(mask):
            unit.error = DIVERGENT
            self.event(unit.annotation_id, "divergent", DIVERGENT)
            return False
        return True

    def _accept(
        self, unit: Row, mask: Path, action: RepairKind, detail: str = ""
    ) -> None:
        """Mark the unit done and upload whatever is still missing."""
        unit.processed = True
        unit.mask_digest = file_digest(mask)
        self.training_set.save_local_table()
        if upload_unit(self.training_set, unit, mask):
            self.changed = True
            self.event(unit.annotation_id, action, detail)


def reconcile(
    training_set: TrainingSet,
    *,
    prune_duplicates: bool = False,
    cancel: Callable[[], bool] | None = None,
) -> bool:
    """Repair an opened set's ``rows`` (loaded from the OMERO table).

    Rows only the local CSV has (e.g. images added when the table update failed)
    are added back. Changed rows are written to the table and the local CSV.

    Returns:
        Whether anything was repaired.
    """
    rows = training_set.rows
    local_rows = training_set.read_local_table()
    local_known = local_rows is not None
    local_rows = local_rows or {}
    remote_ids = {a.annotation_id for a in rows}
    remote_flags = {a.annotation_id: _record(a) for a in rows}
    for annotation_id, row in local_rows.items():
        if annotation_id not in remote_ids:
            rows.append(row.model_copy(update={"processed": False}))
            remote_flags[annotation_id] = None

    fixer = _Reconciler(training_set, prune_duplicates)
    units = list(rows)
    for done, unit in enumerate(units, start=1):
        check_cancel(training_set, cancel, "reconcile")
        local = local_rows.get(unit.annotation_id)
        mask = mask_path(training_set.bundle_dir, unit.annotation_id)
        step = decide(
            remote_flags[unit.annotation_id], _record(local), mask.exists(), local_known
        )
        if step == "mirror":
            fixer.mirror(unit, mask)
        elif step == "verify":
            assert local is not None
            fixer.verify(unit, local, mask)
        elif step == "find_tagged":
            fixer.find_tagged(unit, mask)
        elif step == "reset":
            fixer.event(unit.annotation_id, "reset", "marked done locally, but no mask")
        if local is not None and not unit.processed and local.error and not unit.error:
            unit.error = local.error
        training_set.events(Progress("reconcile", done, len(units)))

    training_set.save_local_table()
    if fixer.changed or len(rows) > len(remote_ids):
        training_set.write_table()
    return fixer.changed
