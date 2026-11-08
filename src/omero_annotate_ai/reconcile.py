"""Bring the local copy and OMERO back in step when a training set is resumed.

The write order (see :mod:`omero_annotate_ai.session`) means a crash leaves one of
a few known states. On resume each unit is compared across the OMERO table row, the
local ``tracking_table.csv`` row and the mask file, and :func:`decide` picks a step:

=================  =========================================================
``excluded``       OMERO has the unit excluded: nothing to repair. Uploads or a
                   mask found for it are reported, never adopted - unless the
                   local copy shows the same session annotated it afterwards
                   (a commit after excluding, interrupted before the table
                   write): then the steps below repair it like any annotation.
``exclude``        Only the local copy has the unit excluded (the table write
                   failed): the exclusion is written to the table.
``mirror``         OMERO already has the unit: keep its row; report a local
                   mask whose pixels differ from the digest OMERO recorded -
                   unless a newer tagged upload holds exactly that mask (a
                   review that stopped halfway): then keep the newer upload,
                   delete the older one and update the table.
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
An excluded unit (``annotation_type: excluded``) has no mask, uploads or tag by
design, so it is never "needs review".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tifffile

from .bundle import atomic_write, file_digest, input_path, mask_path, pixel_digest
from .claims import is_live
from .events import Progress, ReconcileAction, RepairKind
from .provenance import describe_change, file_code, image_code
from .rows import EXCLUDED, keep_local_codes
from .session import NEEDS_REVIEW, check_cancel, upload_unit

if TYPE_CHECKING:
    from collections.abc import Callable

    from .rows import Row
    from .trainingset import TrainingSet

Step = Literal[
    "excluded", "exclude", "mirror", "verify", "find_tagged", "reset", "nothing"
]


@dataclass(frozen=True, slots=True)
class Record:
    """What one side (the OMERO table or the local CSV) knows about a unit."""

    processed: bool
    label_id: int | None
    roi_id: int | None
    excluded: bool = False
    session: int = -1


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
    if remote is not None and remote.excluded:
        annotated_after = (
            local is not None
            and local.session == remote.session
            and (local.label_id is not None or mask_present)
        )
        if not annotated_after:
            return "excluded"
    if remote is not None and remote.processed and remote.label_id is not None:
        return "mirror"
    if (
        local is not None
        and local.excluded
        and (remote is None or not remote.processed)
        and not mask_present
    ):
        return "exclude"
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
    return Record(
        annotation.processed,
        annotation.label_id,
        annotation.roi_id,
        annotation.excluded,
        annotation.session,
    )


DIVERGENT = NEEDS_REVIEW + "the local mask differs from the uploaded label file"


def _read(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(path))


def _with_change(message: str, change: str) -> str:
    """``message``, plus how much the content changed when ISCC codes tell."""
    return f"{message} ({change})" if change else message


@dataclass(slots=True)
class _Reconciler:
    training_set: TrainingSet
    prune_duplicates: bool
    changed: bool = False
    repaired: set[str] = field(default_factory=set[str])

    def event(self, annotation_id: str, action: RepairKind, detail: str = "") -> None:
        self.training_set.events(ReconcileAction(annotation_id, action, detail))

    def mirror(self, unit: Row, local: Row | None, mask: Path) -> None:
        """OMERO has the unit; finish a review, or report local files that differ."""
        if unit.mask_digest and mask.exists() and file_digest(mask) != unit.mask_digest:
            if (label_id := self._newer_label(unit, mask)) is not None:
                self._supersede(unit, local, mask, label_id)
                return
            self.event(
                unit.annotation_id,
                "divergent",
                _with_change(
                    "the local mask differs from the one uploaded to OMERO; kept OMERO's",
                    describe_change(unit.mask_iscc, _read(mask), mask=True),
                ),
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
                _with_change(
                    "the local input image differs from the one annotated",
                    describe_change(unit.input_iscc, _read(image), mask=False),
                ),
            )

    def _newer_label(self, unit: Row, mask: Path) -> int | None:
        """A tagged label file newer than the table's whose pixels are the mask's."""
        if unit.image_id is None or unit.label_id is None:
            return None
        writes, conn = self.training_set.omero_writes, self.training_set.conn
        tag = self.training_set.upload_tag(unit.annotation_id)
        labels, _ = writes.find_tagged(conn, unit.image_id, tag)
        digest = file_digest(mask)
        for label_id in sorted((i for i in labels if i > unit.label_id), reverse=True):
            pixels = writes.download_label(conn, label_id)  # OMERO IDs only grow
            if pixels is not None and pixel_digest(pixels) == digest:
                return label_id
        return None

    def _supersede(
        self, unit: Row, local: Row | None, mask: Path, label_id: int
    ) -> None:
        """Keep a review's newer upload; delete the older label file and ROI."""
        writes, conn = self.training_set.omero_writes, self.training_set.conn
        old_label, old_roi = unit.label_id, unit.roi_id
        unit.label_id = label_id
        new_roi = local.roi_id if local is not None else None
        if new_roi not in (None, old_roi) and writes.object_exists(
            conn, "Roi", new_roi
        ):
            unit.roi_id = new_roi
        else:
            assert unit.image_id is not None
            tag = self.training_set.upload_tag(unit.annotation_id)
            _, rois = writes.find_tagged(conn, unit.image_id, tag)
            unit.roi_id = self._keep_first(
                unit, "Roi", [r for r in rois if r != old_roi]
            )
        if local is not None:
            unit.session = local.session
            unit.annotation_updated_at = local.annotation_updated_at
        for kind, object_id in (("FileAnnotation", old_label), ("Roi", old_roi)):
            if object_id is not None and writes.object_exists(conn, kind, object_id):
                writes.delete_objects(conn, kind, [object_id])
                self.event(unit.annotation_id, "pruned", f"replaced {kind} {object_id}")
        self._accept(unit, mask, "adopted", f"reviewed label {label_id}")

    def excluded(self, unit: Row, local: Row | None, mask: Path, scan: bool) -> None:
        """Leave an excluded unit excluded; report uploads or a mask found for it.

        ``scan``: look for tagged uploads (the local copy is gone).
        """
        found: list[str] = []
        if local is not None and local.label_id is not None:
            found.append(f"label {local.label_id} (session {local.session})")
        if mask.exists():
            found.append(f"mask {mask.name}")
        if scan and unit.image_id is not None:
            tag = self.training_set.upload_tag(unit.annotation_id)
            labels, rois = self.training_set.omero_writes.find_tagged(
                self.training_set.conn, unit.image_id, tag
            )
            found += [f"label {i}" for i in labels] + [f"ROI {i}" for i in rois]
        if found:
            self.event(
                unit.annotation_id,
                "excluded",
                f"excluded in session {unit.session}; not adopted: {', '.join(found)}",
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
        unit.input_iscc = local.input_iscc or unit.input_iscc
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
            unit.error = _with_change(
                DIVERGENT,
                describe_change(image_code(pixels, mask=True), _read(mask), mask=True),
            )
            self.event(unit.annotation_id, "divergent", unit.error)
            return False
        return True

    def _accept(
        self, unit: Row, mask: Path, action: RepairKind, detail: str = ""
    ) -> None:
        """Mark the unit done and upload whatever is still missing."""
        unit.processed = True
        if unit.annotation_type == EXCLUDED:  # annotated after it was excluded
            unit.annotation_type = self.training_set.annotation_type
        unit.mask_digest = file_digest(mask)
        unit.mask_iscc = file_code(mask, mask=True)
        self.training_set.save_local_table()
        if upload_unit(self.training_set, unit, mask):
            self.changed = True
            self.repaired.add(unit.annotation_id)
            self.event(unit.annotation_id, action, detail)


def reconcile(
    training_set: TrainingSet,
    local_rows: dict[str, Row] | None,
    *,
    prune_duplicates: bool = False,
    cancel: Callable[[], bool] | None = None,
) -> bool:
    """Repair an opened set's ``rows`` (loaded from the OMERO table).

    ``local_rows`` is the local CSV as the last run left it (None: there is none),
    read before opening writes it. Rows only the local CSV has (e.g. images added
    when the table update failed) are added back. Changed rows are written to the
    table and the local CSV.

    Returns:
        Whether anything was repaired.
    """
    rows = training_set.rows
    local_known = local_rows is not None
    local_rows = local_rows or {}
    remote_ids = {a.annotation_id for a in rows}
    remote_flags = {a.annotation_id: _record(a) for a in rows}
    for annotation_id, row in local_rows.items():
        if annotation_id not in remote_ids:
            rows.append(row.model_copy(update={"processed": False}))
            remote_flags[annotation_id] = None

    fixer = _Reconciler(training_set, prune_duplicates)
    hours = training_set.config.workflow.claim_hours
    units = list(rows)
    for done, unit in enumerate(units, start=1):
        check_cancel(training_set, cancel, "reconcile")
        if not unit.processed and is_live(unit.claimed_by, unit.claimed_at, hours):
            continue  # another session is annotating it right now
        local = local_rows.get(unit.annotation_id)
        keep_local_codes(unit, local)
        mask = mask_path(training_set.bundle_dir, unit.annotation_id)
        step = decide(
            remote_flags[unit.annotation_id], _record(local), mask.exists(), local_known
        )
        if step == "excluded":
            fixer.excluded(unit, local, mask, scan=not local_known)
        elif step == "exclude":
            assert local is not None
            rows[done - 1] = local.model_copy()
            fixer.changed = True
            fixer.repaired.add(unit.annotation_id)
            fixer.event(unit.annotation_id, "adopted", "exclusion from the local copy")
        elif step == "mirror":
            fixer.mirror(unit, local, mask)
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
    changed = [r for r in rows if r.annotation_id in fixer.repaired]
    changed += [r for r in rows if r.annotation_id not in remote_ids]
    if changed:
        training_set.write_table(changed)
    return fixer.changed
