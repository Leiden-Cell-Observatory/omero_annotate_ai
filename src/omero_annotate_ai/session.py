"""One annotation session, step by step: prepare a batch, commit its masks, finish.

Every front-end (the notebook training_set, the napari plugin, a script) drives the same
steps, so a mask is always saved in the same order:

1. the mask is moved into the bundle (``annotation_output/{id}_mask.tif``);
2. the unit is accepted locally: ``processed``, session and mask digest go to
   ``tracking_table.csv`` (written atomically);
3. for OMERO sets, the label file is uploaded and its ID saved, then the ROI and
   its ID - each ID straight after its own upload, both tagged with the set's tag;
4. once per commit, the tracking table rows are updated in place.

On OMERO sets a session claims the units of a batch before showing them and
re-reads each unit's row before saving it, so several computers can annotate one set
(see :mod:`omero_annotate_ai.claims`). A unit another session saved first is not
uploaded again: the mask drawn here is kept next to it as a conflict file.

A review session (``review=True``) hands out annotated units with their saved mask.
A changed mask replaces the old one: it is uploaded first (tagged like the first),
its IDs saved, then the old label file and ROI are deleted and the table row
updated. A crash in between leaves both, and resuming keeps the newer one (see
:mod:`omero_annotate_ai.reconcile`). An unchanged mask changes nothing.

A crash between any two steps loses at most one upload, and the tag finds that one
again when the set is resumed (see :mod:`omero_annotate_ai.reconcile`). A failed
upload leaves the unit unannotated, with the error in the local CSV, and the session
carries on with the next unit.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tifffile

from . import claims
from .annotators import Batch, BatchItem
from .bundle import conflict_path, file_digest, fresh_scratch, mask_path
from .errors import Cancelled
from .events import (
    BatchPrepared,
    BatchReleased,
    Progress,
    ReconcileAction,
    RunCancelled,
    SessionFinished,
    SessionStarted,
    Stage,
    UnitCommitted,
    UnitConflict,
    UploadFailed,
)

if TYPE_CHECKING:
    from .rows import Row
    from .trainingset import TrainingSet


NEEDS_REVIEW = "needs review: "
"""Prefix of a unit's ``error`` when its local and uploaded masks differ. Sessions
skip such units until the user resolves them (delete the local mask to accept the
upload, or delete the upload to keep the local mask), then resumes the set."""


@dataclass(frozen=True, slots=True)
class BatchReport:
    """What a :meth:`AnnotationSession.commit` call did with each unit.

    ``committed``: saved (in a review: changed and saved); ``failed``: upload
    failed (stays as it was); ``conflicts``: another session saved it first (its
    upload is kept); ``unchanged``: reviewed without changes.
    """

    committed: tuple[str, ...]
    failed: tuple[str, ...]
    conflicts: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()


def check_cancel(
    training_set: TrainingSet, cancel: Callable[[], bool] | None, stage: Stage
) -> None:
    """Raise :class:`~omero_annotate_ai.errors.Cancelled` if ``cancel()`` says so."""
    if cancel is not None and cancel():
        training_set.events(RunCancelled(stage))
        raise Cancelled(f"Cancelled during {stage}")


def upload_unit(training_set: TrainingSet, annotation: Row, mask_file: Path) -> bool:
    """Steps 3a/3b: upload what is missing (label file, then ROI), saving each ID.

    Returns False, leaving the unit unannotated with the error recorded, if an
    upload fails.
    """
    label_img = np.asarray(tifffile.imread(mask_file))
    common = training_set.upload_fields(annotation)
    writes = training_set.omero_writes
    try:
        if annotation.label_id is None:
            annotation.label_id = writes.upload_label(
                training_set.conn, annotation.image_id, label_img, **common
            )
            training_set.save_local_table()
        if annotation.roi_id is None:
            annotation.roi_id = writes.upload_roi(
                training_set.conn,
                annotation.image_id,
                label_img,
                is_volumetric=annotation.is_volumetric,
                **common,
            )
            training_set.save_local_table()
    except Exception as error:  # noqa: BLE001 - any failure leaves the unit for a retry
        annotation.processed = False
        annotation.error = f"{type(error).__name__}: {error}"
        training_set.save_local_table()
        training_set.events(UploadFailed(annotation.annotation_id, annotation.error))
        return False
    annotation.error = None
    training_set.events(
        UnitCommitted(annotation.annotation_id, annotation.label_id, annotation.roi_id)
    )
    return True


def commit_unit(
    training_set: TrainingSet,
    annotation: Row,
    mask_source: Path,
    session: int,
) -> bool:
    """Save one unit's mask: steps 1-3 of the write order (see the module docstring).

    Returns:
        True when the unit is saved (and, for OMERO sets, uploaded).
    """
    target = mask_path(training_set.bundle_dir, annotation.annotation_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if Path(mask_source).resolve() != target.resolve():
        shutil.move(mask_source, target)  # a rename when both are in the bundle
    annotation.mask_digest = file_digest(target)
    annotation.session = session
    annotation.error = None
    annotation.claimed_by = annotation.claimed_at = ""  # the table write releases it
    annotation.mark_processed(annotation_type=training_set.annotation_type)
    training_set.save_local_table()
    if training_set.is_local:
        training_set.events(UnitCommitted(annotation.annotation_id))
        return True
    return upload_unit(training_set, annotation, target)


def review_unit(
    training_set: TrainingSet, annotation: Row, mask_source: Path, session: int
) -> Literal["revised", "unchanged", "failed"]:
    """Save a reviewed mask: replace the unit's mask and uploads if it changed."""
    digest = file_digest(mask_source)
    if digest == annotation.mask_digest:
        Path(mask_source).unlink(missing_ok=True)
        return "unchanged"
    target = mask_path(training_set.bundle_dir, annotation.annotation_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(mask_source).replace(target)
    old = (("FileAnnotation", annotation.label_id), ("Roi", annotation.roi_id))
    annotation.mask_digest = digest
    annotation.session = session
    annotation.error = None
    annotation.claimed_by = annotation.claimed_at = ""
    annotation.mark_processed(annotation_type=training_set.annotation_type)
    training_set.save_local_table()
    if training_set.is_local:
        training_set.events(UnitCommitted(annotation.annotation_id))
        return "revised"
    pixels = np.asarray(tifffile.imread(target))
    common = training_set.upload_fields(annotation)
    writes, conn = training_set.omero_writes, training_set.conn
    try:  # new uploads first: a crash leaves the old ones, never none
        annotation.label_id = writes.upload_label(
            conn, annotation.image_id, pixels, **common
        )
        training_set.save_local_table()
        annotation.roi_id = writes.upload_roi(
            conn,
            annotation.image_id,
            pixels,
            is_volumetric=annotation.is_volumetric,
            **common,
        )
        training_set.save_local_table()
    except Exception as error:  # noqa: BLE001 - the old uploads stay valid
        annotation.error = f"{type(error).__name__}: {error}"
        training_set.save_local_table()
        training_set.events(UploadFailed(annotation.annotation_id, annotation.error))
        return "failed"
    for kind, object_id in old:
        if object_id is None:
            continue
        try:
            writes.delete_objects(conn, kind, [object_id])
        except Exception as error:  # noqa: BLE001 - a leftover is reported, not fatal
            training_set.events(
                ReconcileAction(
                    annotation.annotation_id,
                    "duplicate",
                    f"old {kind} {object_id} not deleted: {error}",
                )
            )
    training_set.events(
        UnitCommitted(annotation.annotation_id, annotation.label_id, annotation.roi_id)
    )
    return "revised"


class AnnotationSession:
    """Hands out batches of pending units and saves the masks that come back.

    Each pending unit is handed out at most once per session. Batches may be
    committed in parts and committed again (each unit is saved once); units never
    committed are handed back with :meth:`release` and stay pending for the next
    session. With ``review=True`` the annotated units (or those in ``ids``) are
    handed out instead, each with its saved mask. Not thread-safe: call it from one
    thread.
    """

    def __init__(
        self,
        training_set: TrainingSet,
        *,
        batch_size: int | None = None,
        model: str | None = None,
        cancel: Callable[[], bool] | None = None,
        review: bool = False,
        ids: Collection[str] | None = None,
    ) -> None:
        self._set = training_set
        self.review = review
        self._ids = None if ids is None else set(ids)
        self._batch_size = batch_size
        self._cancel = cancel
        self._open: dict[int, set[str]] = {}
        self._scratch: dict[int, Path] = {}
        self._handed_out: set[str] = set()  # each unit at most once per session
        self._finished = False
        self.session = training_set.begin_session(
            model, batch_size, kind="review" if review else "annotate"
        )
        self.token = claims.token(self.session, training_set.user)
        self.annotated = 0
        training_set.events(SessionStarted(self.session, len(self._pending())))

    def _pending(self) -> list[Row]:
        hours = self._set.config.workflow.claim_hours
        return [
            a
            for a in self._set.rows
            if a.processed == self.review
            and (self._ids is None or a.annotation_id in self._ids)
            and a.annotation_id not in self._handed_out
            and not (a.error or "").startswith(NEEDS_REVIEW)
            and not (
                a.claimed_by != self.token
                and claims.is_live(a.claimed_by, a.claimed_at, hours)
            )
        ]

    def _choose(self) -> list[Row]:
        """The next batch's units; on OMERO sets, claimed for this session."""
        if self._set.is_local:
            pending = self._pending()
            return pending[: self._batch_size or len(pending)]
        self._set.refresh()
        pending = self._pending()
        size = self._batch_size or len(pending)
        taken = set(
            claims.claim(
                self._set,
                self.token,
                [a.annotation_id for a in pending],
                size,
                annotated=self.review,
            )
        )
        return [a for a in pending if a.annotation_id in taken]

    def next_batch(self) -> Batch | None:
        """Load the next pending units (pixels, context channels, input images).

        Returns None when nothing is left to hand out.
        """
        chosen = self._choose()
        if not chosen:
            return None
        items: list[BatchItem] = []
        for done, annotation in enumerate(chosen, start=1):
            check_cancel(self._set, self._cancel, "load")
            items.append(self._set.prepare_item(annotation, with_mask=self.review))
            self._set.events(Progress("load", done, len(chosen)))
        self._set.save_local_table()  # records the input digests
        index = self._set.next_batch_index()
        scratch = fresh_scratch(self._set.bundle_dir, self.session, index)
        batch = self._set.make_batch(index, items, scratch)
        ids = tuple(item.annotation_id for item in items)
        self._open[index] = set(ids)
        self._handed_out.update(ids)
        self._scratch[index] = scratch
        self._set.events(BatchPrepared(index, ids))
        return batch

    def commit(
        self,
        batch: Batch,
        result: Mapping[str, Path],
        *,
        publish_config: bool = True,
    ) -> BatchReport:
        """Save the masks in ``result`` (``{annotation_id: file}``), then update the table.

        ``result`` may cover part of the batch, and a unit already committed is
        skipped, so calling this again is safe. With ``publish_config=False`` the
        config on OMERO is only replaced at :meth:`finish`.

        Raises:
            TableWriteError: The tracking table could not be updated (the uploads
                are saved locally and tagged; resuming repairs the table).
        """
        waiting = self._open.get(batch.index, set())
        by_id = {a.annotation_id: a for a in self._set.rows}
        todo = [i.annotation_id for i in batch.items if i.annotation_id in result]
        todo = [annotation_id for annotation_id in todo if annotation_id in waiting]
        outcome: dict[str, list[str]] = {
            "revised": [],
            "failed": [],
            "conflict": [],
            "unchanged": [],
        }
        for done, annotation_id in enumerate(todo, start=1):
            check_cancel(self._set, self._cancel, "upload")
            row, mask = by_id[annotation_id], result[annotation_id]
            if self._saved_elsewhere(row, mask):
                outcome["conflict"].append(annotation_id)
            elif self.review:
                outcome[review_unit(self._set, row, mask, self.session)].append(
                    annotation_id
                )
            else:
                ok = commit_unit(self._set, row, mask, self.session)
                outcome["revised" if ok else "failed"].append(annotation_id)
            waiting.discard(annotation_id)
            self._set.events(Progress("upload", done, len(todo)))
        committed = outcome["revised"]
        self.annotated += len(committed)
        self._unclaim(set(outcome["unchanged"]))

        if committed and not self._set.is_local:
            self._set.write_table([by_id[i] for i in committed])
            if publish_config:
                self._set.publish_config()
        self._set.save_config()
        if not waiting:
            self._close(batch)
        return BatchReport(
            tuple(committed),
            tuple(outcome["failed"]),
            tuple(outcome["conflict"]),
            tuple(outcome["unchanged"]),
        )

    def _saved_elsewhere(self, row: Row, mask_source: Path) -> bool:
        """Whether another session saved the unit first (OMERO sets; re-reads the row).

        If so the table's row is taken over and the mask is kept as a conflict file.
        """
        if self._set.is_local:
            return False
        annotation_id = row.annotation_id
        theirs = self._set.current_rows([annotation_id]).get(annotation_id)
        if theirs is None or not theirs.processed:
            return False
        if self.review and theirs.label_id == row.label_id:
            return False  # nobody else changed it since it was handed out
        keep = conflict_path(self._set.bundle_dir, annotation_id, self.session)
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(mask_source, keep)
        rows = self._set.rows
        rows[[r.annotation_id for r in rows].index(annotation_id)] = theirs
        self._set.save_local_table()
        self._set.events(UnitConflict(annotation_id, str(keep)))
        return True

    def release(self, batch: Batch) -> None:
        """Hand back the batch's uncommitted units (they stay pending) and clean up."""
        left = self._open.get(batch.index, set())
        if left:
            self._unclaim(left)
            self._set.events(BatchReleased(batch.index, tuple(sorted(left))))
        self._close(batch)

    def _unclaim(self, annotation_ids: set[str]) -> None:
        if not self._set.is_local and annotation_ids:
            claims.release(self._set, self.token, sorted(annotation_ids))

    def _close(self, batch: Batch) -> None:
        self._open.pop(batch.index, None)
        self._scratch.pop(batch.index, None)
        shutil.rmtree(batch.mask_dir, ignore_errors=True)

    def finish(
        self,
        note: str | None = None,
        *,
        status: Literal["finished", "interrupted"] = "finished",
    ) -> int:
        """End the session: record it (with ``note``) in the config and publish it.

        Returns the number of units annotated. Calling it again does nothing.
        """
        if self._finished:
            return self.annotated
        self._finished = True
        for index, scratch in list(self._scratch.items()):  # still-open batches
            self._unclaim(self._open.pop(index, set()))
            shutil.rmtree(scratch, ignore_errors=True)
        self._scratch.clear()
        self._set.end_session(self.session, self.annotated, note, status)
        self._set.events(SessionFinished(self.session, self.annotated, note))
        return self.annotated
