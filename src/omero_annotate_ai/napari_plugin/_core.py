"""The slice of the core training-set API the napari plugin uses, plus a fake.

TEMPORARY: delete this module when the core step API lands (0.5.0 core PRs 3c/3d:
``TrainingSet``, ``AnnotationSession``, events in ``omero_annotate_ai.events``,
``Batch``/``AnnotatorResult`` in ``omero_annotate_ai.annotators.base``). The names
and semantics below follow that plan, so the switch is an import change:

- ``Session.commit`` accepts a result for a subset of the batch and may be called
  again for the same batch (one call per accepted image); units it does not cover
  stay pending.
- ``Session.release`` hands the uncommitted units of a batch back (skipped images
  come back in a later session).
- ``Session.finish`` records the session with its note.

``FakeBackend`` implements the same contract on synthetic images in a local
folder. It backs the "demo training set" button and the tests.
"""

from __future__ import annotations

import shutil
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

Category = Literal["training", "validation", "test"]
CATEGORIES: tuple[Category, ...] = ("training", "validation", "test")


@dataclass(frozen=True, slots=True)
class UnitInfo:
    """What the plugin shows about one annotation unit (a subset of the core Unit)."""

    annotation_id: str
    image_name: str
    category: Category
    timepoint: int = 0
    z_slice: int = 0
    channel: int = 0
    well: str = ""
    field: str = ""

    def describe(self) -> str:
        """One line for the dock: where the image comes from and which plane."""
        place = " / ".join(p for p in (self.image_name, self.well, self.field) if p)
        return f"{place} · t{self.timepoint} c{self.channel} z{self.z_slice} [{self.category}]"


@dataclass(frozen=True, slots=True)
class BatchItem:
    unit: UnitInfo
    pixels: np.ndarray


@dataclass(frozen=True, slots=True)
class Batch:
    index: int
    items: tuple[BatchItem, ...]
    mask_dir: Path
    embedding_dir: Path
    model_type: str


@dataclass(frozen=True, slots=True)
class AnnotatorResult:
    masks: Mapping[str, Path]


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: int
    started: str
    n_annotated: int
    status: Literal["open", "finished", "interrupted"]
    note: str = ""


@dataclass(frozen=True, slots=True)
class SetSummary:
    key: str
    name: str
    storage: Literal["omero", "local"]
    counts: Mapping[Category, tuple[int, int]]
    sessions: tuple[SessionRecord, ...]
    model: str


@dataclass(frozen=True, slots=True)
class UnitCommitted:
    annotation_id: str
    label_id: int | None = None
    roi_id: int | None = None


@dataclass(frozen=True, slots=True)
class UploadFailed:
    annotation_id: str
    error: str


Event = UnitCommitted | UploadFailed
EventSink = Callable[[Event], None]


def series_mask_name(position: int) -> str:
    """File name the core collects for the mask of batch item ``position``."""
    return f"seg_{position:05}.tif"


class Session(Protocol):
    def next_batch(self) -> Batch | None: ...
    def commit(self, batch: Batch, result: AnnotatorResult) -> None: ...
    def release(self, batch: Batch) -> None: ...
    def finish(
        self, note: str, *, status: Literal["finished", "interrupted"] = "finished"
    ) -> SessionRecord: ...


class TrainingSet(Protocol):
    def summary(self) -> SetSummary: ...
    def start_session(self) -> Session: ...


class Backend(Protocol):
    def list_sets(self) -> list[SetSummary]: ...
    def open(self, key: str, events: EventSink) -> TrainingSet: ...


# --- fake implementation ----------------------------------------------------------


def synthetic_image(seed: int, shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """A uint16 image with a few bright round 'nuclei' on a noisy background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    image = rng.normal(200, 20, shape)
    for _ in range(8):
        cy, cx = rng.integers(20, shape[0] - 20), rng.integers(20, shape[1] - 20)
        radius = rng.integers(8, 18)
        image[(yy - cy) ** 2 + (xx - cx) ** 2 < radius**2] += 1500
    return np.clip(image, 0, 65535).astype(np.uint16)


@dataclass
class _FakeUnit:
    info: UnitInfo
    processed: bool = False
    session: int | None = None


@dataclass
class _FakeState:
    root: Path
    name: str
    units: list[_FakeUnit]
    sessions: list[SessionRecord] = field(default_factory=list[SessionRecord])
    batch_size: int = 3
    model: str = "vit_b_lm"


class FakeSession:
    def __init__(self, state: _FakeState, session_id: int, events: EventSink) -> None:
        self._state = state
        self._id = session_id
        self._events = events
        self._batches = 0
        self._in_flight: set[str] = set()

    def next_batch(self) -> Batch | None:
        pending = [
            u
            for u in self._state.units
            if not u.processed and u.info.annotation_id not in self._in_flight
        ]
        if not pending:
            return None
        chosen = pending[: self._state.batch_size]
        self._in_flight.update(u.info.annotation_id for u in chosen)
        scratch = self._state.root / ".scratch" / f"s{self._id}-b{self._batches}"
        (scratch / "embeddings").mkdir(parents=True, exist_ok=True)
        batch = Batch(
            index=self._batches,
            items=tuple(
                BatchItem(
                    u.info, synthetic_image(zlib.crc32(u.info.annotation_id.encode()))
                )
                for u in chosen
            ),
            mask_dir=scratch,
            embedding_dir=scratch / "embeddings",
            model_type=self._state.model,
        )
        self._batches += 1
        return batch

    def commit(self, batch: Batch, result: AnnotatorResult) -> None:
        out = self._state.root / "annotation_output"
        out.mkdir(exist_ok=True)
        by_id = {u.info.annotation_id: u for u in self._state.units}
        for annotation_id, mask in result.masks.items():
            unit = by_id[annotation_id]
            if unit.processed:  # idempotent per unit
                continue
            mask.replace(out / f"{annotation_id}_mask.tif")
            unit.processed, unit.session = True, self._id
            self._in_flight.discard(annotation_id)
            self._events(UnitCommitted(annotation_id))

    def release(self, batch: Batch) -> None:
        for item in batch.items:
            self._in_flight.discard(item.unit.annotation_id)
        shutil.rmtree(batch.mask_dir, ignore_errors=True)

    def finish(
        self, note: str, *, status: Literal["finished", "interrupted"] = "finished"
    ) -> SessionRecord:
        record = SessionRecord(
            id=self._id,
            started=self._state.sessions[-1].started,
            n_annotated=sum(u.session == self._id for u in self._state.units),
            status=status,
            note=note,
        )
        self._state.sessions[-1] = record
        return record


class FakeTrainingSet:
    def __init__(self, state: _FakeState, events: EventSink) -> None:
        self._state = state
        self._events = events

    def summary(self) -> SetSummary:
        return _summarize(self._state)

    def start_session(self) -> Session:
        session_id = len(self._state.sessions) + 1
        started = datetime.now(UTC).isoformat(timespec="seconds")
        self._state.sessions.append(SessionRecord(session_id, started, 0, "open"))
        return FakeSession(self._state, session_id, self._events)


def _summarize(state: _FakeState) -> SetSummary:
    counts: dict[Category, tuple[int, int]] = {
        c: (
            sum(u.processed for u in state.units if u.info.category == c),
            sum(u.info.category == c for u in state.units),
        )
        for c in CATEGORIES
    }
    return SetSummary(
        key=str(state.root),
        name=state.name,
        storage="local",
        counts=counts,
        sessions=tuple(state.sessions),
        model=state.model,
    )


class FakeBackend:
    """One local demo training set with synthetic images."""

    def __init__(
        self,
        root: Path,
        splits: Sequence[Category] = ("training",) * 4 + ("validation",) * 2,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        units = [
            _FakeUnit(UnitInfo(f"demo{i:03d}_0_0", f"demo_image_{i}", category))
            for i, category in enumerate(splits)
        ]
        self._state = _FakeState(root=root, name="Demo training set", units=units)

    def list_sets(self) -> list[SetSummary]:
        return [_summarize(self._state)]

    def open(self, key: str, events: EventSink) -> TrainingSet:
        if key != str(self._state.root):
            raise KeyError(key)
        return FakeTrainingSet(self._state, events)
