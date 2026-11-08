"""Claims: several computers annotating one OMERO training set without collisions.

Before a session shows a unit in napari it claims the unit in the tracking table:
``claimed_by`` gets the session's token (``<session>@<user>/<host>``) and
``claimed_at`` the time (UTC). Other sessions skip units with a live claim. A claim
is released when the unit is committed or handed back, and expires after
``workflow.claim_hours`` - so a crashed session never blocks a unit for good.
Opening a set releases the claims an earlier run on the same computer (same user
and host) left behind.

OMERO.tables has no compare-and-swap, so claiming is a read-modify-write of the
claim cells only (:func:`~omero_annotate_ai.omero.tracking_table.update_cells`):
two sessions claiming the same unit within the same instant could both win. The
commit check in :mod:`omero_annotate_ai.session` (re-read the row, never upload
over a unit another session saved) keeps that case from duplicating uploads.
Local sets have no claims: their folder belongs to one computer.
"""

from __future__ import annotations

import hashlib
import socket
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trainingset import TrainingSet


def host_id() -> str:
    """Short, stable ID of this computer (a hash: the host name is not stored)."""
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]


def owner(user: str | None) -> str:
    """``<user>/<host>``: whose claims a run on this computer may release."""
    return f"{user or 'unknown'}/{host_id()}"


def token(session: int, user: str | None) -> str:
    """The claim a session writes: ``<session>@<user>/<host>``."""
    return f"{session}@{owner(user)}"


def now() -> str:
    """The current time as stored in ``claimed_at``."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def is_live(claimed_by: Any, claimed_at: Any, hours: float) -> bool:
    """Whether a claim is held and younger than ``hours`` (unreadable: expired)."""
    if not claimed_by:
        return False
    try:
        since = datetime.fromisoformat(str(claimed_at))
    except ValueError:
        return False
    return datetime.now(UTC) - since < timedelta(hours=hours)


def claim(
    training_set: TrainingSet, session_token: str, keys: Sequence[str], limit: int
) -> list[str]:
    """Claim up to ``limit`` of ``keys`` (in order) that are free; returns them.

    Free means: not annotated, and not claimed by a live session other than
    ``session_token``.
    """
    hours = training_set.config.workflow.claim_hours
    stamp = now()
    taken = 0

    def take(row: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal taken
        if taken >= limit or row.get("processed"):
            return None
        holder = row.get("claimed_by") or ""
        if holder != session_token and is_live(holder, row.get("claimed_at"), hours):
            return None
        taken += 1
        return {"claimed_by": session_token, "claimed_at": stamp}

    return training_set.update_cells(keys, take)


def release(
    training_set: TrainingSet, holder: str, keys: Sequence[str], *, suffix: bool = False
) -> list[str]:
    """Clear the claims ``holder`` holds on ``keys``; returns the keys released.

    With ``suffix=True`` every claim ending in ``holder`` is cleared (an
    :func:`owner`: all sessions of one user on one computer).
    """

    def clear(row: dict[str, Any]) -> dict[str, Any] | None:
        held = str(row.get("claimed_by") or "")
        mine = held.endswith(f"@{holder}") if suffix else held == holder
        return {"claimed_by": "", "claimed_at": ""} if held and mine else None

    return training_set.update_cells(keys, clear)
