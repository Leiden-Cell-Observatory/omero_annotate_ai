"""Saved logins: long-lived OMERO session tokens, one per server and user.

Logging in with a password can leave a *token* behind: a separate OMERO session that
stays valid as long as it is used at least once every ``idle_days``. The next time,
notebooks and the napari plugin join that session instead of asking for the password
again. The password itself is never stored.

Tokens are kept in ``~/.omero-annotate-ai/tokens.json`` (readable by the user only).
A token made with Glencoe's ``omero-user-token set`` (``~/.omero_user_token/token``,
or ``$OMERO_USER_TOKEN_DIR``) is offered as well. Anyone who can read a token can use
the OMERO account until it is forgotten or expires: :func:`forget_login` ends the
session on the server too.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from omero.gateway import BlitzGateway

DEFAULT_IDLE_DAYS = 7
_DAY_MS = 24 * 3600 * 1000


@dataclass(frozen=True)
class SavedLogin:
    """A token for one server and user."""

    host: str
    port: int | None
    username: str
    key: str
    secure: bool = True
    created: str = ""
    source: Literal["omero-annotate-ai", "omero-user-token"] = "omero-annotate-ai"

    @property
    def label(self) -> str:
        server = self.host if self.port is None else f"{self.host}:{self.port}"
        user = self.username or "token"
        when = f", saved {self.created[:10]}" if self.created else ""
        return f"{user} @ {server}{when}" + (
            " (omero-user-token)" if self.source == "omero-user-token" else ""
        )


def tokens_file() -> Path:
    return Path.home() / ".omero-annotate-ai" / "tokens.json"


def _read_ours() -> list[SavedLogin]:
    try:
        records = json.loads(tokens_file().read_text())
    except (OSError, ValueError):
        return []
    return [SavedLogin(**record) for record in records]


def _write_ours(logins: list[SavedLogin]) -> None:
    path = tokens_file()
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Created readable by the user only, before the token is written into it.
    fd = os.open(
        tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR
    )
    with os.fdopen(fd, "w") as handle:
        json.dump([asdict(login) for login in logins], handle, indent=2)
    tmp.replace(path)


def _omero_user_token() -> SavedLogin | None:
    """The token of ``omero-user-token set`` (``key@host:port``), if there is one."""
    directory = os.getenv("OMERO_USER_TOKEN_DIR") or Path.home() / ".omero_user_token"
    try:
        token = (Path(directory) / "token").read_text().strip()
        key, server = token.split("@", 1)
        host, port = server.rsplit(":", 1)
        return SavedLogin(host, int(port), "", key, source="omero-user-token")
    except (OSError, ValueError):
        return None


def saved_logins() -> list[SavedLogin]:
    """The saved logins, most recent first, then an ``omero-user-token`` one."""
    ours = sorted(_read_ours(), key=lambda login: login.created, reverse=True)
    theirs = _omero_user_token()
    return ours + ([theirs] if theirs is not None else [])


def save_login(
    conn: Any,
    host: str,
    port: int | None,
    *,
    secure: bool = True,
    idle_days: float = DEFAULT_IDLE_DAYS,
) -> SavedLogin:
    """Make a token on a logged-in connection and save it for ``host``.

    Replaces an earlier token of the same server and user (and ends its session).
    """
    context = conn.getEventContext()
    session = conn.getSessionService().createUserSession(
        0, int(idle_days * _DAY_MS), context.groupName
    )
    login = SavedLogin(
        host=host,
        port=port,
        username=str(context.userName),
        key=str(session.getUuid().getValue()),
        secure=secure,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    kept = []
    for old in _read_ours():
        if (old.host, old.port, old.username) == (host, port, login.username):
            _end_session(old)
        else:
            kept.append(old)
    _write_ours([login, *kept])
    return login


def forget_login(login: SavedLogin) -> None:
    """End the token's session on the server and remove it from this computer.

    An ``omero-user-token`` token is only ended, not removed: its file belongs to
    that tool (``omero-user-token`` makes a new one).
    """
    _end_session(login)
    if login.source == "omero-annotate-ai":
        _write_ours([old for old in _read_ours() if old.key != login.key])


def _end_session(login: SavedLogin) -> None:
    """Close the token's session on the server; an expired or unreachable one is fine."""
    kwargs: dict[str, Any] = {"host": login.host, "secure": login.secure}
    if login.port is not None:
        kwargs["port"] = login.port
    conn = BlitzGateway(**kwargs)
    with contextlib.suppress(Exception):  # expired, or server unreachable
        if conn.connect(sUuid=login.key):
            conn.close(hard=True)
            return
    with contextlib.suppress(Exception):
        conn.close()
