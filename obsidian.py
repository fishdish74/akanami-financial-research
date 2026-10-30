"""Safe append-only filesystem writes into the Obsidian vault."""

from __future__ import annotations

import errno
import os
import tempfile
from pathlib import Path

from config import writes_enabled


def write_new_note_atomic(destination: Path, content: str, *, allowed_root: Path) -> None:
    """Create a new Markdown note atomically, guarded at the final write boundary."""
    if not writes_enabled():
        raise PermissionError("Research-file writes are disabled")
    if destination.suffix.casefold() != ".md":
        raise ValueError("Research notes must use the .md extension")
    if "\x00" in content:
        raise ValueError("Note content contains a NUL byte")
    root = allowed_root.resolve()
    resolved_parent = destination.parent.resolve()
    if not resolved_parent.is_relative_to(root):
        raise ValueError("Research note destination is outside the allowed vault root")
    if destination.is_symlink():
        raise ValueError("Research note destination cannot be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing note: {destination}")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A hard link is an atomic create-if-absent operation. Unlike os.replace,
            # it remains no-overwrite safe if another process wins a race.
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing note: {destination.name}"
            ) from exc
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
            # Some filesystems do not support hard links. O_EXCL retains the
            # create-if-absent guarantee without broadening overwrite authority.
            destination_fd = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
            try:
                with os.fdopen(destination_fd, "w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                # The path was created by this call; remove only an incomplete
                # fallback file, never a pre-existing destination.
                destination.unlink(missing_ok=True)
                raise
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
