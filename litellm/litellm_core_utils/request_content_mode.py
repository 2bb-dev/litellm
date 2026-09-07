"""Dependency-light protected-mode switch, also usable during logger startup."""

import os
from pathlib import Path

CONFIG_ENV = "OPENORANGE_REQUEST_LOG_ENCRYPTION_CONFIG"
_MARKER_SUFFIX = ".request-log-encryption-required"


def protection_marker_path() -> Path | None:
    path = os.getenv("SPEND_LOG_DURABLE_QUEUE_PATH", "").strip()
    return Path(path + _MARKER_SUFFIX) if path else None


def encryption_enabled() -> bool:
    if os.getenv(CONFIG_ENV, "").strip():
        return True
    marker = protection_marker_path()
    if marker is None:
        return False
    try:
        marker.lstat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        # An inaccessible marker is not authorization to write plaintext.
        return True


def require_protection_marker() -> bool:
    marker = protection_marker_path()
    if marker is None:
        return False
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, b"openorange.request-log.v1\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(marker.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    except FileExistsError:
        return marker.is_file() and not marker.is_symlink()
    except OSError:
        return False
