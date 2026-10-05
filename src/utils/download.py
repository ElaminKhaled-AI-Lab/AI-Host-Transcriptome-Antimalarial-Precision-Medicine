"""Robust HTTP download helpers shared by the 01_download scripts.

Uses only the standard library plus ``requests`` if present. Supports resume,
retries, and streaming to disk so large raw files never load into memory.
"""
from __future__ import annotations

import time
import urllib.request
import urllib.error
from pathlib import Path

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def http_download(
    url: str,
    dest: str | Path,
    *,
    retries: int = 4,
    backoff: float = 3.0,
    timeout: int = 120,
    overwrite: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` with retries. Returns the local path.

    If the destination exists and ``overwrite`` is False, the download is
    skipped (raw data are immutable; re-runs should be cheap).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        log.info("exists, skipping: %s", dest.name)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("GET %s (attempt %d/%d)", url, attempt, retries)
            req = urllib.request.Request(url, headers={"User-Agent": "Malaria_Project/0.1"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                out.flush()
            # Only finalize if the temp file actually exists and has content.
            # (A dropped connection can leave no/zero-byte temp; retry instead of
            #  crashing on a rename of a missing file -- the bug that aborted the
            #  dual-RNA run near completion.)
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise OSError("empty or missing temp file after read")
            tmp.replace(dest)
            log.info("saved: %s (%d bytes)", dest.name, dest.stat().st_size)
            return dest
        # Broad catch: URLError/timeouts/connection resets AND http.client
        # IncompleteRead + any OSError from the temp file, so one bad file never
        # kills the whole download run.
        except Exception as e:  # noqa: BLE001 -- intentional: keep the run alive
            last_err = e
            log.warning("download failed: %s", e)
            try:
                if tmp.exists():
                    tmp.unlink()          # discard partial before the next attempt
            except OSError:
                pass
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_err


def check_url(url: str, timeout: int = 30) -> bool:
    """Return True if a HEAD/GET to ``url`` succeeds (used for pre-flight)."""
    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "Malaria_Project/0.1"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 400
    except Exception:
        # Some servers reject HEAD; fall back to a ranged GET of 1 byte.
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Malaria_Project/0.1", "Range": "bytes=0-0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= resp.status < 400
        except Exception:
            return False
