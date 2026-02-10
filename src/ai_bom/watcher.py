"""Watch mode for AI-BOM — re-scan on file changes.

Uses the watchdog library to monitor filesystem events and
trigger incremental re-scans when source files change.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WATCH_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".tf",
    ".ipynb",
}


def watch_and_scan(
    target: Path,
    callback: Any,
    debounce_seconds: float = 2.0,
) -> None:
    """Watch a directory and trigger callback on changes.

    Args:
        target: Directory to watch.
        callback: Function to call on changes. Receives list of changed paths.
        debounce_seconds: Minimum interval between callbacks.
    """
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        raise ImportError(
            "Watch mode requires the watchdog library. Install with: pip install ai-bom[watch]"
        ) from None

    class _ChangeHandler(FileSystemEventHandler):
        def __init__(self) -> None:
            self.changed_files: list[str] = []
            self.last_trigger = 0.0

        def on_modified(self, event: Any) -> None:
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix in WATCH_EXTENSIONS:
                self.changed_files.append(str(path))

        def on_created(self, event: Any) -> None:
            self.on_modified(event)

        def on_deleted(self, event: Any) -> None:
            if not event.is_directory:
                path = Path(event.src_path)
                if path.suffix in WATCH_EXTENSIONS:
                    self.changed_files.append(str(path))

    handler = _ChangeHandler()
    observer = Observer()
    observer.schedule(handler, str(target), recursive=True)
    observer.start()

    logger.info("Watching %s for changes (Ctrl+C to stop)...", target)

    try:
        while True:
            time.sleep(0.5)
            now = time.time()
            if handler.changed_files and (now - handler.last_trigger) >= debounce_seconds:
                changed = list(set(handler.changed_files))
                handler.changed_files.clear()
                handler.last_trigger = now
                logger.info("Detected %d changed file(s), re-scanning...", len(changed))
                try:
                    callback(changed)
                except Exception as e:
                    logger.error("Re-scan failed: %s", e)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
