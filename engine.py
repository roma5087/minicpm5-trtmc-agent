"""One `trtmc run` invocation, shared by agent.py and precision_compare.py.

trtmc has no persistent/server mode, so every call is a fresh subprocess that
reloads the compiled engine from disk.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

DEFAULT_TIMEOUT_S = 120
_STDERR_TAIL_CHARS = 500


class EngineError(RuntimeError):
    """A failed trtmc invocation. The message carries the exit status and the
    tail of stderr -- never the command line, which embeds the whole prompt."""


def run_trtmc(
    binary: Path,
    bundle: Path,
    runtime_root: Path,
    prompt: str,
    max_new_tokens: int,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict:
    command = [
        str(binary),
        "run",
        str(bundle),
        "--runtime-root",
        str(runtime_root),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--use-chat-template",
        "false",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
        payload = json.loads(completed.stdout)
    except subprocess.CalledProcessError as error:
        stderr_tail = (error.stderr or "").strip()[-_STDERR_TAIL_CHARS:]
        raise EngineError(f"trtmc exited with status {error.returncode}: {stderr_tail or '(no stderr)'}") from error
    except subprocess.TimeoutExpired as error:
        raise EngineError(f"trtmc timed out after {timeout}s") from error
    except (ValueError, OSError) as error:
        # ValueError covers JSONDecodeError, UnicodeDecodeError (invalid UTF-8
        # on stdout) and a NUL byte in the prompt; OSError covers a missing
        # binary and an oversized argv.
        raise EngineError(f"{type(error).__name__}: {error}") from error
    if not isinstance(payload, dict):
        raise EngineError(f"expected a JSON object from trtmc, got {type(payload).__name__}")
    payload["_wall_ms"] = (time.perf_counter() - started) * 1000.0
    return payload
