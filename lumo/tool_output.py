

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile


TOOL_RESULT_INLINE_LIMIT_CHARS = 30_000
TOOL_RESULT_PREVIEW_LIMIT_CHARS = 2_000
TOOL_RESULT_BATCH_LIMIT_CHARS = 200_000
TOOL_RESULT_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024

_SAFE_ARTIFACT_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def safe_artifact_component(value: str, fallback: str = "tool-result") -> str:
    normalized = _SAFE_ARTIFACT_COMPONENT.sub("-", str(value or "").strip()).strip(".-")
    return normalized or fallback


def preview_text(content: str, limit: int = TOOL_RESULT_PREVIEW_LIMIT_CHARS) -> str:
    content = str(content or "")
    if len(content) <= int(limit):
        return content
    prefix = content[: int(limit)]
    last_newline = prefix.rfind("\n")
    if last_newline > int(limit) // 2:
        return prefix[:last_newline]
    return prefix


@dataclass(frozen=True)
class PersistedToolOutput:
    relative_path: str
    original_chars: int
    original_bytes: int
    stored_bytes: int
    artifact_truncated: bool


def build_externalized_output_message(persisted: PersistedToolOutput, preview: str) -> str:
    lines = [
        f"Output exceeded Lumo's inline limit ({persisted.original_chars} chars).",
        f"Full output saved to: {persisted.relative_path}",
    ]
    if persisted.artifact_truncated:
        lines.append(
            f"The artifact is capped at {persisted.stored_bytes} bytes; the remaining output was not saved."
        )
    lines.extend(
        [
            "",
            f"Preview (first {len(preview)} chars):",
            preview or "(empty)",
            "",
            "Use read_file with this path and offset/limit to inspect more. "
            "Do not rerun the originating tool solely to recover omitted output.",
        ]
    )
    return "\n".join(lines)


def truncate_utf8_to_bytes(content: str, limit: int = TOOL_RESULT_ARTIFACT_MAX_BYTES) -> tuple[bytes, bool]:
    encoded = str(content or "").encode("utf-8")
    if len(encoded) <= int(limit):
        return encoded, False
    clipped = encoded[: int(limit)]
    return clipped.decode("utf-8", errors="ignore").encode("utf-8"), True


def remove_file(path: str | Path | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        return


def _count_text_chars(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), ""):
            total += len(chunk)
    return total


def _read_text_prefix(path: Path, limit: int) -> str:
    if limit <= 0:
        return ""
    chunks = []
    remaining = int(limit)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while remaining > 0:
            chunk = handle.read(min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    return "".join(chunks)


@dataclass(frozen=True)
class ShellOutputCapture:
    stdout_path: Path
    stderr_path: Path
    environment_lines: tuple[str, ...]
    returncode: int
    timed_out: bool = False

    def _header(self) -> str:
        return "\n".join([*self.environment_lines, f"exit_code: {self.returncode}"])

    def total_chars(self) -> int:
        return len(self._header()) + len("\nstdout:\n\nstderr:\n") + _count_text_chars(self.stdout_path) + _count_text_chars(self.stderr_path)

    def total_bytes(self) -> int:
        return len(self._header().encode("utf-8")) + len(b"\nstdout:\n\nstderr:\n") + self.stdout_path.stat().st_size + self.stderr_path.stat().st_size

    def diagnostic_text(self, limit: int = TOOL_RESULT_INLINE_LIMIT_CHARS) -> str:
        header = self._header()
        remaining = max(0, int(limit) - len(header) - len("\nstdout:\n\nstderr:\n"))
        stdout_budget = remaining // 2
        stderr_budget = remaining - stdout_budget
        stdout = _read_text_prefix(self.stdout_path, stdout_budget).strip() or "(empty)"
        stderr = _read_text_prefix(self.stderr_path, stderr_budget).strip() or "(empty)"
        if self.timed_out:
            stderr = f"{stderr}\ncommand timed out"
        return "\n".join([header, "stdout:", stdout, "stderr:", stderr])

    def full_text(self) -> str:
        stdout = self.stdout_path.read_text(encoding="utf-8", errors="replace").strip() or "(empty)"
        stderr = self.stderr_path.read_text(encoding="utf-8", errors="replace").strip() or "(empty)"
        if self.timed_out:
            stderr = f"{stderr}\ncommand timed out"
        return "\n".join([self._header(), "stdout:", stdout, "stderr:", stderr])

    def persist(self, destination: Path, limit: int = TOOL_RESULT_ARTIFACT_MAX_BYTES) -> PersistedToolOutput:
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_chars = self.total_chars()
        original_bytes = self.total_bytes()
        if not destination.exists():
            written = 0
            truncated = False
            with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(destination.parent), prefix=destination.name + ".", suffix=".tmp") as handle:
                for value in (self._header() + "\nstdout:\n",):
                    chunk = value.encode("utf-8")
                    remaining = int(limit) - written
                    if remaining <= 0:
                        truncated = True
                        break
                    handle.write(chunk[:remaining])
                    written += min(len(chunk), remaining)
                    truncated = truncated or len(chunk) > remaining
                if not truncated:
                    for source in (self.stdout_path, self.stderr_path):
                        if source == self.stderr_path:
                            marker = b"\nstderr:\n"
                            remaining = int(limit) - written
                            if remaining <= 0:
                                truncated = True
                                break
                            handle.write(marker[:remaining])
                            written += min(len(marker), remaining)
                            if len(marker) > remaining:
                                truncated = True
                                break
                        with source.open("rb") as input_handle:
                            while True:
                                remaining = int(limit) - written
                                if remaining <= 0:
                                    truncated = True
                                    break
                                chunk = input_handle.read(min(64 * 1024, remaining))
                                if not chunk:
                                    break
                                handle.write(chunk)
                                written += len(chunk)
                            if truncated:
                                break
                temp_name = handle.name
            Path(temp_name).replace(destination)
        stored_bytes = destination.stat().st_size
        return PersistedToolOutput(
            relative_path=str(destination),
            original_chars=original_chars,
            original_bytes=original_bytes,
            stored_bytes=stored_bytes,
            artifact_truncated=stored_bytes < original_bytes,
        )

    def cleanup(self) -> None:
        remove_file(self.stdout_path)
        remove_file(self.stderr_path)
