"""Background shell task management for long-running tool execution."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .workspace import clip, now

STATUS_RUNNING = "running"
STATUS_EXITED = "exited"
STATUS_FAILED = "failed"
STATUS_STOPPED = "stopped"

STREAM_STDOUT = "stdout"
STREAM_STDERR = "stderr"
STREAM_BOTH = "both"
VALID_STREAMS = {STREAM_STDOUT, STREAM_STDERR, STREAM_BOTH}
TASK_LIST_DEFAULT_LIMIT = 20
TASK_LIST_MAX_LIMIT = 100
TASK_LIST_STATUSES = {"all", STATUS_RUNNING, STATUS_EXITED, STATUS_FAILED, STATUS_STOPPED}

_POLL_CACHE = {}


@dataclass(frozen=True)
class BackgroundTaskRecord:
    task_id: str
    run_id: str
    command: str
    cwd: str
    status: str
    pid: int
    started_at: str
    finished_at: str
    return_code: int | None
    timeout: int
    stdout_path: str
    stderr_path: str

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "run_id": self.run_id,
            "command": self.command,
            "cwd": self.cwd,
            "status": self.status,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "return_code": self.return_code,
            "timeout": self.timeout,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }


def _normalize_record(data):
    return BackgroundTaskRecord(
        task_id=str(data.get("task_id", "")).strip(),
        run_id=str(data.get("run_id", "")).strip(),
        command=str(data.get("command", "")).strip(),
        cwd=str(data.get("cwd", "")).strip(),
        status=str(data.get("status", STATUS_FAILED)).strip() or STATUS_FAILED,
        pid=int(data.get("pid", 0) or 0),
        started_at=str(data.get("started_at", "")).strip(),
        finished_at=str(data.get("finished_at", "")).strip(),
        return_code=None if data.get("return_code", None) is None else int(data.get("return_code")),
        timeout=int(data.get("timeout", 0) or 0),
        stdout_path=str(data.get("stdout_path", "")).strip(),
        stderr_path=str(data.get("stderr_path", "")).strip(),
    )


class BackgroundTaskManager:
    def __init__(self, run_store, workspace_root):
        self.run_store = run_store
        self.workspace_root = Path(workspace_root)

    def start(self, run_id, task_id, command, cwd, env, timeout):
        stdout_path = self.run_store.background_task_stdout_path(run_id, task_id)
        stderr_path = self.run_store.background_task_stderr_path(run_id, task_id)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)

        stdout_handle = stdout_path.open("w", encoding="utf-8", errors="replace")
        stderr_handle = stderr_path.open("w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            str(command),
            cwd=str(cwd),
            shell=True,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        stdout_handle.close()
        stderr_handle.close()
        _POLL_CACHE[task_id] = proc
        record = BackgroundTaskRecord(
            task_id=str(task_id),
            run_id=str(run_id),
            command=str(command),
            cwd=str(cwd),
            status=STATUS_RUNNING,
            pid=int(proc.pid),
            started_at=now(),
            finished_at="",
            return_code=None,
            timeout=int(timeout),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        self.run_store.write_background_task(record)
        return record

    def get(self, task_id):
        record = self.run_store.find_background_task(task_id)
        if record is None:
            raise ValueError(f"unknown background task: {task_id}")
        refreshed = self.refresh(record)
        return refreshed

    def refresh(self, record):
        record = _normalize_record(record.to_dict() if isinstance(record, BackgroundTaskRecord) else record)
        if record.status != STATUS_RUNNING:
            return record
        if self._timed_out(record):
            return self._mark_timeout(record)
        if self._is_running(record):
            return record
        return self._finalize(record)

    def stop(self, task_id):
        record = self.get(task_id)
        if record.status != STATUS_RUNNING:
            return record, False
        self._terminate_process(record.pid)
        finished = self._finalize(record, forced_status=STATUS_STOPPED)
        return finished, True

    def read_output(self, task_id, offset=0, limit=4000, stream=STREAM_STDOUT):
        record = self.get(task_id)
        if stream not in VALID_STREAMS:
            raise ValueError(f"stream must be one of: {', '.join(sorted(VALID_STREAMS))}")
        offset = int(offset)
        limit = int(limit)
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stdout_text = self._read_text(Path(record.stdout_path))
        stderr_text = self._read_text(Path(record.stderr_path))
        if stream == STREAM_STDOUT:
            source_text = stdout_text
        elif stream == STREAM_STDERR:
            source_text = stderr_text
        else:
            source_text = self._combine_streams(stdout_text, stderr_text)
        chunk = source_text[offset : offset + limit]
        next_offset = offset + len(chunk)
        has_more = next_offset < len(source_text)
        return {
            "record": record,
            "stream": stream,
            "offset": offset,
            "limit": limit,
            "chunk": chunk,
            "next_offset": next_offset,
            "has_more": has_more,
            "total_chars": len(source_text),
        }

    def list_tasks(self, run_id, offset=0, limit=TASK_LIST_DEFAULT_LIMIT, status="all"):
        offset = int(offset)
        limit = int(limit)
        status = str(status or "all").strip().lower() or "all"
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if limit > TASK_LIST_MAX_LIMIT:
            raise ValueError(f"limit must be <= {TASK_LIST_MAX_LIMIT}")
        if status not in TASK_LIST_STATUSES:
            raise ValueError(f"status must be one of: {', '.join(sorted(TASK_LIST_STATUSES))}")

        records = []
        for payload in self.run_store.list_background_tasks(run_id):
            record = self.refresh(payload)
            if status != "all" and record.status != status:
                continue
            records.append(record)
        records.sort(key=lambda item: item.started_at or "", reverse=True)
        visible = records[offset : offset + limit]
        next_offset = offset + len(visible)
        has_more = next_offset < len(records)
        return {
            "run_id": str(run_id),
            "status": status,
            "offset": offset,
            "limit": limit,
            "visible": visible,
            "total": len(records),
            "next_offset": next_offset,
            "has_more": has_more,
        }

    def summarize_run_tasks(self, run_id, limit=5):
        records = []
        counts = {
            STATUS_RUNNING: 0,
            STATUS_EXITED: 0,
            STATUS_FAILED: 0,
            STATUS_STOPPED: 0,
        }
        for payload in self.run_store.list_background_tasks(run_id):
            record = self.refresh(payload)
            counts[record.status] = counts.get(record.status, 0) + 1
            records.append(record)
        records.sort(key=lambda item: item.started_at or "", reverse=True)
        return {
            "run_id": str(run_id),
            "total": len(records),
            "counts": counts,
            "recent": records[: max(0, int(limit))],
        }

    @staticmethod
    def _read_text(path):
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _combine_streams(stdout_text, stderr_text):
        sections = []
        sections.append("stdout:")
        sections.append(stdout_text.rstrip() or "(empty)")
        sections.append("")
        sections.append("stderr:")
        sections.append(stderr_text.rstrip() or "(empty)")
        return "\n".join(sections).strip()

    def _timed_out(self, record):
        if int(record.timeout or 0) <= 0:
            return False
        started = self._parse_epoch(record.started_at)
        if started is None:
            return False
        return (time.time() - started) > int(record.timeout)

    @staticmethod
    def _parse_epoch(value):
        if not value:
            return None
        try:
            from datetime import datetime

            return datetime.fromisoformat(str(value)).timestamp()
        except Exception:
            return None

    def _mark_timeout(self, record):
        self._terminate_process(record.pid)
        return self._finalize(record, forced_status=STATUS_FAILED, forced_return_code=-1)

    def _is_running(self, record):
        proc = _POLL_CACHE.get(record.task_id)
        if proc is not None:
            if proc.poll() is None:
                return True
            return False
        if record.pid <= 0:
            return False
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"$p = Get-Process -Id {int(record.pid)} -ErrorAction SilentlyContinue; if ($p) {{ 'running' }}",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception:
            return False
        return "running" in result.stdout

    def _finalize(self, record, forced_status="", forced_return_code=None):
        proc = _POLL_CACHE.get(record.task_id)
        return_code = forced_return_code
        if return_code is None and proc is not None:
            return_code = proc.poll()
        if return_code is None and record.return_code is not None:
            return_code = record.return_code
        status = forced_status or (STATUS_EXITED if int(return_code or 0) == 0 else STATUS_FAILED)
        finalized = BackgroundTaskRecord(
            task_id=record.task_id,
            run_id=record.run_id,
            command=record.command,
            cwd=record.cwd,
            status=status,
            pid=record.pid,
            started_at=record.started_at,
            finished_at=record.finished_at or now(),
            return_code=None if return_code is None else int(return_code),
            timeout=record.timeout,
            stdout_path=record.stdout_path,
            stderr_path=record.stderr_path,
        )
        self.run_store.write_background_task(finalized)
        if finalized.status != STATUS_RUNNING:
            _POLL_CACHE.pop(record.task_id, None)
        return finalized

    @staticmethod
    def _terminate_process(pid):
        if pid <= 0:
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception:
            pass
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except Exception:
            pass


def format_task_start_text(record):
    summary = (
        f"Started background shell task {record.task_id} for {clip(record.command, 120)}; "
        f"status {record.status}."
    )
    lines = [
        f"task_id: {record.task_id}",
        f"status: {record.status}",
        f"pid: {record.pid}",
        f"timeout: {record.timeout}",
        f"command: {record.command}",
        f"<summary-for-history>{summary}</summary-for-history>",
    ]
    return "\n".join(lines)


def format_task_output_text(output_data):
    record = output_data["record"]
    stream = output_data["stream"]
    chunk = output_data["chunk"]
    next_offset = output_data["next_offset"]
    total_chars = output_data["total_chars"]
    has_more = bool(output_data["has_more"])
    summary = (
        f"Checked background task {record.task_id} {stream}; status {record.status}; "
        f"return_code {record.return_code if record.return_code is not None else 'None'}; "
        f"showing chars {output_data['offset']}-{max(next_offset - 1, output_data['offset'])}."
    )
    lines = [
        f"task_id: {record.task_id}",
        f"status: {record.status}",
        f"return_code: {record.return_code if record.return_code is not None else '(running)'}",
        f"stream: {stream}",
        f"offset: {output_data['offset']}",
        f"next_offset: {next_offset}",
        f"total_chars: {total_chars}",
        "output:",
        chunk.strip() or "(empty)",
    ]
    if has_more:
        lines.append(
            f"<tool_reminder>This task output is only a partial page. Continue with task_output using offset {next_offset} to read more.</tool_reminder>"
        )
    lines.append(f"<summary-for-history>{summary}</summary-for-history>")
    return "\n".join(lines)


def format_task_stop_text(record, stopped):
    action = "Stopped" if stopped else "Task already finished"
    summary = f"{action} background task {record.task_id}; status {record.status}."
    lines = [
        f"task_id: {record.task_id}",
        f"status: {record.status}",
        f"stopped: {'yes' if stopped else 'no'}",
        f"return_code: {record.return_code if record.return_code is not None else '(none)'}",
        f"<summary-for-history>{summary}</summary-for-history>",
    ]
    return "\n".join(lines)


def format_task_list_text(listing):
    run_id = str(listing["run_id"])
    status = str(listing["status"])
    total = int(listing["total"])
    offset = int(listing["offset"])
    next_offset = int(listing["next_offset"])
    visible = list(listing["visible"])
    has_more = bool(listing["has_more"])
    lines = [
        f"# task list run_id: {run_id}",
        f"# status filter: {status}",
        f"# showing {len(visible)} of {total} tasks from offset {offset}",
    ]
    if visible:
        for record in visible:
            return_code = record.return_code if record.return_code is not None else "(running)"
            command_preview = clip(record.command, 120)
            lines.append(
                f"{record.task_id} | status={record.status} | pid={record.pid} | return_code={return_code} | started_at={record.started_at} | command={command_preview}"
            )
    else:
        lines.append("(no tasks)")
    if has_more:
        lines.append(
            f"<tool_reminder>This task list is only a partial page. Continue with task_list using offset {next_offset} to view more current-run tasks.</tool_reminder>"
        )
    lines.append(
        f"<summary-for-history>Listed current-run background tasks with status filter {status}; total {total}; showing {len(visible)} from offset {offset}.</summary-for-history>"
    )
    return "\n".join(lines)
