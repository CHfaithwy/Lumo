

import json
import tempfile
from pathlib import Path

from .tool_output import PersistedToolOutput, TOOL_RESULT_ARTIFACT_MAX_BYTES, safe_artifact_component, truncate_utf8_to_bytes


def _run_id(value):
    if hasattr(value, "run_id"):
        return value.run_id
    return str(value)


class RunStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id):
        return self.root / _run_id(run_id)

    def task_state_path(self, run_id):
        return self.run_dir(run_id) / "task_state.json"

    def trace_path(self, run_id):
        return self.run_dir(run_id) / "trace.jsonl"

    def report_path(self, run_id):
        return self.run_dir(run_id) / "report.json"

    def request_path(self, run_id, index):
        return self.run_dir(run_id) / f"request{int(index)}.json"

    def response_path(self, run_id, index):
        return self.run_dir(run_id) / f"response{int(index)}.json"

    def auxiliary_request_path(self, run_id, kind, index):
        return self.run_dir(run_id) / f"aux_{str(kind)}_{int(index)}_request.json"

    def auxiliary_response_path(self, run_id, kind, index):
        return self.run_dir(run_id) / f"aux_{str(kind)}_{int(index)}_response.json"

    def task_dir(self, run_id):
        return self.run_dir(run_id) / "tasks"

    def latest_git_diff_path(self, run_id):
        return self.run_dir(run_id) / "latest_git_diff.patch"

    def shell_repair_log_path(self, run_id, index):
        return self.run_dir(run_id) / f"shell_repair_{int(index)}.log"

    def tool_results_dir(self, run_id):
        return self.run_dir(run_id) / "tool-results"

    def tool_result_path(self, run_id, call_id, suffix=".txt"):
        safe_call_id = safe_artifact_component(call_id)
        safe_suffix = ".json" if str(suffix) == ".json" else ".txt"
        return self.tool_results_dir(run_id) / f"{safe_call_id}{safe_suffix}"

    def background_task_meta_path(self, run_id, task_id):
        return self.task_dir(run_id) / f"{str(task_id)}.json"

    def background_task_stdout_path(self, run_id, task_id):
        return self.task_dir(run_id) / f"{str(task_id)}.stdout.log"

    def background_task_stderr_path(self, run_id, task_id):
        return self.task_dir(run_id) / f"{str(task_id)}.stderr.log"

    def start_run(self, task_state):


        run_dir = self.run_dir(task_state)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.write_task_state(task_state)
        return run_dir

    def write_task_state(self, task_state):
        path = self.task_state_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, task_state.to_dict())
        return path

    def append_trace(self, task_state, event):
        path = self.trace_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)


        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
        return path

    def write_report(self, task_state, report):
        path = self.report_path(task_state)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, report)
        return path

    def write_request(self, task_state, index, request):
        path = self.request_path(task_state, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, dict(request or {}))
        return path

    def write_response(self, task_state, index, response):
        path = self.response_path(task_state, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(path, dict(response or {}))
        return path

    def write_auxiliary_exchange(self, task_state, kind, index, request, response):
        request_path = self.auxiliary_request_path(task_state, kind, index)
        response_path = self.auxiliary_response_path(task_state, kind, index)
        request_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_json_atomic(request_path, dict(request or {}))
        self._write_json_atomic(response_path, dict(response or {}))
        return request_path, response_path

    def write_latest_git_diff(self, run_id, content):
        path = self.latest_git_diff_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
        return path

    def write_shell_repair_log(self, run_id, index, content):
        path = self.shell_repair_log_path(run_id, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
        return path

    def write_tool_result(self, run_id, call_id, content, suffix=".txt"):
        path = self.tool_result_path(run_id, call_id, suffix=suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = str(content or "")
        payload, artifact_truncated = truncate_utf8_to_bytes(text, TOOL_RESULT_ARTIFACT_MAX_BYTES)
        if not path.exists():
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=str(path.parent),
                prefix=path.name + ".",
                suffix=".tmp",
            ) as handle:
                handle.write(payload)
                temp_name = handle.name
            Path(temp_name).replace(path)
        stored_bytes = path.stat().st_size
        return PersistedToolOutput(
            relative_path=str(path),
            original_chars=len(text),
            original_bytes=len(text.encode("utf-8")),
            stored_bytes=stored_bytes,
            artifact_truncated=artifact_truncated,
        )

    def write_captured_tool_result(self, run_id, call_id, capture):
        path = self.tool_result_path(run_id, call_id, suffix=".txt")
        return capture.persist(path, TOOL_RESULT_ARTIFACT_MAX_BYTES)

    def load_task_state(self, task_id):
        return json.loads(self.task_state_path(task_id).read_text(encoding="utf-8"))

    def load_report(self, task_id):
        return json.loads(self.report_path(task_id).read_text(encoding="utf-8"))

    def write_background_task(self, record):
        run_id = record.run_id if hasattr(record, "run_id") else record["run_id"]
        path = self.background_task_meta_path(run_id, record.task_id if hasattr(record, "task_id") else record["task_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        self._write_json_atomic(path, payload)
        return path

    def load_background_task(self, run_id, task_id):
        path = self.background_task_meta_path(run_id, task_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def find_background_task(self, task_id):
        task_id = str(task_id).strip()
        if not task_id:
            return None
        candidates = sorted(self.root.glob(f"*/tasks/{task_id}.json"), reverse=True)
        if not candidates:
            return None
        return json.loads(candidates[0].read_text(encoding="utf-8"))

    def list_background_tasks(self, run_id):
        task_dir = self.task_dir(run_id)
        if not task_dir.is_dir():
            return []
        records = []
        for path in sorted(task_dir.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return records

    def _write_json_atomic(self, path, payload):


        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            temp_name = handle.name
        Path(temp_name).replace(path)
