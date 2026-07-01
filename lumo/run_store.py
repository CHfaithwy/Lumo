"""运行工件落盘。

session.json 负责保存“可恢复的会话状态”；RunStore 负责保存“单次运行的审计工件”，
例如 task_state、trace 和 report。两者分开后，恢复现场和复盘证据不会混在一起。
"""

import json
import tempfile
from pathlib import Path


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

    def prompt_path(self, run_id, index):
        return self.run_dir(run_id) / f"prompt{int(index)}.md"

    def task_dir(self, run_id):
        return self.run_dir(run_id) / "tasks"

    def latest_git_diff_path(self, run_id):
        return self.run_dir(run_id) / "latest_git_diff.patch"

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

    def write_prompt(self, task_state, index, prompt):
        path = self.prompt_path(task_state, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(prompt), encoding="utf-8")
        return path

    def write_latest_git_diff(self, run_id, content):
        path = self.latest_git_diff_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
        return path

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
