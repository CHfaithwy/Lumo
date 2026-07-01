"""Narrow context passed from runtime into tool functions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class ToolContext:
    root: Path
    path_resolver: Callable[[str], Path]
    shell_env_provider: Callable[[], dict]
    depth: int
    max_depth: int
    spawn_delegate: Callable[[dict], str]
    background_task_starter: Callable[[dict], str]
    background_task_reader: Callable[[dict], str]
    background_task_stopper: Callable[[dict], str]
    background_task_lister: Callable[[dict], str]
    background_task_lookup: Callable[[str], dict | None]
    git_diff_artifact_writer: Callable[[str], str] | None = None

    def path(self, raw_path):
        return self.path_resolver(str(raw_path))

    def shell_env(self):
        return self.shell_env_provider()

    def start_background_task(self, args):
        return self.background_task_starter(args)

    def read_background_task(self, args):
        return self.background_task_reader(args)

    def stop_background_task(self, args):
        return self.background_task_stopper(args)

    def list_background_tasks(self, args):
        return self.background_task_lister(args)

    def find_background_task(self, task_id):
        return self.background_task_lookup(str(task_id))

    def write_git_diff_artifact(self, content):
        if self.git_diff_artifact_writer is None:
            return ""
        return str(self.git_diff_artifact_writer(str(content)))
