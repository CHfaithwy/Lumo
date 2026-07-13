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
    skill_loader: Callable[[dict], str] | None = None
    skill_catalog_provider: Callable[[], object] | None = None
    todo_writer: Callable[[dict], str] | None = None
    git_diff_artifact_writer: Callable[[str], str] | None = None
    shell_command_preparer: Callable[[str], object] | None = None

    def path(self, raw_path):
        return self.path_resolver(str(raw_path))

    def shell_env(self):
        return self.shell_env_provider()

    def prepare_shell_command(self, command):
        if self.shell_command_preparer is None:
            return {
                "original_command": str(command),
                "command": str(command),
                "env": self.shell_env(),
                "python_env_used": False,
                "python_env_path": "",
                "python_executable": "",
                "environment_status": "",
            }
        prepared = self.shell_command_preparer(str(command))
        if hasattr(prepared, "__dict__"):
            return dict(prepared.__dict__)
        return dict(prepared)

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

    def load_skill(self, args):
        if self.skill_loader is None:
            return "error: skill loader is not available"
        return str(self.skill_loader(args))

    def skill_catalog(self):
        if self.skill_catalog_provider is None:
            return None
        return self.skill_catalog_provider()

    def write_todos(self, args):
        if self.todo_writer is None:
            return "error: todo writer is not available"
        return str(self.todo_writer(args))

    def write_git_diff_artifact(self, content):
        if self.git_diff_artifact_writer is None:
            return ""
        return str(self.git_diff_artifact_writer(str(content)))
