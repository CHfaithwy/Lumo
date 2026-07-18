

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


EXPLICIT_ENV_PATTERN = re.compile(
    r"(?i)(?:^|[;&|]\s*)(?:uv|poetry|conda|pipenv|hatch)\s+(?:run|exec)|"
    r"(?:^|[;&|]\s*)(?:tox|nox)(?:\s|$)|"
    r"(?:^|[;&|]\s*)[^;&|\s]*(?:\.venv|venv)[/\\](?:Scripts|bin)[/\\]python(?:\.exe)?(?:\s|$)"
)
PYTHON_COMMAND_PATTERN = re.compile(
    r"(?i)(?P<prefix>^|&&|\|\||;)\s*(?P<exe>python(?:3(?:\.\d+)?)?(?:\.exe)?|py(?:\.exe)?)(?=\s|$)"
)
PYTHON_SCRIPT_PATTERN = re.compile(
    r"(?i)(?P<prefix>^|&&|\|\||;)\s*(?P<script>(?:\"[^\"]+\.py\"|'[^']+\.py'|[^\s;&|]+\.py))(?=\s|$)"
)
SHELL_OPERATOR_PATTERN = re.compile(r"(?:&&|\|\||[;|]|`|\$\()")
PACKAGE_REQUIREMENT_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?(?:[<>=!~]{1,2}[A-Za-z0-9*+_.-]+(?:,[<>=!~]{1,2}[A-Za-z0-9*+_.-]+)*)?$"
)
ALLOWED_PIP_FLAGS = {"--upgrade", "--pre", "--only-binary=:all:"}


class PythonEnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedCommand:
    original_command: str
    command: str
    env: dict
    python_env_used: bool
    python_env_path: str = ""
    python_executable: str = ""
    environment_status: str = ""


class PythonEnvironmentManager:
    def __init__(self, workspace_root):
        self.root = Path(workspace_root).resolve()
        self.env_dir = self.root / ".lumo" / "python-env"
        self.manifest_path = self.env_dir / "lumo-environment.json"
        self._lock = threading.Lock()
        self.created_count = 0
        self.reused_count = 0

    @property
    def python_executable(self):
        if os.name == "nt":
            return self.env_dir / "Scripts" / "python.exe"
        return self.env_dir / "bin" / "python"

    @staticmethod
    def is_managed_python_command(command):
        text = str(command or "").strip()
        if not text or EXPLICIT_ENV_PATTERN.search(text):
            return False
        return bool(PYTHON_COMMAND_PATTERN.search(text) or PYTHON_SCRIPT_PATTERN.search(text))

    def prepare(self, command, base_env):
        original = str(command or "").strip()
        env = dict(base_env or {})
        if not self.is_managed_python_command(original):
            return PreparedCommand(original, original, env, False)
        status = self.ensure()
        executable = str(self.python_executable)
        rewritten = self._rewrite_python_command(original, executable)
        env.pop("PYTHONHOME", None)
        env.update(self.environment_variables(env))
        return PreparedCommand(
            original_command=original,
            command=rewritten,
            env=env,
            python_env_used=True,
            python_env_path=self.relative_env_path(),
            python_executable=executable,
            environment_status=status,
        )

    def ensure(self):
        with self._lock:
            if self._is_valid():
                self.reused_count += 1
                return "reused"
            self._remove_owned_environment()
            self.env_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "venv", str(self.env_dir)],
                    cwd=str(self.root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                )
            except Exception as exc:
                raise PythonEnvironmentError(f"could not create Python environment: {exc}") from exc
            if result.returncode != 0 or not self.python_executable.is_file():
                detail = (result.stderr or result.stdout or "unknown venv error").strip()
                raise PythonEnvironmentError(f"could not create Python environment: {detail}")
            self._write_manifest()
            self.created_count += 1
            return "created"

    def environment_variables(self, base_env=None):
        base_env = dict(base_env or {})
        bin_dir = self.python_executable.parent
        existing_path = str(base_env.get("PATH", os.environ.get("PATH", "")))
        result = {
            "VIRTUAL_ENV": str(self.env_dir),
            "PATH": str(bin_dir) + (os.pathsep + existing_path if existing_path else ""),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        if os.name == "nt" and os.environ.get("SYSTEMROOT"):
            result["SYSTEMROOT"] = str(base_env.get("SYSTEMROOT") or os.environ["SYSTEMROOT"])
        return result

    def validate_repair_command(self, command):
        text = str(command or "").strip()
        if not text or SHELL_OPERATOR_PATTERN.search(text):
            raise ValueError("repair command contains unsupported shell syntax")
        try:
            tokens = shlex.split(text, posix=os.name != "nt")
        except ValueError as exc:
            raise ValueError(f"repair command could not be parsed: {exc}") from exc
        tokens = [self._strip_quotes(token) for token in tokens]
        if len(tokens) >= 3 and self._is_python_token(tokens[0]) and tokens[1:3] == ["-m", "ensurepip"]:
            if tokens[3:] not in (["--upgrade"], []):
                raise ValueError("ensurepip repair only supports --upgrade")
            return [str(self.python_executable), "-m", "ensurepip", *tokens[3:]]
        if len(tokens) < 5 or not self._is_python_token(tokens[0]) or tokens[1:4] != ["-m", "pip", "install"]:
            raise ValueError("repair command must use python -m pip install or python -m ensurepip")
        requirements = []
        for token in tokens[4:]:
            if token.startswith("-"):
                if token not in ALLOWED_PIP_FLAGS:
                    raise ValueError(f"pip flag is not allowed for automatic repair: {token}")
                continue
            if not PACKAGE_REQUIREMENT_PATTERN.fullmatch(token):
                raise ValueError(f"package requirement is not allowed for automatic repair: {token}")
            requirements.append(token)
        if not requirements:
            raise ValueError("repair command must include at least one named package")
        return [str(self.python_executable), "-m", "pip", "install", *tokens[4:]]

    def run_repair(self, command, base_env, timeout=300):
        argv = self.validate_repair_command(command)
        env = dict(base_env or {})
        env.pop("PYTHONHOME", None)
        env.update(self.environment_variables(env))
        return subprocess.run(
            argv,
            cwd=str(self.root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(timeout),
        ), argv

    def relative_env_path(self):
        return self.env_dir.relative_to(self.root).as_posix()

    def summary(self):
        return {
            "path": self.relative_env_path(),
            "python_executable": str(self.python_executable),
            "exists": self._is_valid(),
            "created_count": self.created_count,
            "reused_count": self.reused_count,
        }

    def _is_valid(self):
        if not self.python_executable.is_file() or not self.manifest_path.is_file():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return (
            str(manifest.get("base_executable", "")) == str(Path(sys.executable).resolve())
            and str(manifest.get("python_version", "")) == self._python_version()
        )

    def _write_manifest(self):
        payload = {
            "base_executable": str(Path(sys.executable).resolve()),
            "python_version": self._python_version(),
            "platform": platform.platform(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _remove_owned_environment(self):
        if not self.env_dir.exists():
            return
        try:
            self.env_dir.resolve().relative_to((self.root / ".lumo").resolve())
        except ValueError as exc:
            raise PythonEnvironmentError("refusing to remove Python environment outside .lumo") from exc
        shutil.rmtree(self.env_dir)

    @staticmethod
    def _python_version():
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    @staticmethod
    def _strip_quotes(value):
        text = str(value)
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            return text[1:-1]
        return text

    @staticmethod
    def _is_python_token(value):
        return bool(re.fullmatch(r"(?i)(?:python(?:3(?:\.\d+)?)?(?:\.exe)?|py(?:\.exe)?)", str(value)))

    @staticmethod
    def _quoted_executable(path):
        if os.name == "nt":
            return subprocess.list2cmdline([str(path)])
        return shlex.quote(str(path))

    def _rewrite_python_command(self, command, executable):
        quoted = self._quoted_executable(executable)

        def replace_python(match):
            return f"{match.group('prefix')} {quoted}".lstrip() if not match.group("prefix") else f"{match.group('prefix')} {quoted}"

        rewritten = PYTHON_COMMAND_PATTERN.sub(replace_python, command)

        def replace_script(match):
            prefix = match.group("prefix")
            return f"{prefix} {quoted} {match.group('script')}".lstrip() if not prefix else f"{prefix} {quoted} {match.group('script')}"

        return PYTHON_SCRIPT_PATTERN.sub(replace_script, rewritten)
