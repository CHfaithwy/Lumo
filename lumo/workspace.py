"""工作区快照工具。

这个模块负责在 agent 按需读文件之前，先给它一份便宜的“仓库第一印象”。
这份快照刻意保持小而稳定：主要包含 Git 事实和少量白名单项目文档。
"""

import subprocess
import textwrap
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MAX_TOOL_OUTPUT = 4000
AGENT_STATE_DIR = ".lumo"
# 这些文件最可能直接影响 agent 的行动方式。
# 我们不会预加载整个仓库，只会先给模型一小份“导航包”。
MAX_PROJECT_TREE_ENTRIES = 100
IGNORED_PATH_NAMES = {
    ".git",
    AGENT_STATE_DIR,
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


def project_tree(root, max_entries=MAX_PROJECT_TREE_ENTRIES):
    root = Path(root).resolve()
    included = {root}
    current_level_dirs = [root]
    total_entries = 0
    stopped_at_depth = None

    def visible_children(path):
        try:
            entries = [
                item
                for item in path.iterdir()
                if item.name not in IGNORED_PATH_NAMES and not item.name.startswith(".tmp")
            ]
        except (OSError, PermissionError):
            return []
        entries.sort(key=lambda item: (item.is_file(), item.name.lower()))
        return entries

    depth = 0
    while current_level_dirs:
        depth += 1
        next_level = []
        level_children = []
        for directory in current_level_dirs:
            children = visible_children(directory)
            level_children.extend(children)
            next_level.extend(item for item in children if item.is_dir() and not item.is_symlink())

        if total_entries + len(level_children) > max_entries:
            stopped_at_depth = depth
            break

        total_entries += len(level_children)
        included.update(level_children)
        current_level_dirs = next_level

    def render(path, depth):
        lines = []
        for item in visible_children(path):
            if item not in included:
                continue
            suffix = "/" if item.is_dir() and not item.is_symlink() else ""
            lines.append(f"{'  ' * depth}- {item.name}{suffix}")
            if item.is_dir() and not item.is_symlink():
                lines.extend(render(item, depth + 1))
        return lines

    lines = [".", *render(root, 1)]
    if stopped_at_depth is not None:
        lines.append(
            f"...[truncated before depth {stopped_at_depth}; next level would exceed {max_entries} entries]"
        )
    return "\n".join(lines)


def indent_block(text, prefix):
    lines = str(text).splitlines() or [""]
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in lines)


class WorkspaceContext:
    def __init__(self, cwd, repo_root, branch, default_branch, status, recent_commits, project_docs):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd, repo_root_override=None):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                )
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = (
            Path(repo_root_override).resolve()
            if repo_root_override is not None
            else Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        )
        # Keep project_docs as a lightweight navigation map instead of loading file contents.
        docs = {"directory_tree": project_tree(cwd)}
        # 同时扫描 repo_root 和 cwd，这样在子目录启动时也能看到本地文档；
        # 但用相对路径做 key，避免同一份文档被重复收集。
        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=(
                lambda branch: branch[len("origin/") :] if branch.startswith("origin/") else branch
            )(git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        # 这段文本会被塞进 prompt prefix，作为相对稳定的基线上下文。
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        doc_lines = []
        for path, snippet in self.project_docs.items():
            doc_lines.append(f"  - {path}:")
            doc_lines.append(indent_block(snippet, "    "))
        docs = "\n".join(doc_lines) or "  - none"
        return "\n".join(
            [
                "Workspace:",
                f"- cwd: {self.cwd}",
                f"- repo_root: {self.repo_root}",
                f"- branch: {self.branch}",
                f"- default_branch: {self.default_branch}",
                "- status:",
                indent_block(self.status, "  "),
                "- recent_commits:",
                indent_block(commits, "  "),
                "- project_docs:",
                docs,
            ]
        ).strip()

    def fingerprint(self):
        # 这个指纹用来判断仓库状态是否发生了足够大的变化，
        # 从而决定是否需要重建缓存中的 prompt prefix。
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
