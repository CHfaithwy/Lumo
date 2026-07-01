"""Read-only Git helpers for workspace inspection tools."""

from pathlib import Path
import subprocess


GIT_COMMAND_TIMEOUT = 20
GIT_STATUS_DEFAULT_LIMIT = 200
GIT_STATUS_MAX_LIMIT = 2000
GIT_DIFF_DEFAULT_LIMIT = 300
GIT_DIFF_MAX_LIMIT = 4000
GIT_DIFF_INLINE_TRANSCRIPT_MAX_LINES = 100
GIT_DIFF_MODES = {"workspace", "staged", "unstaged"}


def ensure_git_repository(workspace_root):
    workspace_root = Path(workspace_root).resolve()
    result = _run_git(workspace_root, ["rev-parse", "--show-toplevel"], check=False)
    if result.returncode != 0:
        raise RuntimeError("current workspace is not a git repository")
    return Path(result.stdout.strip()).resolve()


def git_status_text(workspace_root, target_path, offset=0, limit=GIT_STATUS_DEFAULT_LIMIT):
    repo_root = ensure_git_repository(workspace_root)
    display_path = _display_path(repo_root, target_path)
    pathspec = _pathspec(repo_root, target_path)
    result = _run_git(
        repo_root,
        ["status", "--porcelain=v1", "--branch", "--", pathspec],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(_git_error("git status", result))

    branch, entries, counts = _parse_status_output(result.stdout)
    total = len(entries)
    visible = entries[offset : offset + limit]
    shown = len(visible)
    lines = [
        f"# git status path: {display_path}",
        f"# branch: {branch}",
        f"# workspace: {'clean' if total == 0 else 'dirty'}",
        (
            f"# staged {counts['staged']}, unstaged {counts['unstaged']}, "
            f"untracked {counts['untracked']}, deleted {counts['deleted']}"
        ),
    ]
    if visible:
        lines.extend(f"{entry['code']} {entry['path']}" for entry in visible)
    elif total == 0:
        lines.append("(clean)")
    else:
        lines.append("(no changed paths at this offset)")

    if offset + shown < total:
        lines.append(
            f"<tool_reminder>This git status result is only a partial page. Continue with git_status using offset {offset + shown} to inspect more changed paths.</tool_reminder>"
        )
    summary = (
        f"Checked git status for {display_path}; branch {branch}; "
        f"staged {counts['staged']}, unstaged {counts['unstaged']}, "
        f"untracked {counts['untracked']}, deleted {counts['deleted']}; "
        f"showing {shown} of {total} changed paths from offset {offset}."
    )
    lines.append(f"<summary-for-history>{summary}</summary-for-history>")
    return "\n".join(lines)


def git_diff_text(
    workspace_root,
    target_path,
    mode="workspace",
    offset=1,
    limit=GIT_DIFF_DEFAULT_LIMIT,
    artifact_writer=None,
    inline_transcript_line_limit=GIT_DIFF_INLINE_TRANSCRIPT_MAX_LINES,
):
    repo_root = ensure_git_repository(workspace_root)
    display_path = _display_path(repo_root, target_path)
    pathspec = _pathspec(repo_root, target_path)
    tracked_patch, tracked_files = _tracked_diff(repo_root, pathspec, mode)
    untracked_patch = ""
    untracked_files = set()
    skipped_files = []
    if mode == "workspace":
        untracked_paths = _untracked_paths(repo_root, pathspec)
        untracked_files = set(untracked_paths)
        patch_parts = []
        for relative_path in untracked_paths:
            patch, skipped = _synthetic_untracked_patch(repo_root, relative_path)
            if skipped:
                skipped_files.append(relative_path)
                continue
            if patch:
                patch_parts.append(patch)
        if skipped_files:
            patch_parts.extend(
                f"# skipped non-text untracked file: {relative_path}" for relative_path in skipped_files
            )
        untracked_patch = "\n\n".join(part for part in patch_parts if part.strip())

    body_text = _join_patch_sections(tracked_patch, untracked_patch)
    body_lines = body_text.splitlines() if body_text else []
    total_lines = len(body_lines)
    visible = body_lines[offset - 1 : offset - 1 + limit] if total_lines and offset <= total_lines else []
    shown = len(visible)
    shown_start = offset if shown else 0
    shown_end = offset + shown - 1 if shown else 0
    changed_files = sorted(tracked_files | untracked_files)

    header_lines = [
        f"# git diff path: {display_path}",
        f"# mode: {mode}",
        f"# changed files: {len(changed_files)}",
    ]
    full_line_count = len(header_lines) + (len(body_lines) if body_lines else 1)
    if artifact_writer and full_line_count > int(inline_transcript_line_limit):
        artifact_text = "\n".join(
            [
                *header_lines,
                f"# total patch lines: {total_lines}",
                body_text if body_text else "(no matching changes)",
            ]
        ).strip()
        artifact_path = ""
        try:
            artifact_path = str(artifact_writer(artifact_text)).strip()
        except Exception:
            artifact_path = ""
        if artifact_path:
            lines = [
                *header_lines,
                f"# total patch lines: {total_lines}",
                f"This git diff is larger than {int(inline_transcript_line_limit)} lines and was written to {artifact_path}.",
                f"想知道文件改动请找文件：{artifact_path}",
                (
                    f"<tool_reminder>This git diff was externalized because it is too large for transcript reuse. "
                    f"If you need the full patch, use read_file on {artifact_path}.</tool_reminder>"
                ),
            ]
            summary = (
                f"Diffed {display_path} in {mode} mode; changed {len(changed_files)} files; "
                f"externalized {total_lines} patch lines to {artifact_path}."
            )
            lines.append(f"<summary-for-history>{summary}</summary-for-history>")
            return "\n".join(lines)

    lines = [
        *header_lines,
        f"# showing lines {shown_start}-{shown_end} of at least {total_lines}",
    ]
    if visible:
        lines.extend(visible)
    elif total_lines == 0:
        lines.append("(no matching changes)")
    else:
        lines.append("(no diff lines at this offset)")

    if shown and shown_end < total_lines:
        lines.append(
            f"<tool_reminder>This git diff is only a partial page. Continue with git_diff using offset {shown_end + 1} to inspect more patch lines, or narrow the path.</tool_reminder>"
        )
    summary = (
        f"Diffed {display_path} in {mode} mode; changed {len(changed_files)} files; "
        f"showing lines {shown_start}-{shown_end} of {total_lines}."
    )
    lines.append(f"<summary-for-history>{summary}</summary-for-history>")
    return "\n".join(lines)


def _run_git(repo_root, args, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=GIT_COMMAND_TIMEOUT,
    )
    if check and result.returncode != 0:
        raise RuntimeError(_git_error("git", result))
    return result


def _git_error(label, result):
    stderr = str(result.stderr or "").strip()
    stdout = str(result.stdout or "").strip()
    detail = stderr or stdout or "git command failed"
    return f"{label} failed: {detail}"


def _display_path(repo_root, target_path):
    target_path = Path(target_path).resolve()
    if target_path == repo_root:
        return "."
    return target_path.relative_to(repo_root).as_posix()


def _pathspec(repo_root, target_path):
    return _display_path(repo_root, target_path)


def _parse_status_output(output):
    branch = "(unknown)"
    entries = []
    counts = {"staged": 0, "unstaged": 0, "untracked": 0, "deleted": 0}
    for raw_line in str(output).splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("## "):
            branch = line[3:].strip()
            continue
        code = line[:2]
        path = line[3:].replace("\\", "/")
        if code == "??":
            counts["untracked"] += 1
        else:
            if code[0] != " ":
                counts["staged"] += 1
            if code[1] != " ":
                counts["unstaged"] += 1
            if "D" in code:
                counts["deleted"] += 1
        entries.append({"code": code, "path": path})
    return branch, entries, counts


def _tracked_diff(repo_root, pathspec, mode):
    has_head = _has_head(repo_root)
    if mode == "staged":
        patch = _run_git(repo_root, ["diff", "--cached", "--no-color", "--no-ext-diff", "--", pathspec]).stdout
        files = _name_only(repo_root, ["diff", "--cached", "--name-only", "--", pathspec])
        return patch.strip(), files
    if mode == "unstaged":
        patch = _run_git(repo_root, ["diff", "--no-color", "--no-ext-diff", "--", pathspec]).stdout
        files = _name_only(repo_root, ["diff", "--name-only", "--", pathspec])
        return patch.strip(), files
    if has_head:
        patch = _run_git(repo_root, ["diff", "HEAD", "--no-color", "--no-ext-diff", "--", pathspec]).stdout
        files = _name_only(repo_root, ["diff", "HEAD", "--name-only", "--", pathspec])
        return patch.strip(), files
    patch = _join_patch_sections(
        _run_git(repo_root, ["diff", "--cached", "--no-color", "--no-ext-diff", "--", pathspec]).stdout.strip(),
        _run_git(repo_root, ["diff", "--no-color", "--no-ext-diff", "--", pathspec]).stdout.strip(),
    )
    files = _name_only(repo_root, ["diff", "--cached", "--name-only", "--", pathspec]) | _name_only(
        repo_root, ["diff", "--name-only", "--", pathspec]
    )
    return patch.strip(), files


def _has_head(repo_root):
    result = _run_git(repo_root, ["rev-parse", "--verify", "HEAD"], check=False)
    return result.returncode == 0


def _name_only(repo_root, args):
    output = _run_git(repo_root, args).stdout
    return {line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()}


def _untracked_paths(repo_root, pathspec):
    output = _run_git(repo_root, ["ls-files", "--others", "--exclude-standard", "--", pathspec]).stdout
    return [line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()]


def _synthetic_untracked_patch(repo_root, relative_path):
    file_path = repo_root / Path(relative_path)
    if not file_path.is_file():
        return "", False
    raw = file_path.read_bytes()
    if b"\x00" in raw:
        return "", True
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "", True

    lines = [
        f"diff --git a/{relative_path} b/{relative_path}",
        "new file mode 100644",
        "--- /dev/null",
        f"+++ b/{relative_path}",
    ]
    content_lines = text.splitlines()
    if content_lines:
        lines.append(f"@@ -0,0 +1,{len(content_lines)} @@")
        lines.extend(f"+{line}" for line in content_lines)
        if not text.endswith("\n"):
            lines.append(r"\ No newline at end of file")
    return "\n".join(lines), False


def _join_patch_sections(*parts):
    rendered = [str(part).strip() for part in parts if str(part).strip()]
    return "\n\n".join(rendered)
