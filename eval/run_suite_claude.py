"""Run the workflow evaluation suite with Claude Code in isolated workspaces."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.run_suite import (  # noqa: E402
    AttemptState,
    build_model_client,
    collect_evaluated_artifacts,
    deterministic_check,
    failure_label,
    judge_prompt,
    load_json,
    load_tasks,
    patched_environment,
    prepare_workspace,
    sanitize_temp_paths,
    utc_now,
    write_json,
)
from eval.validate import validate_all  # noqa: E402
from lumo.config import load_project_env  # noqa: E402
from lumo.security import redact_text  # noqa: E402


RESULT_SCHEMA_VERSION = "claude-workflow-eval-v1"
JUDGE_PREFLIGHT_TIMEOUT_SECONDS = 60
EXECUTION_EVIDENCE_MAX_COMMANDS = 32
EXECUTION_EVIDENCE_MAX_OUTPUT_CHARS = 6000
EXECUTION_EVIDENCE_SECRET_PATTERN = re.compile(
    r"(?i)(?:(?:api[_ -]?key|token|secret|password)\s*(?:=|:)\s*)\S+|\bsk-[A-Za-z0-9_-]{6,}"
)
JUDGE_PREFLIGHT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ready"],
    "properties": {"ready": {"type": "boolean"}},
}


def usage_totals(items: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    totals = {key: 0 for key in keys}
    for item in items:
        for key in keys:
            totals[key] += int(item.get(key, 0) or 0)
    totals["total_tokens"] = sum(totals.values())
    return totals


def normalize_usage(value: Any) -> dict[str, int]:
    source = value if isinstance(value, dict) else {}
    return {
        key: int(source.get(key, 0) or 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }


def claude_stream_details(stdout: str) -> dict[str, Any]:
    events = []
    usage = []
    trace = []
    final_answer = ""
    cost_usd = 0.0
    final_usage = None
    models = set()
    pending_shell_events: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        if message.get("model"):
            models.add(str(message["model"]))
        event_usage = message.get("usage") if isinstance(message.get("usage"), dict) else None
        if event.get("type") == "assistant" and event_usage is not None:
            usage.append(normalize_usage(event_usage))
        for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = str(block.get("name", ""))
            tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
            if tool_name.lower() == "bash":
                shell_event = {
                    "event": "tool_executed",
                    "name": "run_shell",
                    "args": {"command": str(tool_input.get("command", ""))},
                    "tool_status": "requested",
                    "claude_tool_name": tool_name,
                }
                trace.append(shell_event)
                pending_shell_events[str(block.get("id", ""))] = shell_event
        for block in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            shell_event = pending_shell_events.get(str(block.get("tool_use_id", "")))
            if shell_event is None:
                continue
            result = event.get("tool_use_result") if isinstance(event.get("tool_use_result"), dict) else {}
            output = "\n".join(str(value) for value in (result.get("stdout"), result.get("stderr")) if value)
            if not output:
                output = str(block.get("content", ""))
            shell_event["result"] = str(output)
            shell_event["exit_code"] = 1 if bool(block.get("is_error")) else 0
            shell_event["tool_status"] = "error" if bool(block.get("is_error")) else "ok"
        if event.get("type") == "result":
            if isinstance(event.get("usage"), dict):
                final_usage = normalize_usage(event["usage"])
            if isinstance(event.get("modelUsage"), dict):
                models.update(str(name) for name in event["modelUsage"])
            final_answer = str(event.get("result", final_answer) or final_answer)
            try:
                cost_usd += float(event.get("total_cost_usd", 0) or 0)
            except (TypeError, ValueError):
                pass
    if final_usage is not None:
        usage = [final_usage]
    return {
        "event_count": len(events),
        "final_answer": final_answer,
        "trace": trace,
        "usage_records": usage,
        "usage": usage_totals(usage),
        "cost_usd": cost_usd,
        "models": sorted(models),
    }


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    else:
        process.kill()


def claude_executable() -> str:
    candidates = ("claude.cmd", "claude") if os.name == "nt" else ("claude",)
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError("Claude Code executable was not found on PATH")


def run_claude(root: Path, prompt: str, env: dict[str, str], args: argparse.Namespace, timeout: int) -> dict[str, Any]:
    command = [
        claude_executable(),
        "--print",
        "--allow-dangerously-skip-permissions",
        "--permission-mode",
        "bypassPermissions",
        "--output-format",
        "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.max_budget_usd is not None:
        command.extend(["--max-budget-usd", str(args.max_budget_usd)])
    command.append(prompt)
    started = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        creationflags=creationflags,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_tree(process)
        stdout, stderr = process.communicate()
    details = claude_stream_details(stdout)
    details.update(
        {
            "command": command[:-1] + ["<task-prompt>"],
            "return_code": process.returncode,
            "stderr": stderr,
            "stdout": stdout,
            "timed_out": timed_out,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    )
    return details


def preflight_rubric_judge(args: argparse.Namespace) -> dict[str, Any] | None:
    """Fail before the suite when a required rubric judge cannot be reached."""
    client = build_model_client(args.judge_provider, args.judge_model, JUDGE_PREFLIGHT_TIMEOUT_SECONDS)
    if args.judge_provider != "ollama" and not str(getattr(client, "api_key", "")):
        raise RuntimeError(f"rubric judge provider '{args.judge_provider}' has no API key after loading .env")
    request = {
        "instructions": "Return only JSON conforming to the supplied schema.",
        "messages": [{"role": "user", "content": "Return {\"ready\": true}."}],
    }
    response = asyncio.run(
        client.complete_structured_async(
            request,
            JUDGE_PREFLIGHT_SCHEMA,
            max_new_tokens=64,
            name="rubric_judge_preflight",
            reasoning_effort="minimal",
        )
    )
    if not isinstance(response, dict) or response.get("ready") is not True:
        raise RuntimeError("rubric judge preflight returned an invalid readiness response")
    return {
        "provider": args.judge_provider,
        "model": args.judge_model or "configured-default",
        "usage": normalize_usage(getattr(client, "last_completion_metadata", {})),
    }


def trace_matches(event: dict[str, Any], spec: dict[str, Any]) -> bool:
    if event.get("event") != spec.get("event"):
        return False
    if spec.get("tool_name") is not None and event.get("name") != spec["tool_name"]:
        return False
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    if spec.get("command_contains") is not None and spec["command_contains"] not in str(args.get("command", "")):
        return False
    if spec.get("command_equals") is not None and spec["command_equals"] != str(args.get("command", "")):
        return False
    return True


def evaluate_trace_check(check_type: str, spec: dict[str, Any], trace: list[dict[str, Any]]) -> tuple[bool, str]:
    count = sum(trace_matches(event, spec) for event in trace)
    if check_type == "trace_event_absent":
        return count == 0, f"matched {count} forbidden Claude tool events"
    expected = spec.get("count", spec.get("count_across_runs"))
    if expected is not None:
        return count == int(expected), f"matched {count}, expected {expected}"
    minimum = int(spec.get("min_count", spec.get("min_count_across_runs", 1)))
    return count >= minimum, f"matched {count}, expected at least {minimum}"


def score_judge_output(task: dict[str, Any], raw: Any) -> dict[str, Any]:
    criteria_by_id = {item["id"]: item for item in task["evaluation"]["rubric"]["criteria"]}
    scores = {}
    for item in raw.get("criteria", []) if isinstance(raw, dict) else []:
        criterion_id = str(item.get("id", ""))
        if criterion_id in criteria_by_id:
            scores[criterion_id] = max(0, min(4, int(item.get("score", 0))))
    normalized = sum((scores.get(criterion_id, 0) / 4) * criterion["weight"] for criterion_id, criterion in criteria_by_id.items())
    result = dict(raw or {})
    result["task_id"] = task["id"]
    result["normalized_score"] = round(normalized, 2)
    result["pass"] = normalized >= float(task["evaluation"]["rubric"]["threshold"])
    result["missing_criteria"] = sorted(set(criteria_by_id) - set(scores))
    return result


def redact_execution_evidence(value: Any) -> str:
    redacted = redact_text(str(value or ""), env=os.environ)
    return EXECUTION_EVIDENCE_SECRET_PATTERN.sub("[REDACTED]", redacted)


def collect_claude_execution_evidence(task: dict[str, Any], trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    judge = task["evaluation"].get("judge", {})
    if not judge.get("include_execution_evidence", False):
        return []
    evidence = []
    for event in trace:
        if event.get("event") != "tool_executed" or event.get("name") != "run_shell":
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        command = redact_execution_evidence(args.get("command", ""))
        if not command:
            continue
        evidence.append(
            {
                "command": command,
                "exit_code": event.get("exit_code") if isinstance(event.get("exit_code"), int) else None,
                "output": redact_execution_evidence(event.get("result", ""))[:EXECUTION_EVIDENCE_MAX_OUTPUT_CHARS],
            }
        )
        if len(evidence) >= EXECUTION_EVIDENCE_MAX_COMMANDS:
            break
    return evidence


def collect_claude_rubric_materials(
    state: AttemptState, trace: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    submissions = {}
    for relative in state.task["evaluation"]["judge"]["artifact_paths"]:
        path = state.root / relative
        submissions[relative] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "[missing]"
    references = {
        str(item["path"]): str(item["content"])
        for item in state.task["setup"]["workspace_files"]
    }
    return submissions, references, collect_claude_execution_evidence(state.task, trace)


def run_one_judge(
    task: dict[str, Any],
    submissions: dict[str, str],
    references: dict[str, str],
    execution_evidence: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    client = build_model_client(args.judge_provider, args.judge_model, args.timeout)
    schema = load_json(EVAL_ROOT / "schema" / "judge-output.schema.json")
    request = {
        "instructions": "Return only JSON conforming to the supplied rubric output schema.",
        "messages": [{"role": "user", "content": judge_prompt(task, submissions, references, execution_evidence, checks)}],
    }
    raw = asyncio.run(
        client.complete_structured_async(
            request,
            schema,
            max_new_tokens=4096,
            name="rubric_judge",
            reasoning_effort="medium",
        )
    )
    return score_judge_output(task, raw), normalize_usage(getattr(client, "last_completion_metadata", {}))


def run_rubric_judges(
    state: AttemptState, checks: list[dict[str, Any]], claude_trace: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, int]]:
    submissions, references, execution_evidence = collect_claude_rubric_materials(state, claude_trace)
    judges = []
    usages = []
    for _ in range(2):
        judge, usage = run_one_judge(state.task, submissions, references, execution_evidence, checks, args)
        judges.append(judge)
        usages.append(usage)
    threshold = float(state.task["evaluation"]["rubric"]["threshold"])
    if (judges[0]["normalized_score"] >= threshold) != (judges[1]["normalized_score"] >= threshold):
        judge, usage = run_one_judge(state.task, submissions, references, execution_evidence, checks, args)
        judges.append(judge)
        usages.append(usage)
    aggregate = statistics.median(item["normalized_score"] for item in judges)
    return ({"judges": judges, "normalized_score": aggregate, "pass": aggregate >= threshold, "threshold": threshold}, usage_totals(usages))


def evaluate_attempt(state: AttemptState, claude_trace: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, int]]:
    checks = []
    private_mounted = False
    rubric_check = None
    for check in state.task["evaluation"]["checks"]:
        if check["type"] == "rubric_score":
            rubric_check = check
            continue
        if check["type"] in {"trace_event", "trace_event_absent"}:
            passed, detail = evaluate_trace_check(check["type"], check["spec"], claude_trace)
        else:
            passed, detail, private_mounted = deterministic_check(state, check, private_mounted)
        checks.append({"id": check["id"], "type": check["type"], "required": check["required"], "passed": passed, "detail": detail})
    judge_usage = usage_totals([])
    rubric = None
    if rubric_check is not None:
        deterministic_pass = all(item["passed"] for item in checks if item["required"])
        if deterministic_pass:
            rubric, judge_usage = run_rubric_judges(state, checks, claude_trace, args)
            passed = bool(rubric["pass"])
            detail = f"median score={rubric['normalized_score']}, threshold={rubric['threshold']}"
        else:
            passed = False
            detail = "rubric skipped because deterministic gates failed"
        checks.append({"id": rubric_check["id"], "type": "rubric_score", "required": True, "passed": passed, "detail": detail})
    return checks, rubric, judge_usage


def run_attempt(task: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    attempt_dir = output_dir / "tasks" / task["id"] / "rep-1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    marker = attempt_dir / ".active-workspace.json"
    error = ""
    error_traceback = ""
    status = "completed"
    checks: list[dict[str, Any]] = []
    rubric = None
    claude: dict[str, Any] = {"usage": usage_totals([]), "trace": [], "final_answer": ""}
    judge_usage = usage_totals([])
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="claude-eval-") as temporary_name:
        temporary_path = Path(temporary_name)
        write_json(marker, {"primary": str(temporary_path)})
        state = AttemptState(task=task, root=temporary_path, output_dir=attempt_dir)
        try:
            prepare_workspace(state)
            env = os.environ.copy()
            env.update(state.env)
            claude = run_claude(temporary_path, task["model_input"]["turns"][0]["prompt"], env, args, int(task["runner"]["timeout_seconds"]))
            (attempt_dir / "claude-stream.jsonl").write_text(claude.pop("stdout"), encoding="utf-8")
            (attempt_dir / "claude-stderr.log").write_text(str(claude.get("stderr", "")), encoding="utf-8")
            if claude["timed_out"]:
                raise TimeoutError(f"Claude attempt exceeded configured timeout of {task['runner']['timeout_seconds']} seconds")
            if claude["return_code"] != 0:
                raise RuntimeError(f"Claude exited with code {claude['return_code']}: {claude.get('stderr', '').strip()[-1000:]}")
            with patched_environment(state.env):
                checks, rubric, judge_usage = evaluate_attempt(state, claude["trace"], args)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            status = "inconclusive"
            error = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
        artifacts = collect_evaluated_artifacts(state, attempt_dir)
        passed = status == "completed" and bool(checks) and all(item["passed"] for item in checks if item["required"])
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task_id": task["id"],
            "suite": task["suite"],
            "category": task["dataset_category"],
            "difficulty": task["difficulty"],
            "repetition": 1,
            "provider": "claude_code",
            "model": args.model or "configured-default",
            "approval_policy": "bypassPermissions",
            "started_at": utc_now(),
            "duration_seconds": round(time.monotonic() - started, 3),
            "status": "inconclusive" if status == "inconclusive" else ("passed" if passed else "failed"),
            "passed": passed,
            "error": error,
            "error_traceback": error_traceback,
            "failure_label": failure_label(task, checks, error),
            "checks": checks,
            "rubric": rubric,
            "final_answer": claude.get("final_answer", ""),
            "claude": {key: value for key, value in claude.items() if key not in {"stderr"}},
            "judge_usage": judge_usage,
            "evaluated_artifacts": artifacts,
            "temporary_workspace_deleted": True,
        }
        result = sanitize_temp_paths(result, [temporary_path])
    result["temporary_workspace_deleted"] = not temporary_path.exists()
    write_json(attempt_dir / "result.json", result)
    marker.unlink(missing_ok=True)
    return result


def aggregate(tasks: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    results = []
    task_results = {}
    for task in tasks:
        path = output_dir / "tasks" / task["id"] / "rep-1" / "result.json"
        if not path.is_file():
            continue
        result = load_json(path)
        results.append(result)
        status = result.get("status", "passed" if result.get("passed") else "failed")
        task_results[task["id"]] = {
            "first_run_pass": bool(result["passed"]),
            "first_run_status": status,
            "first_run_evaluable": status != "inconclusive",
            "completed_repetitions": 1,
        }

    def breakdown(field: str) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for task in tasks:
            result = task_results.get(task["id"])
            if result is None:
                continue
            group = groups.setdefault(str(task[field]), {"tasks": 0, "evaluable_tasks": 0, "inconclusive_tasks": 0, "passes": 0})
            group["tasks"] += 1
            if result["first_run_evaluable"]:
                group["evaluable_tasks"] += 1
                group["passes"] += int(result["first_run_pass"])
            else:
                group["inconclusive_tasks"] += 1
        for group in groups.values():
            group["success_rate"] = group["passes"] / group["evaluable_tasks"] if group["evaluable_tasks"] else 0
            group["completion_rate"] = group["evaluable_tasks"] / group["tasks"] if group["tasks"] else 0
        return groups

    agent_usage = usage_totals([item.get("claude", {}).get("usage", {}) for item in results])
    run_config_path = output_dir / "run-config.json"
    run_config = load_json(run_config_path) if run_config_path.is_file() else {}
    preflight = run_config.get("rubric_judge_preflight") or {}
    judge_preflight_usage = normalize_usage(preflight.get("usage", {}))
    judge_usage = usage_totals([item.get("judge_usage", {}) for item in results] + [judge_preflight_usage])
    reported_models = sorted({model for item in results for model in item.get("claude", {}).get("models", [])})
    evaluable_results = [item for item in results if item.get("status", "passed" if item.get("passed") else "failed") != "inconclusive"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "task_count": len(tasks),
        "attempt_count": len(results),
        "evaluable_first_run_tasks": len(evaluable_results),
        "inconclusive_first_run_tasks": len(results) - len(evaluable_results),
        "completion_rate": len(evaluable_results) / len(results) if results else 0,
        "first_run_passes": sum(bool(item["passed"]) for item in evaluable_results),
        "exact_success_rate": (sum(bool(item["passed"]) for item in evaluable_results) / len(evaluable_results)) if evaluable_results else 0,
        "agent_usage": agent_usage,
        "judge_usage": judge_usage,
        "judge_preflight_usage": judge_preflight_usage,
        "agent_cost_usd": round(sum(float(item.get("claude", {}).get("cost_usd", 0) or 0) for item in results), 6),
        "reported_models": reported_models,
        "temporary_cleanup_failures": [item["task_id"] for item in results if not item.get("temporary_workspace_deleted")],
        "by_category": breakdown("dataset_category"),
        "by_difficulty": breakdown("difficulty"),
        "task_results": task_results,
    }


def render_report(summary: dict[str, Any], tasks: list[dict[str, Any]], output_dir: Path) -> str:
    lines = [
        "# Claude Code Workflow 首轮评测报告",
        "",
        f"- 生成时间：{summary['generated_at']}",
        f"- 已产生结果：{summary['attempt_count']} / {summary['task_count']}",
        f"- 可评估任务：{summary['evaluable_first_run_tasks']}；未定结果：{summary['inconclusive_first_run_tasks']}",
        f"- 首轮正确率：{summary['first_run_passes']} / {summary['evaluable_first_run_tasks']} ({summary['exact_success_rate']:.2%})",
        f"- 完成率：{summary['completion_rate']:.2%}",
        "- 审批模式：Claude Code `--permission-mode bypassPermissions`（仅限临时工作区）",
        "- 每题在独立 `claude-eval-*` 临时工作区运行，完成后删除。",
        f"- Claude stream 报告的模型标识：{', '.join(summary['reported_models']) or '未提供'}",
        "",
        "## Token 使用量",
        "",
        "| 调用方 | 输入 | 输出 | 缓存写入 | 缓存读取 | 合计 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, usage in (("Claude Agent", summary["agent_usage"]), ("Rubric Judge", summary["judge_usage"])):
        lines.append(
            f"| {name} | {usage['input_tokens']} | {usage['output_tokens']} | {usage['cache_creation_input_tokens']} | {usage['cache_read_input_tokens']} | {usage['total_tokens']} |"
        )
    lines.extend(["", f"Claude Agent 报告的总费用：${summary['agent_cost_usd']:.6f}", "", "## 分类统计", "", "| 分类 | 任务数 | 可评估 | 未定 | 通过 | 正确率 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for group in (summary["by_category"], summary["by_difficulty"]):
        for name, value in group.items():
            lines.append(f"| {name} | {value['tasks']} | {value['evaluable_tasks']} | {value['inconclusive_tasks']} | {value['passes']} | {value['success_rate']:.2%} |")
    lines.extend(["", "## 未通过或未定任务", ""])
    failures = []
    for task in tasks:
        path = output_dir / "tasks" / task["id"] / "rep-1" / "result.json"
        if path.is_file():
            result = load_json(path)
            if not result["passed"]:
                failures.append(result)
    if not failures:
        lines.append("无。")
    else:
        for result in failures:
            failed_checks = ", ".join(item["id"] for item in result.get("checks", []) if item["required"] and not item["passed"])
            lines.append(f"- `{result['task_id']}`：{result.get('status', 'failed')}；{result.get('failure_label') or 'unclassified'}；未通过检查：{failed_checks or '-'}；错误：{result.get('error') or '-'}")
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=EVAL_ROOT / "results" / "claude-workflows-latest")
    parser.add_argument("--fresh-output", action="store_true")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--category", choices=("state", "code", "subjective"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--judge-provider", choices=("openai", "anthropic", "deepseek", "ollama"), default="openai")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--timeout", type=int, default=300, help="Judge provider request timeout in seconds.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_project_env(REPO_ROOT)
    validate_all()
    tasks = [task for task in load_tasks() if task["suite"] == "workflows"]
    if args.task:
        wanted = set(args.task)
        tasks = [task for task in tasks if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            raise ValueError(f"unknown workflow task IDs: {', '.join(sorted(missing))}")
    if args.category:
        tasks = [task for task in tasks if task["dataset_category"] == args.category]
    rubric_judge_preflight = None
    if any(check["type"] == "rubric_score" for task in tasks for check in task["evaluation"]["checks"]):
        rubric_judge_preflight = preflight_rubric_judge(args)
    output_dir = args.output.resolve()
    results_root = (EVAL_ROOT / "results").resolve()
    try:
        output_dir.relative_to(results_root)
    except ValueError as exc:
        raise ValueError("--output must be below eval/results") from exc
    if args.fresh_output:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run-config.json",
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "created_at": utc_now(),
            "agent": "claude_code",
            "model": args.model or "configured-default",
            "approval_policy": "bypassPermissions",
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model or "configured-default",
            "rubric_judge_preflight": rubric_judge_preflight,
            "selected_tasks": [task["id"] for task in tasks],
            "repetitions": 1,
        },
    )
    for index, task in enumerate(tasks, start=1):
        result_path = output_dir / "tasks" / task["id"] / "rep-1" / "result.json"
        if result_path.is_file():
            print(f"[{index}/{len(tasks)}] {task['id']}: skipped existing", flush=True)
            continue
        print(f"[{index}/{len(tasks)}] {task['id']}: running", flush=True)
        result = run_attempt(task, args, output_dir)
        print(f"[{index}/{len(tasks)}] {task['id']}: {result.get('status', 'passed' if result['passed'] else 'failed').upper()} ({result['duration_seconds']}s)", flush=True)
    summary = aggregate(tasks, output_dir)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary, tasks, output_dir), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("task_count", "attempt_count", "first_run_passes", "exact_success_rate", "agent_usage", "judge_usage")}, ensure_ascii=False))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
