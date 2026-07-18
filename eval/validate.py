"""Static validation for the Lumo evaluation datasets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple


EVAL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVAL_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lumo.tools import legal_tool_names  # noqa: E402


CATALOG_PATH = EVAL_ROOT / "catalog.json"
REFERENCES_PATH = EVAL_ROOT / "references.json"
KNOWN_TOOLS = legal_tool_names()
REQUIRED_SCENARIOS = {"common", "boundary", "recovery"}
CHINESE_PATTERN = re.compile(r"[\u3400-\u9fff]")

FORBIDDEN_MODEL_KEYS = {
    "answer",
    "checks",
    "coverage",
    "evaluation",
    "expected",
    "oracle",
    "private_files",
    "rubric",
    "runner",
    "setup",
    "solution",
}

ALLOWED_RUNNER_KINDS = {
    "live_agent",
    "multi_session_live_agent",
    "interrupt_resume_live_agent",
    "scripted_probe",
}
ALLOWED_SESSION_POLICIES = {"new", "same", "resume"}
ALLOWED_METHODS = {"state", "unit_tests", "llm_rubric", "composite"}
ALLOWED_TRACKS = {"tool_matrix", "mechanism", "workflow"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


class ValidationSummary(NamedTuple):
    dataset_count: int
    task_count: int
    core_task_count: int
    workflow_task_count: int
    tool_matrix_task_count: int
    mechanism_task_count: int


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.relative_to(EVAL_ROOT)}: invalid JSON: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_relative_path(raw_path: str, context: str) -> str:
    require(isinstance(raw_path, str) and raw_path.strip(), f"{context}: path must be non-empty")
    require("\\" not in raw_path, f"{context}: paths must use forward slashes: {raw_path!r}")
    path = PurePosixPath(raw_path)
    require(not path.is_absolute(), f"{context}: path must be relative: {raw_path!r}")
    require(".." not in path.parts, f"{context}: path may not escape the workspace: {raw_path!r}")
    require(raw_path not in {".", ""}, f"{context}: path must name a file or directory")
    return path.as_posix()


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from walk_keys(nested)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def _contains_chinese(text: str) -> bool:
    return bool(CHINESE_PATTERN.search(str(text)))


def _term_present(text: str, term: str) -> bool:
    if term.isascii():
        return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", text, re.IGNORECASE))
    return term in text


def find_forbidden_prompt_terms(text: str, prompt_policy: dict[str, Any]) -> list[str]:
    terms = set(KNOWN_TOOLS)
    terms.update(str(item) for item in prompt_policy.get("forbidden_internal_terms", []))
    return sorted(term for term in terms if term and _term_present(text, term))


def validate_coverage(
    task: dict[str, Any],
    suite: str,
    context: str,
    mechanism_names: set[str],
) -> None:
    coverage = task.get("coverage")
    require(isinstance(coverage, dict), f"{context}: coverage must be an object")
    track = coverage.get("track")
    require(track in ALLOWED_TRACKS, f"{context}: invalid coverage track")
    if suite == "core":
        require(track in {"tool_matrix", "mechanism"}, f"{context}: core tasks must use a core coverage track")
    else:
        require(track == "workflow", f"{context}: workflow tasks must use the workflow track")

    primary = coverage.get("primary_tools")
    supporting = coverage.get("supporting_tools")
    require(isinstance(primary, list) and 2 <= len(primary) <= 3, f"{context}: primary_tools must contain 2-3 tools")
    require(len(primary) == len(set(primary)), f"{context}: primary_tools contains duplicates")
    require(isinstance(supporting, list), f"{context}: supporting_tools must be a list")
    require(len(supporting) == len(set(supporting)), f"{context}: supporting_tools contains duplicates")
    require(not (set(primary) & set(supporting)), f"{context}: primary and supporting tools must be disjoint")
    unknown = sorted((set(primary) | set(supporting)) - KNOWN_TOOLS)
    require(not unknown, f"{context}: coverage contains unknown tools: {unknown}")

    allowed = task.get("model_input", {}).get("allowed_tools")
    require(isinstance(allowed, list), f"{context}: model_input.allowed_tools must be a list")
    require(set(allowed) == set(primary) | set(supporting), f"{context}: allowed_tools must equal primary_tools + supporting_tools")

    scenario = coverage.get("scenario_class")
    require(scenario in REQUIRED_SCENARIOS, f"{context}: invalid scenario_class")
    require(coverage.get("prompt_language") == "zh-CN", f"{context}: prompt_language must be zh-CN")

    mechanisms = coverage.get("mechanisms")
    require(isinstance(mechanisms, list), f"{context}: mechanisms must be a list")
    seen_mechanisms: set[tuple[str, str]] = set()
    for index, item in enumerate(mechanisms):
        item_context = f"{context}.coverage.mechanisms[{index}]"
        require(isinstance(item, dict), f"{item_context}: mechanism must be an object")
        name = str(item.get("name", "")).strip()
        mechanism_scenario = str(item.get("scenario", "")).strip()
        require(name in mechanism_names, f"{item_context}: unknown mechanism {name!r}")
        require(mechanism_scenario in REQUIRED_SCENARIOS, f"{item_context}: invalid mechanism scenario")
        key = (name, mechanism_scenario)
        require(key not in seen_mechanisms, f"{context}: duplicate mechanism coverage {key}")
        seen_mechanisms.add(key)


def validate_model_input(task: dict[str, Any], context: str, prompt_policy: dict[str, Any]) -> None:
    model_input = task.get("model_input")
    require(isinstance(model_input, dict), f"{context}: model_input must be an object")
    leaked_keys = sorted(set(walk_keys(model_input)) & FORBIDDEN_MODEL_KEYS)
    require(not leaked_keys, f"{context}: model_input contains private-looking keys: {leaked_keys}")

    allowed_tools = model_input.get("allowed_tools")
    require(isinstance(allowed_tools, list) and allowed_tools, f"{context}: allowed_tools must be non-empty")
    require(len(allowed_tools) == len(set(allowed_tools)), f"{context}: allowed_tools contains duplicates")
    unknown_tools = sorted(set(allowed_tools) - KNOWN_TOOLS)
    require(not unknown_tools, f"{context}: unknown allowed tools: {unknown_tools}")

    turns = model_input.get("turns")
    require(isinstance(turns, list) and turns, f"{context}: model_input.turns must be non-empty")
    seen_turn_ids: set[str] = set()
    for index, turn in enumerate(turns):
        turn_context = f"{context}.model_input.turns[{index}]"
        require(isinstance(turn, dict), f"{turn_context}: turn must be an object")
        turn_id = str(turn.get("id", "")).strip()
        require(turn_id, f"{turn_context}: id must be non-empty")
        require(turn_id not in seen_turn_ids, f"{context}: duplicate turn id {turn_id!r}")
        seen_turn_ids.add(turn_id)
        require(turn.get("session_policy") in ALLOWED_SESSION_POLICIES, f"{turn_context}: invalid session_policy")
        prompt = str(turn.get("prompt", "")).strip()
        require(prompt, f"{turn_context}: prompt must be non-empty")
        require(_contains_chinese(prompt), f"{turn_context}: prompt must contain Chinese text")
        prompt_hits = find_forbidden_prompt_terms(prompt, prompt_policy)
        require(not prompt_hits, f"{turn_context}: prompt contains internal terms: {prompt_hits}")

        notes = turn.get("environment_notes", [])
        require(isinstance(notes, list), f"{turn_context}: environment_notes must be a list")
        for note_index, note in enumerate(notes):
            note = str(note)
            if not note.strip():
                continue
            require(_contains_chinese(note), f"{turn_context}.environment_notes[{note_index}]: note must contain Chinese text")
            note_hits = find_forbidden_prompt_terms(note, prompt_policy)
            require(not note_hits, f"{turn_context}.environment_notes[{note_index}]: note contains internal terms: {note_hits}")


def validate_setup(task: dict[str, Any], context: str) -> None:
    setup = task.get("setup")
    require(isinstance(setup, dict), f"{context}: setup must be an object")
    workspace_files = setup.get("workspace_files")
    private_files = setup.get("private_files")
    directories = setup.get("directories")
    actions = setup.get("actions")
    require(isinstance(workspace_files, list), f"{context}: workspace_files must be a list")
    require(isinstance(private_files, list), f"{context}: private_files must be a list")
    require(isinstance(directories, list), f"{context}: directories must be a list")
    require(isinstance(actions, list), f"{context}: actions must be a list")

    workspace_paths: set[str] = set()
    private_paths: set[str] = set()
    for kind, files, paths in (
        ("workspace_files", workspace_files, workspace_paths),
        ("private_files", private_files, private_paths),
    ):
        for index, item in enumerate(files):
            file_context = f"{context}.setup.{kind}[{index}]"
            require(isinstance(item, dict), f"{file_context}: file must be an object")
            path = validate_relative_path(item.get("path"), file_context)
            require(path not in paths, f"{context}: duplicate {kind} path {path!r}")
            paths.add(path)
            require(isinstance(item.get("content"), str), f"{file_context}: content must be a string")
    require(not (workspace_paths & private_paths), f"{context}: workspace/private file paths overlap")

    seen_directories: set[str] = set()
    for index, raw_path in enumerate(directories):
        path = validate_relative_path(raw_path, f"{context}.setup.directories[{index}]")
        require(path not in seen_directories, f"{context}: duplicate directory path {path!r}")
        seen_directories.add(path)
    for index, action in enumerate(actions):
        action_context = f"{context}.setup.actions[{index}]"
        require(isinstance(action, dict), f"{action_context}: action must be an object")
        require(str(action.get("type", "")).strip(), f"{action_context}: action type must be non-empty")
        require(isinstance(action.get("spec"), dict), f"{action_context}: action spec must be an object")


def validate_runner(task: dict[str, Any], context: str) -> None:
    runner = task.get("runner")
    require(isinstance(runner, dict), f"{context}: runner must be an object")
    require(runner.get("kind") in ALLOWED_RUNNER_KINDS, f"{context}: invalid runner kind")
    for field in ("repetitions", "max_steps", "timeout_seconds"):
        value = runner.get(field)
        require(isinstance(value, int) and value >= 1, f"{context}: runner.{field} must be >= 1")
    require(isinstance(runner.get("required_harness_features"), list), f"{context}: required_harness_features must be a list")
    if runner.get("kind") == "interrupt_resume_live_agent":
        require(isinstance(runner.get("interrupt"), dict), f"{context}: interrupt runner requires interrupt config")
        require(any(turn.get("session_policy") == "resume" for turn in task["model_input"]["turns"]), f"{context}: interrupt runner needs a resume turn")
    if runner.get("kind") == "scripted_probe":
        require(isinstance(runner.get("scripted_model"), dict), f"{context}: scripted probe requires scripted_model")


def validate_evaluation(task: dict[str, Any], context: str, suite: str | None = None, category: str | None = None) -> None:
    evaluation = task.get("evaluation")
    require(isinstance(evaluation, dict), f"{context}: evaluation must be an object")
    require(evaluation.get("method") in ALLOWED_METHODS, f"{context}: invalid evaluation method")
    require(evaluation.get("pass_policy") == "all_required", f"{context}: pass_policy must be all_required")
    checks = evaluation.get("checks")
    require(isinstance(checks, list) and checks, f"{context}: evaluation checks must be non-empty")
    check_ids: set[str] = set()
    for index, check in enumerate(checks):
        check_context = f"{context}.evaluation.checks[{index}]"
        require(isinstance(check, dict), f"{check_context}: check must be an object")
        check_id = str(check.get("id", "")).strip()
        require(check_id and check_id not in check_ids, f"{check_context}: check ID must be unique and non-empty")
        check_ids.add(check_id)
        require(str(check.get("type", "")).strip(), f"{check_context}: type must be non-empty")
        require(check.get("required") is True, f"{check_context}: every declared check must be required")
        require(isinstance(check.get("spec"), dict), f"{check_context}: spec must be an object")

    rubric = evaluation.get("rubric")
    judge = evaluation.get("judge")
    if evaluation.get("method") == "llm_rubric":
        require(isinstance(rubric, dict), f"{context}: llm_rubric method requires rubric")
        require(isinstance(judge, dict), f"{context}: llm_rubric method requires judge")
    if rubric is not None:
        criteria = rubric.get("criteria") if isinstance(rubric, dict) else None
        require(isinstance(criteria, list) and criteria, f"{context}: rubric criteria must be non-empty")
        weights = [criterion.get("weight") for criterion in criteria if isinstance(criterion, dict)]
        require(all(isinstance(weight, (int, float)) for weight in weights), f"{context}: rubric weights must be numeric")
        require(abs(sum(weights) - 100) < 1e-9, f"{context}: rubric weights must sum to 100")
    if judge is not None:
        config = EVAL_ROOT / str(judge.get("config", ""))
        require(config.is_file(), f"{context}: judge config does not exist: {judge.get('config')!r}")
    if suite == "workflows" and category == "subjective":
        prohibited = {"file_content_contains", "file_content_excludes"}
        check_types = {str(check.get("type", "")) for check in checks}
        require(not (check_types & prohibited), f"{context}: subjective tasks must use semantic rubric evidence instead of text-match gates")
        require(isinstance(rubric, dict) and isinstance(judge, dict), f"{context}: subjective tasks require a rubric judge")


def validate_dataset(
    path: Path,
    catalog_entry: dict[str, Any],
    reference_ids: set[str],
    mechanism_names: set[str],
    prompt_policy: dict[str, Any],
    seen_task_ids: set[str],
) -> list[dict[str, Any]]:
    data = load_json(path)
    rel_path = path.relative_to(EVAL_ROOT).as_posix()
    require(isinstance(data, dict), f"{rel_path}: dataset must be an object")
    require(data.get("schema_version") == "2.0", f"{rel_path}: unsupported schema_version")
    require(data.get("suite") == catalog_entry.get("suite"), f"{rel_path}: suite does not match catalog")
    require(data.get("category") == catalog_entry.get("category"), f"{rel_path}: category does not match catalog")
    require(str(data.get("description", "")).strip(), f"{rel_path}: description must be non-empty")
    tasks = data.get("tasks")
    require(isinstance(tasks, list) and tasks, f"{rel_path}: tasks must be non-empty")
    require(len(tasks) == catalog_entry.get("task_count"), f"{rel_path}: task count does not match catalog")

    suite = str(data["suite"])
    category = str(data["category"])
    for index, task in enumerate(tasks):
        context = f"{rel_path}.tasks[{index}]"
        require(isinstance(task, dict), f"{context}: task must be an object")
        task_id = str(task.get("id", "")).strip()
        expected_prefix = "core" if suite == "core" else "workflow"
        require(task_id.startswith(f"{expected_prefix}.{category}."), f"{context}: ID does not match suite/category")
        require(task_id not in seen_task_ids, f"{context}: duplicate global task ID {task_id!r}")
        seen_task_ids.add(task_id)
        require(str(task.get("title", "")).strip(), f"{context}: title must be non-empty")
        require(str(task.get("objective", "")).strip(), f"{context}: objective must be non-empty")
        require(task.get("difficulty") in ALLOWED_DIFFICULTIES, f"{context}: invalid difficulty")
        inspirations = task.get("benchmark_inspiration")
        require(isinstance(inspirations, list) and inspirations, f"{context}: benchmark_inspiration must be non-empty")
        require(not (set(inspirations) - reference_ids), f"{context}: unknown benchmark references")
        if suite == "workflows" and category == "code":
            case = task.get("case_inspiration")
            require(isinstance(case, dict), f"{context}: code workflow requires case_inspiration")
            require(str(case.get("repository", "")).strip(), f"{context}: case_inspiration.repository must be non-empty")
            require(str(case.get("pattern", "")).strip(), f"{context}: case_inspiration.pattern must be non-empty")
            require(str(case.get("url", "")).startswith("https://github.com/"), f"{context}: case_inspiration.url must be a GitHub URL")
        require(isinstance(task.get("failure_labels"), list) and task["failure_labels"], f"{context}: failure_labels must be non-empty")

        validate_model_input(task, context, prompt_policy)
        validate_coverage(task, suite, context, mechanism_names)
        validate_setup(task, context)
        validate_runner(task, context)
        validate_evaluation(task, context, suite, category)
    return tasks


def validate_core_coverage(core_tasks: list[dict[str, Any]], core_policy: dict[str, Any]) -> None:
    expected_core = int(core_policy.get("core_task_count", 0))
    tool_matrix_tasks = [task for task in core_tasks if task["coverage"]["track"] == "tool_matrix"]
    mechanism_tasks = [task for task in core_tasks if task["coverage"]["track"] == "mechanism"]
    require(len(core_tasks) == expected_core, f"core coverage: expected {expected_core} tasks, found {len(core_tasks)}")
    require(len(tool_matrix_tasks) == int(core_policy.get("tool_matrix_task_count", 0)), "core coverage: tool matrix task count mismatch")
    require(len(mechanism_tasks) == int(core_policy.get("mechanism_task_count", 0)), "core coverage: mechanism task count mismatch")
    require(all(task["runner"]["kind"] == "live_agent" for task in tool_matrix_tasks), "core coverage: tool matrix tasks must be live_agent")

    required_scenarios = set(core_policy.get("required_tool_scenarios", []))
    require(required_scenarios == REQUIRED_SCENARIOS, "core coverage: required tool scenarios must be common/boundary/recovery")
    tool_occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task in tool_matrix_tasks:
        scenario = task["coverage"]["scenario_class"]
        for tool_name in task["coverage"]["primary_tools"]:
            tool_occurrences[tool_name].append((task["id"], scenario))
    for tool_name in sorted(KNOWN_TOOLS):
        occurrences = tool_occurrences.get(tool_name, [])
        scenarios = {scenario for _, scenario in occurrences}
        require(len(occurrences) == 3, f"core coverage: {tool_name} must be primary in exactly 3 tool-matrix tasks")
        require(scenarios == REQUIRED_SCENARIOS, f"core coverage: {tool_name} must cover common/boundary/recovery")

    required_mechanisms = set(core_policy.get("required_mechanisms", []))
    mechanism_occurrences: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for task in core_tasks:
        for item in task["coverage"]["mechanisms"]:
            mechanism_occurrences[item["name"]].append((task["id"], item["scenario"]))
    require(set(mechanism_occurrences) == required_mechanisms, "core coverage: mechanism registry and task coverage differ")
    for name in sorted(required_mechanisms):
        occurrences = mechanism_occurrences[name]
        scenarios = {scenario for _, scenario in occurrences}
        task_ids = {task_id for task_id, _ in occurrences}
        require(len(task_ids) >= 3, f"core coverage: {name} needs at least 3 distinct tasks")
        require(scenarios == REQUIRED_SCENARIOS, f"core coverage: {name} must cover common/boundary/recovery")


def validate_workflow_coverage(workflow_tasks: list[dict[str, Any]], workflow_policy: dict[str, Any]) -> None:
    expected_total = int(workflow_policy.get("workflow_task_count", 0))
    require(
        len(workflow_tasks) == expected_total,
        f"workflow coverage: expected {expected_total} tasks, found {len(workflow_tasks)}",
    )
    required_categories = workflow_policy.get("required_categories")
    require(isinstance(required_categories, dict) and required_categories, "workflow coverage: required_categories must be an object")

    tasks_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in workflow_tasks:
        parts = str(task.get("id", "")).split(".")
        require(len(parts) >= 4, f"workflow coverage: malformed task ID {task.get('id')!r}")
        tasks_by_category[parts[1]].append(task)
    require(
        set(tasks_by_category) == set(required_categories),
        "workflow coverage: task categories do not match the catalog",
    )

    for category, category_policy in required_categories.items():
        require(isinstance(category_policy, dict), f"workflow coverage: {category} policy must be an object")
        tasks = tasks_by_category[category]
        expected_count = int(category_policy.get("task_count", 0))
        require(
            len(tasks) == expected_count,
            f"workflow coverage: {category} expected {expected_count} tasks, found {len(tasks)}",
        )
        expected_difficulties = category_policy.get("difficulty_counts")
        require(isinstance(expected_difficulties, dict), f"workflow coverage: {category} difficulty_counts must be an object")
        require(
            set(expected_difficulties) == ALLOWED_DIFFICULTIES,
            f"workflow coverage: {category} must declare easy/medium/hard counts",
        )
        actual_difficulties = Counter(task["difficulty"] for task in tasks)
        expected_counter = Counter({name: int(count) for name, count in expected_difficulties.items()})
        require(
            actual_difficulties == expected_counter,
            f"workflow coverage: {category} difficulty distribution mismatch; "
            f"expected {dict(expected_counter)}, found {dict(actual_difficulties)}",
        )


def validate_all() -> ValidationSummary:
    for required_path in (
        EVAL_ROOT / "schema" / "dataset.schema.json",
        EVAL_ROOT / "schema" / "judge-output.schema.json",
        EVAL_ROOT / "judge" / "rubric-judge.json",
    ):
        require(required_path.is_file(), f"missing required file: {required_path.relative_to(EVAL_ROOT)}")
        load_json(required_path)

    references = load_json(REFERENCES_PATH)
    require(references.get("schema_version") == "2.0", "references.json: unsupported schema_version")
    reference_items = references.get("references")
    require(isinstance(reference_items, list) and reference_items, "references.json: references must be non-empty")
    reference_ids = {str(item.get("id", "")).strip() for item in reference_items if isinstance(item, dict)}
    require(len(reference_ids) == len(reference_items) and all(reference_ids), "references.json: reference IDs must be unique")

    catalog = load_json(CATALOG_PATH)
    require(catalog.get("schema_version") == "2.0", "catalog.json: unsupported schema_version")
    entries = catalog.get("datasets")
    require(isinstance(entries, list) and entries, "catalog.json: datasets must be non-empty")
    core_policy = catalog.get("core_coverage")
    workflow_policy = catalog.get("workflow_coverage")
    prompt_policy = catalog.get("prompt_policy")
    require(isinstance(core_policy, dict), "catalog.json: core_coverage must be an object")
    require(isinstance(workflow_policy, dict), "catalog.json: workflow_coverage must be an object")
    require(isinstance(prompt_policy, dict), "catalog.json: prompt_policy must be an object")
    mechanism_names = set(core_policy.get("required_mechanisms", []))

    seen_paths: set[str] = set()
    seen_task_ids: set[str] = set()
    all_tasks: list[dict[str, Any]] = []
    core_tasks: list[dict[str, Any]] = []
    workflow_tasks: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        context = f"catalog.json.datasets[{index}]"
        require(isinstance(entry, dict), f"{context}: entry must be an object")
        rel_path = validate_relative_path(entry.get("path"), context)
        require(rel_path not in seen_paths, f"catalog.json: duplicate dataset path {rel_path!r}")
        seen_paths.add(rel_path)
        dataset_path = EVAL_ROOT / rel_path
        require(dataset_path.is_file(), f"{context}: dataset does not exist: {rel_path}")
        tasks = validate_dataset(dataset_path, entry, reference_ids, mechanism_names, prompt_policy, seen_task_ids)
        all_tasks.extend(tasks)
        if entry.get("suite") == "core":
            core_tasks.extend(tasks)
        else:
            workflow_tasks.extend(tasks)

    require(len(all_tasks) == catalog.get("task_count"), "catalog.json: total task_count does not match datasets")
    validate_core_coverage(core_tasks, core_policy)
    validate_workflow_coverage(workflow_tasks, workflow_policy)
    discovered = {
        path.relative_to(EVAL_ROOT).as_posix()
        for folder in (EVAL_ROOT / "core", EVAL_ROOT / "workflows")
        for path in folder.glob("*.json")
    }
    require(discovered == seen_paths, "catalog.json: dataset paths do not match discovered files")
    return ValidationSummary(
        dataset_count=len(entries),
        task_count=len(all_tasks),
        core_task_count=len(core_tasks),
        workflow_task_count=len(workflow_tasks),
        tool_matrix_task_count=sum(task["coverage"]["track"] == "tool_matrix" for task in core_tasks),
        mechanism_task_count=sum(task["coverage"]["track"] == "mechanism" for task in core_tasks),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the validation summary as JSON.")
    args = parser.parse_args(argv)
    try:
        summary = validate_all()
    except ValueError as exc:
        print(f"eval validation failed: {exc}", file=sys.stderr)
        return 1
    payload = {"status": "ok", **summary._asdict()}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"eval validation ok: {summary.task_count} tasks in {summary.dataset_count} datasets "
            f"({summary.core_task_count} core, {summary.workflow_task_count} workflows; "
            f"{summary.tool_matrix_task_count} tool-matrix, {summary.mechanism_task_count} mechanism)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
