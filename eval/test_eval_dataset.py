import copy
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "eval" / "validate.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("lumo_eval_validator", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_catalog_and_tasks(validator):
    catalog = validator.load_json(validator.CATALOG_PATH)
    tasks = []
    for entry in catalog["datasets"]:
        data = validator.load_json(validator.EVAL_ROOT / entry["path"])
        tasks.extend(data["tasks"])
    return catalog, tasks


def test_eval_dataset_is_structurally_valid():
    validator = load_validator_module()
    summary = validator.validate_all()

    assert summary.dataset_count == 9
    assert summary.task_count == 78
    assert summary.core_task_count == 33
    assert summary.workflow_task_count == 45
    assert summary.tool_matrix_task_count == 18
    assert summary.mechanism_task_count == 15


def test_every_dataset_matches_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    validator = load_validator_module()
    schema = validator.load_json(validator.EVAL_ROOT / "schema" / "dataset.schema.json")
    catalog = validator.load_json(validator.CATALOG_PATH)

    for entry in catalog["datasets"]:
        dataset = validator.load_json(validator.EVAL_ROOT / entry["path"])
        jsonschema.Draft202012Validator(schema).validate(dataset)


def test_eval_catalog_covers_every_dataset_file():
    validator = load_validator_module()
    catalog = validator.load_json(validator.CATALOG_PATH)
    catalog_paths = {entry["path"] for entry in catalog["datasets"]}
    discovered_paths = {
        path.relative_to(validator.EVAL_ROOT).as_posix()
        for folder in (validator.EVAL_ROOT / "core", validator.EVAL_ROOT / "workflows")
        for path in folder.glob("*.json")
    }

    assert catalog_paths == discovered_paths


@pytest.mark.parametrize("primary_count", [1, 4])
def test_primary_tool_count_outside_two_to_three_is_rejected(primary_count):
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    task = copy.deepcopy(tasks[0])
    candidates = sorted(validator.KNOWN_TOOLS)
    task["coverage"]["primary_tools"] = candidates[:primary_count]
    task["coverage"]["supporting_tools"] = []
    task["model_input"]["allowed_tools"] = candidates[:primary_count]

    with pytest.raises(ValueError, match="primary_tools must contain 2-3 tools"):
        validator.validate_coverage(
            task,
            "core",
            "task",
            set(catalog["core_coverage"]["required_mechanisms"]),
        )


def test_primary_tool_missing_from_allowed_tools_is_rejected():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    task = copy.deepcopy(tasks[0])
    task["model_input"]["allowed_tools"].remove(task["coverage"]["primary_tools"][0])

    with pytest.raises(ValueError, match="allowed_tools must equal"):
        validator.validate_coverage(
            task,
            "core",
            "task",
            set(catalog["core_coverage"]["required_mechanisms"]),
        )


def test_english_only_prompt_is_rejected():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    task = copy.deepcopy(tasks[0])
    task["model_input"]["turns"][0]["prompt"] = "Inspect the repository and report the result."

    with pytest.raises(ValueError, match="must contain Chinese text"):
        validator.validate_model_input(task, "task", catalog["prompt_policy"])


def test_internal_term_in_user_prompt_is_rejected():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    task = copy.deepcopy(tasks[0])
    task["model_input"]["turns"][0]["prompt"] = "请从 checkpoint 继续完成任务。"

    with pytest.raises(ValueError, match="contains internal terms"):
        validator.validate_model_input(task, "task", catalog["prompt_policy"])


def test_missing_delegate_tool_scenario_is_rejected():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    core_tasks = copy.deepcopy([task for task in tasks if task["id"].startswith("core.")])
    for task in core_tasks:
        if task["coverage"]["track"] == "tool_matrix":
            task["coverage"]["primary_tools"] = [
                name for name in task["coverage"]["primary_tools"] if name != "delegate"
            ]

    with pytest.raises(ValueError, match="delegate must be primary"):
        validator.validate_core_coverage(core_tasks, catalog["core_coverage"])


def test_mechanism_missing_boundary_scenario_is_rejected():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    core_tasks = copy.deepcopy([task for task in tasks if task["id"].startswith("core.")])
    for task in core_tasks:
        task["coverage"]["mechanisms"] = [
            item
            for item in task["coverage"]["mechanisms"]
            if not (item["name"] == "context_compression" and item["scenario"] == "boundary")
        ]

    with pytest.raises(ValueError, match="context_compression"):
        validator.validate_core_coverage(core_tasks, catalog["core_coverage"])


def test_workflow_category_count_mismatch_is_rejected():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    workflow_tasks = copy.deepcopy([task for task in tasks if task["id"].startswith("workflow.")])
    workflow_tasks.pop(next(index for index, task in enumerate(workflow_tasks) if task["id"].startswith("workflow.state.")))
    policy = copy.deepcopy(catalog["workflow_coverage"])
    policy["workflow_task_count"] -= 1

    with pytest.raises(ValueError, match="state expected 15 tasks"):
        validator.validate_workflow_coverage(workflow_tasks, policy)


def test_workflow_difficulty_imbalance_is_rejected():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    workflow_tasks = copy.deepcopy([task for task in tasks if task["id"].startswith("workflow.")])
    task = next(
        item
        for item in workflow_tasks
        if item["id"].startswith("workflow.code.") and item["difficulty"] == "easy"
    )
    task["difficulty"] = "hard"

    with pytest.raises(ValueError, match="code difficulty distribution mismatch"):
        validator.validate_workflow_coverage(workflow_tasks, catalog["workflow_coverage"])


def test_scripted_probes_still_have_natural_chinese_prompts():
    validator = load_validator_module()
    catalog, tasks = load_catalog_and_tasks(validator)
    scripted = [task for task in tasks if task["runner"]["kind"] == "scripted_probe"]

    assert scripted
    for task in scripted:
        validator.validate_model_input(task, task["id"], catalog["prompt_policy"])


def test_core_turns_contain_only_user_prompts():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)

    for task in tasks:
        if not task["id"].startswith("core."):
            continue
        for turn in task["model_input"]["turns"]:
            assert "environment_notes" not in turn, task["id"]


def test_every_core_task_has_a_required_non_trace_outcome_check():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)

    for task in tasks:
        if not task["id"].startswith("core."):
            continue
        outcome_checks = [
            check
            for check in task["evaluation"]["checks"]
            if check["required"] and check["type"] not in {"trace_event", "trace_event_absent"}
        ]
        assert outcome_checks, task["id"]


def test_workflow_turns_contain_only_user_prompts():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)

    for task in tasks:
        if not task["id"].startswith("workflow."):
            continue
        for turn in task["model_input"]["turns"]:
            assert "environment_notes" not in turn, task["id"]


def test_code_workflows_declare_clean_room_github_case_inspiration():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)

    code_tasks = [task for task in tasks if task["id"].startswith("workflow.code.")]
    assert len(code_tasks) == 15
    for task in code_tasks:
        case = task["case_inspiration"]
        assert case["repository"]
        assert case["pattern"]
        assert case["url"].startswith("https://github.com/")


def test_subjective_workflows_require_path_and_line_evidence():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)

    subjective_tasks = [task for task in tasks if task["id"].startswith("workflow.subjective.")]
    assert len(subjective_tasks) == 15
    for task in subjective_tasks:
        prompt = task["model_input"]["turns"][0]["prompt"]
        assert "相对路径:行号" in prompt, task["id"]


def test_subjective_workflows_use_semantic_rubrics_and_extended_watchdogs():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)

    subjective_tasks = [task for task in tasks if task["id"].startswith("workflow.subjective.")]
    assert len(subjective_tasks) == 15
    for task in subjective_tasks:
        check_types = {check["type"] for check in task["evaluation"]["checks"]}
        assert not ({"file_content_contains", "file_content_excludes"} & check_types), task["id"]
        assert task["setup"]["workspace_files"], task["id"]
        assert task["runner"]["timeout_seconds"] >= 480, task["id"]


def test_reworked_workflow_prompts_disclose_required_capabilities():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)
    by_id = {task["id"]: task for task in tasks}

    path_filtering = by_id["workflow.code.path-filtering.v1"]
    assert "`*` 不跨路径分隔符" in path_filtering["model_input"]["turns"][0]["prompt"]
    assert "`**`" in path_filtering["model_input"]["turns"][0]["prompt"]
    assert "不含通配符的模式继续按路径子串匹配" in path_filtering["model_input"]["turns"][0]["prompt"]
    hidden = path_filtering["setup"]["private_files"][0]["content"]
    assert "include='*.py') == ['a.py']" in hidden

    expectations = {
        "workflow.subjective.deployment-readiness-review.v1": ["分阶段发布", "暂停条件", "待确认"],
        "workflow.subjective.test-strategy.v1": ["并发与重试", "CI 执行位置", "测试责任归属", "发布门禁"],
        "workflow.subjective.database-latency-incident.v1": ["流量或负载控制", "canary 门禁", "自动回滚阈值"],
    }
    for task_id, phrases in expectations.items():
        prompt = by_id[task_id]["model_input"]["turns"][0]["prompt"]
        assert all(phrase in prompt for phrase in phrases), task_id

    onboarding_judge = by_id["workflow.subjective.repo-onboarding.v1"]["evaluation"]["judge"]
    assert onboarding_judge["include_execution_evidence"] is True


def test_subjective_text_match_gate_is_rejected():
    validator = load_validator_module()
    _, tasks = load_catalog_and_tasks(validator)
    task = copy.deepcopy(next(task for task in tasks if task["id"].startswith("workflow.subjective.")))
    task["evaluation"]["checks"].insert(
        1,
        {"id": "lexical_gate", "type": "file_content_contains", "required": True, "spec": {"path": "incident-report.md", "all": ["03:12"]}},
    )

    with pytest.raises(ValueError, match="semantic rubric evidence"):
        validator.validate_evaluation(task, task["id"], "workflows", "subjective")


def test_reworked_core_fixture_oracles_match_generated_data(tmp_path):
    from eval.run_suite import generate_monthly_records, generate_refund_records

    validator = load_validator_module()
    _, all_tasks = load_catalog_and_tasks(validator)
    tasks = {task["id"]: task for task in all_tasks}

    archive = tasks["core.archive.long-result-common.v1"]
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    materialize_task(archive, archive_root)
    completed = subprocess.run(
        ["python", "emit-inventory.py"],
        cwd=archive_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0
    final_inventory = {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(r"warehouse-(\d+) stage=FINAL inventory=(\d+)", completed.stdout)
    }
    assert final_inventory == {17: 4811, 42: 9923, 88: 12007}
    assert "5000" in (archive_root / "policy.md").read_text(encoding="utf-8")

    monthly = tasks["core.context.long-history-common.v1"]
    monthly_root = tmp_path / "monthly"
    generate_monthly_records(monthly_root)
    totals = {}
    for path in sorted((monthly_root / "records").glob("2026-*.txt")):
        total = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = dict(part.split("=", 1) for part in line.split())
            if fields["status"] != "cancelled":
                total += int(fields["amount"])
        totals[path.stem] = total
    assert totals == required_check(monthly, "totals")["spec"]["value"]
    assert sum(totals.values()) == required_check(monthly, "grand_total")["spec"]["value"]

    refunds = tasks["core.context.compression-failure-recovery.v1"]
    refunds_root = tmp_path / "refunds"
    generate_refund_records(refunds_root)
    counts = Counter()
    for path in (refunds_root / "records").glob("*.log"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = dict(part.split("=", 1) for part in line.split())
            counts[fields["reason"]] += 1
    expected_reasons = required_check(refunds, "reasons")["spec"]["value"]
    assert [counts[item["category"]] for item in expected_reasons] == [item["count"] for item in expected_reasons]


def materialize_task(task, root, include_private=False):
    for relative in task["setup"]["directories"]:
        (root / relative).mkdir(parents=True, exist_ok=True)
    files = list(task["setup"]["workspace_files"])
    if include_private:
        files.extend(task["setup"]["private_files"])
    for item in files:
        path = root / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item["content"], encoding="utf-8")


def required_check(task, check_id):
    return next(check for check in task["evaluation"]["checks"] if check["id"] == check_id)


def json_path_value(value, expression):
    assert expression.startswith("$")
    tokens = [dot or bracket for dot, bracket in re.findall(r"\.([A-Za-z0-9_/-]+)|\['([^']+)'\]", expression[1:])]
    for token in tokens:
        value = value[token]
    return value


def assert_state_checks(task, root):
    for check in task["evaluation"]["checks"]:
        check_type = check["type"]
        spec = check["spec"]
        path = root / spec["path"] if "path" in spec else None
        if check_type == "file_exists":
            assert path.exists(), check["id"]
        elif check_type == "file_absent":
            assert not path.exists(), check["id"]
        elif check_type == "file_content_equals":
            assert path.read_text(encoding="utf-8") == spec["content"], check["id"]
        elif check_type in {"file_content_contains", "file_content_excludes"}:
            content = path.read_text(encoding="utf-8")
            expected = []
            if "text" in spec:
                expected.append(spec["text"])
            expected.extend(spec.get("all", []))
            insensitive = []
            if "text_case_insensitive" in spec:
                insensitive.append(spec["text_case_insensitive"])
            insensitive.extend(spec.get("all_case_insensitive", []))
            if check_type == "file_content_contains":
                assert all(item in content for item in expected), check["id"]
                assert all(item.lower() in content.lower() for item in insensitive), check["id"]
            else:
                assert all(item not in content for item in expected), check["id"]
                assert all(item.lower() not in content.lower() for item in insensitive), check["id"]
        elif check_type == "json_value_equals":
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert json_path_value(payload, spec["json_path"]) == spec["value"], check["id"]
        elif check_type == "command_exit_code":
            if "command" not in spec:
                continue
            completed = subprocess.run(
                spec["command"],
                cwd=root,
                shell=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            assert completed.returncode == spec["exit_code"], f"{check['id']}\n{completed.stdout}\n{completed.stderr}"
        elif check_type == "trace_event":
            continue
        else:
            raise AssertionError(f"unsupported state oracle check: {check_type}")


def test_state_task_oracles_reach_declared_final_state(tmp_path):
    validator = load_validator_module()
    data = validator.load_json(validator.EVAL_ROOT / "workflows" / "state.json")
    tasks = {task["id"]: task for task in data["tasks"]}

    cleanup = tasks["workflow.state.selective-cleanup.v1"]
    cleanup_root = tmp_path / "cleanup"
    materialize_task(cleanup, cleanup_root)
    for line in (cleanup_root / "cleanup-manifest.txt").read_text(encoding="utf-8").splitlines():
        (cleanup_root / line).unlink()
    for check in cleanup["evaluation"]["checks"]:
        path = cleanup_root / check["spec"]["path"]
        if check["type"] == "file_absent":
            assert not path.exists()
        elif check["type"] == "file_content_equals":
            assert path.read_text(encoding="utf-8") == check["spec"]["content"]

    migration = tasks["workflow.state.config-migration.v1"]
    migration_root = tmp_path / "migration"
    materialize_task(migration, migration_root)
    config = json.loads((migration_root / "config.json").read_text(encoding="utf-8"))
    config["orders_endpoint"] = "/api/v2/orders"
    config["retry_limit"] = config.pop("legacy_retry")
    (migration_root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for relative in ("src/client.py", "README.md"):
        path = migration_root / relative
        path.write_text(path.read_text(encoding="utf-8").replace("/api/v1/orders", "/api/v2/orders"), encoding="utf-8")
    completed = subprocess.run(
        ["python", "validate.py"],
        cwd=migration_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    reconciliation = tasks["workflow.state.dataset-reconciliation.v1"]
    reconcile_root = tmp_path / "reconciliation"
    materialize_task(reconciliation, reconcile_root)
    with (reconcile_root / "data/orders.csv").open(encoding="utf-8", newline="") as handle:
        orders = list(csv.DictReader(handle))
    refunds = json.loads((reconcile_root / "data/refunds.json").read_text(encoding="utf-8"))
    paid = {row["order_id"]: int(row["amount"]) for row in orders if row["status"] == "paid"}
    paid_refunds = {order_id: 0 for order_id in paid}
    for refund in refunds:
        if refund["order_id"] in paid_refunds:
            paid_refunds[refund["order_id"]] += int(refund["amount"])
    oracle = {
        "gross_paid": sum(paid.values()),
        "refunded": sum(paid_refunds.values()),
        "net_revenue": sum(paid.values()) - sum(paid_refunds.values()),
        "paid_order_count": len(paid),
        "fully_refunded_order_ids": sorted(
            order_id for order_id, amount in paid.items() if paid_refunds[order_id] == amount
        ),
    }
    for check_id, key in (
        ("gross", "gross_paid"),
        ("refunded", "refunded"),
        ("net", "net_revenue"),
        ("count", "paid_order_count"),
        ("full_refunds", "fully_refunded_order_ids"),
    ):
        assert oracle[key] == required_check(reconciliation, check_id)["spec"]["value"]


def test_expanded_state_task_oracles_reach_declared_final_state(tmp_path):
    validator = load_validator_module()
    data = validator.load_json(validator.EVAL_ROOT / "workflows" / "state.json")
    tasks = {task["id"]: task for task in data["tasks"]}
    solved = []

    task = tasks["workflow.state.static-assets-migration.v1"]
    root = tmp_path / "assets"
    materialize_task(task, root)
    for name in ("app.css", "logo.txt"):
        shutil.move(root / "static" / name, root / "public" / "assets" / name)
    index = root / "public" / "index.html"
    index.write_text(index.read_text(encoding="utf-8").replace("/static/", "/assets/"), encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.env-example-normalization.v1"]
    root = tmp_path / "env"
    materialize_task(task, root)
    expected = required_check(task, "example_exact")["spec"]["content"]
    (root / ".env.example").write_text(expected, encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.documentation-link-repair.v1"]
    root = tmp_path / "docs"
    materialize_task(task, root)
    readme = root / "README.md"
    content = readme.read_text(encoding="utf-8")
    content = content.replace("docs/setup.md", "docs/getting-started.md").replace("guide/api.md", "docs/api.md")
    readme.write_text(content, encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.log-retention-cleanup.v1"]
    root = tmp_path / "logs"
    materialize_task(task, root)
    for name in ("app-2026-07-12.log", "app-2026-07-13.log"):
        (root / "logs" / name).unlink()
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.localization-sync.v1"]
    root = tmp_path / "locale"
    materialize_task(task, root)
    source = json.loads((root / "locales" / "en-US.json").read_text(encoding="utf-8"))
    target = json.loads((root / "locales" / "zh-CN.json").read_text(encoding="utf-8"))
    glossary = json.loads((root / "glossary.json").read_text(encoding="utf-8"))
    normalized = {key: target[key] if key in target else glossary[key] for key in source}
    (root / "locales" / "zh-CN.json").write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.lockfile-workspace-cleanup.v1"]
    root = tmp_path / "lock"
    materialize_task(task, root)
    lock = json.loads((root / "workspace-lock.json").read_text(encoding="utf-8"))
    lock["workspaces"].pop("packages/legacy-ui")
    (root / "workspace-lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.api-fixture-regeneration.v1"]
    root = tmp_path / "fixtures"
    materialize_task(task, root)
    for path in (root / "fixtures" / "users").glob("*.json"):
        user = json.loads(path.read_text(encoding="utf-8"))
        first_name, last_name = user.pop("name").split(" ", 1)
        user.update(schema_version=2, first_name=first_name, last_name=last_name, active=True)
        path.write_text(json.dumps(user, indent=2) + "\n", encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.monorepo-package-rename.v1"]
    root = tmp_path / "monorepo"
    materialize_task(task, root)
    shutil.move(root / "packages" / "auth-client", root / "packages" / "identity-client")
    for path in root.rglob("*"):
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            content = content.replace("@pico/auth-client", "@pico/identity-client")
            content = content.replace("packages/auth-client", "packages/identity-client")
            path.write_text(content, encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.database-seed-reconciliation.v1"]
    root = tmp_path / "seed"
    materialize_task(task, root)
    with (root / "exports" / "customers.csv").open(encoding="utf-8", newline="") as handle:
        customer_rows = list(csv.DictReader(handle))
    customers_by_email = {}
    for row in customer_rows:
        email = row["email"].lower()
        if email not in customers_by_email or row["updated_at"] > customers_by_email[email]["updated_at"]:
            customers_by_email[email] = row
    customers = sorted(
        ({"id": row["id"], "email": email, "name": row["name"]} for email, row in customers_by_email.items()),
        key=lambda item: item["id"],
    )
    customer_id_by_email = {item["email"]: item["id"] for item in customers}
    with (root / "exports" / "orders.csv").open(encoding="utf-8", newline="") as handle:
        order_rows = list(csv.DictReader(handle))
    orders = []
    orphan_count = 0
    for row in order_rows:
        if row["status"] != "paid":
            continue
        customer_id = customer_id_by_email.get(row["customer_email"].lower())
        if customer_id is None:
            orphan_count += 1
            continue
        orders.append({
            "id": row["id"],
            "customer_id": customer_id,
            "amount_cents": int(Decimal(row["amount"]) * 100),
        })
    seed = {"customers": customers, "orders": sorted(orders, key=lambda item: item["id"]), "summary": {"orphan_orders_discarded": orphan_count}}
    (root / "db" / "seed.json").write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.release-artifact-packaging.v1"]
    root = tmp_path / "release"
    materialize_task(task, root)
    payloads = {
        "app.py": (root / "dist" / "app.py").read_bytes(),
        "config/default.json": (root / "dist" / "config.json").read_bytes(),
    }
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    (root / "release" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(root / "release" / "pico-1.4.0.zip", "w") as archive:
        for name, content in payloads.items():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.media-metadata-normalization.v1"]
    root = tmp_path / "media"
    materialize_task(task, root)
    items = []
    rejected = []
    for path in (root / "media").glob("*.json"):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        if not (root / "media" / sidecar["file"]).is_file():
            rejected.append(path.name)
            continue
        raw_date = sidecar["date"]
        date_format = "%Y/%m/%d" if "/" in raw_date else "%B %d, %Y" if "," in raw_date else "%Y-%m-%d"
        items.append({
            "file": sidecar["file"],
            "date": datetime.strptime(raw_date, date_format).strftime("%Y-%m-%d"),
            "tags": sorted({tag.lower() for tag in sidecar["tags"]}),
            "width": int(sidecar["width"]),
            "height": int(sidecar["height"]),
        })
    catalog = {"items": sorted(items, key=lambda item: item["file"]), "rejected": sorted(rejected)}
    (root / "media" / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    assert_state_checks(task, root)
    solved.append(task["id"])

    task = tasks["workflow.state.deployment-manifest-migration.v1"]
    root = tmp_path / "manifests"
    materialize_task(task, root)
    (root / "deploy" / "deployment.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: pico-api\nspec:\n  replicas: 2\n  selector:\n    matchLabels:\n      app: pico-api\n  template:\n    metadata:\n      labels:\n        app: pico-api\n    spec:\n      containers:\n      - name: api\n        image: pico/api:1.4.0\n        ports:\n        - containerPort: 8080\n",
        encoding="utf-8",
    )
    (root / "deploy" / "ingress.yaml").write_text(
        "apiVersion: networking.k8s.io/v1\nkind: Ingress\nmetadata:\n  name: pico-api\nspec:\n  rules:\n  - host: api.pico.test\n    http:\n      paths:\n      - path: /\n        pathType: Prefix\n        backend:\n          service:\n            name: pico-api\n            port:\n              number: 80\n",
        encoding="utf-8",
    )
    assert_state_checks(task, root)
    solved.append(task["id"])

    expanded_ids = {task_id for task_id in tasks if task_id not in {
        "workflow.state.selective-cleanup.v1",
        "workflow.state.config-migration.v1",
        "workflow.state.dataset-reconciliation.v1",
    }}
    assert set(solved) == expanded_ids


def test_code_task_hidden_tests_accept_reference_solutions(tmp_path):
    validator = load_validator_module()
    data = validator.load_json(validator.EVAL_ROOT / "workflows" / "code.json")
    solutions = {
        "workflow.code.config-precedence.v1": (
            "src/config_loader.py",
            "def load_config(defaults, file_values, env_values):\n"
            "    config = dict(defaults)\n"
            "    config.update(file_values)\n"
            "    config.update(env_values)\n"
            "    return config\n",
        ),
        "workflow.code.retry-idempotency.v1": (
            "src/delivery.py",
            "def deliver_with_retry(prepare, send, attempts=3):\n"
            "    payload = prepare()\n"
            "    last_error = None\n"
            "    for _ in range(attempts):\n"
            "        try:\n"
            "            return send(payload)\n"
            "        except RuntimeError as exc:\n"
            "            last_error = exc\n"
            "    raise last_error\n",
        ),
            "workflow.code.path-filtering.v1": (
                "src/path_filter.py",
                "from fnmatch import fnmatchcase\n"
                "from functools import lru_cache\n\n"
                "def _matches(path, pattern):\n"
                "    if not any(char in pattern for char in '*?['):\n"
                "        return pattern in path\n"
                "    path_parts = path.split('/') if path else []\n"
                "    pattern_parts = pattern.split('/') if pattern else []\n"
                "    @lru_cache(maxsize=None)\n"
                "    def match(path_index, pattern_index):\n"
                "        if pattern_index == len(pattern_parts):\n"
                "            return path_index == len(path_parts)\n"
                "        part = pattern_parts[pattern_index]\n"
                "        if part == '**':\n"
                "            return any(match(next_index, pattern_index + 1) for next_index in range(path_index, len(path_parts) + 1))\n"
                "        return path_index < len(path_parts) and fnmatchcase(path_parts[path_index], part) and match(path_index + 1, pattern_index + 1)\n"
                "    return match(0, 0)\n\n"
                "def select_paths(paths, include='*', exclude=None):\n"
                "    selected = []\n"
                "    for path in paths:\n"
                "        normalized = path.replace('\\\\', '/')\n"
                "        if not _matches(normalized, include):\n"
            "            continue\n"
            "        if exclude and _matches(normalized, exclude):\n"
            "            continue\n"
            "        selected.append(path)\n"
            "    return selected\n",
        ),
        "workflow.code.pagination-boundary.v1": (
            "src/pagination.py",
            "def paginate(items, page, page_size):\n"
            "    if page < 1 or page_size < 1:\n"
            "        raise ValueError('invalid pagination')\n"
            "    start = (page - 1) * page_size\n"
            "    return items[start:start + page_size]\n",
        ),
        "workflow.code.cache-ttl-boundary.v1": (
            "src/cache.py",
            "class TTLCache:\n"
            "    def __init__(self, now):\n"
            "        self._now = now\n"
            "        self._items = {}\n\n"
            "    def put(self, key, value, ttl):\n"
            "        self._items[key] = (value, self._now(), ttl)\n\n"
            "    def get(self, key):\n"
            "        item = self._items.get(key)\n"
            "        if item is None:\n"
            "            return None\n"
            "        value, created, ttl = item\n"
            "        if self._now() - created < ttl:\n"
            "            return value\n"
            "        self._items.pop(key, None)\n"
            "        return None\n",
        ),
        "workflow.code.upload-size-validation.v1": (
            "src/uploads.py",
            "def accept_upload(chunks, max_bytes):\n"
            "    accepted = []\n"
            "    total = 0\n"
            "    for chunk in chunks:\n"
            "        total += len(chunk)\n"
            "        if total > max_bytes:\n"
            "            raise ValueError('upload too large')\n"
            "        accepted.append(chunk)\n"
            "    return b''.join(accepted)\n",
        ),
        "workflow.code.cli-boolean-parsing.v1": (
            "src/options.py",
            "def parse_debug(value):\n"
            "    if isinstance(value, bool):\n"
            "        return value\n"
            "    if isinstance(value, str):\n"
            "        normalized = value.lower()\n"
            "        if normalized in {'true', '1', 'yes', 'on'}:\n"
            "            return True\n"
            "        if normalized in {'false', '0', 'no', 'off'}:\n"
            "            return False\n"
            "    raise ValueError('invalid boolean')\n",
        ),
        "workflow.code.frontend-stale-state.v1": (
            "src/counter.js",
            "export function applyClicks(state, clicks) {\n"
            "  let next = state;\n"
            "  for (let i = 0; i < clicks; i += 1) {\n"
            "    next += 1;\n"
            "  }\n"
            "  return next;\n"
            "}\n",
        ),
        "workflow.code.request-validation.v1": (
            "src/api.py",
            "def create_user(payload):\n"
            "    email = payload.get('email', '')\n"
            "    email = email.strip() if isinstance(email, str) else ''\n"
            "    age = payload.get('age')\n"
            "    if not email or '@' not in email:\n"
            "        raise ValueError('invalid email')\n"
            "    if isinstance(age, bool) or not isinstance(age, int) or age < 18:\n"
            "        raise ValueError('invalid age')\n"
            "    return {'email': email, 'age': age}\n",
        ),
        "workflow.code.async-handler-errors.v1": (
            "src/handler.js",
            "export const wrap = (handler) => (req, res, next) =>\n"
            "  Promise.resolve().then(() => handler(req, res)).catch(next);\n",
        ),
        "workflow.code.transaction-rollback.v1": (
            "src/transfers.py",
            "def transfer(db, sender, recipient, amount):\n"
            "    if amount <= 0:\n"
            "        raise ValueError('invalid amount')\n"
            "    with db:\n"
            "        sender_row = db.execute('SELECT balance FROM accounts WHERE id = ?', (sender,)).fetchone()\n"
            "        recipient_row = db.execute('SELECT balance FROM accounts WHERE id = ?', (recipient,)).fetchone()\n"
            "        if sender_row is None or recipient_row is None:\n"
            "            raise ValueError('account missing')\n"
            "        if sender_row[0] < amount:\n"
            "            raise ValueError('insufficient funds')\n"
            "        db.execute('UPDATE accounts SET balance = balance - ? WHERE id = ?', (amount, sender))\n"
            "        db.execute('UPDATE accounts SET balance = balance + ? WHERE id = ?', (amount, recipient))\n"
            "    return tuple(db.execute('SELECT balance FROM accounts WHERE id IN (?, ?) ORDER BY id', (sender, recipient)).fetchall())\n",
        ),
        "workflow.code.async-retry-cancellation.v1": (
            "src/retry.py",
            "async def retry(operation, attempts=3):\n"
            "    if attempts < 1:\n"
            "        raise ValueError('attempts must be positive')\n"
            "    last_error = None\n"
            "    for _ in range(attempts):\n"
            "        try:\n"
            "            return await operation()\n"
            "        except Exception as exc:\n"
            "            last_error = exc\n"
            "    raise last_error\n",
        ),
        "workflow.code.websocket-reconnect.v1": (
            "src/socket-client.js",
            "export class SocketClient {\n"
            "  constructor(onMessage) {\n"
            "    this.onMessage = onMessage;\n"
            "    this.socket = null;\n"
            "    this.handleMessage = (value) => this.onMessage(value);\n"
            "  }\n"
            "  connect(socket) {\n"
            "    if (this.socket === socket) return;\n"
            "    if (this.socket) this.socket.off('message', this.handleMessage);\n"
            "    this.socket = socket;\n"
            "    socket.on('message', this.handleMessage);\n"
            "  }\n"
            "}\n",
        ),
        "workflow.code.concurrent-memoization.v1": (
            "src/memo.py",
            "from threading import Event, RLock\n\n"
            "def memoize(function):\n"
            "    cache = {}\n"
            "    lock = RLock()\n"
            "    in_flight = {}\n"
            "    def wrapped(key):\n"
            "        with lock:\n"
            "            if key in cache:\n"
            "                return cache[key]\n"
            "            event = in_flight.get(key)\n"
            "            owner = event is None\n"
            "            if owner:\n"
            "                event = Event()\n"
            "                in_flight[key] = event\n"
            "        if not owner:\n"
            "            event.wait()\n"
            "            return wrapped(key)\n"
            "        try:\n"
            "            value = function(key)\n"
            "        except BaseException:\n"
            "            with lock:\n"
            "                in_flight.pop(key, None)\n"
            "                event.set()\n"
            "            raise\n"
            "        with lock:\n"
            "            cache[key] = value\n"
            "            in_flight.pop(key, None)\n"
            "            event.set()\n"
            "        return value\n"
            "    return wrapped\n",
        ),
        "workflow.code.frontend-keyed-state.v1": (
            "src/reconcile.js",
            "export function reconcile(previous, items, createState) {\n"
            "  const byId = new Map(previous.map((item) => [item.id, item.state]));\n"
            "  return items.map((item) => ({\n"
            "    ...item,\n"
            "    state: byId.has(item.id) ? byId.get(item.id) : createState(item),\n"
            "  }));\n"
            "}\n",
        ),
    }

    for task in data["tasks"]:
        root = tmp_path / task["id"]
        materialize_task(task, root, include_private=True)
        relative_path, content = solutions[task["id"]]
        (root / relative_path).write_text(content, encoding="utf-8")
        command = required_check(task, "all_tests")["spec"]["command"]
        argv = ["node", "--test"] if command == "npm test" else ["python", "-m", "pytest", "-q", "tests", "private_tests"]
        completed = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert completed.returncode == 0, f"{task['id']}\n{completed.stdout}\n{completed.stderr}"


def test_code_task_baselines_pass_public_tests_and_fail_hidden_bug_tests(tmp_path):
    validator = load_validator_module()
    data = validator.load_json(validator.EVAL_ROOT / "workflows" / "code.json")

    for task in data["tasks"]:
        root = tmp_path / task["id"]
        materialize_task(task, root)
        is_node = required_check(task, "all_tests")["spec"]["command"] == "npm test"
        public_node_tests = [
            item["path"] for item in task["setup"]["workspace_files"] if item["path"].startswith("tests/") and item["path"].endswith(".mjs")
        ]
        public_argv = ["node", "--test", *public_node_tests] if is_node else ["python", "-m", "pytest", "-q", "tests"]
        public = subprocess.run(
            public_argv,
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert public.returncode == 0, f"public baseline failed: {task['id']}\n{public.stdout}\n{public.stderr}"

        materialize_task(task, root, include_private=True)
        private_node_tests = [
            item["path"] for item in task["setup"]["private_files"] if item["path"].endswith(".mjs")
        ]
        full_argv = ["node", "--test", *public_node_tests, *private_node_tests] if is_node else ["python", "-m", "pytest", "-q", "tests", "private_tests"]
        hidden = subprocess.run(
            full_argv,
            cwd=root,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        assert hidden.returncode != 0, f"hidden bug tests did not expose the defect: {task['id']}"
