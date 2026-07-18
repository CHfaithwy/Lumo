

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .context_manager import _context_units
from .workspace import AGENT_STATE_DIR, clip, now


SKILLS_DIR = "skills"
CATEGORY_FILE_NAME = "CATEGORY.md"
SKILL_FILE_NAME = "SKILL.md"
SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
SKILL_LIST_BUDGET_UNITS = 1500
SKILL_CATEGORY_CATALOG_BUDGET_UNITS = 1500
SKILL_CATEGORY_ROUTING_THRESHOLD = 50
TASK_SKILLS_CONTENT_BUDGET_UNITS = 5000
MIN_SKILL_DESCRIPTION_UNITS = 8
MAX_ROUTED_SKILL_CATEGORIES = 2


@dataclass(frozen=True)
class CategoryInfo:
    name: str
    description: str
    path: str


@dataclass(frozen=True)
class SkillInfo:
    category: str
    name: str
    description: str
    path: str

    @property
    def qualified_name(self) -> str:
        return f"{self.category}/{self.name}"


@dataclass(frozen=True)
class SkillCatalog:
    categories: tuple[CategoryInfo, ...] = ()
    skills: tuple[SkillInfo, ...] = ()

    def category_names(self) -> set[str]:
        return {category.name for category in self.categories}

    def skills_for_categories(self, names) -> list[SkillInfo]:
        selected = {str(name).strip() for name in names or ()}
        return [skill for skill in self.skills if skill.category in selected]

    def find_skill(self, qualified_name: str) -> SkillInfo | None:
        for skill in self.skills:
            if skill.qualified_name == qualified_name:
                return skill
        return None


def should_route_skill_categories(catalog: SkillCatalog) -> bool:
    return len(catalog.skills) > SKILL_CATEGORY_ROUTING_THRESHOLD


def skills_root(root: Path) -> Path:
    return Path(root) / AGENT_STATE_DIR / SKILLS_DIR


def ensure_skills_dir(root: Path) -> Path:
    path = skills_root(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except Exception:
        return str(path)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    text = str(text)
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}, text
    fields = {}
    for line in lines[1:closing_index]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            fields[key] = value
    return fields, "\n".join(lines[closing_index + 1 :])


def _fallback_description(markdown: str) -> str:
    paragraph = []
    for line in str(markdown).splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        if stripped in {"---", "..."}:
            continue
        paragraph.append(stripped)
    return " ".join(paragraph).strip()


def _description_from_file(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    frontmatter, markdown = _split_frontmatter(content)
    description = str(frontmatter.get("description", "")).strip() or _fallback_description(markdown)
    return " ".join(description.split())


def parse_category_file(root: Path, category_dir: Path) -> CategoryInfo | None:
    name = category_dir.name
    if not SKILL_NAME_PATTERN.fullmatch(name):
        return None
    category_path = category_dir / CATEGORY_FILE_NAME
    if not category_path.is_file():
        return None
    description = _description_from_file(category_path)
    if not description:
        return None
    return CategoryInfo(
        name=name,
        description=description,
        path=_relative_path(root, category_path),
    )


def parse_skill_file(root: Path, category: CategoryInfo, skill_dir: Path) -> SkillInfo | None:
    name = skill_dir.name
    if not SKILL_NAME_PATTERN.fullmatch(name):
        return None
    skill_path = skill_dir / SKILL_FILE_NAME
    if not skill_path.is_file():
        return None
    description = _description_from_file(skill_path) or "(no description)"
    return SkillInfo(
        category=category.name,
        name=name,
        description=description,
        path=_relative_path(root, skill_path),
    )


def discover_skill_catalog(root: Path) -> SkillCatalog:
    root = Path(root)
    base = ensure_skills_dir(root)
    categories = []
    skills = []
    try:
        entries = sorted(base.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return SkillCatalog()
    for category_dir in entries:
        if not category_dir.is_dir():
            continue
        category = parse_category_file(root, category_dir)
        if category is None:
            continue
        categories.append(category)
        try:
            skill_entries = sorted(category_dir.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for skill_dir in skill_entries:
            if not skill_dir.is_dir():
                continue
            skill = parse_skill_file(root, category, skill_dir)
            if skill is not None:
                skills.append(skill)
    return SkillCatalog(categories=tuple(categories), skills=tuple(skills))


def normalize_routed_categories(requested, catalog: SkillCatalog) -> list[str]:
    categories, _valid = validate_routed_categories(requested, catalog)
    return categories


def validate_routed_categories(requested, catalog: SkillCatalog) -> tuple[list[str], bool]:
    values = [str(value).strip() for value in list(requested or [])]
    if not values:
        return [], False
    if values == ["none"]:
        return [], True
    if "none" in values or len(values) > MAX_ROUTED_SKILL_CATEGORIES:
        return [], False
    if len(set(values)) != len(values):
        return [], False
    if any(not SKILL_NAME_PATTERN.fullmatch(value) for value in values):
        return [], False
    known_categories = catalog.category_names()
    if any(value not in known_categories for value in values):
        return [], False
    return values, True


def _truncate_by_units(text: str, max_units: int) -> str:
    text = str(text)
    if max_units <= 0:
        return ""
    if _context_units(text) <= max_units:
        return text


    low, high = 0, len(text)
    while low < high:
        midpoint = (low + high + 1) // 2
        if _context_units(text[:midpoint]) <= max_units:
            low = midpoint
        else:
            high = midpoint - 1
    return text[:low].rstrip()


def _render_described_listing(title: str, entries, budget_units: int, name_of, description_of, truncated_note: str) -> str:
    entries = list(entries or [])
    if not entries:
        return f"{title}:\n- none"
    full_lines = [f"- {name_of(entry)}: {description_of(entry)}" for entry in entries]
    full_text = f"{title}:\n" + "\n".join(full_lines)
    if _context_units(full_text) <= int(budget_units):
        return full_text

    name_only_lines = [f"- {name_of(entry)}" for entry in entries]
    name_only_text = f"{title}:\n" + "\n".join(name_only_lines)
    available = max(0, int(budget_units) - _context_units(name_only_text) - len(entries))
    per_description_units = available // max(1, len(entries))
    if per_description_units < MIN_SKILL_DESCRIPTION_UNITS:
        return f"{title}:\n# descriptions omitted due to budget\n" + "\n".join(name_only_lines)

    lines = [f"{title}:", truncated_note]
    for entry in entries:
        description = description_of(entry)
        clipped = _truncate_by_units(description, per_description_units)
        if clipped and clipped != description:
            clipped += "..."
        lines.append(f"- {name_of(entry)}: {clipped or '(description omitted)'}")
    return "\n".join(lines)


def render_category_catalog(catalog: SkillCatalog, budget_units: int = SKILL_CATEGORY_CATALOG_BUDGET_UNITS) -> str:
    return _render_described_listing(
        "Skill categories",
        catalog.categories,
        budget_units,
        lambda category: category.name,
        lambda category: category.description,
        "# descriptions truncated evenly due to category catalog budget",
    )


def render_skill_catalog(catalog: SkillCatalog) -> str:

    if not catalog.categories:
        return "Skills:\n- no valid categories found under .lumo/skills"

    skills_by_category = {}
    for skill in catalog.skills:
        skills_by_category.setdefault(skill.category, []).append(skill)

    lines = ["Skills:"]
    for category in catalog.categories:
        lines.append(f"\n{category.name}: {category.description}")
        category_skills = skills_by_category.get(category.name, [])
        if not category_skills:
            lines.append("- no skills")
            continue
        for skill in category_skills:
            lines.append(f"- {skill.qualified_name}: {skill.description}")
    lines.append(f"\nTotal: {len(catalog.skills)} skills across {len(catalog.categories)} categories")
    return "\n".join(lines)


def render_skill_listing(skills: list[SkillInfo], budget_units: int = SKILL_LIST_BUDGET_UNITS, *, no_match: bool = False) -> str:
    if not skills and no_match:
        return "Skills:\n- none matched this request"
    return _render_described_listing(
        "Skills",
        skills,
        budget_units,
        lambda skill: skill.qualified_name,
        lambda skill: skill.description,
        "# descriptions truncated evenly due to skill listing budget",
    )


def load_skill_content(root: Path, qualified_name: str, catalog: SkillCatalog | None = None) -> tuple[SkillInfo, str]:
    requested = str(qualified_name or "").strip()
    category, separator, name = requested.partition("/")
    if not separator or not SKILL_NAME_PATTERN.fullmatch(category) or not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError("skill name must use the category/skill-name format")
    catalog = catalog or discover_skill_catalog(root)
    skill = catalog.find_skill(requested)
    if skill is None:
        raise ValueError(f"unknown skill: {requested}")
    path = Path(root) / skill.path
    try:
        resolved = path.resolve()
        resolved.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError("skill path escapes workspace") from exc
    if not resolved.is_file():
        raise ValueError(f"skill file not found: {skill.path}")
    content = resolved.read_text(encoding="utf-8", errors="replace")
    return skill, content


def render_task_skill_contexts(task_skills: list[dict]) -> dict[str, dict]:

    active = [item for item in task_skills or [] if isinstance(item, dict) and str(item.get("call_id", "")).strip()]
    if not active:
        return {}
    prepared = []
    for item in active:
        name = str(item.get("qualified_name") or item.get("name", "")).strip()
        path = str(item.get("path", "")).strip()
        skill_root = str(item.get("skill_root", "")).strip()
        if not skill_root and path:
            skill_root = Path(path).parent.as_posix()
        args = str(item.get("args", "")).strip()
        content = str(item.get("content", ""))
        header = (
            f"Loaded task skill: {name}\n"
            f"Source: {path}\n"
            f"Skill root: {skill_root}\n"
            "Relative paths in this skill are relative to Skill root; use full workspace-relative paths in tool calls.\n"
            f"Args: {args or '(none)'}\n"
            "Apply these instructions only while completing the current task:\n"
        )
        prepared.append({"call_id": str(item["call_id"]), "name": name, "header": header, "content": content})

    total_content_units = sum(_context_units(item["content"]) for item in prepared)
    truncation_marker = "\n...[task skill content truncated due to budget]"
    if total_content_units <= TASK_SKILLS_CONTENT_BUDGET_UNITS:
        per_skill_budget = None
    else:


        marker_units = _context_units(truncation_marker)
        per_skill_budget = max(
            0,
            (TASK_SKILLS_CONTENT_BUDGET_UNITS - marker_units * len(prepared)) // len(prepared),
        )

    rendered = {}
    for item in prepared:
        content = item["content"]
        header = item["header"]
        rendered_content = content if per_skill_budget is None else _truncate_by_units(content, per_skill_budget)
        truncated = per_skill_budget is not None and rendered_content != content
        if truncated:
            rendered_content = rendered_content.rstrip() + truncation_marker
        rendered[item["call_id"]] = {
            "name": item["name"],
            "content": f"{header}{rendered_content}".strip(),
            "truncated": truncated,
        }
    return rendered


def format_use_skill_result(skill: SkillInfo, args: str) -> str:
    args_text = str(args or "").strip()
    skill_root = Path(skill.path).parent.as_posix()
    lines = [
        f"skill_loaded: {skill.qualified_name}",
        f"description: {skill.description}",
        f"source: {skill.path}",
        f"skill_root: {skill_root}",
        "path_rule: Relative paths in this skill are relative to skill_root; use full workspace-relative paths in tool calls.",
        f"args: {args_text or '(none)'}",
        (
            f"<summary-for-history>Loaded skill {skill.qualified_name} from {skill.path}; "
            f"description: {clip(skill.description, 160)}.</summary-for-history>"
        ),
    ]
    return "\n".join(lines)


def task_skill_record(skill: SkillInfo, content: str, args: str) -> dict:
    return {
        "category": skill.category,
        "name": skill.name,
        "qualified_name": skill.qualified_name,
        "path": skill.path,
        "skill_root": Path(skill.path).parent.as_posix(),
        "description": skill.description,
        "args": str(args or ""),
        "content": str(content),
        "loaded_at": now(),
    }
