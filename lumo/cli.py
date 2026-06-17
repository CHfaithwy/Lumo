"""命令行入口。

这个模块负责把“用户怎么启动 Lumo”翻译成 runtime 能理解的对象：
解析参数、挑模型后端、构建工作区快照、恢复或新建 session，
最后进入 one-shot 或交互式循环。
"""

import argparse
import os
import re
import shutil
import sys
import textwrap
import unicodedata
from pathlib import Path

from .config import load_project_env, provider_env
from .providers.clients import AnthropicCompatibleModelClient, OllamaModelClient, OpenAICompatibleModelClient
from .runtime import Pico, SessionStore
from .workspace import AGENT_STATE_DIR, WorkspaceContext

DEFAULT_SECRET_ENV_NAMES = (
    "PICO_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_API_TOKEN",
    "PICO_ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "PICO_DEEPSEEK_API_KEY",
    "DEEPSEEK_API_KEY",
    "PICO_RIGHT_CODES_API_KEY",
    "RIGHT_CODES_API_KEY",
    "GITHUB_PAT",
    "GH_PAT",
)

WELCOME_ART = tuple(
    "  ".join(parts)
    for parts in (
        ("██     ", "██   ██", "██   ██", "███████"),
        ("██     ", "██   ██", "███ ███", "██   ██"),
        ("██     ", "██   ██", "███████", "██   ██"),
        ("██     ", "██   ██", "██ █ ██", "██   ██"),
        ("██     ", "██   ██", "██   ██", "██   ██"),
        ("██     ", "██   ██", "██   ██", "██   ██"),
        ("███████", "███████", "██   ██", "███████"),
    )
)
WELCOME_NAME = "Lumo"
WELCOME_SUBTITLE = "Code is cheap, Show me your talk"
# WELCOME_STATUS = "calm shell, ready for work"
HELP_DETAILS = textwrap.dedent(
    """\
    Commands:
    /help    Show this help message.
    /memory  Show the agent's distilled working memory.
    /session Show the path to the saved session file.
    /reset   Clear the current session history and memory.
    /exit    Exit the agent.
    """
).strip()


DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://www.right.codes/codex/v1"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_ANTHROPIC_BASE_URL = "https://www.right.codes/claude/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
SECRET_ENV_NAMES_VAR = "PICO_SECRET_ENV_NAMES"
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
ANSI_RESET = "\x1b[0m"
WELCOME_BORDER_COLOR = "\x1b[38;2;125;211;252m"
WELCOME_GRADIENT_START = (255, 255, 255)
WELCOME_GRADIENT_END = (125, 211, 252)


def _strip_ansi(text):
    return ANSI_PATTERN.sub("", str(text))


def _ansi_rgb(red, green, blue):
    return f"\x1b[38;2;{red};{green};{blue}m"


def _color(text, ansi_color):
    return f"{ansi_color}{text}{ANSI_RESET}"


def _gradient_text(text, start_rgb=WELCOME_GRADIENT_START, end_rgb=WELCOME_GRADIENT_END):
    text = str(text)
    span = max(1, len(text) - 1)
    parts = []
    for index, char in enumerate(text):
        if char == " ":
            parts.append(char)
            continue
        ratio = index / span
        rgb = tuple(round(start + (end - start) * ratio) for start, end in zip(start_rgb, end_rgb))
        parts.append(_color(char, _ansi_rgb(*rgb)))
    return "".join(parts)


def _terminal_char_width(char):
    return 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1


def _terminal_width(text):
    return sum(_terminal_char_width(char) for char in _strip_ansi(text))


def _terminal_ljust(text, width):
    text = str(text)
    return text + " " * max(0, width - _terminal_width(text))


def _terminal_center(text, width):
    text = str(text)
    padding = max(0, width - _terminal_width(text))
    left = padding // 2
    right = padding - left
    return " " * left + text + " " * right


def _take_terminal_width(text, width, from_end=False):
    chars = reversed(str(text)) if from_end else str(text)
    selected = []
    used = 0
    for char in chars:
        char_width = _terminal_char_width(char)
        if used + char_width > width:
            break
        selected.append(char)
        used += char_width
    if from_end:
        selected.reverse()
    return "".join(selected)


def _terminal_middle(text, limit):
    text = str(text).replace("\n", " ")
    if _terminal_width(text) <= limit:
        return text
    if limit <= 3:
        return _take_terminal_width(text, limit)
    left = (limit - 3) // 2
    right = limit - 3 - left
    return _take_terminal_width(text, left) + "..." + _take_terminal_width(text, right, from_end=True)


def _effective_model(args, provider):
    # 模型选择优先级：
    # 1. 用户显式传入 --model
    # 2. provider 对应的环境变量
    # 3. 代码里的默认值
    explicit_model = getattr(args, "model", None)
    if explicit_model:
        return explicit_model
    if provider == "openai":
        model = provider_env("PICO_OPENAI_MODEL", ("OPENAI_MODEL",))
        if model:
            return model
        return DEFAULT_OPENAI_MODEL
    if provider == "anthropic":
        model = provider_env("PICO_ANTHROPIC_MODEL", ("ANTHROPIC_MODEL",))
        if model:
            return model
        return DEFAULT_ANTHROPIC_MODEL
    if provider == "deepseek":
        model = provider_env("PICO_DEEPSEEK_MODEL", ("DEEPSEEK_MODEL",))
        if model:
            return model
        return DEFAULT_DEEPSEEK_MODEL
    return DEFAULT_OLLAMA_MODEL


def _configured_secret_names(args):
    configured_secret_names = set(DEFAULT_SECRET_ENV_NAMES)
    configured_secret_names.update(str(name).upper() for name in args.secret_env_names)
    extra_names = os.environ.get(SECRET_ENV_NAMES_VAR, "")
    if extra_names.strip():
        configured_secret_names.update(
            item.strip().upper()
            for item in extra_names.split(",")
            if item.strip()
        )
    return sorted(configured_secret_names)


def _build_model_client(args):
    provider = getattr(args, "provider", "deepseek")
    # CLI 只负责把 provider 选择翻译成具体 client。
    # 真正的提示词格式、缓存支持、HTTP 协议差异，都封装在 models.py 里。
    if provider == "openai":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_OPENAI_API_BASE", ("OPENAI_API_BASE",), DEFAULT_OPENAI_BASE_URL)
        api_key = provider_env(
            "PICO_OPENAI_API_KEY",
            ("OPENAI_API_KEY", "PICO_RIGHT_CODES_API_KEY", "RIGHT_CODES_API_KEY", "PICO_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        )
        return OpenAICompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "anthropic":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_ANTHROPIC_API_BASE", ("ANTHROPIC_API_BASE",), DEFAULT_ANTHROPIC_BASE_URL)
        api_key = provider_env(
            "PICO_ANTHROPIC_API_KEY",
            ("ANTHROPIC_API_KEY", "PICO_RIGHT_CODES_API_KEY", "RIGHT_CODES_API_KEY", "PICO_OPENAI_API_KEY", "OPENAI_API_KEY"),
        )
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )
    if provider == "deepseek":
        model = _effective_model(args, provider)
        base_url = getattr(args, "base_url", None) or provider_env("PICO_DEEPSEEK_API_BASE", ("DEEPSEEK_API_BASE",), DEFAULT_DEEPSEEK_BASE_URL)
        api_key = provider_env("PICO_DEEPSEEK_API_KEY", ("DEEPSEEK_API_KEY",))
        return AnthropicCompatibleModelClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=args.temperature,
            timeout=getattr(args, "openai_timeout", getattr(args, "ollama_timeout", 300)),
        )

    model = _effective_model(args, provider)
    host = getattr(args, "host", DEFAULT_OLLAMA_HOST)
    return OllamaModelClient(
        model=model,
        host=host,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.ollama_timeout,
    )


def build_welcome(agent, model, host):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        body = _terminal_middle(text, width - 4)
        return _border_row(_terminal_ljust(body, width - 4))

    def divider(style="solid"):
        char = "═" if style == "strong" else "─"
        left = "╔" if style == "strong" else "├"
        right = "╗" if style == "strong" else "┤"
        return _color(left + char * (width - 2) + right, WELCOME_BORDER_COLOR)

    def bottom_divider():
        return _color("╚" + "═" * (width - 2) + "╝", WELCOME_BORDER_COLOR)

    def _border_row(body):
        return f"{_color('║', WELCOME_BORDER_COLOR)} {body} {_color('║', WELCOME_BORDER_COLOR)}"

    def center(text):
        body = _terminal_middle(text, inner)
        return _border_row(_terminal_center(body, inner))

    def cell(label, value, size):
        body = _terminal_middle(f"{label:<9} {value}", size)
        return _terminal_ljust(body, size)

    def pair(left_label, left_value, right_label, right_value):
        left = cell(left_label, left_value, left_width)
        right = cell(right_label, right_value, right_width)
        return _border_row(f"{left}{' ' * gap}{right}")

    line = divider("strong")
    rows = [center(_gradient_text(text)) for text in WELCOME_ART]
    rows.extend(
        [
            center(WELCOME_NAME),
            center(WELCOME_SUBTITLE),
            # center(WELCOME_STATUS),
            divider(),
            row(""),
            row("WORKSPACE  " + _terminal_middle(agent.workspace.cwd, inner - 11)),
            pair("MODEL", model, "BRANCH", agent.workspace.branch),
            pair("APPROVAL", agent.approval_policy, "SESSION", agent.session["id"]),
            row(""),
        ]
    )
    return "\n".join([line, *rows, bottom_divider()])


def build_agent(args):
    """根据 CLI 参数装配出一个可运行的 Pico 实例。

    为什么存在：
    命令行参数只是字符串和开关，runtime 需要的是已经装配好的对象图：
    model client、workspace snapshot、session store、secret 配置等。
    这个函数负责把“启动参数”翻译成“agent 运行现场”。

    输入 / 输出：
    - 输入：`argparse` 解析后的 `args`
    - 输出：一个新的 `Pico`，或一个从旧 session 恢复出来的 `Pico`

    在 agent 链路里的位置：
    它是整个程序启动链路里最靠近 runtime 的装配点。`main()` 先调它，
    得到 agent 后，后面无论是 one-shot 还是 REPL 模式，都会落到 `ask()`。
    """
    # 这里是 CLI 到 runtime 的装配点：
    # 先采集工作区快照和加载项目级环境，再整理 secret 名单、模型后端和 session。
    workspace = WorkspaceContext.build(args.cwd)
    # 加载项目级环境变量，覆盖系统环境。这些环境变量可能会被后续的模型调用用到，
    load_project_env(workspace.repo_root)
#   “整理出一份最终要被当成敏感信息处理的环境变量名列表。”
# 也就是说，后面程序在写 trace、report、session 之类内容时，看到这些环境变量名，就会把它们脱敏，不直接暴露真实值。
    configured_secret_names = _configured_secret_names(args)
    store = SessionStore(Path(workspace.repo_root) / AGENT_STATE_DIR / "sessions")
    # return OpenAICompatibleModelClient
    model = _build_model_client(args)
    session_id = args.resume
    if session_id == "latest":
        session_id = store.latest()
    if session_id:
        return Pico.from_session(
            model_client=model,
            workspace=workspace,
            session_store=store,
            session_id=session_id,
            approval_policy=args.approval,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            secret_env_names=configured_secret_names,
        )
    return Pico(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy=args.approval,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        secret_env_names=configured_secret_names,
    )


def _maybe_evolve_durable_memory(agent, reason):
    result = agent.evolve_durable_memory(reason=reason)
    if result.get("status") == "failed":
        print(f"durable memory evolution failed: {result.get('error', '')}", file=sys.stderr)
    return result


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal coding agent for DeepSeek, OpenAI-compatible, Anthropic-compatible, or Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--provider", choices=("ollama", "openai", "anthropic", "deepseek"), default="openai", help="Model backend to use.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override. Defaults to qwen3.5:4b for Ollama, PICO_OPENAI_MODEL for openai, PICO_ANTHROPIC_MODEL for anthropic, and PICO_DEEPSEEK_MODEL for deepseek when set.",
    )
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST, help="Ollama server URL.")
    parser.add_argument("--base-url", default=None, help="Provider API base URL for deepseek, openai, or anthropic.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--openai-timeout", type=int, default=300, help="OpenAI-compatible request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session id to resume or 'latest'.")
    parser.add_argument("--approval", choices=("ask", "auto", "never"), default="ask", help="Approval policy for risky tools.")
    parser.add_argument(
        "--secret-env-name",
        dest="secret_env_names",
        action="append",
        default=[],
        help="Extra environment variable names to treat as secrets for trace/report redaction.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool/model iterations per request.")
    parser.add_argument("--max-new-tokens", type=int, default=8192, help="Maximum model output tokens per step.")
    parser.add_argument("--no-memory-evolution", action="store_true", help="Disable session-end durable memory evolution.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature sent to Ollama.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value sent to Ollama.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    agent = build_agent(args)
    if args.resume and not args.no_memory_evolution:
        _maybe_evolve_durable_memory(agent, "resume_activation")
    # 欢迎页里显示的 MODEL  gpt-5.4
    model = getattr(agent.model_client, "model", getattr(args, "model", DEFAULT_OLLAMA_MODEL))
    # base_url 或 host 都行，优先级：model_client 里如果有就用它的，否则用 CLI 参数里的，再没有就用默认值。
    host = getattr(agent.model_client, "host", getattr(agent.model_client, "base_url", getattr(args, "host", DEFAULT_OLLAMA_HOST)))
    print(build_welcome(agent, model=model, host=host))

    if args.prompt:
        # one-shot 模式：只跑一次 ask，不进入 REPL 循环。
        prompt = " ".join(args.prompt).strip()
        if prompt:
            print()
            try:
                print(agent.ask(prompt))
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            finally:
                if not args.no_memory_evolution:
                    _maybe_evolve_durable_memory(agent, "session_end")
        return 0

    while True:
        # 交互模式：每次读取一条用户输入，交给同一个 agent，
        # 因此 session history 和 working memory 会跨轮延续。
        try:
            user_input = input("\nLUMO> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            if not args.no_memory_evolution:
                _maybe_evolve_durable_memory(agent, "session_end")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            if not args.no_memory_evolution:
                _maybe_evolve_durable_memory(agent, "session_end")
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(agent.session_path)
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue

        print()
        try:
            print(agent.ask(user_input))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
