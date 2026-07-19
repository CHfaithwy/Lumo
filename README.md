<p align="center">
  <img src="assets/image.png" alt="Lumo logo" width="360">
</p>
Lumo 是一个运行在本地代码仓库里的轻量级 Coding Agent。它可以在你的项目目录中读取文件、搜索代码、修改文件、运行命令，并通过多轮对话持续完成代码理解、问题排查、功能修改和项目整理等任务，支持持久记忆、trace自进化、分层skill。


## 安装

需要 Python 3.10 或更高版本。

进入项目根目录后安装：

```bash
pip install -e .
```

安装完成后可以使用：

```bash
lumo --help
```

如果你修改了项目源码，通常不需要重新安装；如果修改了 `pyproject.toml` 里的依赖或命令入口，再重新执行一次：

```bash
pip install -e .
```

## 分层 Skill

可复用工作流放在当前工作区的以下目录中：

```text
.lumo/skills/<类别>/<skill 名称>/SKILL.md
```

每个类别目录需要在同一层放置一个 `CATEGORY.md`，说明该类别的适用场景，并列出类别中的 skill：

```text
.lumo/skills/DocumentsAndAnalysis/
  CATEGORY.md
  pdf/SKILL.md
  docx/SKILL.md
```

`CATEGORY.md` 推荐按以下格式书写：

```md
---
description: 文档创建、读取和分析相关的工作流。
---

Skills:
- pdf
- docx
```

类别名和 skill 名仅使用英文字母、数字、`-` 或 `_`。`description` 应简洁说明类别用途；`Skills:` 描述当前目录下的所有skill名称。

使用 `/skills` 可以按类别查看当前工作区的全部 skill。

## Trace2Skill

Lumo 会从已经发生的工具轨迹中筛选可复用经验。只有任务正常完成、代码修改后已有测试通过、todo 已完成且流程包含多个步骤时，才会进入提炼队列。测试命令、退出码、修改顺序和 evidence ID 均由 runtime 从 trace 中确定；模型只负责将这些事实抽象为工作流。

CLI 空闲时会批量进行一次结构化提炼，每个进程会话最多调用一次模型，不会恢复旧工作区、重复执行任务或额外运行测试。生成的 Skill 保存在：

```text
.lumo/skills/learned/<skill 名称>/
  SKILL.md
  metadata.json
```

新 Skill 以 `probation` 状态参与现有 Skill 路由。后续任务使用它并通过测试后会积累验证成功次数；持续失败则自动禁用。使用 `--no-trace2skill` 可以关闭观察和提炼。

## 配置模型

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写你要使用的模型服务。

OpenAI-compatible 示例：

```env
LUMO_OPENAI_API_BASE=
LUMO_OPENAI_MODEL=
```

DeepSeek 示例：

```env
LUMO_DEEPSEEK_API_BASE=
LUMO_DEEPSEEK_API_KEY=
LUMO_DEEPSEEK_MODEL=
```

Anthropic-compatible 示例：

```env
LUMO_ANTHROPIC_API_BASE=
LUMO_ANTHROPIC_API_KEY=
LUMO_ANTHROPIC_MODEL=
```

如果你使用本地 Ollama，可以不配置 API key，直接指定 provider：

```bash
lumo --provider ollama --host http://127.0.0.1:11434 --model qwen3.5:4b
```

注意：不要把真实 `.env` 提交到 Git 仓库。

## 使用方式

在当前目录启动交互模式：

```bash
lumo --provider openai --cwd .
```

指定另一个项目目录：

```bash
lumo --provider openai --cwd E:\your\project
```

执行一次性任务：

```bash
lumo --provider openai --cwd . "帮我总结这个项目的结构"
```

恢复最近一次会话：

```bash
lumo --resume latest
```

常用参数：

```bash
lumo --provider openai
lumo --provider deepseek
lumo --provider anthropic
lumo --provider ollama
lumo --model gpt-5.4
lumo --base-url https://XXXX/v1
lumo --approval auto
lumo --max-steps 8
```

## 交互命令

进入 Lumo 交互模式后，可以使用这些命令：

```text
/help      查看帮助
/memory    查看当前记忆摘要
/skills    按类别查看当前工作区的所有 skill
/session   查看当前 session 文件路径
/reset     清空当前会话状态
/exit      退出
```


## 项目指令 lumo.md

你可以在项目根目录创建 `lumo.md`，用来告诉 Lumo 这个项目的固定规则。

示例：

```md
# Lumo Instructions

- 使用中文回答。
- 修改代码前先阅读相关文件。
- 写完代码后说明修改了哪些内容。
- 如果运行了测试，请在最终回复中说明测试命令。
```


## 本地数据

Lumo 会在当前工作区下创建 `.lumo/` 目录，用来保存本地运行数据：

```text
.lumo/
  sessions/    会话记录
  runs/        每次运行的 task_state.json、trace.jsonl
  memory/      跨会话长期记忆
  skills/      人工 Skill 与 Trace2Skill 生成的 learned Skill
  python-env/  普通 Python shell 命令使用的工作区隔离环境
```



