<p align="center">
  <img src="assets/image.png" alt="Lumo logo" width="360">
</p>
Lumo 是一个运行在本地代码仓库里的轻量级 Coding Agent。它可以在你的项目目录中读取文件、搜索代码、修改文件、运行命令，并通过多轮对话持续完成代码理解、问题排查、功能修改和项目整理等任务。


## 主要能力

- 支持本地代码仓库分析、代码修改、命令执行和多轮任务协作。
- 支持 OpenAI-compatible、Anthropic-compatible、DeepSeek 和 Ollama 等模型后端。
- 支持交互模式和一次性任务模式。
- 支持继续上一次会话，使用 `--resume latest` 可以恢复最近的工作。
- 支持通过 `lumo.md` 写入项目级指令，例如代码风格、测试习惯和注意事项。
- 运行数据默认保存在 `.lumo/`，不会和业务代码混在一起。

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

## 测试

检查评测数据、fixture、公开测试与隐藏回归测试的定义：

```bash
python -m pytest eval/test_eval_dataset.py
python eval/validate.py
```

使用 Lumo 完成全部 78 道评测任务的单轮首跑：

```bash
python eval/run_suite.py --output eval/results/latest --fresh-output --provider openai --repetitions 1 --timeout 420
```

使用 Claude Code 完成 45 道 workflow 任务的单轮首跑：

```bash
python eval/run_suite_claude.py --output eval/results/claude-workflows-latest --fresh-output --timeout 420
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
  runs/        每次运行的 trace、report、requestN.json 和 responseN.json 审计工件；长工具输出保存在 tool-results/
  memory/      跨会话长期记忆
  python-env/  普通 Python shell 命令使用的工作区隔离环境
```



