
当前题库定义了 78 道题：33 道核心机制题和 45 道完整工作流题。


目录结构：

```text
eval/
├─ core/        核心机制和工具组合题
├─ workflows/   完整工作流题
├─ schema/      JSON 格式规则
├─ judge/       主观题评分规则
├─ results/     已跑完的考试记录
├─ catalog.json 题库总目录
├─ run_suite.py 考试执行器
├─ validate.py  题目数据检查器
├─ references.json 题目设计参考来源
└─ README.md    评测说明书
```

| 目录 | 是什么 | 举例 |
| --- | --- | --- |
| [core](E:/pico/eval/core) | 考 Lumo 自身核心能力，而非单纯完成业务 | 能否跨会话记住约定、被中断后继续、长结果处理、后台任务、委派等 |
| [workflows](E:/pico/eval/workflows) | 考完整真实工作任务 | 删除文件、修代码 bug、写架构审查报告 |
| [schema](E:/pico/eval/schema) | 规定题目 JSON 和裁判输出 JSON 必须长什么样 | 防止有人漏写 `evaluation`、把隐藏测试暴露给 Agent |
| [judge](E:/pico/eval/judge) | 主观题的打分规范 | 要求评审模型给证据引用、按 rubric 打分 |
| [results](E:/pico/eval/results) | 每次实际考试后的保存记录 | 哪题第几次通过、生成了什么文件、失败原因、检查结果 |

核心题在 [core](E:/pico/eval/core) 下分成 6 组：

| 文件 | 测什么 | 题数 |
| --- | --- | ---: |
| [tools.json](E:/pico/eval/core/tools.json) | 多个工具组合使用，覆盖常用、边界、恢复场景 | 18 |
| [memory.json](E:/pico/eval/core/memory.json) | 跨会话稳定约定、约定更新、秘密不泄露 | 3 |
| [checkpoint.json](E:/pico/eval/core/checkpoint.json) | 中断后继续、外部修改后重新读取、非幂等操作只做一次 | 3 |
| [archive.json](E:/pico/eval/core/archive.json) | 长工具结果的归档、事实保留、异常恢复 | 3 |
| [context.json](E:/pico/eval/core/context.json) | 长历史压缩后是否保留目标、调用依赖、失败恢复 | 3 |
| [protocol.json](E:/pico/eval/core/protocol.json) | 调用 ID 关联、参数校验、重复调用控制 | 3 |

[workflows](E:/pico/eval/workflows) 是 45 道完整任务，每类 15 道，且各有 5 道简单、5 道中等、5 道困难：

| 文件 | 怎么判分 | 例子 |
| --- | --- | --- |
| [state.json](E:/pico/eval/workflows/state.json) | 比对最终文件和数据状态 | 只删除清单里的临时文件，保留受保护文件 |
| [code.json](E:/pico/eval/workflows/code.json) | 跑公开和隐藏单元测试，必须全部通过 | 修复路径 glob 匹配、异步重试、事务回滚 |
| [subjective.json](E:/pico/eval/workflows/subjective.json) | 结构/完整性检查 + 基于可信证据的语义 rubric | 无障碍审查、依赖升级风险、事故复盘、架构报告 |


```text
results/latest/
├─ run-config.json   
├─ summary.json      
├─ report.md         
├─ final-report.md   
└─ tasks/
   └─ <任务 ID>/
      └─ rep-<n>/
         ├─ result.json   
         └─ artifacts/    
```

要理解某题为什么没有通过，优先打开对应 `result.json`，看五个字段：

- `status`：`passed`（通过）、`failed`（已完成但验收不通过）或 `inconclusive`（安全看门狗、网络或评审服务等外部问题导致无法评估）。
- `passed`：本次是否通过。
- `checks`：哪条检查没过。
- `failure_label`：归类为工具错误、任务未完成、测试回归、环境错误等。
- `error`：运行时异常，例如这次遇到的 HTTP 502。

首轮报告同时给出两个不能混用的指标：

- **正确率** = `passed / 可评估任务`；可评估任务只包括 `passed` 和 `failed`，不把外部故障误算成内容错误。
- **完成率** = `可评估任务 / 已运行任务`；它反映运行是否顺利结束。

`inconclusive` 不是通过，也不是内容失败；修复外部问题后应重新运行。对配置为三轮的任务，`pass^3` 只在三轮均可评估时计算。

主观题的可信证据默认包括公开物化夹具和 Agent 提交工件。少数明确启用 `include_execution_evidence` 的题目还会向 judge 提供 harness 记录的 `run_shell` 命令、退出码和截断输出；这些内容经过脱敏，不包含隐藏测试、验收规则或模型自述。

## 运行测试

先检查题库结构、fixture、公开测试和隐藏回归测试：

```bash
python -m pytest eval/test_eval_dataset.py
python eval/validate.py
```

运行 Lumo 全部 78 题的单轮首跑：

```bash
python eval/run_suite.py --output eval/results/latest --fresh-output --provider openai --repetitions 1 --timeout 420
```

运行 Claude Code 全部 45 道 workflow 题的单轮首跑：

```bash
python eval/run_suite_claude.py --output eval/results/claude-workflows-latest --fresh-output --timeout 420
```
