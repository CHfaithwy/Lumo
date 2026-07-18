# Lumo 评测任务设计说明

本文说明 `eval/` 中 78 道题为什么这样设计，以及它们分别从哪些方面检验 Lumo。它不是给 Agent 的任务提示，也不包含私有断言、隐藏测试或参考答案。

表格中的任务名使用分组内短 ID，以减少重复文本。例如“工具组合”表中的 `repo-inventory-common` 完整 ID 是 `core.tools.repo-inventory-common.v1`；“代码修复”表中的 `path-filtering` 完整 ID 是 `workflow.code.path-filtering.v1`。每一行与 `catalog.json` 中的一道题一一对应。

## 一句话概览

- `core/` 的 33 题：验证 Lumo 自己的机制是否真的工作，例如中断恢复、长上下文、调用关联和长期约定。
- `workflows/` 的 45 题：验证 Agent 能否完成真实工程工作，例如安全改文件、修 bug、写技术报告。
- 每题都只让 Agent 看见公开工作区、当前任务和允许使用的操作；评测规则、私有文件和参考答案只在任务结束后使用。

## 统一测试流程

```text
读取题目定义
  -> 在临时目录物化公开文件
  -> Agent 根据中文任务执行
  -> 保存运行记录、最终工件和操作轨迹
  -> 挂载私有验证材料或调用评审
  -> 写入 result.json、summary.json 和报告
  -> 删除临时工作区
```

题目有三种设计情境：

| 情境 | 设计意图 |
| --- | --- |
| `common` | 常见的日常任务，确认基本工作流顺畅。 |
| `boundary` | 大输入、边界值、规模或参数限制，确认不会只在简单样例上工作。 |
| `recovery` | 路径失效、中断、外部变化或错误发生后，确认能重新观察环境并继续。 |

## 如何判分

| 题型 | 主要验收方式 | 重点防止的问题 |
| --- | --- | --- |
| 核心机制题 | 最终结果 + 运行状态/轨迹 + 专门机制断言 | 只看最终答案正确，却没有真正触发目标机制。 |
| 状态任务 | 文件、JSON、清单、引用或包产物的精确最终状态 | 多删文件、漏改关联文件、输出不可复现。 |
| 代码任务 | 公开与私有单元测试都通过，并保护既有行为 | 只针对公开 happy path 打补丁。 |
| 主观任务 | 工件存在和不可变性等硬检查 + 基于公开证据的 rubric | 空泛报告、伪造证据、遗漏关键业务约束。 |

主观题的 Judge 只看公开夹具、提交工件和少数任务启用的脱敏执行证据。它不会把 Agent 的自述当作“已经执行过命令”的证明。

## 主要能力维度

1. **任务理解与规划**：能否把多步任务拆开、按依赖顺序完成并验证。
2. **工作区操作安全性**：能否精确读写、只改允许范围、保护敏感配置和无关文件。
3. **信息定位与证据使用**：能否在大量文件、日志或代码中找到真正相关的信息，并引用事实而非猜测。
4. **工程正确性**：能否处理数据结构、跨文件一致性、边界条件、并发、事务和回归测试。
5. **长任务可靠性**：能否在中断、历史变长、路径陈旧、后台任务或重复调用后正确继续。
6. **沟通质量**：能否把事实、建议和待确认项分开，产出可执行、可审查的技术文档。

## Core：机制与工具组合（33 题）

核心题不是单纯考“最后文件对不对”，而是专门构造必须依赖某项机制才能稳定完成的条件。

### 工具组合（18 题）

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `repo-inventory-common` | 盘点仓库配置。 | 目录浏览、模式定位和精确读取能否协同。 |
| `large-evidence-boundary` | 在大量证据文件中找事实。 | 搜索分页后能否回读正确原文，而不是只看首屏。 |
| `discovery-recovery` | 原先路径已失效。 | 能否重新列目录、搜索内容并定位替代文件。 |
| `regression-diff-common` | 审阅当前改动。 | 搜索、状态检查、差异检查和敏感信息保护。 |
| `large-diff-boundary` | 大型 Git 变更集。 | 差异分页、外置结果和后续回读是否正确。 |
| `git-root-recovery` | 旧工作路径不再可用。 | 能否重新找到 Git 根目录并审阅真实未提交改动。 |
| `project-workflow-common` | 按项目规范做小改动。 | 是否按需读取分层规范，并遵守审批与工作区边界。 |
| `large-catalog-boundary` | 大型规范目录中的复杂修改。 | 分层规范路由、计划和精确修改能否同时成立。 |
| `workflow-delegation-recovery` | 多来源分析需要重新分工。 | 是否把独立分析正确委派，并在变化后继续整合。 |
| `script-migration-common` | 迁移脚本并验证。 | 多步规划、脚本修改和受管 Python 环境。 |
| `large-output-boundary` | 命令输出很长且可能含秘密。 | 长输出处理、受管环境和脱敏边界。 |
| `concurrent-edit-recovery` | 修改目标在执行中被外部更新。 | 是否重新读取最新文本，而不是按陈旧内容硬改。 |
| `background-monitor-common` | 启动后台作业并等待完成。 | 后台启动、输出读取和状态监控。 |
| `background-paging-boundary` | 后台日志很多且需要停止。 | 日志分页、定位尾部状态和受控终止。 |
| `command-recovery` | 前台长测试不适合继续等待。 | 能否切换后台执行并恢复读取结果。 |
| `cleanup-delegation-common` | 清理旧服务并独立分析。 | 停止操作的安全边界与独立子任务分工。 |
| `task-catalog-boundary` | 从大量历史作业中筛选目标。 | 大量列表分页和有界委派。 |
| `background-cleanup-recovery` | 遗留作业有的已退出、有的仍运行。 | 停止操作是否幂等，不会把已结束状态当错误。 |

### 长期记忆（3 题）

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `cross-session-convention` | 新会话继续同一项目。 | 稳定项目约定能否跨会话被正确应用。 |
| `convention-supersession-boundary` | 项目约定被更新。 | 新值能否替代旧值，且不产生重复或冲突记忆。 |
| `secret-workspace-isolation-recovery` | 任务中混入敏感值和临时状态。 | 不持久化秘密、不跨项目污染，也不把临时细节当长期事实。 |

### 检查点与继续执行（3 题）

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `partial-migration-common` | 多步迁移中途被打断。 | 恢复后只完成剩余步骤，不从头破坏已完成工作。 |
| `stale-file-boundary` | 中断期间文件被外部修改。 | 恢复时重新读取，避免基于旧内容继续推理。 |
| `exactly-once-recovery` | 已经启动过有副作用的作业。 | 恢复后不重复执行非幂等操作。 |

### 长结果归档（3 题）

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `long-result-common` | 不可过滤的长命令结果里有少量关键事实。 | 归档后能否保留精确事实并与后续信息合并。 |
| `parallel-budget-boundary` | 多个长结果同时出现。 | 每个结果是否独立关联；总量超限时是否按规则外置。 |
| `invalid-event-recovery` | 归档事件缺失、为空或重复。 | 异常事件不会误改历史，任务仍可继续。 |

### 长上下文（3 题）

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `long-history-common` | 对话历史很长。 | 缩减后是否仍保留原始目标和关键约束。 |
| `pending-closure-boundary` | 长结果仍被未结算调用依赖。 | 压缩时不会拆散调用与结果的依赖闭包。 |
| `compression-failure-recovery` | 压缩请求明确超出上下文窗口。 | 只丢弃完整旧回合、保留最新摘要，并继续完成工作。 |

### 调用协议控制（3 题）

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `parallel-correlation-common` | 并行读取多个来源。 | 调用编号、结果和原请求能否正确对应。 |
| `strict-arguments-boundary` | 参数未知或数值极大。 | 未知字段拒绝、合法大值归一化和审计记录。 |
| `duplicate-call-recovery` | 同轮出现重复副作用调用。 | 重复调用只执行一次，后续真实工作不被阻塞。 |

## Workflows：完整工程工作流（45 题）

工作流题把 Agent 放进一次性临时仓库中。它们更接近真实开发工作：既要完成目标，也要保护无关状态、运行验证并留下可检查的工件。

### 状态与文件迁移（15 题）

这组主要测“改完后的世界是否精确正确”，而不是只测某一条命令是否执行过。

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `selective-cleanup` | 按清单清理旧文件。 | 只删除指定项，保护文件绝不能误删。 |
| `config-migration` | 同步迁移服务配置、代码和文档。 | 跨文件一致性、无关设置保留和验证。 |
| `dataset-reconciliation` | 合并订单与退款数据。 | 文件发现、结构化计算和机器可验收的汇总输出。 |
| `static-assets-migration` | 移动静态资源。 | 路径移动、引用更新和无关文件保护。 |
| `env-example-normalization` | 清理示例环境变量。 | 精确配置修正，同时不碰开发者本地秘密。 |
| `documentation-link-repair` | 修复文档树中的坏链接。 | 根据真实文件证据改链接，而非凭名称猜测。 |
| `log-retention-cleanup` | 执行日志保留策略。 | 日期边界、选择性删除和审计材料保护。 |
| `localization-sync` | 同步多语言 key。 | JSON 对齐，补缺不覆盖已有正确翻译。 |
| `lockfile-workspace-cleanup` | 删除失效 workspace 条目。 | 根据当前 monorepo 布局精确修改锁文件。 |
| `api-fixture-regeneration` | 按新契约更新 API fixture。 | 多文件契约迁移与字段一致性。 |
| `monorepo-package-rename` | 重命名 monorepo 包。 | 包名、导入、路径和 workspace 配置同时正确。 |
| `database-seed-reconciliation` | 从杂乱导出重建 seed 数据。 | 去重、外键关系、过滤规则和确定性排序。 |
| `release-artifact-packaging` | 构建发布包。 | 包内容、manifest、校验和和源文件保护。 |
| `media-metadata-normalization` | 将侧车元数据汇总为目录。 | 批量标准化、输入校验和确定性聚合。 |
| `deployment-manifest-migration` | 迁移 Kubernetes API。 | 多资源语义迁移和部署配置一致性。 |

### 代码修复（15 题）

这组采用真实的小仓库 bug 和私有回归测试。重点是“修对问题且不破坏旧行为”。

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `config-precedence` | 环境配置优先级错误。 | 最小补丁、配置覆盖顺序和完整测试。 |
| `retry-idempotency` | 重试准备重复叠加状态。 | 多次重试下保持幂等，不只修首次调用。 |
| `path-filtering` | include/exclude 通配过滤错误。 | 相对路径、`*` 与 `**` 语义及排除规则。 |
| `pagination-boundary` | 一页起始编号错误。 | 一基页码、边界校验和末页行为。 |
| `cache-ttl-boundary` | TTL 临界点失效错误。 | 时钟边界的精确判断。 |
| `upload-size-validation` | 流式上传大小校验不完整。 | 按字节累计、分块输入和安全上限。 |
| `cli-boolean-parsing` | CLI 布尔值被错误解析。 | 显式 true/false、兼容性和默认行为。 |
| `frontend-stale-state` | 批量更新读取到旧前端状态。 | 函数式更新和状态批处理语义。 |
| `request-validation` | 后端输入校验可绕过。 | 空白输入、Python 类型边界和拒绝策略。 |
| `async-handler-errors` | 异步包装漏掉同步异常。 | 同步/异步错误都只转交一次。 |
| `transaction-rollback` | 转账可能半完成。 | SQLite 事务原子性、回滚和完整验证。 |
| `async-retry-cancellation` | 重试吞掉取消异常。 | 异常分类、取消传播和重试次数。 |
| `websocket-reconnect` | 重连后监听器重复。 | 生命周期清理与重复事件防护。 |
| `concurrent-memoization` | 并发计算重复执行。 | 线程安全、缓存和异常语义。 |
| `frontend-keyed-state` | 列表重排后组件状态错位。 | 稳定 key、插入和重排后的状态保持。 |

### 证据型技术报告（15 题）

这组不要求改业务代码，而要求基于仓库材料写能指导决策的文档。先检查工件存在和受保护文件未变，再由两位 Judge 按公开证据和 rubric 评分。

| 任务 | 模拟场景 | 主要测点 |
| --- | --- | --- |
| `incident-diagnosis` | 数据库连接事故复盘。 | 从日志、指标和发布记录建立因果链，并提出可执行修复。 |
| `architecture-decision` | 选择持久化作业传输方案。 | 约束驱动决策，而不是偏好驱动选型。 |
| `repo-onboarding` | 为新开发者写启动指南。 | 仓库检查、命令准确性、前置条件和验证事实区分。 |
| `frontend-accessibility-audit` | 审查结账页无障碍风险。 | 以代码证据识别键盘、标签、错误提示和对比度问题。 |
| `dependency-upgrade-risk` | 评估 React 19 升级。 | 清单、供应商说明和实际调用点驱动的风险与回退。 |
| `repository-architecture-report` | 解释订单服务架构。 | 准确追踪调用链、职责边界、配置和测试位置。 |
| `data-quality-assessment` | 分析客户导出质量。 | 精确统计、业务影响、导入前拦截和修复顺序。 |
| `deployment-readiness-review` | 审查生产部署准备度。 | 配置、健康检查、连接、资源、副本、发布暂停与回退演练。 |
| `api-migration-plan` | 规划兼容 API 迁移。 | 契约差异、客户端发现、分阶段切换和回退。 |
| `test-strategy` | 为续费和优惠券规则设计测试。 | 把变更文档中的精确业务规则转成单元、集成、并发与发布验证。 |
| `observability-gap-analysis` | 分析订单 worker 可观测性缺口。 | 从日志、指标、看板和 SLO 设计可诊断信号。 |
| `performance-investigation` | 调查商品搜索变慢。 | 用剖析、执行计划和发布时间线提出可证伪诊断与低风险缓解。 |
| `database-latency-incident` | 分析生产数据库延迟和连接池耗尽。 | 因果分析、流量控制、沟通、canary 门禁和回退阈值。 |
| `queue-backlog-incident` | 邮件队列积压并重复发送。 | 吞吐、超时、重试、幂等和分阶段修复的联合推理。 |
| `security-configuration-review` | 审查生产 Web 配置安全。 | 跨配置层找可利用路径，并给出可验证、可回退的修复顺序。 |

## 这套题不测什么

- 不以 Agent 最终自然语言自述作为成功证据；文件、测试、状态和轨迹才是主要依据。
- 不要求 Agent 猜测隐藏断言；代码题的隐藏测试只用于确认修复不是对公开样例过拟合。
- 不把服务端故障、网络故障或安全看门狗中断直接算作内容错误；它们会标记为 `inconclusive` 并单列。
- 不把一次偶然成功当作稳定能力。题目可配置多轮，报告会区分首轮正确率和连续多轮全通过率。

## 如何使用结果

1. 先看分类正确率：能区分是机制、状态修改、代码修复还是技术报告薄弱。
2. 再看单题的 `result.json`：确认失败来自任务理解、实现、验证、环境还是 Judge。
3. 对主观题，阅读 `artifacts/` 中的实际文档，再看 Judge 引用的证据；不要只看最终分数。
4. 对 `inconclusive`，修复外部条件后重跑，不把它混入模型能力结论。
