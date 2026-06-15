"""Agent control loop extracted from the runtime facade."""

import time

from .checkpoint import CHECKPOINT_NONE_STATUS, CHECKPOINT_PARTIAL_STALE_STATUS, CHECKPOINT_WORKSPACE_MISMATCH_STATUS
from .task_state import TaskState
from .workspace import clip, now


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    def run(self, user_message):
        agent = self.agent
        run_started_at = time.monotonic()
        # “把这次用户刚说的话，记成当前任务摘要，放进 agent 的工作记忆里。”
        agent.memory.set_task_summary(user_message)
        # 把一条完整会话消息追加到 session["history"] 里，并立刻持久化到 session 文件。实现就在你贴的这段 record() 里。
        agent.record({"role": "user", "content": user_message, "created_at": now()})
        """
        它本质上是“这次运行的状态快照”，里面会记录：
            这次任务是谁 task_id ：你输入：帮我检查 pico 为什么在 hermes-agent 下 401 这句话本身，就是“用户任务”
            这次运行是谁 run_id ：agent 接下来开始组 prompt、调模型、读文件、记 trace、最后给答案，这整段过程，就是“实际运行”
            用户原始请求是什么 user_request
            现在状态是 running/completed/stopped/failed
            调了多少次工具 tool_steps
            模型尝试了多少轮 attempts
            最后为什么停下 stop_reason
            最终答案是什么 final_answer
        """
        task_state = TaskState.create(run_id=agent.new_run_id(), task_id=agent.new_task_id(), user_request=user_message)
        
        """
        把本轮任务启动时的恢复/续跑状态，记进 task_state，后面写 trace、report 时能看见。
        当前会话是不是从旧 session 恢复来的? 当前 workspace 有没有变化? checkpoint 对不对得上? memory/file summary 有没有过期
        workspace ： 在 [pico/runtime.py (line 213)]的 refresh_prefix() 里比较：
        payload = {
            "cwd": self.cwd,
            "repo_root": self.repo_root,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
        }
        除了文件 freshness 外，还会比较 runtime_identity，字段定义在 [pico/checkpoint.py (line 12)] 的 RUNTIME_IDENTITY_KEYS：
        checkpoint：如果这次运行中途停了，我以后要恢复时，需要知道当时在干嘛、关键文件是哪些、环境是不是还一样。
        {
            "checkpoint_id": "ckpt_ab12cd34",
            "current_goal": "帮我排查 401 原因",
            "current_blocker": "",
            "next_step": "Decide the next action after read_file.",
            "key_files": [
                {"path": "pico/cli.py", "freshness": "sha256..."},
                {"path": "pico/config.py", "freshness": "sha256..."}
            ],
            "summary": "tool_executed: 帮我排查 401 原因",
            "runtime_identity": {...}
        }
        file summary ： 对文件当前内容算一个 sha256 哈希，当成“新鲜度指纹”。
        evaluate_resume_state() 里，会拿 checkpoint 里的 key_files 挨个检查：
        """
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        # 把当前任务状态挂到 agent 上
        agent.current_task_state = task_state
        # 给这次运行创建落盘目录
        agent.current_run_dir = agent.run_store.start_run(task_state)
        # 在 trace 里写下‘本次运行开始了’的第一条事件
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )

        tool_steps = 0
        attempts = 0
        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)

        # 这是 agent 的主循环，可以按“感知 -> 决策 -> 行动 -> 记录”来理解：
        # 1. 感知：重新组 prompt，把当前状态整理给模型看
        # 2. 决策：让模型返回一个工具调用，或一个最终答案
        # 3. 行动：如果是工具调用，就执行工具
        # 4. 记录：把结果写回 history / task_state / trace / memory
        # 然后进入下一轮，直到停机条件满足
        while tool_steps < agent.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            # 把当前这次运行的任务状态，实时写到磁盘上的 task_state.json 里。
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()

            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            # 如果系统发现当前恢复状态是“部分过期” (partial-stale) ，就立刻补建一个新的 checkpoint，并把这次动作记到运行状态和 trace 里。
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                # 立刻创建一个新的 checkpoint
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="freshness_mismatch")
                # 把更新后的 task_state 写回 .pico/runs/<run_id>/task_state.json
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            # 这轮在评估恢复状态时发现，当前环境和上一次 checkpoint 记录的环境不一致
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                # 既然旧 runtime identity 对不上了，就基于当前新环境重新建一个 checkpoint
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            # 如果这一轮 prompt 为了塞进预算而发生了上下文压缩，就立刻创建一个 checkpoint。
            if prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="context_reduction")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            prompt_cache_key = None
            prompt_cache_retention = None
            # prefix_state.hash hash 的其实是You are pico...Rules:Tools:Valid response examples:workspace.text()
            if getattr(agent.model_client, "supports_prompt_cache", False):
                # 只有后端明确支持时，才把稳定前缀的 hash 作为 cache key 发出去。
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            raw = agent.model_client.complete(
                prompt,
                agent.max_new_tokens,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                # 把后端返回的 usage/cache 统计并回 prompt_metadata，
                # 方便统一写入 report 和 trace。
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            kind, payload = agent.parse(raw)
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )
            """
            payload 是一个字典，长这样：
            {
                "name": "read_file",
                "args": {
                    "path": "README.md",
                    "start": 1,
                    "end": 80
                }
            }
            """
            if kind == "tool":
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                task_state.record_tool(name)
                tool_started_at = time.monotonic()
                tool_result = agent.execute_tool(name, args)
                result = tool_result.content
                agent.record(
                    {
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "content": result,
                        "created_at": now(),
                    }
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(tool_result.metadata or {}),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, user_message, trigger="tool_executed")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "tool_executed",
                    },
                )
                continue

            if kind == "retry":
                agent.record({"role": "assistant", "content": payload, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue

            final = (payload or raw).strip()
            agent.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            agent.promote_durable_memory(user_message, final)
            checkpoint = agent.create_checkpoint(task_state, user_message, trigger="run_finished")
            agent.run_store.write_task_state(task_state)
            agent.emit_trace(
                task_state,
                "checkpoint_created",
                {
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "trigger": "run_finished",
                },
            )
            agent.emit_trace(
                task_state,
                "run_finished",
                {
                    "status": task_state.status,
                    "stop_reason": task_state.stop_reason,
                    "final_answer": final,
                    "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                },
            )
            agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
            return final

        if attempts >= max_attempts and tool_steps < agent.max_steps:
            final = "Stopped after too many malformed model responses without a valid tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        agent.record({"role": "assistant", "content": final, "created_at": now()})
        """
        只有当用户请求里有这种意图时才会尝试写入,通常不会写 durable memory。：
        英文：capture / remember / save / store / persist / note
        中文：记住 / 保存 / 记录 / 沉淀 / 长期记忆 / 持久记忆
        MEMORY.md
        topics\*.md
        """
        agent.promote_durable_memory(user_message, final)
        # task_state.json 关键状态变化后，都会调用一次 write_task_state(...)，把最新状态落盘。
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, user_message, trigger=task_state.stop_reason or "run_stopped")
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_stopped",
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        return final
