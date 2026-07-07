"""Agent control loop extracted from the runtime facade."""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field

from .checkpoint import CHECKPOINT_NONE_STATUS, CHECKPOINT_PARTIAL_STALE_STATUS, CHECKPOINT_WORKSPACE_MISMATCH_STATUS
from .task_state import (
    STOP_REASON_FINAL_ANSWER_CALL_FAILED,
    STOP_REASON_INVALID_TODO_PROTOCOL,
    STOP_REASON_TODO_BLOCKED_WAITING_FOR_USER,
    TaskState,
)
from .tool_executor import strip_tool_hints
from .workspace import clip, now


@dataclass
class ProgressState:
    logical_steps_used: int = 0
    raw_model_attempts: int = 0
    raw_tool_calls: int = 0
    started_chain_keys: set = field(default_factory=set)
    last_chain_cursor_by_key: dict = field(default_factory=dict)


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    def run(self, user_message):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(user_message))

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(self.run_async(user_message))
            except BaseException as exc:
                result["error"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    async def run_async(self, user_message):
        return await self._run(user_message)

    async def _complete_model_async(self, prompt, prompt_cache_key=None, prompt_cache_retention=None):
        agent = self.agent
        return await agent.complete_text_async(
            prompt,
            max_new_tokens=agent.max_new_tokens,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )

    @staticmethod
    def _todo_lookup(todos):
        return {str(item.get("id", "")).strip(): item for item in list(todos or [])}

    @staticmethod
    def _todo_is_done(todo):
        return str((todo or {}).get("status", "")).strip() == "done"

    @staticmethod
    def _todos_complete(todos):
        return all(str(item.get("status", "")).strip() == "done" for item in list(todos or []))

    def _normalize_plan(self, plan, original_user_message):
        if not isinstance(plan, dict):
            return self.agent._fallback_request_plan(original_user_message, error="invalid_request_plan_object")
        rewritten_request = str(plan.get("rewritten_request", "")).strip() or str(original_user_message or "").strip()
        todos = []
        seen_ids = set()
        active_ids = []
        for item in list(plan.get("todos", []) or []):
            if not isinstance(item, dict):
                continue
            todo_id = str(item.get("id", "")).strip()
            status = str(item.get("status", "")).strip().lower()
            text = str(item.get("text", "")).strip()
            if not todo_id or todo_id in seen_ids or status not in {"active", "pending"} or not text:
                return self.agent._fallback_request_plan(original_user_message, error="invalid_request_plan_shape")
            seen_ids.add(todo_id)
            todos.append({"id": todo_id, "status": status, "text": text})
            if status == "active":
                active_ids.append(todo_id)
        if not todos or len(active_ids) != 1:
            return self.agent._fallback_request_plan(original_user_message, error="invalid_request_plan_shape")
        return {
            "rewritten_request": rewritten_request,
            "todos": todos,
            "active_todo_id": active_ids[0],
            "valid": bool(plan.get("valid", True)),
            "error": str(plan.get("error", "")).strip(),
            "raw_text": str(plan.get("raw_text", "")).strip(),
        }

    def _apply_todo_update(self, task_state, todo_update):
        operations = list((todo_update or {}).get("operations", []) or [])
        todos = [dict(item) for item in list(task_state.todos or [])]
        lookup = self._todo_lookup(todos)
        changed = False
        blocked_todo_id = ""

        for op in operations:
            action = str(op.get("op", "")).strip()
            todo_id = str(op.get("id", "")).strip()
            if action == "append":
                if not todo_id or todo_id in lookup:
                    return {"valid": False, "error": f"invalid_append:{todo_id}"}
                text = str(op.get("text", "")).strip()
                if not text:
                    return {"valid": False, "error": "empty_append_text"}
                new_item = {"id": todo_id, "status": "pending", "text": text}
                todos.append(new_item)
                lookup[todo_id] = new_item
                changed = True
                continue

            item = lookup.get(todo_id)
            if item is None:
                return {"valid": False, "error": f"unknown_todo_id:{todo_id}"}

            if action == "complete":
                if item.get("status") != "done":
                    item["status"] = "done"
                    changed = True
                continue

            if action == "activate":
                if item.get("status") == "done":
                    return {"valid": False, "error": f"cannot_activate_done:{todo_id}"}
                for current in todos:
                    if current.get("status") == "active":
                        current["status"] = "pending"
                item["status"] = "active"
                changed = True
                continue

            if action == "drop":
                if item.get("status") == "done":
                    return {"valid": False, "error": f"cannot_drop_done:{todo_id}"}
                todos = [current for current in todos if str(current.get("id", "")).strip() != todo_id]
                lookup.pop(todo_id, None)
                changed = True
                continue

            if action == "block":
                if item.get("status") == "done":
                    return {"valid": False, "error": f"cannot_block_done:{todo_id}"}
                for current in todos:
                    if current.get("status") == "active":
                        current["status"] = "pending"
                item["status"] = "blocked"
                blocked_todo_id = todo_id
                changed = True
                continue

            return {"valid": False, "error": f"unsupported_todo_op:{action}"}

        active_ids = [str(item.get("id", "")).strip() for item in todos if str(item.get("status", "")).strip() == "active"]
        blocked_ids = [str(item.get("id", "")).strip() for item in todos if str(item.get("status", "")).strip() == "blocked"]
        if len(active_ids) > 1 or len(blocked_ids) > 1:
            return {"valid": False, "error": "invalid_todo_state_shape"}
        if blocked_ids:
            blocked_todo_id = blocked_ids[0]
            active_todo_id = ""
        else:
            if not active_ids:
                for item in todos:
                    if str(item.get("status", "")).strip() == "pending":
                        item["status"] = "active"
                        active_ids = [str(item.get("id", "")).strip()]
                        changed = True
                        break
            active_todo_id = active_ids[0] if active_ids else ""

        task_state.update_todo_state(
            rewritten_request=task_state.rewritten_request,
            todos=todos,
            active_todo_id=active_todo_id,
            todo_version=int(task_state.todo_version or 0) + 1,
            last_todo_update=json.dumps(operations, ensure_ascii=False),
            blocked_todo_id=blocked_todo_id,
        )
        self.agent.set_transient_todo_state(
            rewritten_request=task_state.rewritten_request,
            todos=todos,
            active_todo_id=active_todo_id,
            blocked_todo_id=blocked_todo_id,
        )
        return {
            "valid": True,
            "changed": changed,
            "todos": todos,
            "active_todo_id": active_todo_id,
            "blocked_todo_id": blocked_todo_id,
            "all_done": self._todos_complete(todos),
        }

    @staticmethod
    def _tool_chain_key(name, args, metadata):
        args = args if isinstance(args, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        if name == "read_file":
            path = str(args.get("path", "")).strip()
            freshness = str(metadata.get("freshness", "")).strip()
            return f"read_file:{path}:{freshness or 'unknown'}"
        if name == "task_output":
            task_id = str(args.get("task_id", "")).strip()
            stream = str(args.get("stream", "stdout")).strip() or "stdout"
            return f"task_output:{task_id}:{stream}"
        if name == "git_diff":
            path = str(args.get("path", ".")).strip() or "."
            mode = str(args.get("mode", "workspace")).strip() or "workspace"
            return f"git_diff:{path}:{mode}"
        if name == "git_status":
            path = str(args.get("path", ".")).strip() or "."
            return f"git_status:{path}"
        if name == "grep":
            pattern = str(args.get("pattern", "")).strip()
            path = str(args.get("path", ".")).strip() or "."
            glob = str(args.get("glob", "")).strip()
            output_mode = str(args.get("output_mode", "content")).strip() or "content"
            return f"grep:{path}:{pattern}:{glob}:{output_mode}"
        if name == "glob":
            return f"glob:{str(args.get('path', '.')).strip() or '.'}:{str(args.get('pattern', '')).strip()}"
        if name == "list_files":
            return f"list_files:{str(args.get('path', '.')).strip() or '.'}"
        return name

    @staticmethod
    def _tool_chain_cursor(name, args, metadata):
        args = args if isinstance(args, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        if name == "read_file":
            read_window = metadata.get("read_window", {}) if isinstance(metadata.get("read_window", {}), dict) else {}
            end_line = int(read_window.get("end_line", 0) or 0)
            return f"line:{end_line}" if end_line > 0 else ""
        if name == "task_output":
            status = str(metadata.get("background_task_status", "")).strip()
            return_code = metadata.get("background_task_return_code")
            next_offset = metadata.get("background_task_next_offset")
            return f"{status}:{return_code}:{next_offset}"
        if name == "git_diff":
            externalized_path = str(metadata.get("externalized_patch_path", "")).strip()
            if externalized_path:
                return f"externalized:{externalized_path}"
            offset = args.get("offset", 1)
            limit = args.get("limit", 300)
            return f"offset:{offset}:limit:{limit}"
        if name in {"git_status", "grep", "glob", "list_files"}:
            offset = args.get("offset", 0)
            limit = args.get("limit", args.get("head_limit", 0))
            return f"offset:{offset}:limit:{limit}"
        return ""

    def _record_tool_progress(self, task_state, progress_state, name, args, metadata):
        chain_key = self._tool_chain_key(name, args, metadata)
        cursor = self._tool_chain_cursor(name, args, metadata)
        task_state.record_raw_tool_call(name)
        progress_state.raw_tool_calls += 1
        if chain_key not in progress_state.started_chain_keys:
            progress_state.started_chain_keys.add(chain_key)
            progress_state.logical_steps_used += 1
            task_state.record_logical_step(name)
        task_state.update_progress_state(chain=chain_key, cursor=cursor, stall_reason="")
        progress_state.last_chain_cursor_by_key[chain_key] = cursor

    async def _request_final_answer(self, task_state, original_user_message):
        completed = []
        for item in list(task_state.todos or []):
            if str(item.get("status", "")).strip() == "done":
                completed.append(f"- [{item.get('id', '')}] {str(item.get('text', '')).strip()}")
        prompt = "\n\n".join(
            [
                "The todo list for this task is complete.",
                "Write the final user-facing answer now.",
                "Do not return XML, <todo_update>, <display>, or <tool>.",
                "Answer directly in the user's language.",
                f"Original user request:\n{original_user_message}",
                f"Rewritten request:\n{str(task_state.rewritten_request or original_user_message).strip()}",
                "Completed todos:\n" + ("\n".join(completed) if completed else "- none"),
                "Transcript:\n" + self.agent.history_text(),
            ]
        )
        raw = await self.agent.complete_text_async(prompt, max_new_tokens=self.agent.max_new_tokens)
        answer = self.agent.extract_answer_text(raw).strip()
        if answer:
            return answer
        return str(raw or "").strip()

    def _finalize_run(self, task_state, original_user_message, final_answer, run_started_at, *, stop_reason=None):
        agent = self.agent
        if final_answer:
            agent.record({"role": "assistant", "content": final_answer, "created_at": now()})
        if stop_reason == STOP_REASON_TODO_BLOCKED_WAITING_FOR_USER:
            task_state.stop(stop_reason, final_answer=final_answer)
        elif stop_reason == STOP_REASON_FINAL_ANSWER_CALL_FAILED:
            task_state.stop(stop_reason, status="failed", final_answer=final_answer)
        elif stop_reason:
            task_state.stop(stop_reason, final_answer=final_answer)
        else:
            task_state.finish_success(final_answer)
        checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger=task_state.stop_reason or "run_finished")
        agent.run_store.write_task_state(task_state)
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "trigger": task_state.stop_reason or "run_finished",
            },
        )
        agent.emit_trace(
            task_state,
            "run_finished",
            {
                "status": task_state.status,
                "stop_reason": task_state.stop_reason,
                "final_answer": final_answer,
                "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
            },
        )
        agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
        agent.clear_transient_todo_state()
        return final_answer

    async def _run(self, user_message):
        agent = self.agent
        run_started_at = time.monotonic()
        original_user_message = str(user_message or "")
        agent.memory.set_task_summary(original_user_message)
        agent.record({"role": "user", "content": original_user_message, "created_at": now()})

        task_state = TaskState.create(
            run_id=agent.new_run_id(),
            task_id=agent.new_task_id(),
            user_request=original_user_message,
        )
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        agent.current_task_state = task_state
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(original_user_message, 300),
            },
        )

        plan = await agent.rewrite_user_message_async(original_user_message)
        plan = self._normalize_plan(plan, original_user_message)
        task_state.update_todo_state(
            rewritten_request=plan["rewritten_request"],
            todos=plan["todos"],
            active_todo_id=plan["active_todo_id"],
            todo_version=1,
            last_todo_update="",
            blocked_todo_id="",
        )
        agent.set_transient_todo_state(
            rewritten_request=plan["rewritten_request"],
            todos=plan["todos"],
            active_todo_id=plan["active_todo_id"],
            blocked_todo_id="",
        )
        rewrite_metadata = dict(getattr(agent, "last_user_request_rewrite", {}) or {})
        if rewrite_metadata.get("enabled"):
            agent.emit_trace(
                task_state,
                "user_request_rewritten",
                {
                    **rewrite_metadata,
                    "rewritten_request": clip(plan["rewritten_request"], 300),
                    "todo_count": len(plan["todos"]),
                    "active_todo_id": plan["active_todo_id"],
                },
            )

        progress_state = ProgressState()
        raw_tool_call_backstop = max(agent.max_steps * 8, agent.max_steps + 24)
        raw_attempt_backstop = max(agent.max_steps * 12, agent.max_steps + 36)
        invalid_protocol_streak = 0

        while True:
            # if progress_state.logical_steps_used >= agent.max_steps:
            #     final = "Stopped after reaching the logical step limit without completing the todo list."
            #     return self._finalize_run(task_state, original_user_message, final, run_started_at, stop_reason="step_limit_reached")
            # if progress_state.raw_tool_calls >= raw_tool_call_backstop:
            #     final = "Stopped after too many tool calls without completing the todo list."
            #     return self._finalize_run(task_state, original_user_message, final, run_started_at, stop_reason="retry_limit_reached")
            if progress_state.raw_model_attempts >= raw_attempt_backstop:
                final = "Stopped after too many model attempts without completing the todo list."
                return self._finalize_run(task_state, original_user_message, final, run_started_at, stop_reason="retry_limit_reached")

            progress_state.raw_model_attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)

            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = await agent._build_prompt_and_metadata_async(task_state.rewritten_request or original_user_message)
            prompt_path = agent.run_store.write_prompt(task_state, progress_state.raw_model_attempts, agent.redact_text(prompt))
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "prompt_path": str(prompt_path),
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="freshness_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "freshness_mismatch"})
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "workspace_mismatch"})
            elif prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="context_reduction")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "context_reduction"})

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
            if getattr(agent.model_client, "supports_prompt_cache", False):
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"

            model_started_at = time.monotonic()
            try:
                raw = await self._complete_model_async(
                    prompt,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                )
            except Exception as exc:
                final = f"Model error: {exc}"
                return self._finalize_run(task_state, original_user_message, final, run_started_at, stop_reason="model_error")

            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
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

            if kind == "retry":
                invalid_protocol_streak += 1
                agent.emit_trace(
                    task_state,
                    "invalid_todo_protocol_turn",
                    {
                        "problem": str((payload or {}).get("problem", "")),
                        "raw_excerpt": clip(str((payload or {}).get("raw_text", "")), 300),
                        "streak": invalid_protocol_streak,
                    },
                )
                if invalid_protocol_streak >= 2:
                    final = "Stopped because the model failed to follow the todo_update protocol."
                    return self._finalize_run(
                        task_state,
                        original_user_message,
                        final,
                        run_started_at,
                        stop_reason=STOP_REASON_INVALID_TODO_PROTOCOL,
                    )
                continue

            invalid_protocol_streak = 0
            todo_update = (payload or {}).get("todo_update", {}) if isinstance(payload, dict) else {}
            todo_result = self._apply_todo_update(task_state, todo_update)
            if not todo_result.get("valid"):
                agent.emit_trace(
                    task_state,
                    "invalid_todo_protocol_turn",
                    {
                        "problem": str(todo_result.get("error", "")),
                        "raw_excerpt": clip(str(raw), 300),
                        "streak": 2,
                    },
                )
                final = "Stopped because the model returned an invalid todo update."
                return self._finalize_run(
                    task_state,
                    original_user_message,
                    final,
                    run_started_at,
                    stop_reason=STOP_REASON_INVALID_TODO_PROTOCOL,
                )

            agent.emit_trace(
                task_state,
                "todo_state_updated",
                {
                    "todo_version": task_state.todo_version,
                    "active_todo_id": task_state.active_todo_id,
                    "blocked_todo_id": task_state.blocked_todo_id,
                    "todos": list(task_state.todos or []),
                    "changed": bool(todo_result.get("changed")),
                },
            )

            if kind == "tool":
                name = str(payload.get("name", "")).strip()
                args = payload.get("args", {}) if isinstance(payload.get("args", {}), dict) else {}
                original_args = dict(args)
                auto_read_decision = None
                if name == "read_file":
                    auto_read_decision = agent.auto_continue_read_file_args(args)
                    if auto_read_decision.get("status") == "continued":
                        args = dict(auto_read_decision.get("args", {}))

                tool_started_at = time.monotonic()
                agent.report_tool_call(name, args)
                if name == "read_file" and auto_read_decision and auto_read_decision.get("status") == "fully_read":
                    tool_result = agent.synthetic_fully_read_result(
                        path=original_args.get("path", ""),
                        requested_offset=auto_read_decision.get("requested_offset", 1),
                        limit=auto_read_decision.get("limit", 1),
                        coverage=auto_read_decision.get("coverage", {}),
                    )
                else:
                    tool_result = agent.execute_tool(name, args, execution_context={})
                if name == "read_file":
                    agent.report_tool_result(name, args, tool_result.metadata, content=tool_result.content)

                result = tool_result.content
                archive_summary = str((tool_result.metadata or {}).get("archive_summary", "")).strip()
                stored_content = result
                if name in {"read_file", "run_shell_bg", "task_output", "task_list", "task_stop"} and archive_summary:
                    stored_content = archive_summary
                elif name in {"git_status", "git_diff"}:
                    stored_content = strip_tool_hints(result)

                tool_record = {
                    "role": "tool",
                    "name": name,
                    "args": args,
                    "content": stored_content,
                    "created_at": now(),
                    "metadata": dict(tool_result.metadata or {}),
                }
                if archive_summary:
                    tool_record["summary"] = archive_summary
                agent.record(tool_record)
                self._record_tool_progress(task_state, progress_state, name, args, tool_result.metadata)
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "tool_executed",
                    {
                        "name": name,
                        "args": args,
                        "requested_args": original_args if name == "read_file" and auto_read_decision else args,
                        "result": clip(result, 500),
                        "duration_ms": int((time.monotonic() - tool_started_at) * 1000),
                        **dict(tool_result.metadata or {}),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="tool_executed")
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

            answer_text = str((payload or {}).get("text", "")).strip() if isinstance(payload, dict) else ""
            raw_text = str((payload or {}).get("raw_text", "")).strip() if isinstance(payload, dict) else str(raw or "").strip()
            display_text = str((payload or {}).get("display_text", "")).strip() if isinstance(payload, dict) else ""

            if kind == "todo_only":
                if str(task_state.blocked_todo_id or "").strip():
                    final = "Current todo is blocked and waiting for user input."
                    return self._finalize_run(
                        task_state,
                        original_user_message,
                        final,
                        run_started_at,
                        stop_reason=STOP_REASON_TODO_BLOCKED_WAITING_FOR_USER,
                    )
                if todo_result.get("all_done"):
                    final = (await self._request_final_answer(task_state, original_user_message)).strip()
                    if not final:
                        final = "Failed to produce a final answer after the todo list was completed."
                        return self._finalize_run(
                            task_state,
                            original_user_message,
                            final,
                            run_started_at,
                            stop_reason=STOP_REASON_FINAL_ANSWER_CALL_FAILED,
                        )
                    return self._finalize_run(task_state, original_user_message, final, run_started_at)
                if display_text:
                    agent.report_assistant_message(display_text, compact=False)
                agent.run_store.write_task_state(task_state)
                continue

            if str(task_state.blocked_todo_id or "").strip():
                final = answer_text or raw_text or "Current todo is blocked and waiting for user input."
                return self._finalize_run(
                    task_state,
                    original_user_message,
                    final,
                    run_started_at,
                    stop_reason=STOP_REASON_TODO_BLOCKED_WAITING_FOR_USER,
                )

            if todo_result.get("all_done"):
                final = answer_text or raw_text
                if not final:
                    final = (await self._request_final_answer(task_state, original_user_message)).strip()
                    if not final:
                        final = "Failed to produce a final answer after the todo list was completed."
                        return self._finalize_run(
                            task_state,
                            original_user_message,
                            final,
                            run_started_at,
                            stop_reason=STOP_REASON_FINAL_ANSWER_CALL_FAILED,
                        )
                return self._finalize_run(task_state, original_user_message, final, run_started_at)

            if answer_text:
                if display_text:
                    agent.report_assistant_message(display_text, compact=False)
                else:
                    agent.report_assistant_message(answer_text)
                agent.record_history_item({"role": "assistant", "content": answer_text, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue
