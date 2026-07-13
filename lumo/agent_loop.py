"""Agent control loop extracted from the runtime facade."""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field

from .checkpoint import CHECKPOINT_NONE_STATUS
from .task_state import (
    STOP_REASON_FINAL_ANSWER_RETURNED,
    STOP_REASON_INVALID_NATIVE_RESPONSE,
    TaskState,
)
from .tool_executor import strip_tool_hints
from .model_protocol import AssistantToolCall, ModelTurnRequest, ToolResultMessage
from .tool_output import TOOL_RESULT_BATCH_LIMIT_CHARS, TOOL_RESULT_INLINE_LIMIT_CHARS
from .workspace import clip, now


@dataclass
class ProgressState:
    logical_steps_used: int = 0
    raw_model_attempts: int = 0
    raw_tool_calls: int = 0
    started_chain_keys: set = field(default_factory=set)
    last_chain_cursor_by_key: dict = field(default_factory=dict)


@dataclass
class PreparedToolResult:
    call: AssistantToolCall
    arguments: dict
    requested_args: dict
    tool_result: object
    record_progress: bool = True
    rejection_error: str = ""


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    def _report_assistant_progress(self, task_state, text, source, last_message):
        message = self.agent.assistant_progress_message(text)
        if not message or message == last_message:
            return last_message
        reported_message = self.agent.report_assistant_message(message, compact=False)
        if not reported_message:
            return last_message
        self.agent.emit_trace(
            task_state,
            "assistant_progress_reported",
            {
                "source": str(source),
                "message": reported_message,
            },
        )
        return reported_message

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

    async def _complete_model_async(self, model_request, prompt_cache_key=None, prompt_cache_retention=None):
        agent = self.agent
        return await agent.complete_turn_async(
            ModelTurnRequest.from_dict(model_request or {}, max_output_tokens=agent.max_new_tokens),
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
        )

    @staticmethod
    def _normalize_null_args(arguments):
        return {key: value for key, value in dict(arguments or {}).items() if value is not None}

    def _structured_tool_output(self, tool_result):
        agent = self.agent
        metadata = agent.redact_artifact(dict(getattr(tool_result, "metadata", {}) or {}))
        return {
            "status": str(metadata.get("tool_status", "ok")),
            "content": agent.redact_text(strip_tool_hints(str(getattr(tool_result, "content", "")))),
            "metadata": metadata,
            "error": str(metadata.get("tool_error_code", "")),
        }

    @staticmethod
    def _is_concurrency_safe(agent, call):
        tool = agent.tools.get(call.name, {})
        return bool(tool.get("concurrency_safe", False)) and not bool(tool.get("risky", True))

    def _has_same_turn_duplicates(self, calls):
        signatures = set()
        for call in calls:
            if call.error or not call.call_id or not call.name or not isinstance(call.arguments, dict):
                continue
            arguments = self._normalize_null_args(call.arguments)
            signature = (call.name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
            if signature in signatures:
                return True
            signatures.add(signature)
        return False

    @staticmethod
    def _prepare_native_rejection(call, arguments, error, content):
        from .tool_executor import ToolExecutionResult

        return PreparedToolResult(
            call=call,
            arguments=dict(arguments or {}),
            requested_args=dict(arguments or {}),
            tool_result=ToolExecutionResult(
                content=content,
                metadata={"tool_status": "rejected", "tool_error_code": error},
            ),
            record_progress=False,
            rejection_error=error,
        )

    async def _execute_native_tool_call(self, task_state, progress_state, call, seen_signatures):
        agent = self.agent
        arguments = self._normalize_null_args(call.arguments)
        requested_args = dict(arguments)
        arguments, argument_normalizations = agent.normalize_tool_arguments(call.name, arguments)
        signature = (call.name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
        if call.error or not call.call_id or not call.name or not isinstance(call.arguments, dict):
            return self._prepare_native_rejection(
                call,
                arguments,
                call.error or "invalid_tool_call",
                "error: invalid native tool call arguments",
            )
        if signature in seen_signatures:
            return self._prepare_native_rejection(
                call,
                arguments,
                "duplicate_same_turn_call",
                f"error: duplicate tool call for {call.name} in the same model turn",
            )
        seen_signatures.add(signature)
        auto_read_decision = None
        if call.name == "read_file":
            auto_read_decision = agent.auto_continue_read_file_args(arguments)
            if auto_read_decision.get("status") == "continued":
                arguments = dict(auto_read_decision.get("args", {}))
        agent.report_tool_call(call.name, arguments)
        if call.name == "read_file" and auto_read_decision and auto_read_decision.get("status") == "fully_read":
            tool_result = agent.synthetic_fully_read_result(
                path=requested_args.get("path", ""), requested_offset=auto_read_decision.get("requested_offset", 1),
                limit=auto_read_decision.get("limit", 1), coverage=auto_read_decision.get("coverage", {}),
            )
        else:
            tool_execution_context = {
                "requested_args": requested_args,
                "argument_normalizations": argument_normalizations,
            }
            if self._is_concurrency_safe(agent, call):
                tool_result = await asyncio.to_thread(
                    agent.execute_tool,
                    call.name,
                    arguments,
                    tool_execution_context,
                )
            else:
                tool_result = agent.execute_tool(
                    call.name,
                    arguments,
                    execution_context=tool_execution_context,
                )
        if call.name in {"run_shell", "task_output"}:
            tool_result = await agent.maybe_repair_shell_failure_async(call.name, arguments, tool_result)
        return PreparedToolResult(
            call=call,
            arguments=arguments,
            requested_args=requested_args,
            tool_result=tool_result,
        )

    def _commit_native_tool_result(self, task_state, progress_state, prepared, externalization_reason=""):
        agent = self.agent
        call = prepared.call
        tool_result = agent.externalize_tool_result(
            call.call_id or "invalid_call",
            call.name or "unknown",
            prepared.tool_result,
            externalization_reason,
        )
        if call.name == "use_skill" and str((tool_result.metadata or {}).get("tool_status", "")) == "ok":
            agent.register_pending_task_skill(call.call_id, task_state)
        if call.name == "read_file":
            agent.report_tool_result(call.name, prepared.arguments, tool_result.metadata, content=tool_result.content)
        output = self._structured_tool_output(tool_result)
        raw_metadata = dict(tool_result.metadata or {})
        stored_metadata = agent.redact_artifact(raw_metadata)
        archive_summary = agent.redact_text(str(raw_metadata.get("archive_summary", "")).strip())
        stored_content = agent.redact_text(strip_tool_hints(str(tool_result.content)))
        if call.name in {"read_file", "use_skill", "todo_write", "run_shell_bg", "task_output", "task_list", "task_stop"} and archive_summary:
            stored_content = archive_summary
        agent.record({
            "role": "tool", "call_id": call.call_id or "invalid_call", "name": call.name or "unknown", "args": prepared.arguments,
            "content": stored_content, "result": output, "created_at": now(), "metadata": stored_metadata,
            **({"summary": archive_summary} if archive_summary else {}),
        })
        if prepared.rejection_error:
            agent.emit_trace(
                task_state,
                "tool_rejected",
                {"call_id": call.call_id or "invalid_call", "name": call.name or "unknown", "error": prepared.rejection_error},
            )
        if prepared.record_progress:
            self._record_tool_progress(task_state, progress_state, call.name, prepared.arguments, tool_result.metadata)
        agent.emit_trace(task_state, "tool_executed", {"call_id": call.call_id, "name": call.name, "args": prepared.arguments, "requested_args": prepared.requested_args, "result": clip(str(tool_result.content), 500), **dict(tool_result.metadata or {})})
        return ToolResultMessage(call_id=call.call_id or "invalid_call", name=call.name or "unknown", output=output)

    async def _execute_native_tool_calls(self, task_state, progress_state, calls):
        agent = self.agent
        calls = list(calls or [])
        seen_signatures = set()
        results = []
        index = 0
        while index < len(calls):
            call = calls[index]
            if not self._is_concurrency_safe(agent, call):
                results.append(await self._execute_native_tool_call(task_state, progress_state, call, seen_signatures))
                index += 1
                continue
            group = []
            while index < len(calls) and self._is_concurrency_safe(agent, calls[index]):
                group.append(calls[index])
                index += 1
            if len(group) == 1 or self._has_same_turn_duplicates(group):
                results.append(await self._execute_native_tool_call(task_state, progress_state, group[0], seen_signatures))
                for duplicate_call in group[1:]:
                    results.append(await self._execute_native_tool_call(task_state, progress_state, duplicate_call, seen_signatures))
                continue
            agent.emit_trace(task_state, "tool_concurrency_batch", {"call_ids": [item.call_id for item in group], "names": [item.name for item in group]})
            results.extend(await asyncio.gather(*(self._execute_native_tool_call(task_state, progress_state, item, seen_signatures) for item in group)))
        reasons = {}
        total_inline_chars = 0
        candidates = []
        for result in results:
            size = agent.tool_result_output_chars(result.call.name, result.tool_result)
            if size > TOOL_RESULT_INLINE_LIMIT_CHARS:
                reasons[id(result)] = "per_result_limit"
            elif size:
                total_inline_chars += size
                candidates.append((size, len(candidates), result))
        if total_inline_chars > TOOL_RESULT_BATCH_LIMIT_CHARS:
            for size, _index, result in sorted(candidates, key=lambda item: (-item[0], item[1])):
                if total_inline_chars <= TOOL_RESULT_BATCH_LIMIT_CHARS:
                    break
                reasons[id(result)] = "message_budget"
                total_inline_chars -= size
        return [
            self._commit_native_tool_result(
                task_state,
                progress_state,
                result,
                reasons.get(id(result), ""),
            )
            for result in results
        ]

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

    def _finalize_run(
        self,
        task_state,
        original_user_message,
        final_answer,
        run_started_at,
        *,
        stop_reason=None,
        success_reason=None,
        completion_mode="",
    ):
        agent = self.agent
        if final_answer:
            agent.record({"role": "assistant", "content": final_answer, "created_at": now()})
        if stop_reason:
            task_state.stop(stop_reason, final_answer=final_answer)
            if completion_mode:
                task_state.completion_mode = str(completion_mode)
        elif success_reason:
            task_state.finish_success(
                final_answer,
                stop_reason=success_reason,
                completion_mode=completion_mode or "native_text_answer",
            )
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
        agent.clear_task_skills(task_state, reason="run_finished")
        agent.clear_transient_todo_state()
        agent.clear_transient_skill_route()
        return final_answer

    async def _run(self, user_message):
        agent = self.agent
        run_started_at = time.monotonic()
        original_user_message = str(user_message or "")
        agent.shell_repair_artifacts = []
        agent.tool_output_stats = {
            "externalized": 0,
            "per_result_limit": 0,
            "message_budget": 0,
            "persistence_failed": 0,
            "original_bytes": 0,
            "stored_bytes": 0,
            "artifact_truncated": 0,
        }
        agent.memory.set_task_summary(original_user_message)
        agent.record({"role": "user", "content": original_user_message, "created_at": now()})
        task_state = TaskState.create(run_id=agent.new_run_id(), task_id=agent.new_task_id(), user_request=original_user_message)
        task_state.resume_status = agent.resume_state.get("status", CHECKPOINT_NONE_STATUS)
        agent.current_task_state = task_state
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.start_task_skills(task_state)
        agent.emit_trace(task_state, "run_started", {"task_id": task_state.task_id, "user_request": clip(original_user_message, 300), "protocol": "native-v1"})

        skill_catalog = agent.refresh_skill_catalog()
        skill_categories = await agent.route_skill_categories_async(original_user_message, skill_catalog=skill_catalog)
        task_state.update_todo_state(todos=[], active_todo_id="", todo_version=0, last_todo_update="", blocked_todo_id="", planning_mode="direct")
        task_state.update_skill_routing(skill_categories)
        agent.set_transient_todo_state(todos=[], active_todo_id="", blocked_todo_id="")
        agent.set_transient_skill_route(skill_catalog, task_state.skill_categories)
        agent.emit_trace(task_state, "skill_categories_routed", dict(getattr(agent, "last_skill_routing", {}) or {}))
        progress_state = ProgressState()
        raw_attempt_backstop = max(agent.max_steps * 12, agent.max_steps + 36)
        last_assistant_progress_message = ""

        while progress_state.raw_model_attempts < raw_attempt_backstop:
            progress_state.raw_model_attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            _prompt, prompt_metadata = await agent._build_prompt_and_metadata_async(original_user_message)
            model_request = dict(agent.last_model_request or {})
            request_path = agent.run_store.request_path(
                task_state,
                progress_state.raw_model_attempts,
            )
            for skill in agent.new_task_skill_injections(
                [
                    str(item.get("source_call_id", ""))
                    for item in model_request.get("messages", [])
                    if item.get("role") == "skill_context"
                ]
            ):
                agent.emit_trace(task_state, "task_skill_injected", {**skill, "request_path": str(request_path)})
            agent.emit_trace(task_state, "request_built", {"request_path": str(request_path), "message_count": len(model_request.get("messages", [])), "tool_count": len(model_request.get("tools", [])), "duration_ms": int((time.monotonic() - prompt_started_at) * 1000)})
            prompt_cache_key = prompt_metadata.get("prompt_cache_key") if getattr(agent.model_client, "supports_prompt_cache", False) else None
            model_started_at = time.monotonic()
            try:
                response = await self._complete_model_async(model_request, prompt_cache_key=prompt_cache_key, prompt_cache_retention="in_memory" if prompt_cache_key else None)
            except Exception as exc:
                agent.run_store.write_request(
                    task_state,
                    progress_state.raw_model_attempts,
                    agent.redact_artifact(
                        {
                            "provider_request": agent.redact_task_skill_artifact(
                                dict(agent.last_provider_request or {})
                            )
                        }
                    ),
                )
                return self._finalize_run(task_state, original_user_message, f"Model error: {exc}", run_started_at, stop_reason="model_error")
            agent.run_store.write_request(
                task_state,
                progress_state.raw_model_attempts,
                agent.redact_artifact(
                    {
                        "provider_request": agent.redact_task_skill_artifact(
                            dict(agent.last_provider_request or {})
                        )
                    }
                ),
            )
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = {**prompt_metadata, **completion_metadata}
            response_path = agent.run_store.write_response(
                task_state,
                progress_state.raw_model_attempts,
                agent.redact_artifact(
                    {
                        "provider_response": dict(
                            agent.last_provider_response or response.raw_response or {}
                        )
                    }
                ),
            )
            agent.emit_trace(task_state, "model_responded", {"response_path": str(response_path), "call_ids": [call.call_id for call in response.tool_calls], "tool_calls": len(response.tool_calls), "text_chars": len(response.text), "parse_errors": list(response.parse_errors), "duration_ms": int((time.monotonic() - model_started_at) * 1000)})

            if response.tool_calls:
                assistant_record = {
                    "role": "assistant", "content": response.text, "tool_calls": [call.to_dict() for call in response.tool_calls],
                    "provider_output_items": list(response.provider_output_items), "created_at": now(),
                }
                agent.record(assistant_record)
                if response.text:
                    last_assistant_progress_message = self._report_assistant_progress(task_state, response.text, "assistant_text", last_assistant_progress_message)
                await self._execute_native_tool_calls(task_state, progress_state, response.tool_calls)
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="tool_executed")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(task_state, "checkpoint_created", {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "tool_executed"})
                continue

            final_text = str(response.text or response.refusal or "").strip()
            if not final_text:
                final_text = "Model returned neither text nor a native tool call."
                return self._finalize_run(task_state, original_user_message, final_text, run_started_at, stop_reason=STOP_REASON_INVALID_NATIVE_RESPONSE)
            return self._finalize_run(task_state, original_user_message, final_text, run_started_at, success_reason=STOP_REASON_FINAL_ANSWER_RETURNED, completion_mode="native_text_answer")

        return self._finalize_run(task_state, original_user_message, "Stopped after too many model attempts.", run_started_at, stop_reason="retry_limit_reached")
