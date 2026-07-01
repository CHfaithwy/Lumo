"""Agent control loop extracted from the runtime facade."""

import asyncio
import threading
import time

from .checkpoint import CHECKPOINT_NONE_STATUS, CHECKPOINT_PARTIAL_STALE_STATUS, CHECKPOINT_WORKSPACE_MISMATCH_STATUS
from .task_state import TaskState
from .tool_executor import strip_tool_hints
from .workspace import clip, now


class AgentLoop:
    def __init__(self, agent):
        self.agent = agent

    async def _force_summary_reply_async(self, prompt):
        fallback_prompt = "\n\n".join(
            [
                prompt,
                "Runtime fallback:",
                "The completion-driven loop decided to stop normal planning because progress is no longer improving reliably.",
                "Stop using tools and write the best direct reply to the user now using only the evidence already available in this prompt.",
                "Respond in the user's language. Be concise and concrete.",
                "If the evidence is sufficient, answer directly.",
                "If the evidence is insufficient, clearly state what can already be concluded and what remains uncertain.",
                "Return plain answer text only. Do not use <tool> tags, <final> tags, or <completion> tags.",
            ]
        )
        return await self._complete_model_async(prompt=fallback_prompt)

    def _coerce_fallback_final(self, raw):
        raw = str(raw or "").strip()
        if not raw:
            return "I could not produce a properly formatted final answer, and the fallback summary was empty."
        if "<final>" in raw:
            final = self.agent.extract(raw, "final").strip()
            if final:
                return self.agent.strip_completion_tags(final).strip()
        return self.agent.strip_completion_tags(raw).strip()

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

    async def _run(self, user_message):
        agent = self.agent
        run_started_at = time.monotonic()
        original_user_message = str(user_message)
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
        effective_user_message = await agent.rewrite_user_message_async(original_user_message)
        rewrite_metadata = dict(getattr(agent, "last_user_request_rewrite", {}) or {})
        if rewrite_metadata.get("enabled"):
            agent.emit_trace(
                task_state,
                "user_request_rewritten",
                {
                    **rewrite_metadata,
                    "rewritten_request": clip(effective_user_message, 300),
                },
            )

        tool_steps = 0
        attempts = 0
        max_attempts = max(agent.max_steps * 3, agent.max_steps + 4)
        consecutive_retry_without_progress = 0
        forced_summary_used = False
        last_completion_score = None
        previous_scored_completion = None
        missing_completion_streak = 0

        while tool_steps < agent.max_steps and attempts < max_attempts:
            attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()

            prompt, prompt_metadata = await agent._build_prompt_and_metadata_async(effective_user_message)
            prompt_path = agent.run_store.write_prompt(task_state, attempts, agent.redact_text(prompt))
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
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif prompt_metadata.get("resume_status") == CHECKPOINT_WORKSPACE_MISMATCH_STATUS:
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(prompt_metadata.get("runtime_identity_mismatch_fields", [])),
                    },
                )
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="workspace_mismatch")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions"):
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="context_reduction")
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
            if getattr(agent.model_client, "supports_prompt_cache", False):
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_retention = "in_memory"
            model_started_at = time.monotonic()
            raw = await self._complete_model_async(
                prompt,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention=prompt_cache_retention,
            )
            completion_metadata = dict(getattr(agent.model_client, "last_completion_metadata", {}) or {})
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            kind, payload = agent.parse(raw)
            current_completion_score = None
            if isinstance(payload, dict):
                current_completion_score = payload.get("completion_score")
            if current_completion_score is not None:
                current_completion_score = int(current_completion_score)
                agent.last_completion_score = current_completion_score
                missing_completion_streak = 0
            else:
                missing_completion_streak += 1
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "previous_completion_score": previous_scored_completion,
                    "current_completion_score": current_completion_score,
                    "missing_completion_streak": missing_completion_streak,
                    "completion_metadata": completion_metadata,
                    "duration_ms": int((time.monotonic() - model_started_at) * 1000),
                },
            )

            if kind == "tool":
                consecutive_retry_without_progress = 0
                tool_steps += 1
                name = payload.get("name", "")
                args = payload.get("args", {})
                original_args = dict(args) if isinstance(args, dict) else {}
                auto_read_decision = None
                if name == "read_file":
                    auto_read_decision = agent.auto_continue_read_file_args(args)
                    if auto_read_decision.get("status") == "continued":
                        args = dict(auto_read_decision.get("args", {}))
                task_state.record_tool(name)
                tool_started_at = time.monotonic()
                if name == "read_file" and auto_read_decision and auto_read_decision.get("status") == "fully_read":
                    tool_result = agent.synthetic_fully_read_result(
                        path=original_args.get("path", ""),
                        requested_offset=auto_read_decision.get("requested_offset", 1),
                        limit=auto_read_decision.get("limit", 1),
                        coverage=auto_read_decision.get("coverage", {}),
                    )
                else:
                    agent.report_tool_call(name, args)
                    tool_result = agent.execute_tool(name, args)
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
                if current_completion_score is not None:
                    last_completion_score = current_completion_score
                    previous_scored_completion = current_completion_score
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

            if kind == "retry":
                consecutive_retry_without_progress += 1
                retry_payload = payload if isinstance(payload, dict) else {"notice": str(payload or ""), "problem": "", "raw_text": ""}
                retry_raw_text = str(retry_payload.get("raw_text", "")).strip()
                if consecutive_retry_without_progress >= 2 and not forced_summary_used:
                    forced_summary_used = True
                    agent.emit_trace(
                        task_state,
                        "forced_summary_requested",
                        {
                            "attempts": attempts,
                            "tool_steps": tool_steps,
                            "reason": "two_consecutive_non_tool_non_final_responses",
                            "previous_completion_score": previous_scored_completion,
                            "current_completion_score": current_completion_score,
                            "missing_completion_streak": missing_completion_streak,
                        },
                    )
                    fallback_started_at = time.monotonic()
                    fallback_raw = await self._force_summary_reply_async(prompt)
                    fallback_final = self._coerce_fallback_final(fallback_raw)
                    agent.record({"role": "assistant", "content": fallback_final, "created_at": now()})
                    task_state.finish_success(fallback_final)
                    checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="forced_summary")
                    agent.run_store.write_task_state(task_state)
                    agent.emit_trace(
                        task_state,
                        "checkpoint_created",
                        {
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "trigger": "forced_summary",
                        },
                    )
                    agent.emit_trace(
                        task_state,
                        "forced_summary_finished",
                        {
                            "duration_ms": int((time.monotonic() - fallback_started_at) * 1000),
                            "final_answer": fallback_final,
                        },
                    )
                    agent.emit_trace(
                        task_state,
                        "run_finished",
                        {
                            "status": task_state.status,
                            "stop_reason": task_state.stop_reason,
                            "final_answer": fallback_final,
                            "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                        },
                    )
                    agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
                    return fallback_final
                if retry_raw_text:
                    agent.report_assistant_message(retry_raw_text)
                    agent.record_history_item({"role": "assistant", "content": retry_raw_text, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue

            answer_payload = payload if isinstance(payload, dict) else {"text": str(payload or "").strip(), "completion_score": current_completion_score, "raw_text": raw}
            answer_text = str(answer_payload.get("text", "")).strip()
            consecutive_retry_without_progress = 0

            stop_reason = ""
            if current_completion_score is not None:
                if previous_scored_completion is not None and current_completion_score < previous_scored_completion:
                    stop_reason = "completion_score_declined"
                elif previous_scored_completion is not None and current_completion_score == previous_scored_completion:
                    stop_reason = "completion_score_unchanged"
                elif current_completion_score >= 95:
                    stop_reason = "completion_score_threshold"
            elif missing_completion_streak >= 2:
                stop_reason = "completion_score_missing_twice"

            if current_completion_score is not None:
                last_completion_score = current_completion_score
                previous_scored_completion = current_completion_score

            if not stop_reason:
                if answer_text:
                    agent.report_assistant_message(answer_text)
                    agent.record_history_item({"role": "assistant", "content": answer_text, "created_at": now()})
                agent.run_store.write_task_state(task_state)
                continue

            if stop_reason in {"completion_score_declined", "completion_score_unchanged", "completion_score_missing_twice"} and not forced_summary_used:
                forced_summary_used = True
                agent.emit_trace(
                    task_state,
                    "forced_summary_requested",
                    {
                        "attempts": attempts,
                        "tool_steps": tool_steps,
                        "reason": stop_reason,
                        "previous_completion_score": last_completion_score,
                        "current_completion_score": current_completion_score,
                        "missing_completion_streak": missing_completion_streak,
                    },
                )
                fallback_started_at = time.monotonic()
                fallback_raw = await self._force_summary_reply_async(prompt)
                fallback_final = self._coerce_fallback_final(fallback_raw)
                agent.record({"role": "assistant", "content": fallback_final, "created_at": now()})
                task_state.finish_success(fallback_final)
                checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="forced_summary")
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "forced_summary",
                    },
                )
                agent.emit_trace(
                    task_state,
                    "forced_summary_finished",
                    {
                        "reason": stop_reason,
                        "duration_ms": int((time.monotonic() - fallback_started_at) * 1000),
                        "final_answer": fallback_final,
                    },
                )
                agent.emit_trace(
                    task_state,
                    "run_finished",
                    {
                        "status": task_state.status,
                        "stop_reason": task_state.stop_reason,
                        "final_answer": fallback_final,
                        "completion_stop_reason": stop_reason,
                        "run_duration_ms": int((time.monotonic() - run_started_at) * 1000),
                    },
                )
                agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
                return fallback_final

            final = answer_text or self._coerce_fallback_final(raw)
            agent.record({"role": "assistant", "content": final, "created_at": now()})
            task_state.finish_success(final)
            checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger="run_finished")
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
                    "completion_stop_reason": stop_reason,
                    "current_completion_score": current_completion_score,
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
        agent.run_store.write_task_state(task_state)
        checkpoint = agent.create_checkpoint(task_state, original_user_message, trigger=task_state.stop_reason or "run_stopped")
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
