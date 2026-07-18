

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
STOP_REASON_INVALID_NATIVE_RESPONSE = "invalid_native_response"
STOP_REASON_STEP_LIMIT_REACHED = "step_limit_reached"
STOP_REASON_RETRY_LIMIT_REACHED = "retry_limit_reached"
STOP_REASON_MODEL_ERROR = "model_error"
STOP_REASON_TOOL_TIMEOUT = "tool_timeout"
STOP_REASON_APPROVAL_DENIED = "approval_denied"
STOP_REASON_DELEGATE_FAILED = "delegate_failed"
STOP_REASON_PERSISTENCE_ERROR = "persistence_error"
STOP_REASON_RESUME_LOAD_ERROR = "resume_load_error"


@dataclass
class TaskState:
    run_id: str
    task_id: str
    user_request: str
    status: str = STATUS_RUNNING
    tool_steps: int = 0
    attempts: int = 0
    logical_steps: int = 0
    raw_tool_calls: int = 0
    raw_attempts: int = 0
    last_tool: str = ""
    last_progress_chain: str = ""
    last_progress_cursor: str = ""
    last_stall_reason: str = ""
    skill_categories: list | None = None
    loaded_skills: list | None = None
    todos: list | None = None
    active_todo_id: str = ""
    todo_version: int = 0
    last_todo_update: str = ""
    blocked_todo_id: str = ""
    planning_mode: str = "direct"
    stop_reason: str = ""
    final_answer: str = ""
    completion_mode: str = ""
    checkpoint_id: str = ""
    resume_status: str = ""

    @classmethod
    def create(cls, task_id, user_request, run_id=""):
        if not run_id:
            run_id = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]
        return cls(run_id=run_id, task_id=task_id, user_request=user_request)

    @classmethod
    def from_dict(cls, data):
        return cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            user_request=str(data.get("user_request", "")),
            status=str(data.get("status", STATUS_RUNNING)),
            tool_steps=int(data.get("tool_steps", 0)),
            attempts=int(data.get("attempts", 0)),
            logical_steps=int(data.get("logical_steps", data.get("tool_steps", 0))),
            raw_tool_calls=int(data.get("raw_tool_calls", data.get("tool_steps", 0))),
            raw_attempts=int(data.get("raw_attempts", data.get("attempts", 0))),
            last_tool=str(data.get("last_tool", "")),
            last_progress_chain=str(data.get("last_progress_chain", "")),
            last_progress_cursor=str(data.get("last_progress_cursor", "")),
            last_stall_reason=str(data.get("last_stall_reason", "")),
            skill_categories=list(data.get("skill_categories", []) or []),
            loaded_skills=list(data.get("loaded_skills", []) or []),
            todos=list(data.get("todos", []) or []),
            active_todo_id=str(data.get("active_todo_id", "")),
            todo_version=int(data.get("todo_version", 0)),
            last_todo_update=str(data.get("last_todo_update", "")),
            blocked_todo_id=str(data.get("blocked_todo_id", "")),
            planning_mode=str(
                data.get("planning_mode", "planned" if list(data.get("todos", []) or []) else "direct")
            ),
            stop_reason=str(data.get("stop_reason", "")),
            final_answer=str(data.get("final_answer", "")),
            completion_mode=str(data.get("completion_mode", "")),
            checkpoint_id=str(data.get("checkpoint_id", "")),
            resume_status=str(data.get("resume_status", "")),
        )

    def record_attempt(self):
        self.attempts += 1
        self.raw_attempts += 1
        return self

    def record_raw_tool_call(self, name):
        self.raw_tool_calls += 1
        self.last_tool = str(name or "")
        return self

    def record_logical_step(self, name):
        self.logical_steps += 1
        self.tool_steps = self.logical_steps
        self.last_tool = str(name or "")
        return self

    def record_tool(self, name):
        return self.record_logical_step(name)

    def update_progress_state(self, chain="", cursor="", stall_reason=""):
        self.last_progress_chain = str(chain or "")
        self.last_progress_cursor = str(cursor or "")
        self.last_stall_reason = str(stall_reason or "")
        return self

    def update_todo_state(
        self,
        todos=None,
        active_todo_id="",
        todo_version=None,
        last_todo_update="",
        blocked_todo_id="",
        planning_mode=None,
    ):
        self.todos = list(todos or [])
        self.active_todo_id = str(active_todo_id or "")
        if todo_version is not None:
            self.todo_version = int(todo_version)
        self.last_todo_update = str(last_todo_update or "")
        self.blocked_todo_id = str(blocked_todo_id or "")
        if planning_mode is not None:
            self.planning_mode = str(planning_mode or "direct")
        return self

    def update_skill_routing(self, skill_categories=None):
        self.skill_categories = [str(name).strip() for name in list(skill_categories or []) if str(name).strip()]
        return self

    def update_loaded_skills(self, loaded_skills=None):
        self.loaded_skills = [dict(item) for item in list(loaded_skills or []) if isinstance(item, dict)]
        return self

    def stop(self, stop_reason, status=STATUS_STOPPED, final_answer=""):
        self.status = status
        self.stop_reason = stop_reason
        if final_answer != "":
            self.final_answer = final_answer
        return self

    def stop_step_limit(self, final_answer=""):
        return self.stop(STOP_REASON_STEP_LIMIT_REACHED, final_answer=final_answer)

    def stop_retry_limit(self, final_answer=""):
        return self.stop(STOP_REASON_RETRY_LIMIT_REACHED, final_answer=final_answer)

    def stop_model_error(self, final_answer=""):
        return self.stop(STOP_REASON_MODEL_ERROR, status=STATUS_FAILED, final_answer=final_answer)

    def finish_success(
        self,
        final_answer,
        stop_reason=STOP_REASON_FINAL_ANSWER_RETURNED,
        completion_mode="native_text_answer",
    ):
        self.status = STATUS_COMPLETED
        self.stop_reason = str(stop_reason or STOP_REASON_FINAL_ANSWER_RETURNED)
        self.final_answer = str(final_answer)
        self.completion_mode = str(completion_mode or "native_text_answer")
        return self

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "user_request": self.user_request,
            "status": self.status,
            "tool_steps": self.tool_steps,
            "attempts": self.attempts,
            "logical_steps": self.logical_steps,
            "raw_tool_calls": self.raw_tool_calls,
            "raw_attempts": self.raw_attempts,
            "last_tool": self.last_tool,
            "last_progress_chain": self.last_progress_chain,
            "last_progress_cursor": self.last_progress_cursor,
            "last_stall_reason": self.last_stall_reason,
            "skill_categories": list(self.skill_categories or []),
            "loaded_skills": [dict(item) for item in list(self.loaded_skills or []) if isinstance(item, dict)],
            "todos": list(self.todos or []),
            "active_todo_id": self.active_todo_id,
            "todo_version": self.todo_version,
            "last_todo_update": self.last_todo_update,
            "blocked_todo_id": self.blocked_todo_id,
            "planning_mode": self.planning_mode,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "completion_mode": self.completion_mode,
            "checkpoint_id": self.checkpoint_id,
            "resume_status": self.resume_status,
        }
