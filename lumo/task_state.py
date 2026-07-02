"""一次 ask() 运行过程中的状态机快照。

它回答的是：这次用户请求当前进行到哪了、调了多少次工具、最后为什么停下。
这个对象会被不断写入 task_state.json，供运行中观察和运行后复盘。
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

STOP_REASON_FINAL_ANSWER_RETURNED = "final_answer_returned"
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
    stop_reason: str = ""
    final_answer: str = ""
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
            stop_reason=str(data.get("stop_reason", "")),
            final_answer=str(data.get("final_answer", "")),
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

    def finish_success(self, final_answer):
        self.status = STATUS_COMPLETED
        self.stop_reason = STOP_REASON_FINAL_ANSWER_RETURNED
        self.final_answer = str(final_answer)
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
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "checkpoint_id": self.checkpoint_id,
            "resume_status": self.resume_status,
        }
