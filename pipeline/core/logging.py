import json
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, Protocol, Optional

from loguru import logger


class PipelineObserver(Protocol):
    def on_run_start(self, name: str, run_id: str): ...

    def on_step_start(self, step_name: str, config: Dict[str, Any], depth: int): ...

    def on_step_end(self, step_name: str, duration: float, tokens: int, state_json: str, depth: int): ...

    def on_artifact(self, label: str, data: Any, depth: int): ...

    def on_run_end(self, duration: float): ...

    def log_summary(self, summary_text: str): ...


class PipelineLogger:
    def __init__(self, run_id: str, debug: bool = True):
        self.debug = debug
        self.run_id = run_id
        self.log_file = None
        self.prompt_log_file = None
        self._pending_prompt_entries = {}
        self._lock = threading.RLock()

        # Reset loguru to clear default handlers
        logger.remove()

        if self.debug:
            # 1. This file is in: .../pipeline/core/logging.py
            current_file = os.path.abspath(__file__)
            core_dir = os.path.dirname(current_file)
            pipeline_dir = os.path.dirname(core_dir)
            project_root = os.path.dirname(pipeline_dir)  # .../ (Root)

            # 2. Force logs to be in Root/logs/
            log_dir = os.path.join(project_root, "logs")
            os.makedirs(log_dir, exist_ok=True)

            # 3. Set the file path
            self.log_file = os.path.join(log_dir, f"pipeline_debug_{run_id}.log")
            self.prompt_log_file = os.path.join(log_dir, f"pipeline_debug_{run_id}_prompts.log")

            # Simple format: Time | Message
            fmt = "<green>{time:H:mm:ss}</green>\n{message}\n"

            logger.add(self.log_file, format=fmt, level="DEBUG")
            logger.add(sys.stderr, format=fmt, level="ERROR")

    def _format_json(self, data: Any) -> str:
        try:
            s = json.dumps(data, indent=2, default=str)
            s = s.replace("\\n", "\n      ")
            return s
        except Exception:
            return str(data)

    def _truncate_large_strings(self, obj: Any, max_len: int = 1000) -> Any:
        if isinstance(obj, str):
            if len(obj) > max_len:
                return obj[:max_len] + f"... [truncated {len(obj) - max_len} chars]"
            return obj
        if isinstance(obj, dict):
            return {k: self._truncate_large_strings(v, max_len) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_large_strings(i, max_len) for i in obj]
        return obj

    def _log(self, text: str, depth: int):
        if not self.debug: return

        step_indent = "   " * depth
        lines = text.splitlines()
        if not lines: return

        ts_pad = "" * 11
        final_msg = f"{step_indent}{lines[0]}"

        for line in lines[1:]:
            final_msg += f"\n{ts_pad}{step_indent}{line}"

        logger.debug(final_msg)

    def _append_prompt_log(self, text: str):
        if not self.debug or not self.prompt_log_file:
            return
        try:
            with self._lock:
                with open(self.prompt_log_file, "a", encoding="utf-8") as f:
                    f.write(text + "\n")
        except Exception as e:
            print(f"Prompt logging error: {e}")

    def _format_tokens_line(self, usage: Optional[Dict[str, Any]]) -> str:
        if not isinstance(usage, dict):
            return "TOKENS: unknown"
        prompt = usage.get("prompt")
        completion = usage.get("completion")
        total = usage.get("total")
        parts = [
            f"prompt={prompt}" if prompt is not None else "prompt=?",
            f"completion={completion}" if completion is not None else "completion=?",
            f"total={total}" if total is not None else "total=?",
        ]
        return "TOKENS: " + " ".join(parts)

    def _write_prompt_entry(self, timestamp: str, content: str, usage: Optional[Dict[str, Any]]):
        tokens_line = self._format_tokens_line(usage)
        msg = (
            f"{timestamp}\n"
            f">>> [LLM Prompt]\n"
            f"{tokens_line}\n"
            f"{content}\n"
            f"{'=' * 80}"
        )
        self._append_prompt_log(msg)

    def _flush_prompt_entry(self, usage: Optional[Dict[str, Any]]):
        if not self.debug or not self.prompt_log_file:
            return
        with self._lock:
            call_id = usage.get("call_id") if isinstance(usage, dict) else None
            entry = None
            if call_id and call_id in self._pending_prompt_entries:
                entry = self._pending_prompt_entries.pop(call_id)
            elif call_id is None and len(self._pending_prompt_entries) == 1:
                entry = self._pending_prompt_entries.pop(next(iter(self._pending_prompt_entries)))
            if not entry:
                return
            self._write_prompt_entry(entry["timestamp"], entry["content"], usage)

    def _flush_pending_prompt_entries(self):
        if not self.debug or not self.prompt_log_file:
            return
        with self._lock:
            if not self._pending_prompt_entries:
                return
            for entry in list(self._pending_prompt_entries.values()):
                self._write_prompt_entry(entry["timestamp"], entry["content"], usage=None)
            self._pending_prompt_entries.clear()

    # -------------------------------------------------------------------------
    # PUBLIC EVENTS
    # -------------------------------------------------------------------------

    def on_run_start(self, name: str, run_id: str):
        if not self.debug: return
        divider = "=" * 80
        msg = f"{divider}\nLAUNCHING PIPELINE: {name} (ID: {run_id})\n{divider}"
        self._log(msg, 0)

    def on_step_start(self, step_name: str, config: Dict[str, Any], depth: int):
        # 1. Format Config
        safe_conf = {k: v for k, v in config.items() if k not in ["debug", "llm_settings"]}
        conf_str = self._format_json(safe_conf)

        # 2. Build Block (Header, Newline, Settings)
        msg = (
            f"START STEP: {step_name}\n"
            f"--- SETTINGS ---\n"
            f"{conf_str}\n"
            f"----------------"
        )
        # Use hierarchy depth for START block so we see nesting
        self._log(msg, 0)

    def on_step_end(self, step_name: str, duration: float, tokens: int, state_json: str, depth: int):
        # 1. Parse & Truncate State
        try:
            state_dict = json.loads(state_json)
            clean_state = self._truncate_large_strings(state_dict)
            clean_json_str = self._format_json(clean_state)
        except Exception:
            clean_json_str = state_json

        # 2. Build Footer
        stats = f"DURATION: {duration:.4f}s"
        if tokens > 0:
            stats += f" | TOKENS: {tokens}"

        divider = "=" * 80

        # 3. Build Block
        msg = (
            f"--- OUTPUT STATE ---\n"
            f"{clean_json_str}\n"
            f"{divider}\n"
            f"FINISHED: {step_name} | {stats}\n"
            f"{divider}"
        )

        self._log(msg, 0)

    def on_artifact(self, label: str, data: Any, depth: int):
        if label == "LLM Prompt":
            if not self.debug:
                return
            with self._lock:
                timestamp = datetime.now().strftime("%H:%M:%S")
                call_id = None
                prompt_data = data
                if isinstance(data, dict):
                    call_id = data.get("call_id")
                    prompt_data = dict(data)
                    prompt_data.pop("call_id", None)
                if isinstance(prompt_data, (dict, list)):
                    content = self._format_json(prompt_data)
                else:
                    content = str(prompt_data)
                if call_id:
                    self._pending_prompt_entries[call_id] = {
                        "timestamp": timestamp,
                        "content": content,
                    }
                else:
                    self._write_prompt_entry(timestamp, content, usage=None)
            return

        if label == "LLM Usage Stats":
            self._flush_prompt_entry(data)

        if isinstance(data, (dict, list)):
            content = self._format_json(data)
        else:
            content = str(data)

        msg = f">>> [ARTIFACT] {label}\n{content}"
        self._log(msg, depth=depth)

    def on_run_end(self, duration: float):
        self._flush_pending_prompt_entries()
        divider = "=" * 80
        msg = f"{divider}\nTOTAL PIPELINE TIME: {duration:.4f}s\n{divider}"
        self._log(msg, 0)

    def log_summary(self, summary_text: str):
        if not self.debug or not self.log_file: return
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write("\n" + summary_text + "\n")
        except Exception as e:
            print(f"Logging error: {e}")
