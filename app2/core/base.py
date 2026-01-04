import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any

from .llm import LLMService
from .models import PipelineState


class PipelineStep(ABC):
    def __init__(self, step_config: Dict[str, Any]):
        self.config = step_config
        self.debug = self.config.get("debug", False)
        self.step_name = self.config.get("name", self.__class__.__name__)
        self.log_file = self.config.get("log_file", "pipeline_debug.log")

        self._llm_service = None

    @property
    def llm(self) -> LLMService:
        if self._llm_service is None:
            # Inject debug settings into the service
            self._llm_service = LLMService(
                self.config.get("llm_settings", {}),
                debug=self.debug,
                log_file=self.log_file
            )
        return self._llm_service

    def run(self, state: PipelineState) -> PipelineState:
        """
        The public interface. Wraps the logic with timing, logging, and stats tracking.
        DO NOT OVERRIDE THIS METHOD IN SUBCLASSES.
        """
        start_time = time.time()
        is_module = isinstance(self, PipelineModule)

        # 1. START LOG (Settings & Header)
        # We write this BEFORE execution so that any artifacts (LLM calls)
        # generated during execution appear physically inside this step's block.
        if self.debug and not is_module:
            self._write_start_log()

        # 2. Run the actual logic
        try:
            new_state = self.execute(state)
        except Exception as e:
            print(f"[ERROR] Step {self.step_name} failed: {e}")
            raise e

        # 3. Measure duration
        duration = time.time() - start_time

        # 4. Capture Tokens
        tokens = 0
        if self._llm_service:
            tokens = self._llm_service.token_usage["total_tokens"]

        # 5. Record timing stats
        new_state.execution_log.append({
            "step": self.step_name,
            "duration": duration,
            "indent": state.depth,
            "is_module": is_module,
            "tokens": tokens
        })

        # 6. END LOG (State & Stats)
        # We write this AFTER execution to close the block.
        if self.debug and not is_module:
            self._write_end_log(new_state, duration, tokens)

        return new_state

    def _write_start_log(self):
        """Writes the Header and Settings block."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        divider = "=" * 160

        # Header Info
        header_parts = [f"[{timestamp}] START STEP: {self.step_name}"]

        if "model" in self.config:
            header_parts.append(f"MODEL: {self.config['model']}")
        if "temperature" in self.config:
            header_parts.append(f"TEMP: {self.config['temperature']}")

        header_str = " | ".join(header_parts)

        # Clean Config for display (remove internal keys)
        config_to_log = self.config.copy()
        config_to_log.pop("debug", None)
        config_to_log.pop("log_file", None)

        settings_dump = json.dumps(config_to_log, indent=2, default=str)
        # Visual formatting for prompts
        settings_dump = settings_dump.replace("\\n", "\n      ")

        log_entry = (
            f"\n{divider}\n"
            f"{header_str}\n"
            f"{divider}\n"
            f"--- STEP SETTINGS ---\n"
            f"{settings_dump}\n"
            f"{divider}\n"  # Closes settings block, ready for artifacts
        )
        self._append_to_file(log_entry)

    def _write_end_log(self, state: PipelineState, duration: float, tokens: int):
        """Writes the Footer and Output State block."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        divider = "=" * 160

        # Build Stats Line
        stats_parts = [f"DURATION: {duration:.4f}s"]
        if tokens > 0:
            stats_parts.append(f"TOKENS: {tokens}")
        stats_line = " | ".join(stats_parts)

        # Format State
        state_json = state.model_dump_json(indent=2)

        log_entry = (
            f"\n--- OUTPUT STATE ---\n"
            f"{state_json}\n"
            f"{divider}\n"
            f"[{timestamp}] FINISHED STEP: {self.step_name} | {stats_line}\n"
            f"{divider}\n"
        )
        self._append_to_file(log_entry)

    def log_artifact(self, label: str, data: Any):
        """
        Call this INSIDE your execute() method to log intermediate data.
        """
        if not self.debug:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Intelligent Formatting
        if isinstance(data, (dict, list)):
            try:
                formatted_data = json.dumps(data, indent=2, default=str)
                # Hack to visually indent JSON newlines
                formatted_data = formatted_data.replace("\\n", "\n")
            except Exception:
                formatted_data = str(data)
        else:
            formatted_data = str(data)

        # Apply indentation to make it look "inside" the log block
        prefix = "      "  # 6 spaces
        formatted_data = "\n".join([f"{prefix}{line}" for line in formatted_data.split("\n")])

        log_entry = (
            f"\n   >>> [ARTIFACT] {self.step_name} @ {timestamp}\n"
            f"   LABEL: {label}\n"
            f"   --------------------------------------------------\n"
            f"{formatted_data}\n"
            f"   --------------------------------------------------\n"
        )
        self._append_to_file(log_entry)

    def _append_to_file(self, text: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except (OSError, PermissionError) as e:
            print(f"[{self.step_name}] Warning: Failed to write to log file '{self.log_file}': {e}")

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        pass


class PipelineModule(PipelineStep):
    """A container that executes a sequence of internal steps."""

    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)
        self.module_name = module_config.get("name", "Unnamed Module")
        self.steps: List[PipelineStep] = []

        from .factory import StepFactory
        parent_debug = self.config.get("debug", False)
        parent_log_file = self.config.get("log_file")

        for step_def in module_config.get("steps", []):
            if "settings" not in step_def:
                step_def["settings"] = {}
            if "debug" not in step_def["settings"]:
                step_def["settings"]["debug"] = parent_debug
            if "log_file" not in step_def["settings"] and parent_log_file:
                step_def["settings"]["log_file"] = parent_log_file

            self.steps.append(StepFactory.create(step_def))

    def execute(self, state: PipelineState) -> PipelineState:
        if self.debug:
            self._log_boundary(f"=== Entering Module: {self.module_name} ===")

        state.depth += 1

        for step in self.steps:
            state = step.run(state)

        state.depth -= 1

        if self.debug:
            self._log_boundary(f"=== Exiting Module: {self.module_name} ===")

        return state

    def _log_boundary(self, msg):
        self._append_to_file(f"\n{msg}\n")