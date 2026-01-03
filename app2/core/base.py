from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time
import json
from datetime import datetime
from .models import PipelineState


class PipelineStep(ABC):
    def __init__(self, step_config: Dict[str, Any]):
        self.config = step_config
        self.debug = self.config.get("debug", False)
        self.step_name = self.config.get("name", self.__class__.__name__)
        self.log_file = self.config.get("log_file", "pipeline_debug.log")

    def run(self, state: PipelineState) -> PipelineState:
        """
        The public interface. Wraps the logic with timing, logging, and stats tracking.
        DO NOT OVERRIDE THIS METHOD IN SUBCLASSES.
        """
        start_time = time.time()

        # 1. Run the actual logic
        try:
            # Execute the subclass's logic
            new_state = self.execute(state)
        except Exception as e:
            print(f"[ERROR] Step {self.step_name} failed: {e}")
            raise e

        # 2. Measure duration
        duration = time.time() - start_time

        is_module = isinstance(self, PipelineModule)

        # 3. Record timing stats
        new_state.execution_log.append({
            "step": self.step_name,
            "duration": duration,
            "indent": state.depth,
            "is_module": is_module
        })

        if self.debug and not is_module:
            self._write_debug_log(new_state, duration)

        return new_state

    def _write_debug_log(self, state: PipelineState, duration: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- 1. BUILD ENHANCED HEADER ---
        # Start with standard info
        header_parts = [
            f"[{timestamp}] STEP: {self.step_name}",
            f"DURATION: {duration:.4f}s"
        ]

        # Extract LLM details if available
        if "model" in self.config:
            header_parts.insert(1, f"MODEL: {self.config['model']}")
        if "temperature" in self.config:
            header_parts.insert(2, f"TEMP: {self.config['temperature']}")

        header_str = " | ".join(header_parts)
        divider = "=" * 80

        # --- 2. FORMAT SETTINGS (Prompts & Config) ---
        # We copy config to avoid mutating the actual object if we needed to filter
        config_to_log = self.config.copy()
        config_to_log.pop("debug", None)
        config_to_log.pop("log_file", None)

        # Dump to string first
        settings_dump = json.dumps(config_to_log, indent=2, default=str)

        # --- VISUAL FORMATTING ---
        # Replace the escaped JSON newline "\n" with a real newline and indentation.
        # This makes prompts appear as blocks of text in the log.
        settings_dump = settings_dump.replace("\\n", "\n      ")

        # --- 3. FORMAT STATE ---
        state_json = state.model_dump_json(indent=2)

        log_entry = (
            f"\n{divider}\n"
            f"{header_str}\n"
            f"{divider}\n"
            f"--- STEP SETTINGS & PROMPT ---\n"
            f"{settings_dump}\n\n"
            f"--- OUTPUT STATE ---\n"
            f"{state_json}\n"
        )

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
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
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{msg}\n")
        except Exception:
            pass