from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time
from datetime import datetime
from .models import PipelineState


class PipelineStep(ABC):
    def __init__(self, step_config: Dict[str, Any]):
        self.config = step_config
        self.debug = self.config.get("debug", False)
        self.step_name = self.config.get("name", self.__class__.__name__)
        self.log_file = "pipeline_debug.log"

    def run(self, state: PipelineState) -> PipelineState:
        start_time = time.time()

        try:
            new_state = self.execute(state)
        except Exception as e:
            # Errors should still be visible in terminal
            print(f"[ERROR] Step {self.step_name} failed: {e}")
            raise e

        duration = time.time() - start_time

        # Record timing stats
        new_state.execution_log.append({
            "step": self.step_name,
            "duration": duration
        })

        if self.debug:
            self._write_debug_log(new_state, duration)

        return new_state

    def _write_debug_log(self, state: PipelineState, duration: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"[{timestamp}] STEP: {self.step_name} | DURATION: {duration:.4f}s"
        divider = "=" * 80
        state_json = state.model_dump_json(indent=2)

        log_entry = f"\n{divider}\n{header}\n{divider}\n{state_json}\n"

        # --- CHANGED: REMOVED PRINT STATEMENT ---
        # print(f"[DEBUG] Finished {self.step_name}...") <--- Deleted

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except (OSError, PermissionError) as e:
            print(f"[{self.step_name}] Warning: Failed to write to log file '{self.log_file}': {e}")

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        pass


class PipelineModule(PipelineStep):
    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)
        self.module_name = module_config.get("name", "Unnamed Module")
        self.steps: List[PipelineStep] = []

        from .factory import StepFactory
        parent_debug = self.config.get("debug", False)

        for step_def in module_config.get("steps", []):
            if "settings" not in step_def:
                step_def["settings"] = {}
            if "debug" not in step_def["settings"]:
                step_def["settings"]["debug"] = parent_debug
            self.steps.append(StepFactory.create(step_def))

    def execute(self, state: PipelineState) -> PipelineState:
        if self.debug:
            self._log_boundary(f"=== Entering Module: {self.module_name} ===")

        for step in self.steps:
            state = step.run(state)

        if self.debug:
            self._log_boundary(f"=== Exiting Module: {self.module_name} ===")

        return state

    def _log_boundary(self, msg):
        # --- CHANGED: REMOVED PRINT STATEMENT ---
        # print(f"[DEBUG] {msg}") <--- Deleted
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{msg}\n")
        except (OSError, PermissionError) as e:
            print(f"[{self.step_name}] Warning: Failed to write to log file '{self.log_file}': {e}")