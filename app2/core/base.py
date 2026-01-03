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
        """
        IMPLEMENT THIS METHOD instead of run().
        Contains the actual step logic.
        """
        pass


class PipelineModule(PipelineStep):
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

            # Propagate Debug
            if "debug" not in step_def["settings"]:
                step_def["settings"]["debug"] = parent_debug

            # Propagate Log File (if not already set by Orchestrator recursion)
            if "log_file" not in step_def["settings"] and parent_log_file:
                step_def["settings"]["log_file"] = parent_log_file

            self.steps.append(StepFactory.create(step_def))

    def execute(self, state: PipelineState) -> PipelineState:
        if self.debug:
            self._log_boundary(f"=== Entering Module: {self.module_name} ===")

        # --- INCREASE DEPTH FOR CHILDREN ---
        state.depth += 1

        for step in self.steps:
            state = step.run(state)

        # --- DECREASE DEPTH AFTER CHILDREN FINISH ---
        state.depth -= 1

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