from abc import ABC, abstractmethod
from typing import List, Dict, Any
import time
import os
from datetime import datetime
from .models import PipelineState


class PipelineStep(ABC):
    def __init__(self, step_config: Dict[str, Any]):
        self.config = step_config
        self.debug = self.config.get("debug", False)
        self.step_name = self.config.get("name", self.__class__.__name__)
        self.log_file = "pipeline_debug.log"

    def run(self, state: PipelineState) -> PipelineState:
        """
        Wraps logic with timing, logging, and stats tracking.
        """
        start_time = time.time()

        try:
            new_state = self.execute(state)
        except Exception as e:
            print(f"[ERROR] Step {self.step_name} failed: {e}")
            raise e

        duration = time.time() - start_time

        # --- NEW: Record timing stats in the state ---
        # We append a simple dict: {"step": name, "duration": seconds}
        new_state.execution_log.append({
            "step": self.step_name,
            "duration": duration
        })

        # Log to file if debug is enabled
        if self.debug:
            self._write_debug_log(new_state, duration)

        return new_state

    def _write_debug_log(self, state: PipelineState, duration: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"[{timestamp}] STEP: {self.step_name} | DURATION: {duration:.4f}s"
        divider = "=" * 80

        # Using model_dump_json for full details
        state_json = state.model_dump_json(indent=2)

        log_entry = f"\n{divider}\n{header}\n{divider}\n{state_json}\n"

        print(f"[DEBUG] Finished {self.step_name} in {duration:.4f}s.")

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception:
            pass

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        pass


class PipelineModule(PipelineStep):
    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)
        self.module_name = module_config.get("name", "Unnamed Module")
        self.steps: List[PipelineStep] = []

        from .factory import StepFactory

        # Inherit debug from the module config itself
        parent_debug = self.config.get("debug", False)

        for step_def in module_config.get("steps", []):
            if "settings" not in step_def:
                step_def["settings"] = {}

            # If child doesn't specify debug, enforce parent's setting
            # (Note: Orchestrator will have already pushed global debug into the module config)
            if "debug" not in step_def["settings"]:
                step_def["settings"]["debug"] = parent_debug

            self.steps.append(StepFactory.create(step_def))

    def execute(self, state: PipelineState) -> PipelineState:
        # We generally don't log module container duration in the list
        # to avoid double counting, but we can log boundaries.
        if self.debug:
            self._log_boundary(f"=== Entering Module: {self.module_name} ===")

        for step in self.steps:
            state = step.run(state)

        if self.debug:
            self._log_boundary(f"=== Exiting Module: {self.module_name} ===")

        return state

    def _log_boundary(self, msg):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{msg}\n")
        except:
            pass