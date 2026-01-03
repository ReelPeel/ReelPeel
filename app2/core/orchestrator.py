from typing import Dict
import time
from .models import PipelineState
from .factory import StepFactory


class PipelineOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name", "Experiment")

        # 1. Capture Global Debug Setting
        self.global_debug = config.get("debug", False)

        # 2. Inject debug flag into all steps before creation
        self.steps = []
        for step_def in config.get("steps", []):
            self._inject_global_debug(step_def)
            self.steps.append(StepFactory.create(step_def))

    def _inject_global_debug(self, step_def: Dict):
        """Recursively sets 'debug' in settings if not present."""
        if "settings" not in step_def:
            step_def["settings"] = {}

        # If not manually overridden in the step, use global
        if "debug" not in step_def["settings"]:
            step_def["settings"]["debug"] = self.global_debug

        # If it's a module, we need to ensure the module passes it down too
        # (The Module class logic we wrote handles the rest, but we ensure the module itself has it)
        if step_def["type"] == "module":
            # We also need to inject it into the children definitions inside the module config
            # because PipelineModule.__init__ reads them.
            if "steps" in step_def["settings"]:
                for child_step in step_def["settings"]["steps"]:
                    self._inject_global_debug(child_step)

    def run(self, initial_state: PipelineState) -> PipelineState:
        print(f"--- Launching Pipeline: {self.name} (Debug={self.global_debug}) ---")
        total_start = time.time()

        state = initial_state
        for step in self.steps:
            state = step.run(state)

        total_duration = time.time() - total_start

        # 3. Print Final Summary
        self._print_summary(state, total_duration)
        return state

    def _print_summary(self, state: PipelineState, total_duration: float):
        print("\n" + "=" * 60)
        print(f" EXECUTION SUMMARY: {self.name}")
        print("=" * 60)
        print(f"{'STEP NAME':<40} | {'DURATION':<10}")
        print("-" * 60)

        # Aggregate times (excluding container modules if they are logged)
        # Note: Our Base class logs *everything*, including Modules.
        # If you want to exclude 'Module' wrappers from the sum, you can filter by class name if desired.
        # For now, we list everything.

        for entry in state.execution_log:
            name = entry["step"]
            dur = entry["duration"]
            print(f"{name:<40} | {dur:.4f}s")

        print("-" * 60)
        print(f"{'TOTAL PIPELINE TIME':<40} | {total_duration:.4f}s")
        print("=" * 60 + "\n")