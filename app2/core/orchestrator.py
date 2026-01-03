from typing import Dict
import time
from .models import PipelineState
from .factory import StepFactory


class PipelineOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name", "Experiment")
        self.global_debug = config.get("debug", False)
        self.log_file = "pipeline_debug.log"  # Define log file here too

        self.steps = []
        for step_def in config.get("steps", []):
            self._inject_global_debug(step_def)
            self.steps.append(StepFactory.create(step_def))

    def _inject_global_debug(self, step_def: Dict):
        if "settings" not in step_def:
            step_def["settings"] = {}
        if "debug" not in step_def["settings"]:
            step_def["settings"]["debug"] = self.global_debug
        if step_def["type"] == "module" and "steps" in step_def["settings"]:
            for child_step in step_def["settings"]["steps"]:
                self._inject_global_debug(child_step)

    def run(self, initial_state: PipelineState) -> PipelineState:
        # Capture launch message
        msg = f"--- Launching Pipeline: {self.name} (Debug={self.global_debug}) ---"
        print(msg)
        self._append_to_log(f"\n{msg}\n")

        total_start = time.time()

        state = initial_state
        for step in self.steps:
            state = step.run(state)

        total_duration = time.time() - total_start

        self._print_summary(state, total_duration)
        return state

    def _print_summary(self, state: PipelineState, total_duration: float):
        # Construct the summary string
        lines = ["\n" + "=" * 60, f" EXECUTION SUMMARY: {self.name}", "=" * 60, f"{'STEP NAME':<40} | {'DURATION':<10}",
                 "-" * 60]

        for entry in state.execution_log:
            # Ensure we handle both string and float correctly
            name = str(entry.get("step", "Unknown"))
            dur = float(entry.get("duration", 0.0))
            lines.append(f"{name:<40} | {dur:.4f}s")

        lines.append("-" * 60)
        lines.append(f"{'TOTAL PIPELINE TIME':<40} | {total_duration:.4f}s")
        lines.append("=" * 60 + "\n")

        final_output = "\n".join(lines)

        # 1. Print to Terminal
        print(final_output)

        # 2. Append to Log File
        self._append_to_log(final_output)

    def _append_to_log(self, text: str):
        """Helper to write to the debug log."""
        if self.global_debug:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(text)
            except (IOError, OSError, PermissionError) as e:
                print(f"[PipelineOrchestrator] Warning: Failed to write to log file '{self.log_file}': {e}")