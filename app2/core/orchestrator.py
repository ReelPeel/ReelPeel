from datetime import datetime
from typing import Dict, List, Any
import time
from .models import PipelineState
from .factory import StepFactory


class PipelineOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name", "Experiment")
        self.global_debug = config.get("debug", False)

        # 1. Handle Run ID
        self.run_id = config.get("run_id")
        if not self.run_id:
            self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_file = f"pipeline_debug_{self.run_id}.log"

        self.steps = []
        for step_def in config.get("steps", []):
            self._inject_globals(step_def)
            self.steps.append(StepFactory.create(step_def))

    def _inject_globals(self, step_def: Dict):
        """
        Recursively injects 'debug' and 'log_file' into step settings.
        """
        if "settings" not in step_def:
            step_def["settings"] = {}

        # Inject Debug (if not overridden)
        if "debug" not in step_def["settings"]:
            step_def["settings"]["debug"] = self.global_debug

        # Inject Log File (always overwrite/ensure consistency for this run)
        step_def["settings"]["log_file"] = self.log_file

        # Recurse for Modules
        if step_def["type"] == "module" and "steps" in step_def["settings"]:
            for child_step in step_def["settings"]["steps"]:
                self._inject_globals(child_step)

    def run(self, initial_state: PipelineState) -> PipelineState:
        # Write header to the new log file
        msg = f"--- Launching Pipeline: {self.name} (ID={self.run_id}) ---"
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
        ordered_log = self._reorder_logs_header_style(state.execution_log)

        # Calculate Total Tokens (summing leaves only to avoid double counting modules)
        total_tokens = sum(e.get("tokens", 0) for e in state.execution_log if not e.get("is_module"))

        # Construct the summary string
        lines = [
            "\n" + "=" * 75,
            f" EXECUTION SUMMARY: {self.name}",
            f" Run ID: {self.run_id}",
            "=" * 75,
            # Updated Header with Tokens
            f"{'STEP NAME':<40} | {'DURATION':<10} | {'TOKENS'}",
            "-" * 75,
        ]

        for entry in ordered_log:
            name = str(entry.get("step", "Unknown"))
            dur = float(entry.get("duration", 0.0))
            tokens = entry.get("tokens", 0)
            indent = entry.get("indent", 0)
            is_module = entry.get("is_module", False)

            # Formatting
            prefix = "   " * indent

            if is_module:
                # Module Header Style
                display_name = f"{prefix}>> {name.upper()}"
                token_str = ""  # Blank for modules
            else:
                display_name = f"{prefix}{name}"
                token_str = f"{tokens}" if tokens > 0 else "-"

            lines.append(f"{display_name:<40} | {dur:.4f}s   | {token_str}")

        lines.append("-" * 75)
        lines.append(f"{'TOTAL PIPELINE TIME':<40} | {total_duration:.4f}s   | {total_tokens}")
        lines.append("=" * 75 + "\n")

        final_output = "\n".join(lines)

        print(final_output)
        self._append_to_log(final_output)

    def _reorder_logs_header_style(self, original_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms the flat execution log (where parents appear AFTER children)
        into a list where parents appear BEFORE children.
        """
        # Buckets for each depth level
        levels = {}

        for entry in original_log:
            depth = entry.get("indent", 0)
            is_module = entry.get("is_module", False)

            # Ensure list exists for this depth
            if depth not in levels:
                levels[depth] = []

            if is_module:
                # This is a Parent closing its block.
                # 1. Retrieve all items currently accumulated at the child level (depth + 1)
                children = levels.get(depth + 1, [])

                # 2. Reset the child level bucket (they have been consumed)
                levels[depth + 1] = []

                # 3. Add the Parent (this entry) BEFORE the children
                levels[depth].append(entry)  # Parent first
                levels[depth].extend(children)  # Then Children
            else:
                # Normal step, just append to current level
                levels[depth].append(entry)

        # The result is whatever is at depth 0 (the root)
        return levels.get(0, [])

    def _append_to_log(self, text: str):
        if self.global_debug:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(text)
            except (OSError, PermissionError) as e:
                print(f"[PipelineOrchestrator] Warning: Failed to write to log file '{self.log_file}': {e}")