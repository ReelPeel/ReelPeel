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

        # Define Layout Widths
        w_name = 50
        w_dur = 15
        w_tok = 10
        total_width = w_name + w_dur + w_tok + 6  # +6 for separators " | "

        # Construct the summary string
        lines = [
            "\n" + "=" * total_width,
            f" EXECUTION SUMMARY: {self.name}",
            f" Run ID: {self.run_id}",
            "=" * total_width,
            # Header
            f"{'STEP NAME':<{w_name}} | {'DURATION':<{w_dur}} | {'TOKENS':<{w_tok}}",
            "-" * total_width,
        ]

        for entry in ordered_log:
            name = str(entry.get("step", "Unknown"))
            dur = float(entry.get("duration", 0.0))
            tokens = entry.get("tokens", 0)
            indent = entry.get("indent", 0)
            is_module = entry.get("is_module", False)

            # Formatting Name
            prefix = "   " * indent
            if is_module:
                display_name = f"{prefix}>> {name.upper()}"
                token_str = ""
            else:
                display_name = f"{prefix}{name}"
                token_str = f"{tokens}" if tokens > 0 else "-"

            # Formatting Duration
            dur_str = f"{dur:.4f}s"

            lines.append(f"{display_name:<{w_name}} | {dur_str:<{w_dur}} | {token_str:<{w_tok}}")

        lines.append("-" * total_width)

        # Footer
        total_dur_str = f"{total_duration:.4f}s"
        lines.append(f"{'TOTAL PIPELINE TIME':<{w_name}} | {total_dur_str:<{w_dur}} | {total_tokens:<{w_tok}}")
        lines.append("=" * total_width + "\n")

        final_output = "\n".join(lines)

        print(final_output)
        self._append_to_log(final_output)

    def _reorder_logs_header_style(self, original_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms the flat execution log (where parents appear AFTER children)
        into a list where parents appear BEFORE children.
        """
        levels = {}

        for entry in original_log:
            depth = entry.get("indent", 0)
            is_module = entry.get("is_module", False)

            if depth not in levels:
                levels[depth] = []

            if is_module:
                children = levels.get(depth + 1, [])
                levels[depth + 1] = []
                levels[depth].append(entry)
                levels[depth].extend(children)
            else:
                levels[depth].append(entry)

        return levels.get(0, [])

    def _append_to_log(self, text: str):
        if self.global_debug:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(text)
            except (OSError, PermissionError) as e:
                print(f"[PipelineOrchestrator] Warning: Failed to write to log file '{self.log_file}': {e}")