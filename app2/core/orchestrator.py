import io
import time
from datetime import datetime
from typing import Dict, List, Any

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .factory import StepFactory
from .models import PipelineState


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
        # --- 1. PREPARE THE DATA ---
        rows = []
        # Ensure we are using the full log
        ordered_log = self._reorder_logs_header_style(state.execution_log)
        total_tokens = 0

        for entry in ordered_log:
            indent = entry.get("indent", 0)
            name = str(entry.get("step", "Unknown"))
            is_module = entry.get("is_module", False)
            tokens = entry.get("tokens", 0)
            duration = float(entry.get("duration", 0.0))

            # Visual Indentation
            padding = "   " * indent

            # Format Rows
            if is_module:
                display_name = Text(f"{padding}>> {name.upper()}", style="bold magenta")
                token_display = ""
            else:
                display_name = f"{padding}{name}"
                token_display = str(tokens) if tokens > 0 else "-"
                total_tokens += tokens

            rows.append([display_name, f"{duration:.4f}s", token_display])

        # --- 2. TABLE FACTORY ---
        # We pass the 'box_style' in so Terminal gets fancy curves, File gets safe ASCII lines
        def create_table(box_style, width=None):
            table = Table(
                title=f"EXECUTION SUMMARY: {self.name}",
                title_justify="left",
                box=box_style,
                width=width,
                show_header=True
            )
            table.add_column("Step Name", justify="left", no_wrap=True)
            table.add_column("Duration", justify="right")
            table.add_column("Tokens", justify="right")

            for row in rows:
                table.add_row(*row)

            table.add_section()
            table.add_row("TOTAL PIPELINE TIME", f"{total_duration:.4f}s", str(total_tokens))
            return table

        # --- 3. PRINT TO TERMINAL (Fancy) ---
        term_console = Console()
        # Use ROUNDED for nice look on screen
        term_table = create_table(box_style=box.ROUNDED)

        # Add colors only for the terminal version
        term_table.columns[0].style = "cyan"
        term_table.columns[1].style = "magenta"
        term_table.columns[2].style = "green"

        term_console.print(term_table)

        # --- 4. PRINT TO LOG FILE (Safe & Wide) ---
        # Use ASCII to ensure borders appear in simple text files (like .log or .txt)
        string_buffer = io.StringIO()
        file_console = Console(file=string_buffer, no_color=True, width=200)

        # box.ASCII ensures you see the grid lines in the file
        file_table = create_table(box_style=box.ROUNDED)
        file_console.print(file_table)

        clean_output = string_buffer.getvalue()

        # Save
        self._append_to_log(clean_output)

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