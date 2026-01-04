import time
import io
from datetime import datetime
from typing import Dict, List, Any

# Rich is still used for the pretty terminal table
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .factory import StepFactory
from .models import PipelineState
from .logging import PipelineLogger
from .service_manager import ensure_pubmed_proxy


class PipelineOrchestrator:
    def __init__(self, config: Dict):
        self.config = config
        self.name = config.get("name", "Experiment")
        self.run_id = config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug = config.get("debug", False)

        # 1. Initialize the Logger Service
        self.logger = PipelineLogger(self.run_id, debug=self.debug)
        ensure_pubmed_proxy()

        # 2. Build Steps
        self.steps = []
        for step_def in config.get("steps", []):
            # Inject global settings
            if "settings" not in step_def: step_def["settings"] = {}
            if "debug" not in step_def["settings"]:
                step_def["settings"]["debug"] = self.debug

            # Create step
            step = StepFactory.create(step_def)

            # INJECT LOGGER
            step.observer = self.logger

            self.steps.append(step)

    def run(self, initial_state: PipelineState) -> PipelineState:
        # Notify Start
        self.logger.on_run_start(self.name, self.run_id)
        print(f"--- Launching Pipeline: {self.name} (ID={self.run_id}) ---")

        total_start = time.time()
        state = initial_state

        # Run Loop
        for step in self.steps:
            # Ensure the logger is set (redundant but safe)
            step.observer = self.logger
            state = step.run(state)

        total_duration = time.time() - total_start

        # Notify End
        self.logger.on_run_end(total_duration)

        # Generate and print summary
        self._print_and_log_summary(state, total_duration)

        return state

    def _print_and_log_summary(self, state: PipelineState, total_duration: float):
        """
        Generates the Rich table, prints it to stdout, and logs it to file.
        """
        # 1. Prepare Data
        ordered_log = self._reorder_logs_header_style(state.execution_log)
        total_tokens = 0
        rows = []

        for entry in ordered_log:
            indent = entry.get("indent", 0)
            name = str(entry.get("step", "Unknown"))
            is_module = entry.get("is_module", False)
            tokens = entry.get("tokens", 0)
            duration = float(entry.get("duration", 0.0))

            total_tokens += tokens

            # Visual Indentation
            padding = "   " * indent

            if is_module:
                display_name = Text(f"{padding}>> {name.upper()}", style="bold magenta")
                token_display = ""
            else:
                display_name = f"{padding}{name}"
                token_display = str(tokens) if tokens > 0 else "-"

            rows.append([display_name, f"{duration:.4f}s", token_display])

        # 2. Create Rich Table
        table = Table(
            title=f"EXECUTION SUMMARY: {self.name}",
            title_justify="left",
            box=box.ROUNDED,
            show_header=True
        )
        table.add_column("Step Name", justify="left", no_wrap=True)
        table.add_column("Duration", justify="right")
        table.add_column("Tokens", justify="right")

        for row in rows:
            table.add_row(*row)

        table.add_section()
        table.add_row("TOTAL", f"{total_duration:.4f}s", str(total_tokens))

        # 3. Print to Terminal
        term_console = Console()
        term_console.print(table)

        # 4. Save to Log File (via Logger Service)
        # We render the table to a string using a wide, non-color console
        string_buffer = io.StringIO()
        file_console = Console(file=string_buffer, no_color=True, width=150)
        file_console.print(table)

        self.logger.log_summary(string_buffer.getvalue())

    def _reorder_logs_header_style(self, original_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transforms the flat execution log into a list where parents appear BEFORE children.
        """
        levels = {}
        for entry in original_log:
            depth = entry.get("indent", 0)
            is_module = entry.get("is_module", False)

            if depth not in levels: levels[depth] = []

            if is_module:
                children = levels.get(depth + 1, [])
                levels[depth + 1] = []  # Consume children
                levels[depth].append(entry)
                levels[depth].extend(children)
            else:
                levels[depth].append(entry)

        return levels.get(0, [])