import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .llm import LLMService
from .models import PipelineState
from .logging import PipelineObserver


def _count_evidence(state: PipelineState) -> Dict[str, Any]:
    total = 0
    by_source: Dict[str, int] = {}
    statements = getattr(state, "statements", None) or []
    for stmt in statements:
        evidence = getattr(stmt, "evidence", None) or []
        total += len(evidence)
        for ev in evidence:
            source_type = getattr(ev, "source_type", None)
            if hasattr(source_type, "value"):
                source_type = source_type.value
            if not source_type:
                source_type = "Unknown"
            by_source[source_type] = by_source.get(source_type, 0) + 1
    return {"total": total, "by_source": by_source}


class PipelineStep(ABC):
    def __init__(self, step_config: Dict[str, Any]):
        self.config = step_config
        self.step_name = self.config.get("name", self.__class__.__name__)
        self.debug = self.config.get("debug", False)

        self._llm_service = None
        self._step_tokens = 0

        # This will be injected by the Orchestrator (or Parent Module)
        self.observer: Optional[PipelineObserver] = None

    @property
    def llm(self) -> LLMService:
        if self._llm_service is None:
            # Inject the observer into the LLM service so it can log usage
            self._llm_service = LLMService(
                self.config.get("llm_settings", {}),
                observer=self.observer
            )
        return self._llm_service

    def _reset_step_tokens(self) -> None:
        self._step_tokens = 0

    def add_step_tokens(self, tokens: int) -> None:
        try:
            self._step_tokens += int(tokens)
        except Exception:
            pass

    def run(self, state: PipelineState) -> PipelineState:
        """
        The standard execution wrapper.
        Handles timing, logging events, and stats tracking.
        DO NOT OVERRIDE. Override execute() instead.
        """
        start_time = time.time()
        evidence_before = _count_evidence(state)

        # 1. Notify Start
        if self.observer:
            self.observer.on_step_start(self.step_name, self.config, state.depth)

        # 2. Execute Logic
        self._reset_step_tokens()
        try:
            new_state = self.execute(state)
        except Exception as e:
            print(f"[ERROR] Step {self.step_name} failed: {e}")
            raise e
        evidence_after = _count_evidence(new_state)

        # 3. Calculate Stats
        duration = time.time() - start_time
        tokens = int(self._llm_service.token_usage["total_tokens"]) if self._llm_service else 0
        tokens += int(self._step_tokens or 0)

        # 4. Notify End
        if self.observer:
            # Serialize state here so the Logger class remains decoupled from Pydantic
            state_json = new_state.model_dump_json(indent=2)
            self.observer.on_step_end(self.step_name, duration, tokens, state_json, state.depth)

        # 5. Record internal execution stats (for the final summary table)
        new_state.execution_log.append({
            "step": self.step_name,
            "duration": duration,
            "tokens": tokens,
            "indent": state.depth,
            "is_module": isinstance(self, PipelineModule),
            "evidence_total_before": evidence_before["total"],
            "evidence_total_after": evidence_after["total"],
            "evidence_by_source_before": evidence_before["by_source"],
            "evidence_by_source_after": evidence_after["by_source"],
        })

        return new_state

    def log_artifact(self, label: str, data: Any):
        """
        Call this inside your execute() method to log intermediate data.
        """
        if self.observer:
            # We assume a default depth of 0 here; the logger will handle context
            # or rely on the fact that artifacts are usually logged during execution
            # where we might want to track 'current_depth' if strict indentation is needed.
            # For now, passing 0 lets the logger decide (or we can modify run() to track depth on self).
            self.observer.on_artifact(label, data, depth=0)

    @abstractmethod
    def execute(self, state: PipelineState) -> PipelineState:
        pass


class PipelineModule(PipelineStep):
    """A container that executes a sequence of internal steps."""

    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)
        self.steps = []

        # Import inside to avoid circular dependency
        from .factory import StepFactory

        for step_def in module_config.get("steps", []):
            # Propagate the debug flag to children
            if "settings" not in step_def: step_def["settings"] = {}
            step_def["settings"]["debug"] = self.debug

            # Create step (but don't inject observer yet, that happens at runtime)
            self.steps.append(StepFactory.create(step_def))

    def execute(self, state: PipelineState) -> PipelineState:
        state.depth += 1

        for step in self.steps:
            # Inject the module's observer into the child step
            step.observer = self.observer
            state = step.run(state)

        state.depth -= 1
        return state
