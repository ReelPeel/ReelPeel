from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .models import PipelineState


class PipelineStep(ABC):
    def __init__(self, step_config: Dict[str, Any]):
        self.config = step_config

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Execute the logic of the step."""
        pass


class PipelineModule(PipelineStep):
    """A container that executes a sequence of internal steps."""

    def __init__(self, module_config: Dict[str, Any]):
        super().__init__(module_config)
        self.module_name = module_config.get("name", "Unnamed Module")
        self.steps: List[PipelineStep] = []

        # Internal steps are initialized by the Factory during orchestrator setup
        from factory import StepFactory
        for step_def in module_config.get("steps", []):
            self.steps.append(StepFactory.create(step_def))

    def run(self, state: PipelineState) -> PipelineState:
        print(f"\n>>> Starting Module: {self.module_name}")
        for step in self.steps:
            state = step.run(state)
        print(f">>> Finished Module: {self.module_name}\n")
        return state