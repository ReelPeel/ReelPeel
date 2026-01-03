from .models import PipelineState
from .factory import StepFactory

class PipelineOrchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.name = config.get("name", "Experiment")
        self.steps = [StepFactory.create(s) for s in config.get("steps", [])]

    def run(self, initial_state: PipelineState) -> PipelineState:
        print(f"--- Launching Pipeline: {self.name} ---")
        state = initial_state
        for step in self.steps:
            state = step.run(state)
        return state