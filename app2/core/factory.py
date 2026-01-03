from typing import Dict, Any


class StepFactory:
    _registry = {}

    @classmethod
    def register(cls, name: str, step_class):
        cls._registry[name] = step_class

    @classmethod
    def create(cls, step_def: Dict[str, Any]):
        step_type = step_def["type"]
        step_config = step_def.get("settings", {})

        # Handle the special 'module' type
        if step_type == "module":
            from base import PipelineModule
            return PipelineModule(step_def)

        step_class = cls._registry.get(step_type)
        if not step_class:
            raise ValueError(f"Step type '{step_type}' not registered.")
        return step_class(step_config)