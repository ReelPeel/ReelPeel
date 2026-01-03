import json
from app2.core.models import PipelineState
from app2.core.orchestrator import PipelineOrchestrator
from app2.test_configs.test_extraction import TEST_CONFIG_V1


def run():
    # 1. Initialize empty state (or state with metadata)
    initial_state = PipelineState()

    # 2. Boot the Orchestrator with our Test Config
    pipeline = PipelineOrchestrator(TEST_CONFIG_V1)

    # 3. Execute
    # The pipeline will flow: MockLoader -> PipelineState(transcript) -> ExtractionStep -> PipelineState(statements)
    final_state = pipeline.run(initial_state)

    # 4. Inspect Output
    print("\n" + "=" * 40)
    print(" PIPELINE RESULTS ")
    print("=" * 40)

    print(f"Transcript: {final_state.transcript[:50]}...")
    print(f"Statements Found: {len(final_state.statements)}\n")

    for stmt in final_state.statements:
        print(f"ID {stmt.id}: {stmt.text}")

    # Save to file for inspection
    with open("result_test_extraction.json", "w") as f:
        f.write(final_state.model_dump_json(indent=2))


if __name__ == "__main__":
    run()