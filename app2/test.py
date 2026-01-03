import json
from app2.core.models import PipelineState
from app2.core.orchestrator import PipelineOrchestrator
from app2.test_configs.test_extraction import FULL_PIPELINE_CONFIG


def main():
    print("Initializing Pipeline...")

    # 1. Initialize State
    state = PipelineState()

    # 2. Load Orchestrator with the Config
    orchestrator = PipelineOrchestrator(FULL_PIPELINE_CONFIG)

    # 3. Run
    try:
        final_state = orchestrator.run(state)

        # 4. Print Summary
        print("\n" + "=" * 50)
        print(" FINAL PIPELINE REPORT ")
        print("=" * 50)

        for stmt in final_state.statements:
            print(f"\n[Statement {stmt.id}]: {stmt.text}")
            print(f"   Query Used: {stmt.query}")
            print(f"   Evidence Found: {len(stmt.evidence)}")

            for idx, ev in enumerate(stmt.evidence, 1):
                print(f"     {idx}. [PMID {ev.pubmed_id}] (Type: {ev.pub_type}, Weight: {ev.weight})")
                print(f"        Summary: {ev.summary[:100]}...")

        # 5. Save Full Output
        with open("final_output.json", "w") as f:
            f.write(final_state.model_dump_json(indent=2))
            print(f"\nFull results saved to 'final_output.json'")

    except Exception as e:
        print(f"\nCRITICAL PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()