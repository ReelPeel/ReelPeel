from pipeline.core.models import PipelineState
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.test_configs.kai_test import FULL_PIPELINE_CONFIG


def print_report(state: PipelineState):
    """
    Helper to pretty-print the final pipeline results.
    """
    print("\n" + "=" * 80)
    print(f" FINAL REPORT | Overall Truthiness: {state.overall_truthiness}")
    print("=" * 80 + "\n")

    if not state.statements:
        print("No statements were processed.")
        return

    for stmt in state.statements:
        # Determine icon based on verdict
        icon = "?"
        if stmt.verdict == "true":
            icon = "✅"
        elif stmt.verdict == "false":
            icon = "❌"
        elif stmt.verdict == "uncertain":
            icon = "⚠️"
    
        print(f"{icon} [ID {stmt.id}] VERDICT: {stmt.verdict.upper()} (Conf: {stmt.confidence})")
        print(f"   Claim: \"{stmt.text}\"")
    
        # Print Rationale (optional, usually long)
        # if stmt.rationale:
        #    print(f"   Rationale: {stmt.rationale[:150]}...")
    
        if stmt.evidence:
            print(f"   Evidence Used ({len(stmt.evidence)}):")
            for ev in stmt.evidence:
                # Show PMID, Type, and Weight
                # Truncate summary to one line
                summary_snippet = (ev.abstract or ev.summary or "No summary")[:80].replace("\n", " ")
                print(f"     • PMID {ev.pubmed_id} [{ev.pub_type}] (Wt: {ev.weight})")
                print(f"       \"{summary_snippet}...\"")
        else:
            print("   (No relevant evidence found)")
    
        print("-" * 60)


def main():
    print("Initializing Pipeline System...")

    # 1. Initialize State
    # (Optional: Pass metadata or existing transcript if skipping step 1)
    state = PipelineState()

    # 2. Boot the Orchestrator
    try:
        orchestrator = PipelineOrchestrator(FULL_PIPELINE_CONFIG)
    except Exception as e:
        print(f"Configuration Error: {e}")
        return

    # 3. Run the Pipeline
    try:
        final_state = orchestrator.run(state)

        # 4. Print the Pretty Report
        print_report(final_state)

        # 5. Save Full JSON for Debugging
        with open("final_output.json", "w") as f:
            f.write(final_state.model_dump_json(indent=2))
        print(f"\nFull structured data saved to 'final_output.json'")

    except Exception as e:
        print(f"\nCRITICAL PIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()