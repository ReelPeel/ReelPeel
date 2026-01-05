from pathlib import Path

from pipeline.core.models import PipelineState, SourceType
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.test_configs.rag_test import RAG_TEST_CONFIG


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

        verdict = (stmt.verdict or "unknown").upper()
        score_display = stmt.score if stmt.score is not None else "N/A"
        print(f"{icon} [ID {stmt.id}] VERDICT: {verdict} (Conf: {score_display})")
        print(f"   Claim: \"{stmt.text}\"")

        # Print Rationale (optional, usually long)
        # if stmt.rationale:
        #    print(f"   Rationale: {stmt.rationale[:150]}...")

        if stmt.evidence:
            print(f"   Evidence Used ({len(stmt.evidence)}):")
            for ev in stmt.evidence:
                # Show PMID, Type, and Weight
                # Truncate summary to one line
                summary_text = (
                    getattr(ev, "abstract", None)
                    or getattr(ev, "text", None)
                    or "No summary"
                )
                summary_snippet = summary_text[:80].replace("\n", " ")
                if ev.stance:
                    stance_label = ev.stance.abstract_label or "N/A"
                    stance_supports = ev.stance.abstract_p_supports
                    stance_refutes = ev.stance.abstract_p_refutes
                    stance_neutral = ev.stance.abstract_p_neutral
                else:
                    stance_label = "N/A"
                    stance_supports = "N/A"
                    stance_refutes = "N/A"
                    stance_neutral = "N/A"

                source_type = getattr(ev, "source_type", None)
                source_value = source_type.value if hasattr(source_type, "value") else source_type
                if source_value == SourceType.PUBMED.value:
                    source_label = "PMID"
                    source_id = getattr(ev, "pubmed_id", None) or "N/A"
                elif source_value == SourceType.EPISTEMONIKOS.value:
                    source_label = "EPIST"
                    source_id = getattr(ev, "epistemonikos_id", None) or "N/A"
                elif source_value == SourceType.RAG.value:
                    source_label = "RAG"
                    source_id = getattr(ev, "chunk_id", None) or "N/A"
                elif getattr(ev, "url", None):
                    source_label = "URL"
                    source_id = Path(ev.url).name
                else:
                    source_label = str(source_value) if source_value is not None else "Unknown"
                    source_id = "N/A"
                pub_type = getattr(ev, "pub_type", None) or "n/a"
                print(
                    f"     • {source_label} {source_id} [{pub_type}] "
                    f"(Wt: {ev.weight}, Rel: {ev.relevance}, Stance: {stance_label}, "
                    f"Stance_prob (s,r,n): {stance_supports}, {stance_refutes}, {stance_neutral}):"
                )
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
        orchestrator = PipelineOrchestrator(RAG_TEST_CONFIG)
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
