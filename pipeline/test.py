from pathlib import Path

from pipeline.core.models import PipelineState
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
                summary_snippet = (ev.abstract or ev.summary or "No summary")[:80].replace("\n", " ")
                if ev.stance:
                    stance_label = ev.stance.abstract_label or ev.stance.summary_label or "N/A"
                    stance_supports = (
                        ev.stance.abstract_p_supports
                        if ev.stance.abstract_p_supports is not None
                        else ev.stance.summary_p_supports
                    )
                    stance_refutes = (
                        ev.stance.abstract_p_refutes
                        if ev.stance.abstract_p_refutes is not None
                        else ev.stance.summary_p_refutes
                    )
                    stance_neutral = (
                        ev.stance.abstract_p_neutral
                        if ev.stance.abstract_p_neutral is not None
                        else ev.stance.summary_p_neutral
                    )
                else:
                    stance_label = "N/A"
                    stance_supports = "N/A"
                    stance_refutes = "N/A"
                    stance_neutral = "N/A"
                source_value = getattr(ev, "source", "unknown")
                source_label = source_value.value if hasattr(source_value, "value") else str(source_value)
                source_id = ev.pubmed_id or (Path(ev.url).name if ev.url else "N/A")
                print(
                    f"     • {source_label} {source_id} [{ev.pub_type}] "
                    f"(Wt: {ev.weight}, Rel: {ev.relevance}, Stance: {stance_label}, "
                    f"Stance_prob (s,r,n): {stance_supports}, {stance_refutes}, {stance_neutral}):"
                )
                print(f"       \"{summary_snippet}...\"")
        else:
            print("   (No relevant evidence found)")

        if getattr(stmt, "guideline_chunks", None):
            print(f"   Guideline Chunks ({len(stmt.guideline_chunks)}):")
            for ch in stmt.guideline_chunks:
                source = Path(ch.source_path).name
                pages = ",".join(str(p) for p in ch.pages) if ch.pages else "n/a"
                chunk_snippet = ch.text[:140].replace("\n", " ")
                print(f"     • {source} p.{pages} (Score: {ch.score:.2f})")
                print(f"       \"{chunk_snippet}...\"")
        else:
            print("   (No guideline matches found)")

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
