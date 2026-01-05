from pipeline.core.models import PipelineState, SourceType
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

        print(f"{icon} [ID {stmt.id}] VERDICT: {stmt.verdict.upper()} (Conf: {stmt.score})")
        print(f"   Claim: \"{stmt.text}\"")

        # Print Rationale (optional, usually long)
        # if stmt.rationale:
        #    print(f"   Rationale: {stmt.rationale[:150]}...")

        if stmt.evidence:
            print(f"   Evidence Used ({len(stmt.evidence)}):")
            for ev in stmt.evidence:
                abstract_snippet = (getattr(ev, "abstract", None) or "No abstract")[:80].replace("\n", " ")
                stance = getattr(ev, "stance", None)
                stance_label = getattr(stance, "abstract_label", None) if stance else None
                stance_s = getattr(stance, "abstract_p_supports", None) if stance else None
                stance_r = getattr(stance, "abstract_p_refutes", None) if stance else None
                stance_n = getattr(stance, "abstract_p_neutral", None) if stance else None

                if getattr(ev, "source_type", None) == SourceType.RAG:
                    chunk_id = getattr(ev, "chunk_id", "N/A")
                    source_path = getattr(ev, "source_path", "unknown")
                    pages = getattr(ev, "pages", []) or []
                    pages_txt = f" p.{','.join(str(p) for p in pages)}" if pages else ""
                    print(
                        f"     • RAG {chunk_id} [{source_path}{pages_txt}] "
                        f"(Wt: {ev.weight}, Rel: {ev.relevance}, "
                        f"Stance: {stance_label}, "
                        f"Stance_prob (s,r,n): {stance_s}, {stance_r}, {stance_n}):"
                    )
                else:
                    pub_id = getattr(ev, "pubmed_id", None) or getattr(ev, "epistemonikos_id", None) or "N/A"
                    pub_type = getattr(ev, "pub_type", None)
                    print(
                        f"     • PMID {pub_id} [{pub_type}] "
                        f"(Wt: {ev.weight}, Rel: {ev.relevance}, "
                        f"Stance: {stance_label}, "
                        f"Stance_prob (s,r,n): {stance_s}, {stance_r}, {stance_n}):"
                    )

                print(f"       \"{abstract_snippet}...\"")
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
