# pipeline.py
import os, re, sys
import time
from datetime import datetime
import json
from .step_1_audio_to_transcript import update_transcript
from .step_2_transcript_to_statement import update_statements
from .step_3_statement_to_query import update_query
from .step_4_query_to_link import query_to_link
from .step_5_link_to_summary import link_to_summary
from .step_6_reduce_to_evidence import reduce_to_evidence
from .step_7_statement_to_truthness import statement_to_truthness
from .step_8_statment_to_score import statement_to_score
from .step_5_1_add_weights import pubtype_to_weight

from ..app2.core.preprompts import *
from .llmconfigs import *


class TeeStream:
    """
    Ein „Tee“, der in mehrere Streams gleichzeitig schreibt.
    """
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

def run_pipeline(tmp_path: str) -> dict:
    """
    1) Transcribe with Whisper
    2) Feed the transcript to an LLM (e.g. fact-check, summary, follow-up Q&A…)
    3) Return a dict you’ll JSON-encode later
    """

    # ─────────────────────────────────────────────────────────────────────────────
    # Log-Datei erzeugen und stdout/stderr „teeben“ (nur Datum + Stunden+Minuten)
    # ─────────────────────────────────────────────────────────────────────────────
    # Format: YYYYMMDD_HHMM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"log_{timestamp}.txt"
    log_file = open(log_filename, "w", encoding="utf-8")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    tee = TeeStream(original_stdout, log_file)
    sys.stdout = tee
    sys.stderr = tee

    # ─────────────────────────────────────────────────────────────────────────────
    # Print LLM Configs
    # ─────────────────────────────────────────────────────────────────────────────
    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- LLM Configs --- ")
    print(" \n --------------------------------------------------------------- \n")

    # Allgemeine Basis-Settings
    print(f"BASE_URL   : {OPENAI_BASE_URL}")
    print(f"API_KEY    : {BASE_API_KEY}")
    print(f"BASE_MODEL : {BASE_MODEL}")
    print(f"BASE_TEMP  : {BASE_TEMP}\n")

    # STEP2
    print(" STEP2:")
    print(f"   MODEL      = {MODEL_2}")
    print(f"   TEMP       = {TEMP_2}")
    print(f"   MAX_TOKENS = {MAX_TOKENS_2}\n")

    # STEP3
    print(" STEP3:")
    print(f"   MODEL      = {MODEL_3}")
    print(f"   TEMP       = {TEMP_3}")
    print(f"   MAX_TOKENS = {MAX_TOKENS_3}\n")

    # STEP5
    print(" STEP5:")
    print(f"   MODEL      = {MODEL_5}")
    print(f"   TEMP       = {TEMP_5}")
    print(f"   MAX_TOKENS = {MAX_TOKENS_5}\n")

    # STEP6
    print(" STEP6:")
    print(f"   MODEL      = {MODEL_6}")
    print(f"   TEMP       = {TEMP_6}")
    print(f"   MAX_TOKENS = {MAX_TOKENS_6}\n")

    # STEP7
    print(" STEP7:")
    print(f"   MODEL      = {MODEL_7}")
    print(f"   TEMP       = {TEMP_7}")
    print(f"   MAX_TOKENS = {MAX_TOKENS_7}\n")


    # ─────────────────────────────────────────────────────────────────────────────
    # Print LLM Preprompts
    # ─────────────────────────────────────────────────────────────────────────────
    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- LLM Preprompts --- ")
    print(" \n --------------------------------------------------------------- \n")

    # STEP2
    print("Preprompt STEP2:")
    print(PROMPT_TMPL_S2)
    print("\n\n")
    # STEP3
    print("Prepromt STEP3:")
    print(PROMPT_TMPL_S3_NARROW_QUERY)
    print("\n\n")
    # STEP5
    print("Preprompt STEP5:")
    print(PROMPT_TMPL_S5)
    print("\n\n")
    # STEP6
    
    print("Preprompt STEP6:")
    print(PROMPT_TMPL_S6)
    print("\n\n")

    # STEP7
    print("Prepromt STEP7:")
    print(PROMPT_TMPL_S7)
    print("\n\n")


    # ─────────────────────────────────────────────────────────────────────────────
    # Pipeline ausführen (alle print() und Fehlermeldungen gehen jetzt in Terminal UND Logdatei)
    # ─────────────────────────────────────────────────────────────────────────────
    start = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step1 --- ")
    print(" \n --------------------------------------------------------------- \n")
    t0 = time.time()
    transcript = update_transcript("app/json_example.json", tmp_path)
    t1 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step2 --- ")
    print(" \n --------------------------------------------------------------- \n")
    statments = update_statements(transcript)
    t2 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step3 --- ")
    print(" \n --------------------------------------------------------------- \n")
    query = update_query(statments)
    t3 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step4 --- ")
    print(" \n --------------------------------------------------------------- \n")
    link = query_to_link(query)
    t4 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step5 --- ")
    print(" \n --------------------------------------------------------------- \n")
    summary = link_to_summary(link)
    t5 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step6 --- ")
    print(" \n --------------------------------------------------------------- \n")
    weighted = pubtype_to_weight(summary, default=0.5)
    t6 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step7 --- ")
    print(" \n --------------------------------------------------------------- \n")
    evidence = reduce_to_evidence(weighted)
    t7 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step8 --- ")
    print(" \n --------------------------------------------------------------- \n")
    truthness = statement_to_truthness(evidence)
    t8 = time.time()

    print(" \n\n\n\n\n\n  --------------------------------------------------------------- \n")
    print(" --- Step9 --- ")
    print(" \n --------------------------------------------------------------- \n")
    scores = statement_to_score(truthness)
    t9 = time.time()

    elapsed = t9 - start

    print(" \n\n\n\n\n\n --------------------------------------------------------------- \n")
    print(" --- Runtime Analysis ---")
    print(" \n --------------------------------------------------------------- \n")
    print(f"update_transcript: {t1 - t0:.6f} seconds")
    print(f"update_statements: {t2 - t1:.6f} seconds")
    print(f"update_query: {t3 - t2:.6f} seconds")
    print(f"query_to_link: {t4 - t3:.6f} seconds")
    print(f"link_to_summary: {t5 - t4:.6f} seconds")
    print(f"pubtype_to_weight: {t6 - t5:.6f} seconds")
    print(f"reduce_to_evidence: {t7 - t6:.6f} seconds")
    print(f"statement_to_truthness: {t8 - t7:.6f} seconds")
    print(f"statement_to_score: {t9 - t8:.6f} seconds")
    print(f"Total elapsed time: {elapsed:.6f} seconds")

    print(" \n\n\n\n\n\n --------------------------------------------------------------- \n")
    print(" --- Final Output (JSON) ---")
    print(" \n --------------------------------------------------------------- \n")
    print(scores)


    # ─────────────────────────────────────────────────────────────────────────────
    # Am Ende: Streams zurücksetzen und Logdatei schließen
    # ─────────────────────────────────────────────────────────────────────────────
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()

    return scores
