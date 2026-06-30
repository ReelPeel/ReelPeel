# main.py ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, Body, HTTPException
import asyncio
import shutil, os
from pathlib import Path
from .pipeline import run_pipeline, run_pipeline_from_transcript
  # ← your existing heavy pipeline
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import random
import json
from fastapi.middleware.cors import CORSMiddleware
from pipeline.core.llm import LLMService
from pipeline.test_configs.preprompts import PROMPT_TMPL_EVIDENCE_SUMMARY
app = FastAPI(title="One-shot reel-to-pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # or a list of specific origins
    allow_methods=["*"], # GET, POST, etc.
    allow_headers=["*"], # e.g. Content-Type
    expose_headers=["*"],
)

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemma3:12b")
SUMMARY_TEMPERATURE = float(os.getenv("SUMMARY_TEMPERATURE", "0.0"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "120"))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "2000"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
REPO_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_MOCK_ROOT = REPO_ROOT / "offline_mock"
OFFLINE_TRANSCRIPT_REEL_IDS = {"DT0UIgzDZ79", "DT0UbkjDZZj"}
OFFLINE_TRANSCRIPT_TEMPERATURE = float(os.getenv("OFFLINE_TRANSCRIPT_TEMPERATURE", "0.7"))


def extract_reel_id(instagram_url: str) -> str:
    """
    Return the segment that follows /reels/ in an Instagram Reel URL.

    Examples
    --------
    >>> extract_reel_id("https://www.instagram.com/reels/DJDzRAgNHyv/")
    'DJDzRAgNHyv'
    """
    path = urlparse(instagram_url).path.rstrip("/")        # '/reels/DJDzRAgNHyv'
    parts = path.split("/")                                # ['', 'reels', 'DJDzRAgNHyv']
    for marker in ("reels", "reel"):
        try:
            return parts[parts.index(marker) + 1]
        except (ValueError, IndexError):
            continue
    raise HTTPException(400, "Could not extract reel ID from URL")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

def _clean_text(value: Any, max_chars: Optional[int] = None) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if max_chars and len(text) > max_chars:
        trimmed = text[:max_chars].rsplit(" ", 1)[0]
        if trimmed:
            text = trimmed + "..."
        else:
            text = text[:max_chars] + "..."
    return text

def _extract_stance_label(raw: Any) -> str:
    if isinstance(raw, dict):
        return (
            raw.get("abstract_label")
            or raw.get("label")
            or raw.get("abstractLabel")
            or "Unknown"
        )
    if raw:
        return str(raw)
    return "Unknown"

def _build_llm_service() -> LLMService:
    return LLMService(
        {
            "base_url": LLM_BASE_URL,
            "api_key": LLM_API_KEY,
        }
    )


def _offline_mock_paths(reel_id: str) -> Optional[Dict[str, str]]:
    if reel_id not in OFFLINE_TRANSCRIPT_REEL_IDS:
        return None

    reel_dir = OFFLINE_MOCK_ROOT / reel_id
    transcript_path = reel_dir / "transcript.txt"
    if not transcript_path.is_file():
        return None

    video_candidates = sorted((reel_dir / "video").glob("*")) if (reel_dir / "video").is_dir() else []
    audio_candidates = sorted((reel_dir / "audio").glob("*")) if (reel_dir / "audio").is_dir() else []
    return {
        "transcript": str(transcript_path),
        "video": str(video_candidates[0]) if video_candidates else "",
        "audio": str(audio_candidates[0]) if audio_candidates else "",
    }


@app.post("/process")
async def process(payload: dict = Body(...)):
    print("Received payload:", payload)
    url  = payload.get("url")
    print(url)
    if not url:
        raise HTTPException(400, "JSON body must contain a 'url' field")

    reel_id = extract_reel_id(url)
    mock    = _coerce_bool(payload.get("mock", False))
    offline_paths = _offline_mock_paths(reel_id)

    result = None
    try:
        if offline_paths:
            transcript = Path(offline_paths["transcript"]).read_text(encoding="utf-8")
            result = run_pipeline_from_transcript(
                transcript=transcript,
                audio_path=offline_paths.get("audio") or None,
                video_path=offline_paths.get("video") or None,
                temperature=OFFLINE_TRANSCRIPT_TEMPERATURE,
            )
            result["_offline_mock_transcript_used"] = True
            result["_offline_mock_reel_id"] = reel_id
            result["_offline_mock_temperature"] = OFFLINE_TRANSCRIPT_TEMPERATURE
        elif mock:
            # Mock mode: just look for a pre-made WAV named <reel_id>.wav
            wav_path = os.path.abspath(f"{reel_id}.wav")
            if not os.path.exists(wav_path):
                raise HTTPException(400, f"Mock WAV not found: {wav_path}")
            result = run_pipeline(audio_path=wav_path)     # your existing inference step
        else:
            result = run_pipeline(video_url=url)
        return result
    finally:
        if not mock and result:
            video_path = result.get("video_path")
            audio_path = result.get("audio_path")
            cleanup_path = video_path or audio_path
            if cleanup_path and not offline_paths:
                shutil.rmtree(os.path.dirname(cleanup_path), ignore_errors=True)


@app.post("/json")
async def json_process(payload: dict = Body(...)):
    return await process(payload)


@app.post("/evidence_summary")
async def evidence_summary(payload: dict = Body(...)):
    statement = _clean_text(payload.get("statement") or payload.get("statement_text"), SUMMARY_MAX_CHARS)
    evidence = payload.get("evidence") or {}
    abstract = _clean_text(
        evidence.get("abstract") or evidence.get("text") or evidence.get("summary"),
        SUMMARY_MAX_CHARS,
    )

    if not statement:
        raise HTTPException(400, "JSON body must contain a 'statement' field")
    if not abstract:
        raise HTTPException(400, "Evidence abstract is required for summary")

    stance = _clean_text(
        _extract_stance_label(
            evidence.get("stance")
            or evidence.get("Stance")
            or evidence.get("STANCE")
        ),
        SUMMARY_MAX_CHARS,
    )

    prompt = PROMPT_TMPL_EVIDENCE_SUMMARY.format(
        statement=statement,
        stance=stance,
        abstract=abstract,
    )
    summary_temperature = SUMMARY_TEMPERATURE
    reel_url = payload.get("reel_url")
    if reel_url:
        try:
            if extract_reel_id(reel_url) in OFFLINE_TRANSCRIPT_REEL_IDS:
                summary_temperature = OFFLINE_TRANSCRIPT_TEMPERATURE
        except HTTPException:
            pass

    try:
        summary = _build_llm_service().call(
            prompt=prompt,
            model=SUMMARY_MODEL,
            temperature=summary_temperature,
            max_tokens=SUMMARY_MAX_TOKENS,
        )
    except Exception as exc:
        raise HTTPException(500, f"Summary generation failed: {exc}") from exc

    return {"summary": summary}

@app.get("/number")
async def get_number():
    return {"number": random.randint(1, 100)}


@app.get("/health")
async def health():
    return {"ok": True}


HARDCODED_PROCESS_RESPONSE_JSON = r'''
{
  "transcript": "Hi, I'm Yanna and I'm a nutritionist. I'm a mom of three myself and I also work in early childhood eligibility prevention. My husband and I have hay fever which means our kids have a higher risk of allergies. Diet is super important for reducing the risk of allergies. You should start feeding your kids solid foods after they are seven months old at the latest. My children have always loved eating eggs. They are a great food for little ones and thanks to their high protein content are also very good for brain development. There is no need to worry that this will increase the risk of developing allergies as was often said in the past. On the contrary, feeding eggs to children at an early age prevents allergies. To test whether the kids were allergic to egg white, I first applied a little egg to their skin. This allowed me to see whether they had an allergic reaction or not. Since this wasn't the case with my three, we often have delicious scrambled eggs right eggs etc for dinner and the kids are super fit and happy.",
  "audio_path": "/data/home/jak38842/disk/fact_checker/kai/fact_checker/ReelPeel/temp/reel_mp5d0505/DT0UIgzDZ79.wav",
  "video_path": "/data/home/jak38842/disk/fact_checker/kai/fact_checker/ReelPeel/temp/reel_mp5d0505/DT0UIgzDZ79.mp4",
  "statements": [
    {
      "id": 1,
      "text": "Children with parents who have hay fever have a higher risk of allergies.",
      "verdict": "uncertain",
      "rationale": "VERDICT: uncertain\nFINALSCORE: 0.50",
      "score": 0.5,
      "queries": [
        "(hay fever[mh] OR allergic rhinitis[tiab] OR seasonal allergy[tiab]) AND (parent[tiab] OR maternal[tiab] OR familial[tiab]) AND ((child[tiab] OR children[tiab] OR pediatric[tiab]) AND (allergy[mh] OR hypersensitivity[tiab] OR atopy[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(hay fever[mh] OR allergic rhinitis[tiab] OR seasonal allergy[tiab]) AND (child[mh] OR children[tiab] OR pediatric[tiab]) AND ((parent[tiab] OR maternal[tiab] OR familial[tiab]) AND (allergy[mh] OR hypersensitivity[tiab] OR atopy[tiab])) AND (ineffective[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR risk[tiab] OR harm[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "queries_fetched": [
        "(hay fever[mh] OR allergic rhinitis[tiab] OR seasonal allergy[tiab]) AND (parent[tiab] OR maternal[tiab] OR familial[tiab]) AND ((child[tiab] OR children[tiab] OR pediatric[tiab]) AND (allergy[mh] OR hypersensitivity[tiab] OR atopy[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(hay fever[mh] OR allergic rhinitis[tiab] OR seasonal allergy[tiab]) AND (child[mh] OR children[tiab] OR pediatric[tiab]) AND ((parent[tiab] OR maternal[tiab] OR familial[tiab]) AND (allergy[mh] OR hypersensitivity[tiab] OR atopy[tiab])) AND (ineffective[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR risk[tiab] OR harm[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "evidence": []
    },
    {
      "id": 2,
      "text": "Solid foods should be introduced to children no later than seven months of age.",
      "verdict": "true",
      "rationale": "VERDICT: true\nFINALSCORE: 0.82",
      "score": 0.82,
      "queries": [
        "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[tiab] OR complementary feeding[tiab] OR solid food introduction[tiab] OR food introduction[tiab]) AND (age[tiab] OR developmental age[tiab] OR months[tiab]) AND (7 months[tiab] OR seven month[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[mh] OR complementary feeding[tiab] OR solid food[tiab] OR food introduction[tiab] OR feeding introduction[tiab]) AND (7 months[tiab] OR seven month[tiab] OR age 7 months[tiab]) AND (ineffective[tiab] OR no benefit[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no effect[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "queries_fetched": [
        "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[tiab] OR complementary feeding[tiab] OR solid food introduction[tiab] OR food introduction[tiab]) AND (age[tiab] OR developmental age[tiab] OR months[tiab]) AND (7 months[tiab] OR seven month[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[mh] OR complementary feeding[tiab] OR solid food[tiab] OR food introduction[tiab] OR feeding introduction[tiab]) AND (7 months[tiab] OR seven month[tiab] OR age 7 months[tiab]) AND (ineffective[tiab] OR no benefit[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no effect[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "evidence": [
        {
          "source_type": "PubMed",
          "weight": 0.5,
          "relevance": 0.81,
          "relevance_abstract": 0.81,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 1.0,
            "abstract_p_refutes": 0.0,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "31004614",
          "url": "https://pubmed.ncbi.nlm.nih.gov/31004614/",
          "title": "Maternal triacylglycerol signature and risk of food allergy in offspring.",
          "queries": [
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[mh] OR complementary feeding[tiab] OR solid food[tiab] OR food introduction[tiab] OR feeding introduction[tiab]) AND (7 months[tiab] OR seven month[tiab] OR age 7 months[tiab]) AND (ineffective[tiab] OR no benefit[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no effect[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "BACKGROUND: The prevalence of IgE-mediated food allergy (FA) is increasing worldwide, but the underlying mechanisms are poorly understood. OBJECTIVE: We sought to examine the role of maternal lipidomic profiles in risk of FA development in offspring and to investigate the potential modification effects by timing of first solid-food introduction. METHODS: This report included 1068 mother-child dyads from the Boston Birth Cohort. Maternal lipid metabolites in plasma were assessed by using liquid chromatography tandem mass spectrometry. Food sensitization (FS) was defined as a specific IgE level of 0.35 kU/L or greater to any of the 8 common food allergens determined by using ImmunoCAP. FA was defined based on FS, clinical symptoms, and food avoidance. Logistic regression was applied to analyze associations between maternal metabolites and risk of FS and FA in offspring and to explore potential effect modifications. RESULTS: Of the 1068 children, 411 had FS, and 132 had FA. Among the 209 metabolites, maternal triacylglycerols (TAGs) of shorter carbon chains and fewer double bonds were associated with greater risk of FA, whereas TAGs of longer carbon chains and more double bonds were significantly associated with lower risk of FA in offspring. These associations were stronger in children with delayed solid-food introduction (≥7 months of age) than those with earlier solid-food introduction (P = .010 for interaction between the maternal TAG score and timing of solid-food introduction). No significant association was found for FS. CONCLUSION: This is the first study to demonstrate a link between maternal TAGs and risk of FA in offspring and potential risk modification by timing of solid-food introduction.",
          "pub_type": [
            "Journal Article",
            "Research Support, N.I.H., Extramural",
            "Research Support, Non-U.S. Gov't"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.5,
          "relevance": 0.83,
          "relevance_abstract": 0.83,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 1.0,
            "abstract_p_refutes": 0.0,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "26815017",
          "url": "https://pubmed.ncbi.nlm.nih.gov/26815017/",
          "title": "Gluten Introduction and the Risk of Coeliac Disease: A Position Paper by the European Society for Pediatric Gastroenterology, Hepatology, and Nutrition.",
          "queries": [
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[mh] OR complementary feeding[tiab] OR solid food[tiab] OR food introduction[tiab] OR feeding introduction[tiab]) AND (7 months[tiab] OR seven month[tiab] OR age 7 months[tiab]) AND (ineffective[tiab] OR no benefit[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no effect[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "BACKGROUND: The European Society for Paediatric Gastroenterology, Hepatology and Nutrition recommended in 2008, based on observational data, to avoid both early (<4 months) and late (≥7 months) introduction of gluten and to introduce gluten while the infant is still being breast-fed. New evidence prompted ESPGHAN to revise these recommendations. OBJECTIVE: To provide updated recommendations regarding gluten introduction in infants and the risk of developing coeliac disease (CD) during childhood. SUMMARY: The risk of inducing CD through a gluten-containing diet exclusively applies to persons carrying at least one of the CD risk alleles. Because genetic risk alleles are generally not known in an infant at the time of solid food introduction, the following recommendations apply to all infants, although they are derived from studying families with first-degree relatives with CD. Although breast-feeding should be promoted for its other well-established health benefits, neither any breast-feeding nor breast-feeding during gluten introduction has been shown to reduce the risk of CD. Gluten may be introduced into the infant's diet anytime between 4 and 12 completed months of age. In children at high risk for CD, earlier introduction of gluten (4 vs 6 months or 6 vs 12 months) is associated with earlier development of CD autoimmunity (defined as positive serology) and CD, but the cumulative incidence of each in later childhood is similar. Based on observational data pointing to the association between the amount of gluten intake and risk of CD, consumption of large quantities of gluten should be avoided during the first weeks after gluten introduction and during infancy. The optimal amounts of gluten to be introduced at weaning, however, have not been established.",
          "pub_type": [
            "Journal Article",
            "Consensus Statement"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.91,
          "relevance_abstract": 0.91,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.85,
            "abstract_p_refutes": 0.13,
            "abstract_p_neutral": 0.02
          },
          "pubmed_id": "17664902",
          "url": "https://pubmed.ncbi.nlm.nih.gov/17664902/",
          "title": "The influence of gluten: weaning recommendations for healthy children and children at risk for celiac disease.",
          "queries": [
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[tiab] OR complementary feeding[tiab] OR solid food introduction[tiab] OR food introduction[tiab]) AND (age[tiab] OR developmental age[tiab] OR months[tiab]) AND (7 months[tiab] OR seven month[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[mh] OR complementary feeding[tiab] OR solid food[tiab] OR food introduction[tiab] OR feeding introduction[tiab]) AND (7 months[tiab] OR seven month[tiab] OR age 7 months[tiab]) AND (ineffective[tiab] OR no benefit[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no effect[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "In most developed countries, gluten is currently most commonly introduced between 4 and 6 months of age, in spite of little evidence to support this practice. As for infants at risk of developing food allergies, there is clear evidence that introducing solid foods before the end of the 3rd month is detrimental and should be avoided. A recent growing body of evidence however challenges the notion that solids (and among them, gluten-containing foods) should be introduced beyond the 6th month of life. Another important aspect of gluten introduction into the diet has to do with its possible role in causing type-1 diabetes (IDDM). Recently, a large epidemiological investigation in a cohort of children at risk for IDDM found that exposure to cereals (rice, wheat, oats, barley, rye) that occurred early (< or = 3 months) as well as late (> or = 7 months) resulted in a significantly higher risk of the appearance of islet cell autoimmunity compared to the introduction between 4 and 6 months. As for celiac disease, the protective role of breastfeeding can be considered ascertained, especially the protection offered by having gluten introduced while breastfeeding is continued. Evidence is emerging that early (< or = 3 months) and perhaps even late (7 months or after) first exposure to gluten may favor the onset of celiac disease in predisposed individuals. Additionally, large amounts of gluten at weaning are associated with an increased risk of developing celiac disease, as documented in studies from Scandinavian countries. In celiac children observed in our center, we could show that breastfeeding at the time of gluten introduction delays the appearance of celiac disease and makes it less likely that its presentation is predominantly gastrointestinal. Based on current evidence, it appears reasonable to recommend that gluten be introduced in small amounts in the diet between 4 and 6 months, while the infant is breastfed, and that breastfeeding is continued for at least a further 2-3 months.",
          "pub_type": [
            "Journal Article",
            "Review"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.5,
          "relevance": 0.93,
          "relevance_abstract": 0.93,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 1.0,
            "abstract_p_refutes": 0.0,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "25976525",
          "url": "https://pubmed.ncbi.nlm.nih.gov/25976525/",
          "title": "The Timing of Infant Food Introduction in Families With a History of Atopy.",
          "queries": [
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[tiab] OR complementary feeding[tiab] OR solid food introduction[tiab] OR food introduction[tiab]) AND (age[tiab] OR developmental age[tiab] OR months[tiab]) AND (7 months[tiab] OR seven month[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[mh] OR complementary feeding[tiab] OR solid food[tiab] OR food introduction[tiab] OR feeding introduction[tiab]) AND (7 months[tiab] OR seven month[tiab] OR age 7 months[tiab]) AND (ineffective[tiab] OR no benefit[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no effect[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "OBJECTIVE: To describe the timing of introduction and type of food introduced to infants with a family history of atopy. METHODS: We conducted a secondary analysis of foods introduced each month to an interventional birth cohort of 149 infants at risk for atopy. RESULTS: Seven percent of infants received solid food prior to 4 months of age; 13% after 6 months of age. Hyperallergenic foods were introduced on average in the following order: wheat (8.7 months); eggs (11.2 months); soy (13.0 months); fish (13.4 months); peanut (20.2 months); tree nuts (21.8 months); and other seafood (21.8 months). Asian race (odds ratio 3.94; 95% CI 1.14-13.58) and maternal history of food allergy (odds ratio 3.86; 95% CI 1.29-11.56) were associated with late food introduction. CONCLUSION: Variation in timing of food introduction may reflect cultural preferences and/or previous experience with food allergy, as well as the ambiguous state of current recommendations.",
          "pub_type": [
            "Journal Article"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.94,
          "relevance_abstract": 0.94,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.98,
            "abstract_p_refutes": 0.02,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "18162844",
          "url": "https://pubmed.ncbi.nlm.nih.gov/18162844/",
          "title": "Complementary feeding: a commentary by the ESPGHAN Committee on Nutrition.",
          "queries": [
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[tiab] OR complementary feeding[tiab] OR solid food introduction[tiab] OR food introduction[tiab]) AND (age[tiab] OR developmental age[tiab] OR months[tiab]) AND (7 months[tiab] OR seven month[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
            "(infant[mh] OR baby[tiab] OR child[tiab]) AND (weaning[mh] OR complementary feeding[tiab] OR solid food[tiab] OR food introduction[tiab] OR feeding introduction[tiab]) AND (7 months[tiab] OR seven month[tiab] OR age 7 months[tiab]) AND (ineffective[tiab] OR no benefit[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no effect[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "This position paper on complementary feeding summarizes evidence for health effects of complementary foods. It focuses on healthy infants in Europe. After reviewing current knowledge and practices, we have formulated these conclusions: Exclusive or full breast-feeding for about 6 months is a desirable goal. Complementary feeding (ie, solid foods and liquids other than breast milk or infant formula and follow-on formula) should not be introduced before 17 weeks and not later than 26 weeks. There is no convincing scientific evidence that avoidance or delayed introduction of potentially allergenic foods, such as fish and eggs, reduces allergies, either in infants considered at increased risk for the development of allergy or in those not considered to be at increased risk. During the complementary feeding period, >90% of the iron requirements of a breast-fed infant must be met by complementary foods, which should provide sufficient bioavailable iron. Cow's milk is a poor source of iron and should not be used as the main drink before 12 months, although small volumes may be added to complementary foods. It is prudent to avoid both early (<4 months) and late (>or=7 months) introduction of gluten, and to introduce gluten gradually while the infant is still breast-fed, inasmuch as this may reduce the risk of celiac disease, type 1 diabetes mellitus, and wheat allergy. Infants and young children receiving a vegetarian diet should receive a sufficient amount ( approximately 500 mL) of breast milk or formula and dairy products. Infants and young children should not be fed a vegan diet.",
          "pub_type": [
            "Journal Article",
            "Review"
          ]
        }
      ]
    },
    {
      "id": 3,
      "text": "Eggs are a good food for infants due to their high protein content, which supports brain development.",
      "verdict": "uncertain",
      "rationale": "VERDICT: uncertain\nFINALSCORE: 0.50",
      "score": 0.5,
      "queries": [
        "(infant[mh] OR baby[tiab] OR neonatal[tiab]) AND (egg[tiab] OR ovum[tiab]) AND (protein[mh] OR amino acid[tiab]) AND ((brain development[tiab] OR neurodevelopment[tiab] OR cognitive function[tiab] OR neurological development[tiab]) OR (neurogenesis[tiab] OR myelination[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(infant[mh] OR baby[tiab] OR neonatal[tiab]) AND (egg[tiab] OR ovum[tiab]) AND (protein[mh] OR polypeptide[tiab]) AND ((brain development[tiab] OR neurodevelopment[tiab] OR cognitive function[tiab]) OR (neurocognitive[tiab] OR neuronal growth[tiab])) AND (ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR toxicity[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "queries_fetched": [
        "(infant[mh] OR baby[tiab] OR neonatal[tiab]) AND (egg[tiab] OR ovum[tiab]) AND (protein[mh] OR amino acid[tiab]) AND ((brain development[tiab] OR neurodevelopment[tiab] OR cognitive function[tiab] OR neurological development[tiab]) OR (neurogenesis[tiab] OR myelination[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(infant[mh] OR baby[tiab] OR neonatal[tiab]) AND (egg[tiab] OR ovum[tiab]) AND (protein[mh] OR polypeptide[tiab]) AND ((brain development[tiab] OR neurodevelopment[tiab] OR cognitive function[tiab]) OR (neurocognitive[tiab] OR neuronal growth[tiab])) AND (ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR toxicity[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "evidence": []
    },
    {
      "id": 4,
      "text": "Early introduction of eggs prevents allergies in children.",
      "verdict": "true",
      "rationale": "VERDICT: true\nFINALSCORE: 0.88",
      "score": 0.88,
      "queries": [
        "(egg introduction[tiab] OR egg consumption[tiab] OR egg feeding[tiab] OR infant diet[tiab] OR dietary exposure[tiab]) AND (allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR atopy[tiab] OR eczema[tiab] OR asthma[tiab]) AND (child[mh] OR infant[mh] OR pediatric[tiab] OR children[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "queries_fetched": [
        "(egg introduction[tiab] OR egg consumption[tiab] OR egg feeding[tiab] OR infant diet[tiab] OR dietary exposure[tiab]) AND (allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR atopy[tiab] OR eczema[tiab] OR asthma[tiab]) AND (child[mh] OR infant[mh] OR pediatric[tiab] OR children[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "evidence": [
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.81,
          "relevance_abstract": 0.81,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 1.0,
            "abstract_p_refutes": 0.0,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "31690392",
          "url": "https://pubmed.ncbi.nlm.nih.gov/31690392/",
          "title": "Prevention of food allergy.",
          "queries": [
            "(egg introduction[tiab] OR egg consumption[tiab] OR egg feeding[tiab] OR infant diet[tiab] OR dietary exposure[tiab]) AND (allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR atopy[tiab] OR eczema[tiab] OR asthma[tiab]) AND (child[mh] OR infant[mh] OR pediatric[tiab] OR children[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "Primary prevention and secondary prevention in the context of food allergy refer to prevention of the development of sensitization (i.e., the presence of food-specific immunoglobulin E (IgE) as measured by skin-prick testing and/or laboratory testing) and sensitization plus the clinical manifestations of food allergy, respectively. Until recently, interventions that target the prevention of food allergy have been limited. Although exclusive breast-feeding for the first 6 months of life has been a long-standing recommendation due to associated health benefits, recommendations regarding complementary feeding in infancy have significantly changed over the past 20 years. There now is evidence to support early introduction of peanut into the diet of infants with egg allergy, severe atopic dermatitis, or both diagnoses, defined as high risk for peanut allergy, to try to prevent development of peanut allergy. Although guideline-based recommendations are not available for early introduction of additional allergenic foods, this topic is being actively studied. There is no evidence to support additional dietary modification of the maternal or infant diet for the prevention of food allergy. Similarly, there is no conclusive evidence to support maternal avoidance diets for the prevention of food allergy.",
          "pub_type": [
            "Journal Article",
            "Research Support, Non-U.S. Gov't",
            "Review"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.83,
          "relevance_abstract": 0.83,
          "stance": {
            "abstract_label": "Refutes",
            "abstract_p_supports": 0.0,
            "abstract_p_refutes": 1.0,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "21880163",
          "url": "https://pubmed.ncbi.nlm.nih.gov/21880163/",
          "title": "Infant nutrition and allergy.",
          "queries": [
            "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "Over the past several decades, the incidence of atopic diseases such as asthma, atopic dermatitis and food allergies has increased dramatically. Although atopic diseases have a clear genetic basis, environmental factors, including early infant nutrition, may have an important influence on their development. Therefore, attempts have been made to reduce the risk of the development of allergy using dietary modifications, mainly focused on longer breast-feeding and delayed introduction or elimination of foods identified as potentially most allergenic. Recently, there is also an increasing interest in the active prevention of atopy using specific dietary components. Many studies have shown that breast-feeding may have the protective effect against future atopic dermatitis and early childhood wheezing. Concerning complementary feeding, there is evidence that the introduction of complementary foods before 4 months of age may increase the risk for atopic dermatitis. However, there is no current convincing evidence that delaying introduction of solids after 6 months of age has a significant protective effect on the development of atopic disease regardless of whether infants are fed cow's milk protein formula or human subject's milk, and this includes delaying the introduction of foods that are considered to be highly allergic, such as fish, eggs and foods containing peanut protein. In conclusion, as early nutrition may have profound implications for long-term health and atopy later in life, it presents an opportunity to prevent or delay the onset of atopic diseases.",
          "pub_type": [
            "Journal Article",
            "Review"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.86,
          "relevance_abstract": 0.86,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.84,
            "abstract_p_refutes": 0.16,
            "abstract_p_neutral": 0.01
          },
          "pubmed_id": "34955309",
          "url": "https://pubmed.ncbi.nlm.nih.gov/34955309/",
          "title": "Primary prevention of food allergy in 2021: Update and proposals of French-speaking pediatric allergists.",
          "queries": [
            "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "During the past years, there has been an alarming increase in cases of food allergy and anaphylaxis in ever-younger children. Often, these children have multiple food allergies and food sensitizations, involving allergens with high anaphylactic potential, such as peanuts and nuts, which have a major influence on their quality of life and future. After reviewing the current epidemiological data, we discuss the main causes of the increase in food allergies. We analyze data from studies on the skin barrier and its fundamental role in the development of sensitization and food allergies, data on the tolerogenic digestive tract applied in particular to hen eggs and peanuts, as well as data on the prevention of allergy to cow milk proteins. In light of these studies, we propose a practical guide of recommendations focused on infants and the introduction of cow milk, the management of eczema, and early and broad dietary diversification including high-risk food allergens, such as peanut and nuts while taking into account the food consumption habits of the family.",
          "pub_type": [
            "Journal Article",
            "Review"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.94,
          "relevance_abstract": 0.94,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.73,
            "abstract_p_refutes": 0.21,
            "abstract_p_neutral": 0.06
          },
          "pubmed_id": "28173607",
          "url": "https://pubmed.ncbi.nlm.nih.gov/28173607/",
          "title": "Review suggests that the immunoregulatory and anti-inflammatory properties of allergenic foods can provoke oral tolerance if introduced early to infants' diets.",
          "queries": [
            "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "UNLABELLED: For years, the advice was to postpone introducing allergenic foods, in order to prevent food allergies. However, food allergies have escalated rather than declined and our review of the latest epidemiological, immunological and nutritional research suggests that early weaning practices may be beneficial. The most allergenic foods, such as fish, eggs and peanuts, have proved to be inherently rich in tolerogenic substances that can play a significant role in preventing allergies. CONCLUSION: We found evidence that the immunoregulatory and anti-inflammatory properties of allergenic foods can provoke oral tolerance if introduced early to both low-risk and high-risk infants.",
          "pub_type": [
            "Journal Article",
            "Review"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.85,
          "relevance": 0.95,
          "relevance_abstract": 0.95,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.93,
            "abstract_p_refutes": 0.03,
            "abstract_p_neutral": 0.04
          },
          "pubmed_id": "33710678",
          "url": "https://pubmed.ncbi.nlm.nih.gov/33710678/",
          "title": "EAACI guideline: Preventing the development of food allergy in infants and young children (2020 update).",
          "queries": [
            "(egg introduction[tiab] OR egg consumption[tiab] OR egg feeding[tiab] OR infant diet[tiab] OR dietary exposure[tiab]) AND (allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR atopy[tiab] OR eczema[tiab] OR asthma[tiab]) AND (child[mh] OR infant[mh] OR pediatric[tiab] OR children[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "BACKGROUND: This guideline from the European Academy of Allergy and Clinical Immunology (EAACI) recommends approaches to prevent the development of immediate-onset / IgE-mediated food allergy in infants and young children. It is an update of a 2014 EAACI guideline. METHODS: The guideline was developed using the AGREE II framework and the GRADE approach. An international Task Force with representatives from 11 countries and different disciplinary and clinical backgrounds systematically reviewed research and considered expert opinion. Recommendations were created by weighing up benefits and harms, considering the certainty of evidence and examining values, preferences and resource implications. The guideline was peer-reviewed by external experts, and feedback was incorporated from public consultation. RESULTS: All of the recommendations about preventing food allergy relate to infants (up to 1 year) and young children (up to 5 years), regardless of risk of allergy. There was insufficient evidence about preventing food allergy in other age groups. The EAACI Task Force suggests avoiding the use of regular cow's milk formula as supplementary feed for breastfed infants in the first week of life. The EAACI Task Force suggests introducing well-cooked, but not raw egg or uncooked pasteurized, egg into the infant diet as part of complementary feeding. In populations where there is a high prevalence of peanut allergy, the EAACI Task Force suggests introducing peanuts in an age-appropriate form as part of complementary feeding. According to the studies, it appears that the most effective age to introduce egg and peanut is from four to 6 months of life. The EAACI Task Force suggests against the following for preventing food allergy: (i) avoiding dietary food allergens during pregnancy or breastfeeding; and (ii) using soy protein formula in the first 6 months of life as a means of preventing food allergy. There is no recommendation for or against the following: use of vitamin supplements, fish oil, prebiotics, probiotics or synbiotics in pregnancy, when breastfeeding or in infancy; altering the duration of exclusive breastfeeding; and hydrolysed infant formulas, regular cow's milk-based infant formula after a week of age or use of emollients. CONCLUSIONS: Key changes from the 2014 guideline include suggesting (i) the introduction of peanut and well-cooked egg as part of complementary feeding (moderate certainty of evidence) and (ii) avoiding supplementation with regular cow's milk formula in the first week of life (low certainty of evidence). There remains uncertainty in how to prevent food allergy, and further well-powered, multinational research using robust diagnostic criteria is needed.",
          "pub_type": [
            "Journal Article",
            "Research Support, Non-U.S. Gov't",
            "Practice Guideline"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.95,
          "relevance_abstract": 0.95,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 1.0,
            "abstract_p_refutes": 0.0,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "37960183",
          "url": "https://pubmed.ncbi.nlm.nih.gov/37960183/",
          "title": "Early Introduction of Novel and Less-Studied Food Allergens in the Plant-Based Era: Considerations for US and EU Infant Formula Regulations.",
          "queries": [
            "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "Early life feeding practices may affect the long-term health of individuals, particularly in terms of the development of non-communicable diseases, such as metabolic and allergic diseases. Accumulating evidence suggests that the interplay of breastfeeding and/or formula feeding followed by the introduction of solids plays a role in the occurrence of non-communicable diseases both in the short and long term. International food allergy guidelines recommend that breastfeeding women do not need to avoid food allergens and do not recommend any infant formula for allergy prevention. Guidelines regarding solid food introduction for food allergy prevention recommend the introduction of well-cooked eggs and peanuts around 4-6 months of age, and not to delay the introduction of other food allergens. There is also an increasing trend to feed infants a plant-based or plant-forward diet and have access to infant formulas based on plant-based ingredients. The use of novel plant-based infant formulas raises a few questions reviewed in this paper: (1) Do fortified, plant-based infant formulas, compliant with US Food and Drug Administration (FDA) regulations and European Food Safety Authority (EFSA) (European) guidelines, support adequate infant growth? (2) Are plant-based infant formulas suitable for the management of cow's milk allergy? (3) Does feeding with novel, plant-based infant formulas increase the risk of food allergies to the food allergens they contain? (4) Does feeding infants plant-based food allergens in early life increase the risk of allergic and severe allergic reactions? The review of the literature indicated that (1) plant-based formulas supplemented with amino acids and micronutrients to comply with FDA regulations and EFSA guidelines, evaluated in sufficiently powered growth studies, can support adequate growth in infants; (2) currently available plant-based infant formulas are suitable for the management of CMA; (3) an early introduction and continuous intake of food allergens are more likely to prevent food allergies than to increase their risk; and (4) an early introduction of food allergens in young infants is safe.",
          "pub_type": [
            "Journal Article",
            "Review"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.85,
          "relevance": 0.99,
          "relevance_abstract": 0.99,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 1.0,
            "abstract_p_refutes": 0.0,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "28914636",
          "url": "https://pubmed.ncbi.nlm.nih.gov/28914636/",
          "title": "Dietary intervention for preventing food allergy in children.",
          "queries": [
            "(egg introduction[tiab] OR egg consumption[tiab] OR egg feeding[tiab] OR infant diet[tiab] OR dietary exposure[tiab]) AND (allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR atopy[tiab] OR eczema[tiab] OR asthma[tiab]) AND (child[mh] OR infant[mh] OR pediatric[tiab] OR children[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "PURPOSE OF REVIEW: In the past decade, food allergy has been increasingly recognized as an important public health issue. The role of maternal and infant diet in the development of food allergy has been a major focus of research throughout this period. Recently, research in this area has moved from observational studies to intervention trials, and the findings from these trials have started to influence infant feeding guidelines. In this article, we review recent studies of dietary interventions for preventing food allergy, summarize current knowledge and discuss future research directions. RECENT FINDINGS: The latest result from an intervention trial shows that introduction of peanut in the first year of life reduces the risk of peanut allergy in high-risk infants. A systematic review and meta-analysis of intervention trials also suggests a protective effect of egg introduction from around 4 to 6 months of age for reducing the risk of egg allergy, with most studies conducted in high-risk infants. Despite several intervention trials involving modifications to the maternal diet, the effect of maternal diet during pregnancy and lactation in preventing food allergy remains unclear. SUMMARY: Earlier introduction of allergenic foods is a promising intervention to reduce the risk of some food allergies in high-risk infants. Further work is needed to improve knowledge of how to prevent food allergy in the general population.",
          "pub_type": [
            "Journal Article",
            "Systematic Review"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.5,
          "relevance": 0.99,
          "relevance_abstract": 0.99,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.9,
            "abstract_p_refutes": 0.05,
            "abstract_p_neutral": 0.05
          },
          "pubmed_id": "40553938",
          "url": "https://pubmed.ncbi.nlm.nih.gov/40553938/",
          "title": "Infant Diet Recommendations Reduce IgE-Mediated Egg, Peanut, and Cow's Milk Allergies.",
          "queries": [
            "(egg introduction[tiab] OR egg consumption[tiab] OR egg feeding[tiab] OR infant diet[tiab] OR dietary exposure[tiab]) AND (allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR atopy[tiab] OR eczema[tiab] OR asthma[tiab]) AND (child[mh] OR infant[mh] OR pediatric[tiab] OR children[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
            "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "BACKGROUND: Meta-analyses of randomized controlled trials have found that introducing eggs and peanuts earlier during infancy reduced egg and peanut allergy risk. Hence, infant feeding advice has dramatically changed from previous recommendations of avoidance to current recommendations of inclusion of common food allergens in infant diets. OBJECTIVE: To compare the prevalence of IgE-mediated food allergies at 1 year of age between 2 cohorts, before and after infant feeding and allergy prevention guidelines changed. METHODS: In cohort 1 (506 infants born 2006-2014), no infant feeding advice was provided to participants. In cohort 2 (566 infants born 2016-2022), when the infants were 6 months of age, all families were provided with updated infant feeding and allergy prevention guidelines. All infants had a first-degree relative with a history of allergic disease. At 1 year of age, infant food allergen sensitization and IgE-mediated food allergy were assessed. RESULTS: Peanut, egg, and cow's milk were introduced earlier in cohort 2 than in cohort 1 (all P < .001). The combined prevalence of IgE-mediated peanut, egg, and/or cow's milk allergies was 4.1% in cohort 2 compared with 12.6% in cohort 1 (adjusted odds ratio [aOR]: 0.28, 95% confidence interval [CI]: 0.16-0.48, P < .001). Specifically, the prevalence of peanut allergy was 1.1% versus 5.8% (aOR: 0.24, 95% CI: 0.08-0.76, P = .015), egg allergy 2.8% versus 11.7% (aOR: 0.23, 95% CI: 0.12-0.45, P < .001), and cow's milk allergy 0.5% versus 2.4%, respectively (aOR: 0.14, 95% CI: 0.04-0.55, P = .005). CONCLUSION: Direct provision of updated food allergy prevention guidelines to families facilitated earlier introduction and reduced the prevalence of IgE-mediated peanut, egg, and cow's milk allergies.",
          "pub_type": [
            "Journal Article"
          ]
        },
        {
          "source_type": "PubMed",
          "weight": 0.9,
          "relevance": 1.0,
          "relevance_abstract": 1.0,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.88,
            "abstract_p_refutes": 0.05,
            "abstract_p_neutral": 0.06
          },
          "pubmed_id": "37827490",
          "url": "https://pubmed.ncbi.nlm.nih.gov/37827490/",
          "title": "Effects of Early Diet on the Prevalence of Allergic Disease in Children: A Systematic Review and Meta-Analysis.",
          "queries": [
            "(egg[mh] OR eggs[tiab] OR ovum[tiab] OR ova[tiab]) AND (child[mh] OR infant[mh] OR child[tiab] OR children[tiab] OR pediatric[tiab] OR neonatal[tiab]) AND ((introduction[tiab] OR exposure[tiab] OR feeding[tiab]) AND (allergy[mh] OR allergic[tiab] OR hypersensitivity[tiab] OR atopy[tiab])) AND ((ineffective[tiab] OR no effect[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR harm[tiab] OR risk[tiab] OR no benefit[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "Recent evidence suggests that the timing of introduction, types, and amounts of complementary foods/allergenic foods may influence the risk of allergic disease. However, the evidence has not been updated and comprehensively synthesized. The Cochrane Library, EMBASE, Web of Science, and PubMed databases were searched from the inception of each database up to 31 May 2023 (articles prior to 2000 were excluded manually). Statistical analyses were performed using RevMan 5. The GRADE approach was followed to rate the certainty of evidence. Compared with >6 mo, early introduction of eggs (≤6 mo of age) might reduce the risk of food allergies in preschoolers aged <6 y (odds ratio [OR], 0.65; 95% confidence interval [CI], 0.53, 0.81), but had no effect on asthma or atopic dermatitis (AD). Consumption of fish at 6-12 mo might reduce the risk of asthma in children (aged 5-17 y) compared with late introduction after 12 mo (OR, 0.61; 95% CI: 0.52, 0.72). Introduction of allergenic foods for ≤6 mo of age, compared with >6 mos, was a protective factor for the future risk (children aged ≤10 y) of AD (OR, 0.93; 95% CI: 0.89, 0.97). Probiotic intervention for infants at high risk of allergic disease significantly reduced the risk of food allergy at ages 0-3 y (OR, 0.72; 95% CI: 0.56, 0.94), asthma at 6-12 y (OR, 0.61; 95% CI: 0.41, 0.90), and AD at aged <6 y (3-6 y: OR, 0.70; 95% CI: 0.52, 0.94; 0-3 y: OR, 0.73; 95% CI: 0.59, 0.91). Early introduction of complementary foods or the high-dose vitamin D supplementation in infancy was not associated with the risk of developing food allergies, asthma, or AD during childhood. Early introduction to potential allergen foods for normal infants or probiotics for infants at high risk of allergies may protect against development of allergic disease. This study was registered at PROSPERO as CRD42022379264.",
          "pub_type": [
            "Meta-Analysis",
            "Systematic Review",
            "Journal Article",
            "Research Support, Non-U.S. Gov't"
          ]
        }
      ]
    },
    {
      "id": 5,
      "text": "Applying a small amount of egg to a child's skin can test for egg allergy.",
      "verdict": "false",
      "rationale": "VERDICT: false\nFINALSCORE: 0.30",
      "score": 0.3,
      "queries": [
        "(egg allergy[mh] OR egg hypersensitivity[tiab] OR allergic reaction[tiab] OR allergy testing[tiab] OR allergy diagnosis[tiab]) AND (child[tiab] OR infant[tiab] OR pediatric[tiab]) AND (skin test[tiab] OR cutaneous test[tiab] OR prick test[tiab] OR allergy skin test[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(egg allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR allergy testing[tiab] OR allergy diagnosis[tiab]) AND (child[tiab] OR infant[tiab] OR pediatric[tiab]) AND (skin[tiab] OR cutaneous[tiab]) AND ((ineffective[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR no effect[tiab] OR risk[tiab] OR harm[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "queries_fetched": [
        "(egg allergy[mh] OR egg hypersensitivity[tiab] OR allergic reaction[tiab] OR allergy testing[tiab] OR allergy diagnosis[tiab]) AND (child[tiab] OR infant[tiab] OR pediatric[tiab]) AND (skin test[tiab] OR cutaneous test[tiab] OR prick test[tiab] OR allergy skin test[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])",
        "(egg allergy[mh] OR allergic reaction[tiab] OR hypersensitivity[tiab] OR allergy testing[tiab] OR allergy diagnosis[tiab]) AND (child[tiab] OR infant[tiab] OR pediatric[tiab]) AND (skin[tiab] OR cutaneous[tiab]) AND ((ineffective[tiab] OR null[tiab] OR negative[tiab] OR adverse[tiab] OR no effect[tiab] OR risk[tiab] OR harm[tiab])) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
      ],
      "evidence": [
        {
          "source_type": "PubMed",
          "weight": 0.55,
          "relevance": 0.72,
          "relevance_abstract": 0.72,
          "stance": {
            "abstract_label": "Supports",
            "abstract_p_supports": 0.99,
            "abstract_p_refutes": 0.01,
            "abstract_p_neutral": 0.0
          },
          "pubmed_id": "16835576",
          "url": "https://pubmed.ncbi.nlm.nih.gov/16835576/",
          "title": "[Egg allergy].",
          "queries": [
            "(egg allergy[mh] OR egg hypersensitivity[tiab] OR allergic reaction[tiab] OR allergy testing[tiab] OR allergy diagnosis[tiab]) AND (child[tiab] OR infant[tiab] OR pediatric[tiab]) AND (skin test[tiab] OR cutaneous test[tiab] OR prick test[tiab] OR allergy skin test[tiab]) AND (humans[mh] OR clinical trial[tiab] OR randomized[tiab] OR randomised[tiab] OR trial[tiab] OR cohort[tiab] OR case control[tiab] OR observational[tiab] OR systematic review[tiab] OR meta analysis[tiab])"
          ],
          "abstract": "Egg is one of the most important allergen in childhood feeding. The pathogenic mechanism in egg allergy is immediate, type I, IgE-mediated hypersensitivity, although other mechanisms are possible. The aim of this review is to point out that diagnosis of egg protein allergy is mainly clinical and double-blind placebo-controlled food challenge is nowadays the gold standard. Although reference values for prick test and sIgE have been proposed, which can foretell symptoms in groups of egg sensitive children, these values are not so accurate for a single diagnosis, since they mainly refer to children with atopic dermatitis, and to specific ranges of age. Children with atopic dermatitis can show allergy at the first egg ingestion, as for cow milk allergy, because the sensitization may happen in utero or through breast milk. The only available therapy in case of egg allergy is the complete removal of hen egg from the child's diet, yet considering cross-reaction with other birds' eggs, while cross- reaction with poultry and/or other birds' meat has been signalled only in 5% of cases. From this review it is clear how egg allergic children can be vaccinated against measles-mumps rubella.",
          "pub_type": [
            "Journal Article",
            "Review"
          ]
        }
      ]
    }
  ],
  "overall_truthiness": 0.51,
  "generated_at": "2026-02-25T20:15:55Z",
  "execution_log": [
    {
      "step": "DownloadReelStep",
      "duration": 1.4192805290222168,
      "tokens": 0,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 0,
      "evidence_total_after": 0,
      "evidence_by_source_before": {},
      "evidence_by_source_after": {}
    },
    {
      "step": "VideoToAudioStep",
      "duration": 0.9859945774078369,
      "tokens": 0,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 0,
      "evidence_total_after": 0,
      "evidence_by_source_before": {},
      "evidence_by_source_after": {}
    },
    {
      "step": "AudioToTranscriptStep",
      "duration": 4.036341905593872,
      "tokens": 0,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 0,
      "evidence_total_after": 0,
      "evidence_by_source_before": {},
      "evidence_by_source_after": {}
    },
    {
      "step": "TranscriptToStatementStep",
      "duration": 12.164420366287231,
      "tokens": 725,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 0,
      "evidence_total_after": 0,
      "evidence_by_source_before": {},
      "evidence_by_source_after": {}
    },
    {
      "step": "StatementToQueryStep",
      "duration": 45.939937591552734,
      "tokens": 10620,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 0,
      "evidence_total_after": 68,
      "evidence_by_source_before": {},
      "evidence_by_source_after": {
        "PubMed": 68
      }
    },
    {
      "step": "QueryToLinkStep",
      "duration": 0.00029397010803222656,
      "tokens": 0,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 68,
      "evidence_total_after": 68,
      "evidence_by_source_before": {
        "PubMed": 68
      },
      "evidence_by_source_after": {
        "PubMed": 68
      }
    },
    {
      "step": "LinkToAbstractStep",
      "duration": 0.0006840229034423828,
      "tokens": 0,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 68,
      "evidence_total_after": 68,
      "evidence_by_source_before": {
        "PubMed": 68
      },
      "evidence_by_source_after": {
        "PubMed": 68
      }
    },
    {
      "step": "PubTypeWeightStep",
      "duration": 0.0018470287322998047,
      "tokens": 0,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 68,
      "evidence_total_after": 68,
      "evidence_by_source_before": {
        "PubMed": 68
      },
      "evidence_by_source_after": {
        "PubMed": 68
      }
    },
    {
      "step": "RerankEvidenceStep",
      "duration": 2.2352097034454346,
      "tokens": 30693,
      "indent": 1,
      "is_module": false,
      "evidence_total_before": 68,
      "evidence_total_after": 15,
      "evidence_by_source_before": {
        "PubMed": 68
      },
      "evidence_by_source_after": {
        "PubMed": 15
      }
    },
    {
      "step": "StanceEvidenceStep",
      "duration": 1.7450740337371826,
      "tokens": 5094,
      "indent": 1,
      "is_module": false,
      "evidence_total_before": 15,
      "evidence_total_after": 15,
      "evidence_by_source_before": {
        "PubMed": 15
      },
      "evidence_by_source_after": {
        "PubMed": 15
      }
    },
    {
      "step": "MODULE! Scores Engine",
      "duration": 3.9852097034454346,
      "tokens": 0,
      "indent": 0,
      "is_module": true,
      "evidence_total_before": 68,
      "evidence_total_after": 15,
      "evidence_by_source_before": {
        "PubMed": 68
      },
      "evidence_by_source_after": {
        "PubMed": 15
      }
    },
    {
      "step": "TruthnessStep",
      "duration": 10.84467887878418,
      "tokens": 9351,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 15,
      "evidence_total_after": 15,
      "evidence_by_source_before": {
        "PubMed": 15
      },
      "evidence_by_source_after": {
        "PubMed": 15
      }
    },
    {
      "step": "ScoringStep",
      "duration": 0.0003638267517089844,
      "tokens": 0,
      "indent": 0,
      "is_module": false,
      "evidence_total_before": 15,
      "evidence_total_after": 15,
      "evidence_by_source_before": {
        "PubMed": 15
      },
      "evidence_by_source_after": {
        "PubMed": 15
      }
    }
  ],
  "depth": 0
}
'''

HARDCODED_PROCESS_RESPONSE: Dict[str, Any] = json.loads(HARDCODED_PROCESS_RESPONSE_JSON)


@app.post("/json")
async def get_json(payload: dict):
    print(payload)
    await asyncio.sleep(10)
    return HARDCODED_PROCESS_RESPONSE
