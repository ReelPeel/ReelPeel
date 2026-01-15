# main.py ──────────────────────────────────────────────────────────────
from fastapi import FastAPI, Body, HTTPException
import shutil, os
from .pipeline import run_pipeline
  # ← your existing heavy pipeline
from .reel_utils import convert_video_to_wav 
from typing import Any, Dict, List
from urllib.parse import urlparse
import random
import json
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="One-shot reel-to-pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # or a list of specific origins
    allow_methods=["*"], # GET, POST, etc.
    allow_headers=["*"], # e.g. Content-Type
    expose_headers=["*"],
)

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
    try:
        return parts[parts.index("reels") + 1]
    except (ValueError, IndexError):
        raise HTTPException(400, "Could not extract reel ID from URL")

@app.post("/process")
async def process(payload: dict = Body(...)):
    print("Received payload:", payload)
    url  = payload.get("url")
    print(url)
    if not url:
        raise HTTPException(400, "JSON body must contain a 'url' field")

    reel_id = extract_reel_id(url)
    mock    = payload.get("mock", False)

    if mock:
        # Mock mode: just look for a pre-made WAV named <reel_id>.wav
        wav_path = os.path.abspath(f"{reel_id}.wav")
        tmp_dir  = os.path.dirname(wav_path)
    else:
        try:
            print(f"Downloading reel from {url}...")
            wav_path = convert_video_to_wav(url)           # your existing helper
            tmp_dir  = os.path.dirname(wav_path)
        except RuntimeError as e:
            raise HTTPException(400, str(e))

    try:
        return run_pipeline(wav_path)                      # your existing inference step
    finally:
        if not mock:                                       # don’t delete local mocks
            shutil.rmtree(tmp_dir, ignore_errors=True)

@app.get("/number")
async def get_number():
    return {"number": random.randint(1, 100)}

@app.post("/json")
async def get_json(payload: dict):
    print(payload)
    data :Dict[str, Any] = {'transcript': "Did you know that starving yourself can reverse your age, repair damaged cells, and increase your brain power with just one easy health hack? It sounds so unrealistic you would think I'm lying to you. But I'm not. I'm not lying. These are all recorded benefits of fasting. There are literally tons of religions and ancient cultures that practice fasting as a way to regulate health, discipline, and spirit. And modern science proves that they were on to something. Just divine wisdom and nerds and lab coats agreeing on something being good for you should be all the evidence that you need. And here's what we know that fasting can do for you. Discard old and damaged cells and replace them with new ones. Purify your brain function with less toxicity flowing through your body. Rases your willpower because it's a form of discipline training. Rases your ketone levels and helps with metabolism. And obviously if you're not eating as much you're going to lose weight. And the research varies on how long you're supposed to fast forward. But to get these types of benefits it sounds like the minimum is 24 hours. I go anywhere from 36 to 48 and I do it once a week. And in the next video I'll walk you through how I do it.", 'statements': [{'id': 1, 'text': 'Fasting can promote cellular repair and renewal by discarding old and damaged cells.', 'verdict': 'true', 'rationale': 'VERDICT: true\nFINALSCORE: 0.95', 'score': 0.95, 'query': 'fasting autophagy cellular renewal', 'evidence': [{'pubmed_id': '36911497', 'url': 'https://pubmed.ncbi.nlm.nih.gov/36911497/', 'abstract': "This review article summarizes research indicating that **intermittent fasting (IF) demonstrates several positive health effects.** Key results show IF can **effectively reduce weight, fasting insulin, and blood glucose levels.** Beyond weight management, IF appears to **enhance antitumor activity of medications and improve neurological function**, potentially alleviating memory deficits.\n\nThe research highlights that IF achieves these benefits by **activating biological pathways promoting autophagy and cell renewal**, ultimately **inhibiting cancer cell proliferation, preventing spread, and delaying senescence (aging).**\n\nWhile promising, the review notes potential adverse effects and limitations relating to age and gender, emphasizing the **need for further, more systematic research** to fully understand IF's health benefits and ensure its safe application. The article aims to provide a foundation for future research and clinical use of IF.", 'relevance': ''}, {'pubmed_id': '32765037', 'url': 'https://pubmed.ncbi.nlm.nih.gov/32765037/', 'abstract': 'This study investigated the effects of puerarin on diabetic nephropathy (DN) in mice. Researchers found that puerarin improved renal function and podocyte health (indicated by increased nephrin, podocin, and podocalyxin expression) in a DN mouse model. \n\nSpecifically, puerarin treatment led to increased autophagy – a cellular “self-cleaning” process – as evidenced by upregulation of autophagy markers Beclin-1, LC3II, and Atg5, and downregulation of p62.  The study demonstrated that puerarin partially reversed the downregulation of the PERK/eIF2α/ATF4 signaling pathway observed in DN mice. \n\nThese results suggest that puerarin exerts a protective effect against DN by modulating autophagy, potentially through influencing the PERK/eIF2α/ATF4 pathway. The researchers conclude that targeting the PERK pathway may be a promising approach for treating diabetic nephropathy.', 'relevance': ''}, {'pubmed_id': '9034581', 'url': 'https://pubmed.ncbi.nlm.nih.gov/9034581/', 'abstract': 'This study investigated the impact of fasting and refeeding on the intestinal structure of laying hens. **Results demonstrated a clear relationship between nutritional status and intestinal villi height.** Villi height decreased with fasting duration across the small intestine (duodenum, jejunum, ileum), with the duodenum and jejunum showing more rapid reductions than the ileum.\n\n**Critically, a single day of refeeding fully restored villi height in the duodenum and jejunum, even after 20 days of fasting.** The ileum showed slower recovery. This indicates a higher absorptive capacity and responsiveness to nutrition in the proximal intestine.\n\n**Fasting reduced both epithelial cell area and cell division, but refeeding quickly stimulated cell renewal, directly impacting villi height.** Furthermore, prolonged fasting induced cellular autophagy (self-digestion) visible in intestinal cells, which was reversed by just one day of refeeding. \n\n**The findings suggest that forced molting is feasible and that a nutrient-rich diet can be reintroduced immediately after fasting periods.** The study also suggests that cellular autophagy observed in normally fed chickens may indicate underlying nutritional stress.', 'relevance': ''}]}, {'id': 2, 'text': 'Fasting may improve brain function by reducing toxicity.', 'verdict': 'uncertain', 'rationale': 'VERDICT: uncertain\nFINALSCORE: 0.65', 'score': 0.65, 'query': 'fasting brain function toxicity reduction', 'evidence': [{'pubmed_id': '37020122', 'url': 'https://pubmed.ncbi.nlm.nih.gov/37020122/', 'abstract': 'This study investigated the therapeutic effects of Cyclo-Z on an Alzheimer’s disease (AD) rat model. Researchers found that inducing AD with Aβ42 oligomers resulted in increased blood glucose, insulin, insulin resistance (HOMA-IR), and phospho-tau levels, alongside decreases in body weight, hippocampal & brain insulin, IRS-Ser612, GSK-3β, and impaired memory. Aβ42 also reduced left temporal spindle and delta power during anesthesia.\n\nTreatment with Cyclo-Z (10mg Zn+2/kg and 0.2mg CHP/kg) for 21 days *reversed* most of these Aβ42-induced changes. Specifically, Cyclo-Z normalized blood glucose, insulin levels, insulin resistance, body weight, hippocampal & brain insulin, IRS-Ser612, GSK-3β, and significantly improved memory function. While phospho-tau levels remained unaffected, Cyclo-Z reduced Aβ42 oligomer levels and restored the reduction in left temporal spindle power. \n\nThese results suggest Cyclo-Z effectively counteracts Aβ oligomer-induced disruptions to the insulin pathway and associated toxicity, potentially improving both cognitive function and neural network dynamics in this AD rat model.', 'relevance': ''}]}, {'id': 3, 'text': 'Fasting can increase ketone levels and support metabolism.', 'verdict': 'true', 'rationale': 'VERDICT: true\nFINALSCORE: 0.95', 'score': 0.95, 'query': 'fasting ketones metabolism support', 'evidence': [{'pubmed_id': '40349316', 'url': 'https://pubmed.ncbi.nlm.nih.gov/40349316/', 'abstract': 'This review article highlights that ketones are far more than just an alternative fuel source during fasting. Research demonstrates ketones fuel not only the brain, but also the heart and skeletal muscle, including during exercise. Importantly, the study reveals ketones function as *signalling molecules*, impacting gene expression and potentially influencing broader physiological processes.\n\nThe central finding is that disruptions in ketone metabolism are linked to pathologies, specifically **heart failure and type 2 diabetes**. The authors propose that *modifying* ketone metabolism could represent a therapeutic strategy for managing these conditions. The review aims to comprehensively examine the multifaceted roles of ketones in whole-body physiology and evaluate the potential of targeting ketone metabolism for therapeutic intervention in cardiometabolic diseases.', 'relevance': ''}, {'pubmed_id': '40129260', 'url': 'https://pubmed.ncbi.nlm.nih.gov/40129260/', 'abstract': 'This study investigated the impact of a 60-hour fast, with and without added exercise, on metabolic and immunological markers in healthy young adults. Results showed both fasting alone (FAST) and fasting with exercise (FEX) effectively reduced glucose and insulin levels, and increased ketone body (BHB) concentrations, with exercise accelerating these metabolic shifts.\n\nRegarding immune responses, both conditions decreased the pro-inflammatory cytokine TNF-α and increased the anti-inflammatory cytokine IL-10. The increase in IL-10 was more pronounced with exercise. There was no significant change in total white blood cell count or major leukocyte populations. \n\nUltimately, the study demonstrated that prolonged fasting elicits an anti-inflammatory effect, but adding exercise to the fast *did not* significantly alter systemic cytokine or leukocyte responses, despite increasing metabolic strain (indicated by higher BHB levels). These findings suggest that while exercise enhances metabolic adaptations during fasting, it doesn’t necessarily amplify the associated immunological changes.', 'relevance': ''}, {'pubmed_id': '39991337', 'url': 'https://pubmed.ncbi.nlm.nih.gov/39991337/', 'abstract': 'This study investigated the efficacy of a milk-based ketogenic diet (KD) in eight infants (1-6 months old) with genetic drug-resistant epilepsy (DRE). Researchers found a **significant reduction in seizure frequency**, decreasing from a mean of 16.5 seizures/day at baseline to **4.6 seizures/day after six months** on the KD (p < 0.001). \n\nAlongside seizure control, the study demonstrated **improved nutritional status** in the infants. Importantly, **biochemical parameters remained stable**, indicating the KD was well-tolerated and safe, with no significant changes in triglycerides or random blood sugar. Consistent high levels of urine ketones confirmed sustained ketosis and diet adherence. \n\nThe results suggest a milk-based KD is an effective and safe treatment option for reducing seizures and supporting nutritional development in infants with genetic DRE. Researchers highlight the importance of consistent monitoring and parental guidance, while advocating for further research with larger groups to optimize dietary protocols.', 'relevance': ''}]}], 'overall_truthiness': 0.85, 'generated_at': '2025-05-17T07:09:39Z'}
    return data