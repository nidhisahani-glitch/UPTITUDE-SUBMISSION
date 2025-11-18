import os
import re
import json
import time
import csv
import math
import pathlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import requests
import pdfplumber
import pandas as pd

# Optional embed imports 
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False
    SentenceTransformer = None
    np = None

# -----------------------
# CONFIG
# -----------------------
LM_STUDIO_HOST = os.environ.get("LM_STUDIO_HOST", "http://192.168.1.34:1234")
MODELS_API = f"{LM_STUDIO_HOST.rstrip('/')}/v1/models"
CHAT_API = f"{LM_STUDIO_HOST.rstrip('/')}/v1/chat/completions"

PDF_ROOT = Path("data/full_contract_pdf")  # root folder with PDFs
MAX_DOCS = 10                             # number of contracts to process; 
OUTPUT_DIR = Path("outputs")
RAW_DIR = OUTPUT_DIR / "raw"
OUTPUT_JSON = OUTPUT_DIR / "results.json"
OUTPUT_CSV = OUTPUT_DIR / "results.csv"

# Chunking tuned for small models
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200

# Semantic search settings (optional). If embedding model not available, will disable.
EMBED_MODEL = "all-MiniLM-L6-v2"
USE_SEMANTIC_PREFILTER = True
TOP_K_PER_CLAUSE = 6    # more candidates for small model
BATCH_SIZE = 4

CLAUSE_TYPES = ["Termination", "Confidentiality", "Liability"]

# Few-shot examples (legal-focused, instructive)
FEW_SHOT_EXAMPLES = [
    {
        "chunk": "Either Party may terminate this Agreement upon thirty (30) days prior written notice. Immediate termination is permitted in the event of material breach not cured within ten (10) days.",
        "json": {
            "Termination": [
                {
                    "text":"Either Party may terminate this Agreement upon thirty (30) days prior written notice.",
                    "start":0,"end":86,"confidence":0.98
                },
                {
                    "text":"Immediate termination is permitted in the event of material breach not cured within ten (10) days.",
                    "start":88,"end":184,"confidence":0.97
                }
            ],
            "Confidentiality": [],
            "Liability": []
        }
    },
    {
        "chunk":"Recipient shall keep all Confidential Information strictly confidential and shall not disclose it to any third party for five (5) years, except as required by law.",
        "json": {
            "Termination": [],
            "Confidentiality": [
                {"text":"Recipient shall keep all Confidential Information strictly confidential and shall not disclose it to any third party for five (5) years", "start":0,"end":130,"confidence":0.96}
            ],
            "Liability": []
        }
    },
    {
        "chunk":"Supplier’s total liability shall not exceed the fees paid by Customer in the preceding twelve (12) months. Supplier shall not be liable for indirect, incidental, or consequential damages.",
        "json": {
            "Termination": [],
            "Confidentiality": [],
            "Liability": [
                {"text":"Supplier’s total liability shall not exceed the fees paid by Customer in the preceding twelve (12) months.", "start":0,"end":116,"confidence":0.95}
            ]
        }
    }
]

# Logging / safety
TIMEOUT = 300  # LM Studio HTTP timeout seconds
SAVE_RAW = True

# -----------------------
# Utilities
# -----------------------
def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def list_first_n_pdfs(root: Path, n: int) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"PDF root not found: {root}")
    pdfs = sorted([p for p in root.rglob("*.pdf")])
    return pdfs[:min(n, len(pdfs))]

def extract_text_from_pdf(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append(text)
    return "\n\n".join(pages)

def normalize_text(text: str) -> str:
    if not text:
        return ""
    # remove lines that are short and repeat frequently
    # Basic cleanup
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"-\n(\w)", r"\1", text)    
    # collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # strip weird unicode
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return text.strip()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[int,int,str]]:
    chunks = []
    start = 0
    L = len(text)
    if L == 0:
        return []
    while start < L:
        end = min(start + size, L)
        chunks.append((start, end, text[start:end]))
        if end >= L:
            break
        start = max(0, end - overlap)
    return chunks

# -----------------------
# LM Studio client
# -----------------------
def get_active_model() -> Optional[str]:
    try:
        r = requests.get(MODELS_API, timeout=5)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") or j.get("models") or []
        if data:
            # sometimes 'data' is a list of model dicts
            if isinstance(data, list):
                first = data[0]
                if isinstance(first, dict):
                    return first.get("id") or first.get("model") or first.get("name")
                return first
            return data
        return None
    except Exception:
        return None

def call_llm(messages: List[Dict[str,str]], temperature: float = 0.0, max_retries: int = 3, timeout: int = TIMEOUT) -> str:
    model_id = get_active_model()
    if not model_id:
        raise RuntimeError("No active model found in LM Studio. Load your GGUF model and restart server.")
    payload = {"model": model_id, "messages": messages, "temperature": temperature}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(CHAT_API, json=payload, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            # extraction of textual content
            content = None
            if isinstance(j, dict):
                choices = j.get("choices") or j.get("results") or j.get("outputs") or []
                if choices:
                    c0 = choices[0]
                    if isinstance(c0, dict):
                        # common shapes
                        msg = c0.get("message") or c0.get("output") or c0
                        if isinstance(msg, dict):
                            content = msg.get("content") or msg.get("text") or msg.get("body")
                        else:
                            content = c0.get("text") or c0.get("content") or None
                    elif isinstance(c0, str):
                        content = c0
            if content is None:
                # try top-level
                content = j.get("text") or j.get("content")
            if content is None:
                raise ValueError("No textual content in model response.")
            return content
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt*1.5)
    raise last_err

# -----------------------
# JSON extraction 
# -----------------------
def extract_json_from_text(text: str) -> Optional[Dict[str,Any]]:
    if not text or not isinstance(text, str):
        return None
    # normalize
    t = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
    # remove common prefixes
    prefixes = [
        "Here is the JSON you requested:", "Sure, here is the JSON:", "I will now provide the JSON:",
        "```json", "```", "JSON:", "Result:"
    ]
    for p in prefixes:
        if t.strip().startswith(p):
            t = t.strip()[len(p):].strip()
    # find first balanced JSON object
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # try to find any {...} substring
        matches = re.findall(r"\{.*\}", t, flags=re.DOTALL)
        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                continue
        return None
    candidate = t[start:end+1]
    # fix common issues: trailing commas
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)
    # ensure keys quoted 
    candidate2 = re.sub(r'([{\[,]\s*)([A-Za-z0-9_]+)\s*:', r'\1"\2":', candidate)
    tries = [candidate, candidate2]
    for c in tries:
        try:
            return json.loads(c)
        except Exception:
            continue
    # try balanced extraction
    stack = []
    for i, ch in enumerate(candidate):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start_idx = stack.pop(0) if False else None  
    # final fallback: regex matches
    matches = re.findall(r"\{(?:[^{}]|\{[^}]*\})*\}", t, flags=re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except Exception:
            continue
    return None

# -----------------------
# Semantic search helper (optional)
# -----------------------
class SemanticSearcher:
    def __init__(self, model_name=EMBED_MODEL):
        self.enabled = False
        if not HAS_SENTENCE_TRANSFORMERS:
            print("[WARN] sentence-transformers not available, semantic prefilter disabled.")
            return
        try:
            print(f"[INFO] Loading embedding model: {model_name} ...")
            self.model = SentenceTransformer(model_name)
            self.enabled = True
        except Exception as e:
            print(f"[WARN] Failed to load embedding model ({model_name}): {e}")
            self.enabled = False

    def embed(self, texts: List[str]):
        if not self.enabled:
            return None
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def cosine_sim(a, b):
        # a: (n, d) b: (m, d) -> (n, m)
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a_norm, b_norm.T)

    def rank_chunks(self, chunks: List[str], queries: List[str], top_k: int = TOP_K_PER_CLAUSE) -> List[int]:
        if not self.enabled or not chunks:
            # fallback: return first K chunk indices
            return list(range(min(top_k, len(chunks))))
        chunk_embs = self.embed(chunks)
        query_embs = self.embed(queries)
        sims = self.cosine_sim(chunk_embs, query_embs)
        max_sim = sims.max(axis=1)
        idxs = np.argsort(-max_sim)[:top_k]
        return [int(i) for i in idxs]

# -----------------------
# Prompt builders (strict JSON)
# -----------------------
def build_clause_prompt(chunk: str) -> str:
    few_shot_str = ""
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_str += json.dumps(ex["json"], ensure_ascii=False) + "\n\n"
    prompt = f"""
You are a legal text extraction engine.
You MUST return VALID JSON ONLY — no prose, no headings, no markdown.

Task:
Extract EXACT sentences from the CHUNK that correspond to:
- Termination conditions
- Confidentiality obligations
- Liability limitations

Rules:
- Do NOT summarize or paraphrase; extract exact text.
- Return only the JSON object matching this schema:
{{
"Termination": [{{"text":"...", "start":0, "end":0, "confidence":0.9}}],
"Confidentiality": [],
"Liability": []
}}
- start/end are offsets relative to the CHUNK (0-based).
- If none, return an empty list [] for that key.
- Confidence should be between 0.0 and 1.0.
- Do NOT include extra keys.

Few-shot JSON examples (format only):
{few_shot_str}

CHUNK:
```{chunk}```
"""
    return prompt

def build_summary_prompt(fulltext: str) -> str:
    prompt = f"""
You are a careful legal summarizer. Produce ONE paragraph of 120-150 words that is strictly based on the provided contract text.
Rules:
- Use ONLY facts from the contract. Do NOT invent or assume.
- If a requested element is missing, write "Not specified" for that element.
- Mention: purpose of agreement, key obligations of parties, notable risks/penalties/termination notes.
- Return ONLY the paragraph (no headings).
Contract:
```{fulltext}```
"""
    return prompt

# -----------------------
# regex backup extractor
# -----------------------
def regex_extract_backup(text: str) -> Dict[str, List[Dict[str, Any]]]:
    results = {k: [] for k in CLAUSE_TYPES}
    if not text:
        return results
    # Patterns: capture up to 300-500 chars after keyword 
    term_pats = [r"(terminate(?:s|d)?[^.]{0,400}\.)", r"(termination[^.]{0,400}\.)"]
    conf_pats = [r"(confidential[^.]{0,400}\.)", r"(non-disclos[^.]{0,400}\.)", r"(Confidential Information[^.]{0,400}\.)"]
    liab_pats = [r"(liabilit[^.]{0,400}\.)", r"(indemnif[^.]{0,400}\.)", r"(limitation[^.]{0,400}\.)"]

    def find(pats, key):
        for p in pats:
            for m in re.finditer(p, text, flags=re.IGNORECASE | re.DOTALL):
                s, e = m.start(), m.end()
                snippet = text[s:e].strip()
                if len(snippet) < 10:
                    continue
                results[key].append({"text": snippet, "start": s, "end": e, "confidence": 0.75})
    find(term_pats, "Termination")
    find(conf_pats, "Confidentiality")
    find(liab_pats, "Liability")
    return results

# -----------------------
# Clause extraction
# -----------------------
def extract_clauses_for_document(fulltext: str, searcher: SemanticSearcher) -> Tuple[Dict[str,Any], Dict[str,List[Dict[str,Any]]]]:
    ensure_dirs()
    chunks_meta = chunk_text(fulltext)
    chunk_texts = [c[2] for c in chunks_meta]
    # semantic prefilter
    selected_indices = set()
    for ctype in CLAUSE_TYPES:
        queries = {
            "Termination": ["terminate", "termination", "notice", "cure period", "involuntary withdrawal"],
            "Confidentiality": ["confidential", "confidentiality", "non-disclosure", "NDA", "Confidential Information"],
            "Liability": ["liability", "limitation of liability", "indemnity", "damages", "cap on liability"]
        }.get(ctype, [ctype])
        idxs = searcher.rank_chunks(chunk_texts, queries, top_k=TOP_K_PER_CLAUSE)
        selected_indices.update(idxs)
    # also include first 2 chunks for context
    for i in range(min(2, len(chunk_texts))):
        selected_indices.add(i)
    selected_indices = sorted(selected_indices)
    metas = []
    prompts = []
    for idx in selected_indices:
        s,e,ch = chunks_meta[idx]
        prompts.append(build_clause_prompt(ch))
        metas.append((idx,s,e,ch))
    # call LLM for each prompt and collect outputs
    responses = []
    for i,p in enumerate(prompts):
        messages = [{"role":"system","content":"ONLY output valid JSON. No explanations."},
                    {"role":"user","content":p}]
        try:
            out = call_llm(messages, temperature=0.0)
        except Exception as ex:
            print(f"[WARN] LLM call failed for chunk {i}: {ex}")
            out = ""
        # save raw
        if SAVE_RAW:
            try:
                fname = RAW_DIR / f"raw_chunk_{metas[i][0]}_{int(time.time()*1000)}.txt"
                with open(fname, "w", encoding="utf-8") as rf:
                    rf.write(out)
            except Exception:
                pass
        responses.append(out)
    # parse & aggregate
    aggregated = {k: [] for k in CLAUSE_TYPES}
    for out, (idx, start, end, chunk) in zip(responses, metas):
        text_out = out.strip()
        # quick cleaning of leading fences
        if text_out.startswith("```"):
            text_out = re.sub(r"^```(?:json)?", "", text_out).strip("` \n")
        parsed = extract_json_from_text(text_out)
        if not isinstance(parsed, dict):
            parsed = {k: [] for k in CLAUSE_TYPES}
        for ctype in CLAUSE_TYPES:
            items = parsed.get(ctype, [])
            if not isinstance(items, list):
                continue
            for it in items:
                t = it.get("text","").strip()
                if not t:
                    continue
                s_rel = int(it.get("start", 0))
                e_rel = int(it.get("end", min(len(chunk), s_rel + len(t))))
                conf = float(it.get("confidence", 0.5))
                # clamp
                s_rel = max(0, min(s_rel, len(chunk)))
                e_rel = max(s_rel, min(e_rel, len(chunk)))
                aggregated[ctype].append({
                    "text": t,
                    "start": start + s_rel,
                    "end": start + e_rel,
                    "confidence": conf,
                    "source_chunk_idx": idx
                })
    # deduplicate by identical text
    for ctype in CLAUSE_TYPES:
        seen = set()
        uniq = []
        for item in aggregated[ctype]:
            key = item.get("text","").strip()[:300]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(item)
        aggregated[ctype] = uniq
    # fallback: for each missing clause, try LLM full-document strong fallback and regex
    for ctype in CLAUSE_TYPES:
        if aggregated[ctype]:
            continue
        # 1) regex on whole doc
        regex_hits = regex_extract_backup(fulltext).get(ctype, [])
        if regex_hits:
            # convert to same structure
            for it in regex_hits:
                it["source_chunk_idx"] = "regex_backup"
            aggregated[ctype].extend(regex_hits)
        # 2) strong doc-level LLM extraction
        if not aggregated[ctype]:
            strong_prompt = f"""
You are a legal clause extractor. Return STRICT JSON ONLY.

Extract only clause type: {ctype}

Return JSON like:
{{ "{ctype}": [{{"text":"...", "start": <int>, "end": <int>, "confidence": <float>}}] }}

Rules:
- Use exact text from the contract.
- start/end are absolute character offsets (0-based) in the full document.
- If none, return an empty list.

Document:
```{fulltext}```
"""
            messages = [{"role":"system","content":"Return ONLY valid JSON."},
                        {"role":"user","content":strong_prompt}]
            try:
                strong_out = call_llm(messages, temperature=0.0)
            except Exception as e:
                strong_out = ""
            if SAVE_RAW:
                try:
                    fname = RAW_DIR / f"raw_strong_{ctype}_{int(time.time()*1000)}.txt"
                    with open(fname, "w", encoding="utf-8") as f:
                        f.write(strong_out)
                except Exception:
                    pass
            parsed = extract_json_from_text(strong_out)
            if isinstance(parsed, dict):
                for it in parsed.get(ctype, []):
                    t = it.get("text","").strip()
                    if not t:
                        continue
                    aggregated[ctype].append({
                        "text": t,
                        "start": it.get("start"),
                        "end": it.get("end"),
                        "confidence": float(it.get("confidence", 0.6)),
                        "source_chunk_idx": "strong_fallback"
                    })
    # pick best by confidence, fallback to longest text if confidence ties
    best = {}
    for ctype in CLAUSE_TYPES:
        lst = aggregated.get(ctype, [])
        if not lst:
            best[ctype] = {"text": None, "start": None, "end": None, "confidence": 0.0}
            continue
        # normalize missing confidence
        for it in lst:
            it["confidence"] = float(it.get("confidence", 0.0))
        lst_sorted = sorted(lst, key=lambda x: (x.get("confidence",0.0), len(x.get("text",""))), reverse=True)
        best_item = lst_sorted[0]
        # sometimes start/end None -> keep text only
        best[ctype] = {
            "text": best_item.get("text"),
            "start": best_item.get("start"),
            "end": best_item.get("end"),
            "confidence": best_item.get("confidence", 0.0)
        }
    return best, aggregated

# -----------------------
# Summary generation 
# -----------------------
def generate_summary(fulltext: str) -> str:
    prompt = build_summary_prompt(fulltext[:6000])  # give model first N char
    messages = [{"role":"system","content":"You summarize contracts strictly from provided text. No invention."},
                {"role":"user","content":prompt}]
    try:
        out = call_llm(messages, temperature=0.0)
    except Exception as e:
        print(f"[WARN] Summary LLM call failed: {e}")
        return "Summary generation failed."

    words = out.strip().split()
    if 120 <= len(words) <= 150:
        return out.strip()
    # reformat request
    messages2 = [{"role":"system","content":"Return exactly 120-150 words. No extra text."},
                {"role":"user","content":"Please reformat the following into 120-150 words exactly:\n\n" + out}]
    try:
        out2 = call_llm(messages2, temperature=0.0)
        words2 = out2.strip().split()
        if 120 <= len(words2) <= 150:
            return out2.strip()
    except Exception:
        pass
    # fallback: trim or pad
    if len(words) > 150:
        return " ".join(words[:150])
    if len(words) < 120:
        return out.strip() + " Not specified."
    return out.strip()

# -----------------------
# Main
# -----------------------
def main():
    ensure_dirs()
    print(f"[INFO] Searching PDFs under: {PDF_ROOT} (first {MAX_DOCS})")
    pdfs = list_first_n_pdfs(PDF_ROOT, MAX_DOCS)
    if not pdfs:
        print("[ERROR] No PDFs found.")
        return
    print(f"[INFO] Found {len(pdfs)} PDFs (processing first {len(pdfs)})")
    # LM Studio model check
    active = get_active_model()
    if not active:
        print(f"[ERROR] No active model in LM Studio at {MODELS_API}. Load your GGUF model and restart.")
        return
    print(f"[INFO] LM Studio active model: {active}")
    # semantic search init
    searcher = SemanticSearcher(EMBED_MODEL) if USE_SEMANTIC_PREFILTER else SemanticSearcher("")  # will disable if transformer not present
    # process pdfs
    results = []
    for pdf_path in pdfs:
        print(f"\n[PROCESS] {pdf_path.name}")
        try:
            raw = extract_text_from_pdf(pdf_path)
            fulltext = normalize_text(raw)
            best, aggregated = extract_clauses_for_document(fulltext, searcher)
        except Exception as e:
            print(f"[ERROR] Clause extraction failed for {pdf_path.name}: {e}")
            best = {k: {"text": None, "start": None, "end": None, "confidence": 0.0} for k in CLAUSE_TYPES}
            aggregated = {k: [] for k in CLAUSE_TYPES}
        try:
            summary = generate_summary(fulltext)
        except Exception as e:
            print(f"[ERROR] Summary generation failed for {pdf_path.name}: {e}")
            summary = "Summary generation failed."
        row = {
            "contract_id": pdf_path.stem,
            "source_file": str(pdf_path),
            "summary": summary,
            "termination_clause": best["Termination"]["text"],
            "termination_start": best["Termination"]["start"],
            "termination_end": best["Termination"]["end"],
            "termination_confidence": best["Termination"]["confidence"],
            "confidentiality_clause": best["Confidentiality"]["text"],
            "confidentiality_start": best["Confidentiality"]["start"],
            "confidentiality_end": best["Confidentiality"]["end"],
            "confidentiality_confidence": best["Confidentiality"]["confidence"],
            "liability_clause": best["Liability"]["text"],
            "liability_start": best["Liability"]["start"],
            "liability_end": best["Liability"]["end"],
            "liability_confidence": best["Liability"]["confidence"],
        }
        results.append(row)
        # small delay
        time.sleep(0.2)
    # write outputs
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[COMPLETE] Results written to:\n - {OUTPUT_JSON}\n - {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
