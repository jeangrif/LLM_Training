# generate_artifacts.py
# Usage: python generate_artifacts.py

import json, csv, re, unicodedata, math, os
from collections import Counter
from statistics import mean

# ==========> PARAMÈTRES HARDCODÉS <==========
JSONL_PATH = "/Users/jeangrifnee/PycharmProjects/LLMTraining/outputs/2025-11-12/14-20-12/eval_metrics.jsonl"
FIELD_PRED = "pred"
FIELD_EM = "exact_match"
FIELD_FAITH = "faithfulness"
FIELD_SEM = "semantic_similarity"
TREAT_MISSING_EM_AS = 0.0
MAX_EXAMPLES_PER_BUCKET = {"correct":3,"near_miss":5,"wrong_supported":5,"hallucination":5,"other_error":5}
CUT_STEPS = list(range(0, 55, 5))  # 0..50
# Seuils pour taxonomie
FAITH_HALLU_MAX, SEM_HALLU_MAX, JACC_HALLU_MAX = 0.40, 0.30, 0.10
SEM_NEARMISS_MIN = 0.80
FAITH_SUPPORTED_MIN, JACC_SUPPORTED_MIN = 0.70, 0.20
# ============================================

# --- Utils
def ensure_dir(p): os.makedirs(p, exist_ok=True)
OUTDIR = os.path.dirname(JSONL_PATH)
ensure_dir(OUTDIR)

def to_float01(x, default=0.0):
    try:
        if isinstance(x, bool): return 1.0 if x else 0.0
        v = float(x); return 1.0 if v > 0.5 else 0.0
    except: return default

def safe_float(x):
    try: return float(x)
    except: return float("nan")

def wilson_ci(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; d = 1+z*z/n
    c = (p + z*z/(2*n))/d
    m = z*math.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/d
    return (max(0.0, c-m), min(1.0, c+m))

# --- "don't know"
DONT_KNOW_PATTERNS = [
    r"\bdon'?t\s+know\b", r"\b(do|does)\s+not\s+know\b", r"\bno\s+idea\b",
    r"\bnot\s+sure\b", r"\bunknown\b", r"\bidk\b",
    r"\bje\s+ne\s+sais\s+pas\b", r"\bje\s+ne\s+peux\s+pas\s+répondre\b",
    r"\bimpossible\s+de\s+dire\b", r"\bne\s+sais\s+pas\b", r"\binconnu\b",
]
PATTERNS = [re.compile(p, re.IGNORECASE) for p in DONT_KNOW_PATTERNS]
def is_dont_know(text):
    if text is None: return False
    t = str(text).strip()
    return bool(t) and any(p.search(t) for p in PATTERNS)

# --- Jaccard & normalisation
STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by","from","at","as","that",
    "is","are","was","were","be","been","it","its","this","these","those","their","there",
    "le","la","les","un","une","des","du","de","d","et","ou","au","aux","dans","par",
    "pour","avec","sur","en","est","sont","ete","été","etre","être","ce","cet","cette","ces","il","elle","ils","elles"
}
WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+")

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def tokens_content(s: str):
    s = normalize_text(s)
    toks = WORD_RE.findall(s)
    return {t for t in toks if t not in STOPWORDS and len(t) >= 2}

def jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b); union = len(a | b)
    return inter/union if union else 0.0

def jacc_per_passage(pred, ctxs):
    vals = []
    if isinstance(ctxs, list):
        seq = ctxs
    else:
        seq = [str(ctxs)]
    for i, c in enumerate(seq):
        vals.append((i, jaccard(tokens_content(pred), tokens_content(c or ""))))
    if not vals: return -1, 0.0
    vals.sort(key=lambda x: x[1], reverse=True)
    return vals[0]  # (idx, value)

# --- Answer-contained / numérique light (pour marquage diagnostic, pas pour le score officiel)
def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("–","-").replace("—","-").replace("%"," % ")
    s = re.sub(r"[^\w\s%.-]", " ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

WORD_NUM = {"thousand":1_000,"million":1_000_000,"billion":1_000_000_000,"mille":1_000,"million_fr":1_000_000,"milliard":1_000_000_000}
def extract_number_with_unit(s: str):
    s = norm(s); has_percent = "%" in s
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(million|billion|thousand|mille|milliard)?", s)
    val = None
    if m:
        num = float(m.group(1).replace(",", "."))
        mult = WORD_NUM.get(m.group(2) or "", 1)
        val = num * mult
    return val, has_percent

UNIT_SYNONYMS = {"download":"time", "downloads":"times", "time":"time", "times":"times"}
def unit_token(s: str):
    s = norm(s)
    for u in UNIT_SYNONYMS:
        if re.search(rf"\b{u}\b", s): return UNIT_SYNONYMS[u]
    return None

def answer_contained_or_numeric_equiv(pred: str, gold: str) -> bool:
    pn, gn = norm(pred), norm(gold)
    if gn and gn in pn: return True
    # pourcentages / intervalles
    g_pct = re.findall(r"\d+(?:[.,]\d+)?\s*%", gn)
    if g_pct and all(norm(t) in pn for t in g_pct): return True
    # nombre + unité lâche
    g_val, g_is_pct = extract_number_with_unit(gold)
    p_val, p_is_pct = extract_number_with_unit(pred)
    if g_val is not None and p_val is not None:
        if g_is_pct and p_is_pct and abs(p_val - g_val) <= 0.005 * max(1.0, g_val):
            return True
        g_unit, p_unit = unit_token(gold), unit_token(pred)
        unit_compatible = (g_unit is None or p_unit is None or g_unit == p_unit)
        same_mag = (abs(p_val - g_val) <= 0.02 * max(1.0, g_val))
        over_in = lambda s: re.search(r"\bover\b|\bplus de\b", norm(s)) is not None
        if unit_compatible and (same_mag or over_in(gold) or over_in(pred)):
            return True
    return False

# =========== PASS 1 : lecture & stats globales ===========
total=0; abstain=0; attempts=0; correct=0; incorrect=0
em_all=[]; em_attempts=[]
faith_attempts=[]; sem_attempts=[]
rows_csv=[]
examples={k:[] for k in ["correct","near_miss","wrong_supported","hallucination","other_error"]}

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line: continue
        total += 1
        obj = json.loads(line)

        pred = obj.get(FIELD_PRED, "")
        em = to_float01(obj.get(FIELD_EM, TREAT_MISSING_EM_AS), TREAT_MISSING_EM_AS)
        faith = safe_float(obj.get(FIELD_FAITH))
        sem = safe_float(obj.get(FIELD_SEM))
        gold = obj.get("answer","")
        query = obj.get("query","")
        ctxs = obj.get("contexts") or []

        em_all.append(em)

        if is_dont_know(pred):
            abstain += 1
            continue

        # tentative
        attempts += 1
        em_attempts.append(em)
        if not math.isnan(faith): faith_attempts.append(faith)
        if not math.isnan(sem): sem_attempts.append(sem)

        # jaccard global + par passage
        ctx_concat = "\n".join(ctxs) if isinstance(ctxs,list) else str(ctxs)
        jac_global = jaccard(tokens_content(pred), tokens_content(ctx_concat))
        ctx_idx_max, jac_max = jacc_per_passage(pred, ctxs)

        # bucketisation
        if em == 1.0:
            bucket = "correct"; correct += 1
        else:
            if (not math.isnan(sem)) and sem >= SEM_NEARMISS_MIN:
                bucket = "near_miss"
            elif ((not math.isnan(faith) and faith >= FAITH_SUPPORTED_MIN) or jac_global >= JACC_SUPPORTED_MIN):
                bucket = "wrong_supported"
            elif ((not math.isnan(faith) and faith <= FAITH_HALLU_MAX) and
                  (not math.isnan(sem) and sem <= SEM_HALLU_MAX) and jac_global <= JACC_HALLU_MAX):
                bucket = "hallucination"
            else:
                bucket = "other_error"
            incorrect += 1

        # flags & diag
        is_numeric_q = bool(re.search(r"\d|%|\b(how many|combien)\b", query.lower()))
        is_list_q    = bool(re.search(r",|\band\b|\bou\b", query.lower()))
        is_contrast_q= bool(re.search(r"\b(diff(erence)?|vs)\b", query.lower()))
        ans_contained = answer_contained_or_numeric_equiv(pred, gold)

        # ligne CSV
        rows_csv.append({
            "id": total, "bucket": bucket, "exact_match": em,
            "faithfulness": faith, "semantic_similarity": sem,
            "jaccard_ctx": jac_global, "jaccard_max": jac_max, "ctx_idx_max": ctx_idx_max,
            "answer_contained": int(ans_contained),
            "query_len": len(query), "pred_len": len(pred), "answer_len": len(str(gold)),
            "is_numeric_q": int(is_numeric_q), "is_list_q": int(is_list_q), "is_contrast_q": int(is_contrast_q),
        })

        # exemples JSONL
        if len(examples[bucket]) < MAX_EXAMPLES_PER_BUCKET[bucket]:
            ex = {
                "query": query, "pred": pred, "answer": gold,
                "exact_match": obj.get(FIELD_EM),
                "faithfulness": obj.get(FIELD_FAITH),
                "semantic_similarity": obj.get(FIELD_SEM),
                "jaccard_ctx": round(jac_global,3), "jaccard_max": round(jac_max,3), "ctx_idx_max": ctx_idx_max,
                "answer_contained": ans_contained,
                "top_context": (ctxs[ctx_idx_max] if isinstance(ctxs,list) and 0<=ctx_idx_max<len(ctxs) else None)
            }
            examples[bucket].append(ex)

# =========== Écritures fichiers ===========
# 1) CSV lignes
csv_path = os.path.join(OUTDIR, "analysis_attempts.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as cf:
    w = csv.writer(cf)
    headers = ["id","bucket","exact_match","faithfulness","semantic_similarity","jaccard_ctx","jaccard_max","ctx_idx_max",
               "answer_contained","query_len","pred_len","answer_len","is_numeric_q","is_list_q","is_contrast_q"]
    w.writerow(headers)
    for r in rows_csv:
        w.writerow([r[h] for h in headers])

# 2) JSONL exemples par bucket
for b, lst in examples.items():
    path = os.path.join(OUTDIR, f"examples_{b}.jsonl")
    with open(path, "w", encoding="utf-8") as jf:
        for ex in lst:
            jf.write(json.dumps(ex, ensure_ascii=False) + "\n")

# 3) taxonomy_counts.csv
cats = Counter([r["bucket"] for r in rows_csv])
tax_path = os.path.join(OUTDIR, "taxonomy_counts.csv")
with open(tax_path, "w", newline="", encoding="utf-8") as tf:
    w = csv.writer(tf)
    w.writerow(["bucket","count","share_attempts","ci95_low","ci95_high"])
    n = attempts
    for k in ["correct","near_miss","wrong_supported","hallucination","other_error"]:
        v = cats.get(k,0); rate = v/n if n else float("nan")
        ci = wilson_ci(v, n)
        w.writerow([k, v, f"{rate:.6f}", f"{ci[0]:.6f}", f"{ci[1]:.6f}"])

# 4) risk-coverage sweeps (sem, faith) sur tentatives
def risk_cov(scores, ems, cut_steps):
    # trie croissant, coupe x% bas, calcule EM_cond
    pairs = [(s,e) for s,e in zip(scores, ems) if not (isinstance(s,float) and math.isnan(s))]
    pairs.sort(key=lambda x: x[0])
    n = len(pairs); rows=[]
    for cut in cut_steps:
        k = math.floor(n*(cut/100))
        kept = pairs[k:] if k < n else []
        cov = len(kept)/n if n else float("nan")
        if kept:
            emc = sum(e for _,e in kept)/len(kept); err = 1-emc
        else:
            emc = float("nan"); err = float("nan")
        rows.append({"cut_%":cut,"coverage":cov,"kept_n":len(kept),"EM_cond":emc,"Err_cond":err})
    return rows

def export_rc(name, rows):
    p = os.path.join(OUTDIR, f"risk_coverage_{name}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["cut_%","coverage","kept_n","EM_cond","Err_cond"])
        for r in rows:
            w.writerow([r["cut_%"], f"{r['coverage']:.6f}", r["kept_n"], f"{r['EM_cond']:.6f}", f"{r['Err_cond']:.6f}"])

ems_attempts_int = [int(v) for v in em_attempts]
if sem_attempts:
    export_rc("sem", risk_cov(sem_attempts, ems_attempts_int, CUT_STEPS))
if faith_attempts:
    export_rc("faith", risk_cov(faith_attempts, ems_attempts_int, CUT_STEPS))

# 5) déciles (sem, faith)
def deciles(scores, ems):
    pairs = [(s,e) for s,e in zip(scores,ems) if not (isinstance(s,float) and math.isnan(s))]
    pairs.sort(key=lambda x:x[0])
    n=len(pairs); rows=[]
    for d in range(10):
        a=math.floor(n*(d/10)); b=math.floor(n*((d+1)/10))
        bucket=pairs[a:b]
        em = (sum(e for _,e in bucket)/len(bucket)) if bucket else float("nan")
        rows.append({"decile":d+1,"n":len(bucket),"EM":em})
    return rows

def export_dec(name, rows):
    p=os.path.join(OUTDIR, f"deciles_{name}.csv")
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["decile","n","EM"])
        for r in rows:
            w.writerow([r["decile"], r["n"], f"{r['EM']:.6f}" if not (isinstance(r['EM'],float) and math.isnan(r['EM'])) else "NaN"])

if sem_attempts:
    export_dec("sem", deciles(sem_attempts, ems_attempts_int))
if faith_attempts:
    export_dec("faith", deciles(faith_attempts, ems_attempts_int))

# 6) summary.txt
em_global = mean(em_all) if em_all else float("nan")
em_cond   = mean(em_attempts) if em_attempts else float("nan")
err_cond  = (1-em_cond) if not math.isnan(em_cond) else float("nan")
hallus = cats.get("hallucination",0); hallu_rate = hallus/attempts if attempts else float("nan")
em_cond_ci = wilson_ci(int(sum(em_attempts)), attempts) if attempts else (float("nan"), float("nan"))
hallu_ci = wilson_ci(hallus, attempts) if attempts else (float("nan"), float("nan"))

with open(os.path.join(OUTDIR,"summary.txt"),"w",encoding="utf-8") as sf:
    sf.write(f"File: {JSONL_PATH}\n")
    sf.write(f"Total: {total} | Abstain: {abstain} ({(abstain/total):.2%}) | Attempts: {attempts} ({(attempts/total):.2%})\n")
    sf.write(f"EM (global): {em_global:.4f}\n")
    sf.write(f"EM_cond (attempts): {em_cond:.4f} | 95% CI: [{em_cond_ci[0]:.4f}, {em_cond_ci[1]:.4f}]\n")
    sf.write(f"Err_cond: {err_cond:.4f}\n")
    sf.write(f"Hallucinations (strict): {hallus}/{attempts} ({hallu_rate:.2%}) | 95% CI: [{hallu_ci[0]:.2%}, {hallu_ci[1]:.2%}]\n")
    sf.write("Artifacts: analysis_attempts.csv, taxonomy_counts.csv, risk_coverage_*.csv, deciles_*.csv, examples_*.jsonl\n")

print("Artifacts written to:", OUTDIR)
