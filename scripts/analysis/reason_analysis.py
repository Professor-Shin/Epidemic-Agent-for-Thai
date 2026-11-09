"""Reason-field analysis behind section 5.4 of report/final_report_TH.md.

Uniqueness is counted on the stripped, lower-cased reason; the keyword sets are deliberately bilingual even
though no model ever answered in Thai (that absence is itself a reported result).

    python3 scripts/analysis/reason_analysis.py
"""
import json, re, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNS = {"llama8b": "C_llama8b_s1_7d79571f", "typhoon8b": "C_typhoon8b_s1_f5137aa1",
        "qwen7b": "C_qwen7b_s1_42b86142", "openthaigpt7b": "C_openthaigpt7b_s1_f06e3ce3",
        "seallm7b": "C_seallm7b_s1_08cc9712"}
KW = {"income/รายได้": r"income|earn|money|wage|pay|รายได้|เงิน|ค่าจ้าง",
      "must work/ต้องทำงาน": r"must work|need to work|cannot work from home|no.{0,10}work.{0,10}home|ต้องทำงาน|ทำงานที่บ้านไม่ได้",
      "protect elderly/ผู้สูงอายุ": r"elder|老|ผู้สูงอายุ|vulnerable",
      "symptoms/อาการ": r"symptom|sick|fever|unwell|อาการ|ป่วย|ไข้",
      "cases rising/เคสเพิ่ม": r"cases? (are )?(rising|increas|high)|outbreak|การระบาด|ผู้ป่วยเพิ่ม",
      "mask/precaution": r"mask|precaution|distanc|hygien|หน้ากาก|ระวัง",
      "stay home/อยู่บ้าน": r"stay (at )?home|remain home|อยู่บ้าน",
      "essential only/จำเป็น": r"essential|necessary|จำเป็น"}

for model, run in RUNS.items():
    recs = [json.loads(l) for l in (ROOT / "outputs" / "runs" / run / "decisions.jsonl").open()]
    n = len(recs)
    lang_th = sum(1 for r in recs if re.search(r"[฀-๿]", r["reason"] or ""))
    lang_cjk = sum(1 for r in recs if re.search(r"[一-鿿]", r["reason"] or ""))
    uniq = len({(r["reason"] or "").strip().lower() for r in recs})
    top = collections.Counter((r["reason"] or "").strip() for r in recs).most_common(3)
    print(f"=== {model}  n={n}  unique reasons={uniq} ({uniq/n:.1%})  Thai script={lang_th/n:.1%}  CJK={lang_cjk/n:.1%}")
    for txt, c in top:
        print(f"    {c:5d} ({c/n:5.1%})  {txt[:100]}")
    hits = {k: sum(1 for r in recs if re.search(p, r["reason"] or "", re.I)) / n for k, p in KW.items()}
    print("    themes:", "  ".join(f"{k} {v:.0%}" for k, v in sorted(hits.items(), key=lambda x: -x[1]) if v >= 0.03))
    print()
