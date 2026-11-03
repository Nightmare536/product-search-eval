# -*- coding: utf-8 -*-
"""把复核好的 CSV 转成正式评测集（sample_data/eval_set.json）。
用法: python build_eval_set.py
说明: 只保留「保留」列填 1 的行；同一提问多处出现会自动合并期望答案。"""
import csv
import io
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE, "候选评测集.csv")
rows = list(csv.DictReader(io.open(csv_path, encoding="utf-8-sig")))

merged = {}
for r in rows:
    keep = (r.get("保留(填1保留/0删除)") or "").strip()
    if keep not in ("1", "y", "Y", "是"):
        continue
    q = (r.get("候选提问") or "").strip()
    if not q:
        continue
    rel = [x.strip() for x in (r.get("期望商品编号(可多个,逗号分隔)") or "").split(",") if x.strip()]
    item = merged.setdefault(q, {"query": q, "type": r.get("场景类型") or "常规", "relevant": []})
    for x in rel:
        if x not in item["relevant"]:
            item["relevant"].append(x)

out = []
for i, (q, item) in enumerate(merged.items(), 1):
    item["qid"] = "q%03d" % i
    out.append(item)

p = os.path.join(BASE, "sample_data", "eval_set.json")
io.open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
kinds = {}
for it in out:
    kinds[it["type"]] = kinds.get(it["type"], 0) + 1
print("已生成评测集: %d 条 -> %s" % (len(out), p))
print("场景分布:", "  ".join("%s=%d" % kv for kv in kinds.items()))
