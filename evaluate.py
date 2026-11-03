# -*- coding: utf-8 -*-
"""
离线评测引擎：跑评测集 -> 算指标 -> 出报告 -> 与上次对比（回归检测）。

用法:
    python evaluate.py                # 跑三个内置检索策略，输出 reports/latest.md
    python evaluate.py --topk 10

指标: Recall@5 / Recall@10 / MRR@10 / NDCG@10
      （评测集里的 relevant 是人工标注的期望商品，二值相关性）
"""
import io
import json
import math
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from retrievers import (Product, KeywordRetriever, SynonymBm25Retriever,   # noqa: E402
                        HybridRrfRetriever, LsaVectorRetriever,
                        IntentBm25Retriever, FullStackRetriever)


def recall_at(ranked, rel, k):
    if not rel:
        return 0.0
    return len(set(ranked[:k]).intersection(rel)) / float(len(rel))


def mrr_at(ranked, rel, k):
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked, rel, k):
    dcg = 0.0
    for i, d in enumerate(ranked[:k], 1):
        if d in rel:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(rel)) + 1))
    return dcg / ideal if ideal else 0.0


PRODUCT_FILE = "products_big.json"


def load(name):
    p = os.path.join(BASE, "sample_data", name)
    return json.load(io.open(p, encoding="utf-8"))


def avg(rows, key):
    return sum(r[key] for r in rows) / float(len(rows)) if rows else 0.0


def run_set(r, items, topk):
    rows = []
    for it in items:
        rel = set(it.get("relevant", []))
        ranked = r.search(it["query"], topk)
        rows.append({
            "qid": it["qid"], "query": it["query"],
            "type": it.get("type", "其他"),
            "rel": sorted(rel), "ranked": ranked,
            "r5": recall_at(ranked, rel, 5),
            "r10": recall_at(ranked, rel, 10),
            "mrr": mrr_at(ranked, rel, 10),
            "ndcg": ndcg_at(ranked, rel, 10),
        })
    return rows


def main():
    topk = 10
    if "--topk" in sys.argv:
        topk = int(sys.argv[sys.argv.index("--topk") + 1])
    f = PRODUCT_FILE
    if "--products" in sys.argv:
        f = sys.argv[sys.argv.index("--products") + 1]
    products = [Product(d) for d in load(f)]
    evalset = load("eval_set.json")
    synonyms = load("synonyms.json")

    retrievers = [
        KeywordRetriever(products, synonyms),
        SynonymBm25Retriever(products, synonyms),
        HybridRrfRetriever(products, synonyms),
        LsaVectorRetriever(products, synonyms),
        IntentBm25Retriever(products, synonyms),
        FullStackRetriever(products, synonyms),
    ]
    if "--only" in sys.argv:
        keep = sys.argv[sys.argv.index("--only") + 1].split(",")
        retrievers = [r for r in retrievers if r.name.split("-")[0] in keep]

    all_rows = {}
    for r in retrievers:
        t0 = time.time()
        rows = run_set(r, evalset, topk)
        all_rows[r.name] = rows
        print("[%s] 耗时 %.0f 毫秒  前10条命中率=%.3f" % (r.name, (time.time() - t0) * 1000, avg(rows, "r10")))

    # ---------- 泛化验证：用户换个说法还认得出来吗 ----------
    # 这 20 条「新说法」提问语义和难例相同、但用词完全不同，写意图词表时没有参考过。
    # 它衡量的是词表（和模型）的**泛化能力**，而不是"对着测试集写规则"能刷到的上限。
    para = None
    para_rows = {}
    try:
        para = load("eval_paraphrase.json")
    except Exception:
        para = None
    if para:
        for r in retrievers:
            para_rows[r.name] = run_set(r, para, topk)
        print("-- 泛化验证（%d 条新说法） --" % len(para))
        for n in para_rows:
            print("[%s] 新说法命中率=%.3f" % (n, avg(para_rows[n], "r10")))

    # ---------- 历史对比 ----------
    rp = os.path.join(BASE, "reports")
    if not os.path.exists(rp):
        os.makedirs(rp)
    hist_path = os.path.join(rp, "history.json")
    history = []
    if os.path.exists(hist_path):
        try:
            history = json.load(io.open(hist_path, encoding="utf-8"))
        except Exception:
            history = []
    prev = history[-1]["summary"] if history else None
    summary = dict((n, {"r5": avg(rows, "r5"), "r10": avg(rows, "r10"),
                        "mrr": avg(rows, "mrr"), "ndcg": avg(rows, "ndcg")})
                   for n, rows in all_rows.items())

    # ---------- 出报告 ----------
    L = []
    L.append("# 检索质量评测报告")
    L.append("")
    L.append("- 时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    L.append("- 商品数：%d　评测 query 数：%d　TopK：%d" % (len(products), len(evalset), topk))
    L.append("- 相关性判定：评测集人工标注的期望商品（二值）")
    L.append("")
    L.append("## 一、总体指标")
    L.append("")
    L.append("| 检索策略 | 前5条命中率(R@5) | 前10条命中率(R@10) | 命中靠前度(MRR) | 排序质量分(NDCG) |")
    L.append("|---|---|---|---|---|")
    for n in all_rows:
        s = summary[n]
        L.append("| %s | %.3f | **%.3f** | %.3f | %.3f |" %
                 (n, s["r5"], s["r10"], s["mrr"], s["ndcg"]))
    L.append("")
    if prev:
        L.append("### 与上次运行对比（前10条命中率）")
        L.append("")
        L.append("| 检索策略 | 上次 | 本次 | 变化 |")
        L.append("|---|---|---|---|")
        for n in summary:
            if n in prev:
                a, b = prev[n]["r10"], summary[n]["r10"]
                d = b - a
                mark = "持平" if abs(d) < 1e-9 else ("+%.3f ↑" % d if d > 0 else "%.3f ↓" % d)
                L.append("| %s | %.3f | %.3f | %s |" % (n, a, b, mark))
        L.append("")
    L.append("## 二、分场景（前10条命中率）")
    L.append("")
    types = []
    for it in evalset:
        t = it.get("type", "其他")
        if t not in types:
            types.append(t)
    L.append("| 场景 | " + " | ".join(all_rows.keys()) + " |")
    L.append("|---" * (len(all_rows) + 1) + "|")
    for t in types:
        cells = []
        for n, rows in all_rows.items():
            sub = [r for r in rows if r["type"] == t]
            cells.append("%.3f" % avg(sub, "r10"))
        L.append("| %s | %s |" % (t, " | ".join(cells)))
    L.append("")
    L.append("## 三、失败案例（最后一版策略前10条一条没命中）")
    L.append("")
    last = list(all_rows.keys())[-1]
    bad = [r for r in all_rows[last] if r["r10"] == 0]
    if bad:
        L.append("| qid | query | 场景 | 期望 | 实际返回 Top3 |")
        L.append("|---|---|---|---|---|")
        for r in bad:
            L.append("| %s | %s | %s | %s | %s |" %
                     (r["qid"], r["query"], r["type"], ",".join(r["rel"]),
                      ",".join(r["ranked"][:3])))
    else:
        L.append("无（每条提问都至少命中了一个期望商品）")
    L.append("")
    L.append("> 失败案例是下一轮优化的入口：看是分词问题、同义词缺词，还是需要语义向量召回。")

    if para:
        L.append("")
        L.append("## 四、泛化验证：用户换个说法还认得出来吗")
        L.append("")
        L.append("这 %d 条是「新说法」提问：语义和难例相同、用词完全不同，"
                 "**写意图词表时没有参考过它们**。" % len(para))
        L.append("它量的是词表/模型的泛化能力，而不是「对着测试集写词表」能刷到的上限 —— "
                 "两个数字要一起看。")
        L.append("")
        L.append("| 检索策略 | 难例·纯意图（写词表时参考过） | 难例·新说法（没参考过） |")
        L.append("|---|---|---|")
        for n in all_rows:
            if n in para_rows:
                hard = [r for r in all_rows[n] if r["type"] == u"难例·纯意图"]
                L.append("| %s | %.3f | %.3f |" % (n, avg(hard, "r10"), avg(para_rows[n], "r10")))
        L.append("")
        bad_p = [r for r in para_rows[last] if r["r10"] == 0]
        L.append("最后一版策略在这批新说法上没命中的 %d 条（**这就是该上大模型做意图解析的地方**）：" % len(bad_p))
        L.append("")
        if bad_p:
            L.append("| qid | 用户的新说法 | 期望商品 | 实际返回 Top3 |")
            L.append("|---|---|---|---|")
            for r in bad_p:
                L.append("| %s | %s | %s | %s |" %
                         (r["qid"], r["query"], ",".join(r["rel"]), ",".join(r["ranked"][:3])))
        else:
            L.append("（全部命中）")
        L.append("")

    md = chr(10).join(L)
    io.open(os.path.join(rp, "latest.md"), "w", encoding="utf-8").write(md)
    history.append({"time": time.time(), "summary": summary})
    io.open(hist_path, "w", encoding="utf-8").write(json.dumps(history, ensure_ascii=False, indent=1))
    io.open(os.path.join(rp, "detail.json"), "w", encoding="utf-8").write(
        json.dumps(all_rows, ensure_ascii=False, indent=1))
    if para:
        io.open(os.path.join(rp, "paraphrase_detail.json"), "w", encoding="utf-8").write(
            json.dumps(para_rows, ensure_ascii=False, indent=1))
    print("报告已写入 reports/latest.md")
    return md


if __name__ == "__main__":
    main()
