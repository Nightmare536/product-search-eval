# -*- coding: utf-8 -*-
u"""
检索失败分层诊断：不猜「为什么失败」，而是**实测「哪种手段能救」**。

一条没命中的提问，有 4 种可能的救法，代价从低到高：
    ① 加深召回深度（topK 10 -> 40）          成本≈0，但必须配后过滤，否则上下文变脏
    ② 词表改写（离线规则，0 调用成本）        成本=写词表的人工
    ③ 大模型在线意图改写                      成本=每次请求 +1 次 LLM 调用
    ④ 换真实 embedding / 补数据               成本最高

本脚本对每条 query **逐个实测** ①②③ 能不能救回来，然后按「最便宜的那个手段」
给它贴一个标签 —— 标签本身就是行动清单，不需要再猜。

    ✅ 命中             向量基线 top10 就有正解
    🟡 A·只差召回想深度   向量基线 top10 没有、加深到 40 就有     -> 加 topK + 加后过滤
    🟠 B·词表改写可救     深挖也没有、词表改写后有               -> 补意图词表（离线可做）
    🔴 C·现有手段全覆盖不到 三个都不行                          -> 必须上大模型在线改写 / 真向量 / 补数据

度量口径：向量基线 = LsaVectorRetriever 的余弦相似度（本项目唯一能离线复现的稠密召回），
0~1 跨 query 可比。它不是真实 embedding，但排序行为和失效模式是同一套。
换 pgvector/Qdrant 后只要替换 score_all()。

用法:
    python diagnose.py
    python diagnose.py --rank-cut 10 --deep-cut 40
"""
import io
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from retrievers import (Product, KeywordRetriever, SynonymBm25Retriever,   # noqa: E402
                        HybridRrfRetriever, LsaVectorRetriever,
                        IntentBm25Retriever, FullStackRetriever,
                        rewrite_intent)

HIT = u"✅ 命中"
A_CLASS = u"🟡 A·只差召回深度"
B_CLASS = u"🟠 B·词表改写可救"
C_CLASS = u"🔴 C·现有手段覆盖不到"
ORDER = [HIT, A_CLASS, B_CLASS, C_CLASS]


def load(name):
    with io.open(os.path.join(BASE, "sample_data", name), encoding="utf-8") as f:
        return json.load(f)


def score_all(vec, query):
    u"""query 与全库每条商品的余弦相似度 -> [(score, pid)]，0~1 可比。"""
    v = vec.encode(query)
    if v is None:
        return [(0.0, p.id) for p in vec.products]
    out = []
    if vec._dvec is not None:                      # numpy 稠密
        sims = vec._dvec.dot(v)
        for i, p in enumerate(vec.products):
            out.append((float(sims[i]), p.id))
    else:                                          # 纯 Python 稀疏
        for i, p in enumerate(vec.products):
            d = vec._sdoc[i]
            if len(d) < len(v):
                s = sum(w * v.get(j, 0.0) for j, w in d.items())
            else:
                s = sum(w * d.get(j, 0.0) for j, w in v.items())
            out.append((float(s), p.id))
    return out


def ranked(scored):
    u"""按相似度降序的 pid 列表（分数<=0 视为未召回，排在最后且不计入名次）"""
    live = [x for x in scored if x[0] > 0]
    dead = [x for x in scored if x[0] <= 0]
    live.sort(key=lambda x: (-x[0], x[1]))
    dead.sort(key=lambda x: x[1])
    return [pid for _, pid in live] + [pid for _, pid in dead]


def first_hit(order_ids, rel, limit):
    for i, pid in enumerate(order_ids[:limit], 1):
        if pid in rel:
            return i
    return None


def main():
    rank_cut, deep_cut = 10, 40
    if "--rank-cut" in sys.argv:
        rank_cut = int(sys.argv[sys.argv.index("--rank-cut") + 1])
    if "--deep-cut" in sys.argv:
        deep_cut = int(sys.argv[sys.argv.index("--deep-cut") + 1])

    products = [Product(d) for d in load("products_big.json")]
    synonyms = load("synonyms.json")
    vec = LsaVectorRetriever(products, synonyms)
    sets = [(u"149 条评测集", load("eval_set.json")),
            (u"20 条泛化集（新说法）", load("eval_paraphrase.json"))]
    print(u"[诊断] 向量模式：%s，商品 %d 条" % (vec.mode, len(products)))

    strategies = [
        (u"1-纯关键词", KeywordRetriever(products, synonyms)),
        (u"2-BM25+同义词", SynonymBm25Retriever(products, synonyms)),
        (u"3-混合RRF", HybridRrfRetriever(products, synonyms)),
        (u"4-向量(LSA)", vec),
        (u"5-词表改写+BM25", IntentBm25Retriever(products, synonyms)),
        (u"6-全栈(上线版)", FullStackRetriever(products, synonyms)),
    ]

    L, detail = [], {}
    L.append(u"# 检索失败分层诊断报告")
    L.append(u"")
    L.append(u"- 诊断时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    L.append(u"- 向量基线：LSA 余弦（%s），商品 %d 条" % (vec.mode, len(products)))
    L.append(u"- 分层规则：**按「最便宜的那个能救回来的手段」贴标签**，不是靠猜原因")
    L.append(u"- 命中线 topK=%d，深挖线 topK=%d" % (rank_cut, deep_cut))
    L.append(u"")

    for set_name, items in sets:
        rows = []
        for it in items:
            q, rel = it["query"], set(it.get("relevant", []))
            sc = score_all(vec, q)
            order = ranked(sc)
            sims = dict((pid, s) for s, pid in sc)
            top1_sim = max(sims.values()) if sims else 0.0

            r_base = first_hit(order, rel, rank_cut)          # 基线 top10 的名次
            r_deep = first_hit(order, rel, deep_cut)          # 加深到 40 的名次
            rel_sim = max([sims.get(p, 0.0) for p in rel] or [0.0])

            rq, added = rewrite_intent(q)                     # ② 词表改写
            r_rw = None
            if added:
                order_rw = ranked(score_all(vec, rq))
                r_rw = first_hit(order_rw, rel, rank_cut)

            hits = {}
            for sname, r in strategies:                       # 现有 6 个策略
                hits[sname] = bool(set(r.search(q, rank_cut)) & rel)

            if r_base is not None:
                cls = HIT
            elif r_deep is not None:
                cls = A_CLASS
            elif r_rw is not None:
                cls = B_CLASS
            else:
                cls = C_CLASS

            rows.append({
                "qid": it.get("qid"), "query": q, "type": it.get("type", u"其他"),
                "rel": sorted(rel), "cls": cls,
                "top1_sim": round(max(0.0, top1_sim), 3), "rel_sim": round(max(0.0, rel_sim), 3),
                "r_base": r_base, "r_deep": r_deep, "r_rw": r_rw,
                "added": added, "rewritten": rq if added else u"",
                "top3": order[:3],
                "hits": hits,
                "any_strategy": any(hits.values()),
            })
        detail[set_name] = rows

        n = len(rows)
        cnt = dict((c, 0) for c in ORDER)
        for r in rows:
            cnt[r["cls"]] += 1

        L.append(u"## %s" % set_name)
        L.append(u"")
        L.append(u"| 分层 | 条数 | 占比 | 该怎么治 |")
        L.append(u"|---|---|---|---|")
        L.append(u"| ✅ 命中 | %d | %.1f%% | 向量基线前 %d 名就有正解，线上没问题 |"
                 % (cnt[HIT], 100.0 * cnt[HIT] / n, rank_cut))
        L.append(u"| 🟡 A·只差召回深度 | %d | %.1f%% | **加 topK 到 %d + 加后过滤**（不能再多喂垃圾） |"
                 % (cnt[A_CLASS], 100.0 * cnt[A_CLASS] / n, deep_cut))
        L.append(u"| 🟠 B·词表改写可救 | %d | %.1f%% | **补意图词表**（离线、零调用成本，但会过拟合） |"
                 % (cnt[B_CLASS], 100.0 * cnt[B_CLASS] / n))
        L.append(u"| 🔴 C·现有手段覆盖不到 | %d | %.1f%% | **上大模型在线改写 / 换真 embedding / 补数据** |"
                 % (cnt[C_CLASS], 100.0 * cnt[C_CLASS] / n))
        L.append(u"")

        # ---- 三个手段各自的救回量（互相独立、不是同义反复）----
        deep_only = len([r for r in rows if r["r_base"] is None and r["r_deep"] is not None])
        rw_only = len([r for r in rows if r["r_base"] is None and r["r_deep"] is None and r["r_rw"] is not None])
        nothing = len([r for r in rows if r["r_base"] is None and r["r_deep"] is None and r["r_rw"] is None])
        rw_cover = len([r for r in rows if r["added"]])
        any_str = len([r for r in rows if r["any_strategy"]])
        empty_ctx = len([r for r in rows if r["r_base"] is None])

        L.append(u"### 三个手段各能救回多少（相互独立实测）")
        L.append(u"")
        L.append(u"| 手段 | 成本 | 救回 | 说明 |")
        L.append(u"|---|---|---|---|")
        L.append(u"| 加深召回 topK 10→%d | ≈0（但得有后过滤） | **%d 条（%.1f%%）** | 正解本来就在近邻，只是被挤出去了 |"
                 % (deep_cut, deep_only, 100.0 * deep_only / n))
        L.append(u"| 词表改写（离线） | 写词表的人工 | **%d 条（%.1f%%）** | 词表命中率 %d/%d，命中规则的那部分里救回 %d 条 |"
                 % (rw_only, 100.0 * rw_only / n, rw_cover, n, rw_only))
        L.append(u"| 以上都不行 | — | **%d 条（%.1f%%）** | 现有词表 + 检索器都覆盖不到 |"
                 % (nothing, 100.0 * nothing / n))
        L.append(u"")
        L.append(u"- **空上下文占比：%.1f%%（%d/%d）** —— 直接把前 %d 条当上下文交给大模型，"
                 u"其中一条正解都没有的比例。这是 RAG 系统最直白的健康度指标，比 R@10 更贴近用户体感。"
                 % (100.0 * empty_ctx / n, empty_ctx, n, rank_cut))
        L.append(u"- 现有 6 个策略里至少有一个能命中的：%d 条（%.1f%%）—— 这是「调检索」这条路的天花板。"
                 % (any_str, 100.0 * any_str / n))
        L.append(u"")

        # ---- 诚实性检查：深挖救回的这批，有多少是真「差一点」，有多少是弱模型的侥幸 ----
        deep_rows = [r for r in rows if r["r_base"] is None and r["r_deep"] is not None]
        if deep_rows:
            solid = [r for r in deep_rows
                     if r["top1_sim"] > 0 and r["rel_sim"] >= 0.5 * r["top1_sim"]]
            zeros = [r for r in deep_rows if r["rel_sim"] <= 0]
            L.append(u"> ⚠️ **这个「深挖救回」要打折看**：%d 条里有 **%d 条的正解相似度是 0**，"
                     u"只有 **%d 条**的正解相似度能达到 top1 的 50%% 以上。"
                     u"正解相似度为 0 却被排进前 %d 名，说明**这个弱向量模型对正解和噪音根本没区分度**——"
                     u"「加深 topK 能救」在这种情况下的真实含义是「扩大撒网」，"
                     u"在真实数据量下同样会把一大堆噪声一起捞进来。"
                     u"**所以「加深 topK」不是一个独立手段，它的成立前提是后过滤。**"
                     % (len(deep_rows), len(zeros), len(solid), deep_cut))
            L.append(u"")

        # ---- 分场景 ----
        types = []
        for r in rows:
            if r["type"] not in types:
                types.append(r["type"])
        L.append(u"### 分场景：哪类提问该用哪种药")
        L.append(u"")
        L.append(u"| 场景 | 条数 | ✅ | 🟡 A | 🟠 B | 🔴 C |")
        L.append(u"|---|---|---|---|---|---|")
        for t in types:
            sub = [r for r in rows if r["type"] == t]
            c = dict((x, 0) for x in ORDER)
            for r in sub:
                c[r["cls"]] += 1
            L.append(u"| %s | %d | %d | %d | %d | %d |" % (t, len(sub), c[HIT], c[A_CLASS], c[B_CLASS], c[C_CLASS]))
        L.append(u"")

        # ---- 明细 ----
        miss = [r for r in rows if r["cls"] != HIT]
        if miss:
            L.append(u"### 未命中明细（%d 条，按该用的手段排序）" % len(miss))
            L.append(u"")
            L.append(u"| qid | 用户的说法 | 期望 | 分层 | top1相似度 | 正解相似度 | 基线名次 | 40名内 | 词表改写 |")
            L.append(u"|---|---|---|---|---|---|---|---|---|")
            for r in sorted(miss, key=lambda x: (ORDER.index(x["cls"]), x["qid"] or "")):
                rw_cell = (u"%s→%s" % (r["r_deep"] or r["r_base"] or u"—", r["r_rw"])) if r["added"] else u"词表未命中"
                L.append(u"| %s | %s | %s | %s | %.2f | %.2f | %s | %s | %s |"
                         % (r["qid"], r["query"], u",".join(r["rel"]), r["cls"],
                            r["top1_sim"], r["rel_sim"],
                            r["r_base"] or u"—",
                            r["r_deep"] or u"—",
                            rw_cell))
            L.append(u"")
        else:
            L.append(u"### 全部命中")
            L.append(u"")

    # ---- 两集合并 ----
    allrows = [r for s in detail.values() for r in s]
    n = len(allrows)
    c = dict((x, 0) for x in ORDER)
    for r in allrows:
        c[r["cls"]] += 1
    L.append(u"## 两集合并（%d 条）" % n)
    L.append(u"")
    L.append(u"| 分层 | 条数 | 占比 |")
    L.append(u"|---|---|---|")
    for cls in ORDER:
        L.append(u"| %s | %d | %.1f%% |" % (cls, c[cls], 100.0 * c[cls] / n))
    L.append(u"")
    L.append(u"## 结论（行动清单）")
    L.append(u"")
    L.append(u"1. **先做 🟡 A 类**：加深 topK 最便宜，但**前提是先有后过滤**——"
             u"否则加深召回只是把垃圾一起喂进 prompt，模型反而更容易编。"
             u"（这就是 Modular RAG 里 `DocumentPostProcessor` 那一格存在的理由。）")
    L.append(u"2. **再做 🟠 B 类**：补词表，零调用成本，但它是在**对着评测集写规则**，"
             u"泛化集上的数字会明显掉下来 —— 两个数字必须一起汇报，不能只报高的那个。")
    L.append(u"3. **🔴 C 类是必须上大模型的地方**：词表 + 弱向量都覆盖不到，"
             u"这也正是「在线意图解析」的正当性来源 —— 不是因为它先进，是因为**别的路都试过了**。")
    L.append(u"")

    md = u"\n".join(L)
    rp = os.path.join(BASE, "reports")
    if not os.path.exists(rp):
        os.makedirs(rp)
    with io.open(os.path.join(rp, "diagnosis.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with io.open(os.path.join(rp, "diagnosis.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(detail, ensure_ascii=False, indent=1))
    print(u"诊断报告已写入 reports/diagnosis.md")
    return md


if __name__ == "__main__":
    main()
