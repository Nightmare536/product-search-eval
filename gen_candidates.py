# -*- coding: utf-8 -*-
"""
半自动标注工具：从商品库自动生成「候选评测集」，你只负责打勾复核。

两个关键设计：
 1) 区分度优先：用现有检索策略跑一遍，标出"只有部分方案能命中"的候选，
    让你优先复核这些 —— 它们才能区分方案好坏。全都命中的区分度最低。
 2) 期望答案成组：同一个型号有多个品牌版本时，**同型号全部算作期望答案**，
    否则"只标了一个品牌版本"会让所有方案显得都很差，评测就不公平了。

用法:
    python gen_candidates.py --products products_big.json --per-product 8
输出:
    候选评测集.csv       用表格软件打开（带 BOM，中文不乱码），在「保留」列填 1 或 0
    候选评测集_预览.md   纯预览
接着:
    python build_eval_set.py      # 读「保留」列，生成 sample_data/eval_set.json
"""
import csv
import io
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from retrievers import (Product, KeywordRetriever, SynonymBm25Retriever,   # noqa: E402
                        HybridRrfRetriever)

SCENES = ["宿舍", "办公室", "家里", "学生", "租房", "出差", "公司"]
# 口语品类词：用户不会说"锅具"，会说"锅"；用正式类目名会让提问和商品标题字面重叠，测不出差异
CAT_SPOKEN = {"锅具": "锅", "耳机": "耳机", "充电": "充电宝", "照明": "灯",
              "外设": "鼠标键盘", "电脑配件": "电脑配件", "厨房小电": "小家电",
              "收纳": "收纳", "杯具": "杯子", "数码": "数码产品"}
TEMPLATES = [
    "想买个{tag}的{cat}", "{scene}用的{cat}有推荐吗", "有没有{tag}的{cat}",
    "求推荐{cat}，主要想要{tag}", "{scene}用，想找个{tag}的{cat}",
    "想买{cat}，主要是要{tag}", "{tag}的{cat}有什么好推荐",
    "想买个便宜点的{cat}，要{tag}", "{scene}用{cat}，希望{tag}",
    "有没有{tag}一点的{cat}",
]
# 意图型：不出现类目名，只给场景和功能，考的是语义理解能力
INTENT = [
    "我想{tag}", "{scene}的时候想{tag}", "{tag}，有推荐吗",
    "{scene}用，想{tag}", "有没有能{tag}的",
]
PARAM_RE = re.compile(r"[0-9][0-9.]*\s*(?:mAh|mah|毫安|W|w|L|l|升|寸|cm|键|口|级)")


def load(name):
    return json.load(io.open(os.path.join(BASE, "sample_data", name), encoding="utf-8"))


def core_phrases(title):
    return [p for p in re.split(r"[\s,，、/|]+", title) if len(p) >= 2]


def model_key(p):
    return (p.tags[0] if p.tags else p.category) or p.category


def main():
    per = 8
    pname = "products_big.json"
    if "--per-product" in sys.argv:
        per = int(sys.argv[sys.argv.index("--per-product") + 1])
    if "--products" in sys.argv:
        pname = sys.argv[sys.argv.index("--products") + 1]

    products = [Product(d) for d in load(pname)]
    synonyms = load("synonyms.json")
    by_id = dict((p.id, p) for p in products)

    # 反向词典：标题里出现"电煮锅"，就允许提问用"小锅"（字面不重叠，才测得出方案差异）
    rev_syn = {}
    for k, vs in synonyms.items():
        for v in vs:
            rev_syn.setdefault(v, k)

    # 同型号分组（期望答案成组）
    groups = {}
    for p in products:
        groups.setdefault(model_key(p), []).append(p.id)

    retrievers = [KeywordRetriever(products, synonyms),
                  SynonymBm25Retriever(products, synonyms),
                  HybridRrfRetriever(products, synonyms)]

    cands = []
    seen = set()
    for p in products:
        cat = p.category or "商品"
        ph = core_phrases(p.title)
        name = p.tags[0] if p.tags else ""
        tags = [x for x in p.tags
                if x and x != cat and x != name and not re.match(r"^[0-9]", x)]
        if not tags:
            tags = [w for w in core_phrases(p.title) if w not in (cat, name)][:2]
        grp = groups[model_key(p)]
        buckets = {}

        def add(kind, q, rel):
            q = (q or '').strip()
            if len(q) >= 2:
                buckets.setdefault(kind, []).append((q, rel))

        items = []

        # 1 关键词型（含品牌 -> 期望就是这一个）
        if ph:
            add("常规", " ".join(ph[:2]), [p.id])
        # 2 去品牌型（不含品牌 -> 期望是同型号全部）
        if p.brand and p.brand in p.title:
            t2 = re.sub(r"\s+", " ", p.title.replace(p.brand, "")).strip()
            add("常规", t2, grp)
        # 3 参数型
        params = PARAM_RE.findall(p.title)
        if params:
            add("参数型", " ".join(x.strip() for x in params) + " " + CAT_SPOKEN.get(cat, cat), grp)
        # 4 同义词型
        for k, vs in synonyms.items():
            if k in p.title and vs:
                add("同义词", vs[0] + " " + CAT_SPOKEN.get(cat, cat), grp)
                break
        # 5 口语化长尾
        for i, tpl in enumerate(TEMPLATES):
            if not tags:
                break
            add("口语化长尾", tpl.format(tag=tags[i % len(tags)], cat=CAT_SPOKEN.get(cat, cat),
                                        scene=SCENES[i % len(SCENES)]), grp)
        # 6 意图型（不含类目名）
        for i, tpl in enumerate(INTENT):
            if not tags:
                break
            add("意图型", tpl.format(tag=tags[i % len(tags)],
                                    scene=SCENES[i % len(SCENES)]), grp)
        # 反向同义词：把标题里的词换成它的"同义口语词"，字面不重叠
        for word, alt in rev_syn.items():
            if word in p.title:
                add("同义词", alt + " " + CAT_SPOKEN.get(cat, cat), grp)
                break

        # 轮转取样：保证各种提问类型都有，总数受控
        order = ["同义词", "意图型", "口语化长尾", "参数型", "常规"]
        while len(items) < per and any(buckets.get(t) for t in order):
            for t in order:
                if len(items) >= per:
                    break
                if buckets.get(t):
                    qq, rr = buckets[t].pop(0)
                    items.append((qq, t, rr))

        for q, kind, rel in items:
            q = q.strip()
            if len(q) < 2 or q in seen:
                continue
            seen.add(q)
            hitn = sum(1 for r in retrievers if set(r.search(q, 10)).intersection(set(rel)))
            cands.append({"query": q, "type": kind, "rel": rel, "hitn": hitn})

    order = {1: 0, 2: 1, 0: 2, 3: 3}
    cands.sort(key=lambda c: (order.get(c["hitn"], 9), c["type"]))

    f = io.open(os.path.join(BASE, "候选评测集.csv"), "w", encoding="utf-8-sig", newline="")
    w = csv.writer(f)
    w.writerow(["编号", "候选提问", "场景类型", "期望商品编号(可多个,逗号分隔)",
                "期望商品示例标题", "命中策略数", "区分度", "保留(填1保留/0删除)"])
    for i, c in enumerate(cands, 1):
        tip = {1: "高(只有部分方案命中,最有价值)", 2: "中(能区分部分方案)",
               0: "偏低(所有方案都没命中,可当难例)", 3: "低(都能命中,区分度差)"}[c["hitn"]]
        w.writerow(["q%03d" % i, c["query"], c["type"], ",".join(c["rel"]),
                    by_id[c["rel"][0]].title, c["hitn"], tip, ""])
    f.close()

    L = ["# 候选评测集（待复核）", "",
         "共生成 **%d** 条候选。请在 `候选评测集.csv` 的「保留」列填 1 或 0，",
         "然后运行 `python build_eval_set.py` 生成正式评测集。", "",
         "**优先复核「区分度=高」的候选**：只有部分方案能命中，它们才是评测集里最值钱的部分。", "",
         "| 编号 | 候选提问 | 场景类型 | 期望商品示例 | 命中策略数 | 区分度 |",
         "|---|---|---|---|---|---|"]
    for i, c in enumerate(cands[:80], 1):
        L.append("| q%03d | %s | %s | %s | %d | %s |" %
                 (i, c["query"], c["type"], by_id[c["rel"][0]].title[:20], c["hitn"],
                  {1: "高", 2: "中", 0: "偏低", 3: "低"}[c["hitn"]]))
    L.append("")
    L.append("（预览只显示前 80 条，完整请看 csv）")
    io.open(os.path.join(BASE, "候选评测集_预览.md"), "w", encoding="utf-8").write(chr(10).join(L))

    stat = {}
    for c in cands:
        stat[c["hitn"]] = stat.get(c["hitn"], 0) + 1
    kinds = {}
    for c in cands:
        kinds[c["type"]] = kinds.get(c["type"], 0) + 1
    print("生成候选 %d 条" % len(cands))
    print("  区分度高(仅部分方案命中): %d" % stat.get(1, 0))
    print("  区分度中                : %d" % stat.get(2, 0))
    print("  偏低(都没命中,可当难例)  : %d" % stat.get(0, 0))
    print("  低(都能命中)            : %d" % stat.get(3, 0))
    print("  场景分布:", "  ".join("%s=%d" % kv for kv in kinds.items()))
    print("已写入 候选评测集.csv / 候选评测集_预览.md")


if __name__ == "__main__":
    main()
