# -*- coding: utf-8 -*-
"""轻量中文分词（不依赖第三方库）—— 商品域专用。

为什么自己写：环境装不上 jieba；而且这里要的不是"通用分词"，
是"商品域分词" —— 词表直接从商品库的类目/标签/品牌/同义词反推，比通用词典更贴业务。

它解决三个真实问题：
  1. 双字切分会把词切碎并制造垃圾词：
       "电煮锅 锅" 去掉空格 -> "电煮锅锅" -> 切成「电煮/煮锅/锅锅」，真正的「锅」丢了
  2. 单字乱命中：「不」「多」「上」这类功能字会命中大量无关商品
     -> 单字停用表直接丢弃；只有「锅」「灯」这类商品域单字才保留兜底（降权 0.5）
  3. 场景词污染：「办公室」和键盘/鼠标标签里的「办公」撞车，把锅具挤掉
     -> 场景词降权（0.35），不参与"这是什么商品"的主判定
"""
import re

STOPWORDS = set(u"的 了 用 有 吗 呢 吧 个 想 买 要 我 你 请问 一下 什么 哪个 怎么 哪些 款 种 和 与 及 还是 或者 就是 一个 有没有 值得 推荐 推荐一下 求推荐 求 帮忙 帮我 来个 想要 需要 看看 大概 大约 左右 一般 普通 那种 这种 之类 好用的 好用 不想 不用 不怕 不在 不能 不会 没有 就能 上去 下来 里面 外面 时候 的话 的话 有点 特别 非常 比较 可以 应该 是不是 什么样 多少 怎样 平时 平常 打算 最近 一直 感觉 觉得 麻烦 顺便 最好 尽量 尽量 有没有 多少 几 求 谢谢 大佬".split())
SINGLE_STOP = set(u"的了用有吗呢吧个想买要我你您他她它们是与及就还或也都太很非常一二三四五六七八九十多少大小上下里面前后左右内外中间旁边头尾能会可以又再别不没无好坏高低长短新旧老真假够满空冷热烫凉湿干亮暗软硬薄厚轻重宽窄深浅平斜正反跟给让把被从向对为于由因所但而且如若则才只更最挺蛮超特着过起来去回到出入开关放拿带装看听吃喝走说打做坐躺睡站洗写画唱玩学试问答求帮找选挑换修拆接插挂贴摆收存清买卖送寄发事人物家点些这那哪什么怎天年月日早午夜次遍趟块元钱斤两尺寸秒数量地新常同直总先再刚快慢")

SCENE = set(u"办公室 办公 宿舍 学生 租房 家里 家用 出差 公司 户外 旅行 生日 礼物 送礼 客厅 卧室 厨房 桌面 床上 通勤 上学 学校 寝室".split())

VOCAB = set()
MAXLEN = 4


def _runs(text):
    out, buf, cn = [], u"", None
    for ch in text:
        is_cn = u"\u4e00" <= ch <= u"\u9fff"
        if cn is None or is_cn == cn:
            buf += ch
            cn = is_cn
        else:
            out.append((buf, cn))
            buf, cn = ch, is_cn
    if buf:
        out.append((buf, cn))
    return out


def init(products, synonyms=None):
    """用商品库 + 同义词词典构建词表"""
    global VOCAB, MAXLEN
    v = set()
    for p in products:
        for s in (getattr(p, "category", ""), getattr(p, "brand", "")):
            if s:
                v.add(s)
        for t in (getattr(p, "tags", None) or []):
            if t:
                v.add(t)
        for run, is_cn in _runs(getattr(p, "title", "") or ""):
            seg = run.strip()
            if seg and len(seg) <= 12:
                v.add(seg)
    for k, vs in (synonyms or {}).items():
        if k:
            v.add(k)
        for x in (vs or []):
            if x:
                v.add(x)
    v |= SCENE
    VOCAB = set(w for w in v if w and w not in STOPWORDS)
    MAXLEN = max([len(w) for w in VOCAB] + [3])


def tokenize(text, with_weight=True):
    """返回 [(词, 权重)]：1.0 正常词 / 0.35 场景词 / 0.5 单字"""
    text = (text or u"").lower()
    res = []
    for run, is_cn in _runs(text):
        if not is_cn:
            # 非中文段要按空格/横线等再拆开，否则 "Type-C 100W" 会粘成 "typec100w"
            for seg in re.split(u"[^0-9a-z.]+", run):
                if seg:
                    res.append((seg, 1.0))
            continue
        i, n = 0, len(run)
        while i < n:
            hit = None
            for L in range(min(MAXLEN, n - i), 1, -1):
                w = run[i:i + L]
                if w in VOCAB:
                    hit = w
                    break
            if hit:
                res.append((hit, 0.35 if hit in SCENE else 1.0))
                i += len(hit)
                continue
            # 命中不了就按单字出；整块是停用词（如"推荐""用的"）就直接跳过
            skipped = False
            for L in (3, 2):
                if run[i:i + L] in STOPWORDS:
                    i += L
                    skipped = True
                    break
            if skipped:
                continue
            ch = run[i]
            if ch not in SINGLE_STOP:
                res.append((ch, 0.5))
            i += 1
    return res if with_weight else [t for t, _ in res]
