# -*- coding: utf-8 -*-
"""三种可插拔检索策略，用来做「改动前后」的指标对比。

分词器（tokenize_cn）是三种策略**共用**的基础组件，相当于 ES 里的 analyzer：
所以"改分词"和"改检索策略"是两类可以分别验证的优化。

   1 KeywordRetriever     —— 第一版：纯关键词命中计数（相当于 ES match）
   2 SynonymBm25Retriever —— 你现在这版：BM25 + 同义词词典（双向）扩展
   3 HybridRrfRetriever   —— 升级版：BM25 + 语义召回 + RRF 融合
                             （"语义召回"此处用词集重合度占位，接真实向量库后替换）
"""
import io
import json
import math
import os
import re
from collections import Counter

import tokenize_cn as TK

# 单字兜底开关（对照实验用）：SC_NO_SINGLE=1 关闭
USE_SINGLE_FALLBACK = os.environ.get("SC_NO_SINGLE") != "1"


class Product(object):
    def __init__(self, d):
        self.id = d["id"]
        self.title = d.get("title", "")
        self.category = d.get("category", "")
        self.brand = d.get("brand", "")
        self.price = d.get("price", 0)
        self.tags = d.get("tags", []) or []

    @property
    def text(self):
        return " ".join([self.title, self.category, self.brand] + list(self.tags))


class Retriever(object):
    name = "base"

    def __init__(self, products, synonyms=None):
        self.products = products
        self.syn = synonyms or {}
        TK.init(products, self.syn)
        self.docs = [TK.tokenize(p.text) for p in products]        # [(词, 权重)]
        self.doc_terms = [set(t for t, _ in d) for d in self.docs]  # 词集合
        self.doc_chars = [set(u"".join(t for t, _ in d)) for d in self.docs]  # 字集合

    def search(self, query, topk=10):
        raise NotImplementedError

    def expand(self, query):
        """同义词扩展，双向：提问用口语词能扩到书面词，反之亦然。"""
        extra = []
        for k, vs in self.syn.items():
            if k and k in query:
                extra.extend(vs)
            for v in (vs or []):
                if v and v in query:
                    extra.append(k)
        return query + (" " + " ".join(extra) if extra else "")

    def hits(self, qterms, i):
        """某条商品命中查询词的加权得分（单字走字符兜底）"""
        terms, chars = self.doc_terms[i], self.doc_chars[i]
        out = []
        for t, w in qterms:
            if t in terms:
                out.append((t, w, True))
            elif USE_SINGLE_FALLBACK and len(t) == 1 and t in chars:
                out.append((t, w, False))
        return out


class KeywordRetriever(Retriever):
    name = "1-纯关键词"

    def search(self, query, topk=10):
        q = TK.tokenize(self.expand(query))
        scored = []
        for i, p in enumerate(self.products):
            s = sum(w for _, w, _ in self.hits(q, i))
            if s > 0:
                scored.append((s, p.id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scored[:topk]]


class SynonymBm25Retriever(Retriever):
    name = "2-BM25+同义词词典"

    def __init__(self, products, synonyms=None):
        Retriever.__init__(self, products, synonyms)
        self.tf = [Counter(t for t, _ in d) for d in self.docs]
        self.df = Counter()
        self.df_char = Counter()
        for i in range(len(self.docs)):
            for t in self.doc_terms[i]:
                self.df[t] += 1
            for c in self.doc_chars[i]:
                self.df_char[c] += 1
        self.N = max(1, len(self.docs))
        self.avgdl = sum(len(d) for d in self.docs) / float(self.N)

    def bm25(self, qtext):
        q = TK.tokenize(qtext)
        k1, b = 1.2, 0.75
        out = []
        for idx in range(len(self.docs)):
            tf = self.tf[idx]
            dl = len(self.docs[idx]) or 1
            s = 0.0
            for t, w in q:
                if USE_SINGLE_FALLBACK and len(t) == 1:   # 单字：字符兜底 + 降权
                    if t not in self.doc_chars[idx]:
                        continue
                    n = self.df_char[t]
                    idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    s += w * 0.6 * max(idf, 0.0)
                    continue
                f = tf.get(t)
                if not f:
                    continue
                n = self.df[t]
                idf = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                s += w * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avgdl))
            if s > 0:
                out.append((s, self.products[idx].id))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def search(self, query, topk=10):
        return [i for _, i in self.bm25(self.expand(query))[:topk]]


class HybridRrfRetriever(Retriever):
    name = "3-混合召回+RRF"

    def __init__(self, products, synonyms=None):
        Retriever.__init__(self, products, synonyms)
        self.bm = SynonymBm25Retriever(products, synonyms)

    def semantic(self, query):
        """语义召回占位实现：词集重合度。
        真实项目里替换成向量库召回（pgvector / Qdrant）。"""
        q = TK.tokenize(query)
        qt = set(t for t, _ in q)
        if not qt:
            return []
        out = []
        for i, p in enumerate(self.products):
            cnt = sum(w for _, w, _ in self.hits(q, i))
            if cnt > 0:
                out.append((cnt / float(len(qt)), p.id))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def search(self, query, topk=10):
        r1 = [i for _, i in self.bm.bm25(self.expand(query))]
        r2 = [i for _, i in self.semantic(query)]
        K = 60
        score = {}
        for r in (r1, r2):
            for rank, did in enumerate(r, 1):
                score[did] = score.get(did, 0.0) + 1.0 / (K + rank)
        ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
        return [i for i, _ in ranked[:topk]]


# ============================================================================
# 意图改写层：把「生活场景描述」映射成「商品域品类词」
# ----------------------------------------------------------------------------
# 难例提问里一个商品词都没有（「晚上躺床上听歌不想吵到旁边的人」）——
# 这不是相似度算得不够好，是缺了「知道用户其实想买睡眠耳机」的业务知识。
# 真实系统里这一步交给大模型做意图解析（llm_client.parse_intent）；
# 这里的词表是它的离线兜底版，也是评测里唯一不依赖网络的意图解析实现。
# ============================================================================
_INTENT_RULES = None


def intent_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sample_data", "intent_map.json")


def load_intent_rules(path=None):
    global _INTENT_RULES
    if _INTENT_RULES is None:
        with io.open(path or intent_path(), encoding="utf-8") as f:
            data = json.load(f)
        _INTENT_RULES = [(r[0].split("|"), r[1]) for r in data["规则"]]
    return _INTENT_RULES


def rewrite_intent(query):
    """命中规则就把行业词补进检索串。返回 (改写后的串, 补上的词)。"""
    extra, hit_pats = [], []
    for pats, words in load_intent_rules():
        for p in pats:
            if p and p in query:
                hit_pats.append(p)
                for w in words:
                    if w not in extra:
                        extra.append(w)
                break
    if not extra:
        return query, []
    return query + u" " + u" ".join(extra), extra


# ============================================================================
# 真正的稠密向量召回（离线可复现）
#   有可用的 numpy：TF-IDF(词 + 字符1~3gram) -> 截断SVD 降维(LSA) -> 余弦
#   numpy 不可用/被搞坏：自动退化成纯 Python 的 TF-IDF 余弦（不做降维）
# ----------------------------------------------------------------------------
# 为什么要留纯 Python 兜底：PyCharm 会往 PYTHONPATH 注入它自带的
# pycharm_matplotlib_backend/sitecustomize.py，每次启动都 import matplotlib；
# matplotlib 若是拿 NumPy 1.x 编译的、环境里却是 NumPy 2.x，numpy 会变成半坏状态
# （报 _NoValueType / "NumPy module was reloaded"），连 (M>0).sum(axis=0) 都跑不了。
# 这个坑与检索逻辑无关，但会让服务起不来 —— 所以自己探测 + 兜底。
# 强制走兜底：SC_NO_NUMPY=1
# ============================================================================
_NUMPY_READY = None


def _probe_numpy():
    """探测 numpy 是否真能用（不只是 import 成功，而是那几个操作真能算）"""
    if os.environ.get("SC_NO_NUMPY") == "1":
        return False
    try:
        import numpy as np
        a = np.zeros((3, 4), dtype=np.float64)
        a[0, 0] = 1.0
        (a > 0).sum(axis=0)                      # 就是这一句在坏环境下会抛 TypeError
        np.linalg.svd(np.array([[1.0, 2.0], [3.0, 4.0]]), full_matrices=False)
        return True
    except Exception:
        return False


def numpy_ok():
    global _NUMPY_READY
    if _NUMPY_READY is None:
        _NUMPY_READY = _probe_numpy()
    return _NUMPY_READY


class LsaVectorRetriever(Retriever):
    name = "4-语义向量(LSA)"
    DIM = 128

    def __init__(self, products, synonyms=None, dim=None):
        Retriever.__init__(self, products, synonyms)
        self.dim = dim or self.DIM
        self.mode = u"未初始化"
        self._dvec = None
        self._fit()

    def _features(self, text):
        toks = TK.tokenize(text)
        feats = {}
        for t, _ in toks:
            feats[u"w:" + t] = 1.0
        s = u"".join([t for t, _ in toks])      # 先去掉停用词/场景词，再做字符 n-gram
        for n in (1, 2, 3):
            for i in range(max(0, len(s) - n + 1)):
                feats[u"c%d:%s" % (n, s[i:i + n])] = 1.0
        return feats

    def _fit(self):
        """两种模式共用的部分：词表 + IDF + 稀疏（已归一化）文档向量"""
        docs = [self._features(p.text) for p in self.products]
        vocab = {}
        for d in docs:
            for k in d:
                if k not in vocab:
                    vocab[k] = len(vocab)
        N = len(docs)
        V = max(1, len(vocab))
        df = [0] * V
        for d in docs:
            for k in d:
                df[vocab[k]] += 1
        idf = [math.log((N + 1.0) / (df[j] + 1.0)) + 1.0 for j in range(V)]
        self._vocab, self._idf = vocab, idf
        self._sdoc = []
        for d in docs:
            v, sq = {}, 0.0
            for k in d:
                j = vocab[k]
                v[j] = idf[j]
                sq += idf[j] * idf[j]
            nrm = math.sqrt(sq) or 1.0
            self._sdoc.append(dict((j, w / nrm) for j, w in v.items()))

        if numpy_ok():
            try:
                self._fit_dense(N, V)
                self.mode = u"numpy-SVD(%d维)" % self._k
                return
            except Exception:
                # numpy 半路坏了也不能让服务起不来
                self._dvec = None
        self.mode = u"纯Python-TFIDF余弦(numpy不可用)"

    def _fit_dense(self, N, V):
        import numpy as np
        M = np.zeros((N, V), dtype=np.float64)
        for i, d in enumerate(self._sdoc):
            for j, w in d.items():
                M[i, j] = w
        k = max(1, min(self.dim, min(M.shape) - 1))
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        self._Vt, self._k = Vt[:k], k
        dvec = U[:, :k] * S[:k]
        n2 = np.sqrt((dvec * dvec).sum(axis=1, keepdims=True))
        n2[n2 == 0] = 1.0
        self._dvec = dvec / n2

    # ---------- 纯 Python 路径 ----------
    def _encode_sparse(self, text):
        q, sq = {}, 0.0
        for k in self._features(text):
            j = self._vocab.get(k)
            if j is not None:
                q[j] = self._idf[j]
                sq += self._idf[j] * self._idf[j]
        if not sq:
            return None
        nrm = math.sqrt(sq)
        return dict((j, w / nrm) for j, w in q.items())

    def _search_sparse(self, query, topk):
        q = self._encode_sparse(query)
        if not q:
            return []
        out = []
        for i, d in enumerate(self._sdoc):
            s = 0.0
            if len(d) < len(q):                  # 遍历短的一侧，省点计算
                for j, w in d.items():
                    w2 = q.get(j)
                    if w2:
                        s += w * w2
            else:
                for j, w in q.items():
                    w2 = d.get(j)
                    if w2:
                        s += w * w2
            if s > 0:
                out.append((s, self.products[i].id))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in out[:topk]]

    # ---------- 对外 ----------
    def encode(self, text):
        if self._dvec is None:
            return self._encode_sparse(text)
        import numpy as np
        q = np.zeros(len(self._vocab), dtype=np.float64)
        for k in self._features(text):
            j = self._vocab.get(k)
            if j is not None:
                q[j] = self._idf[j]
        nrm = float(np.sqrt((q * q).sum()))
        if nrm <= 0:
            return None
        v = (q / nrm).dot(self._Vt.T)           # 投影到 k 维隐语义空间
        nv = float(np.sqrt((v * v).sum()))
        return v / nv if nv else None

    def search(self, query, topk=10):
        if self._dvec is None:                  # 纯 Python 兜底
            return self._search_sparse(query, topk)
        import numpy as np
        v = self.encode(query)
        if v is None:
            return []
        sims = self._dvec.dot(v)
        out = []
        for idx in np.argsort(-sims)[:topk]:
            if sims[idx] <= 0:
                break
            out.append(self.products[int(idx)].id)
        return out


class IntentBm25Retriever(SynonymBm25Retriever):
    """只加意图改写，别的都不动 —— 用来单独量出「意图改写」这一层值多少分。"""
    name = "5-意图改写+BM25"

    def search(self, query, topk=10):
        q, _ = rewrite_intent(query)
        return [i for _, i in self.bm25(self.expand(q))[:topk]]


class FullStackRetriever(Retriever):
    """上线组合：意图改写 + BM25 + 稠密向量，三路 RRF 融合。"""
    name = "6-意图改写+BM25+向量+RRF"

    def __init__(self, products, synonyms=None):
        Retriever.__init__(self, products, synonyms)
        self.bm = SynonymBm25Retriever(products, synonyms)
        self.vec = LsaVectorRetriever(products, synonyms)

    def search(self, query, topk=10):
        q, _ = rewrite_intent(query)
        r1 = [i for _, i in self.bm.bm25(self.expand(q))]
        r2 = self.vec.search(q, topk=len(self.products))
        K = 60
        score = {}
        for r in (r1, r2):
            for rank, did in enumerate(r, 1):
                score[did] = score.get(did, 0.0) + 1.0 / (K + rank)
        ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
        return [i for i, _ in ranked[:topk]]
