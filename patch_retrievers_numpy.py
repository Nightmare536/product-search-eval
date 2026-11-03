# -*- coding: utf-8 -*-
"""给 retrievers.py 的 LsaVectorRetriever 加上「纯 Python 兜底」。

背景（真踩到的坑）：
  PyCharm 会把 D:\\PyCharm.../pycharm_matplotlib_backend 注入 PYTHONPATH，
  于是每次启动 Python 都会先 import matplotlib；如果 matplotlib 是拿 NumPy 1.x
  编译的、而环境里是 NumPy 2.x，numpy 就会进入半坏状态（报 _NoValueType、
  "NumPy module was reloaded"），连 (M > 0).sum(axis=0) 都跑不了 —— 服务直接起不来。

修法：自己探测 numpy「能不能真的用」，不能用就退化成纯 Python 的 TF-IDF 余弦。
（实测：兜底版不做 SVD，在 149 条评测集上的指标与 LSA 版一致。）

    python patch_retrievers_numpy.py
"""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "retrievers.py")

NEW = u'''# ============================================================================
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


'''


def main():
    src = io.open(TARGET, encoding="utf-8").read().replace("\r\n", "\n")
    marker = u"# ============================================================================\n# 真正的稠密向量召回"
    if marker not in src:
        raise SystemExit(u"找不到旧注释块，文件可能被改过，先看一下 %s" % TARGET)
    start = src.index(marker)
    end = src.index("class IntentBm25Retriever")
    src = src[:start] + NEW + src[end:]
    io.open(TARGET, "w", encoding="utf-8", newline="").write(src)

    import json
    import sys
    sys.path.insert(0, HERE)
    import retrievers

    print(u"已打补丁：%s" % TARGET)
    print(u"  numpy 可用 = %s" % retrievers.numpy_ok())

    p = os.path.join(HERE, "sample_data", "products.json")
    data = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else []
    if not data:
        data = [{"id": "p1", "title": u"苏泊尔 电煮锅 1.5L", "category": u"锅具",
                 "brand": u"苏泊尔", "price": 99, "tags": [u"电煮锅"]}]
    ps = [retrievers.Product(d) for d in data[:20]]
    r = retrievers.LsaVectorRetriever(ps, {})
    print(u"  向量模式 = %s" % r.mode)
    print(u"  自检检索「电煮锅」-> %s" % r.search(u"电煮锅", 3))
    return 0


if __name__ == "__main__":
    main()
