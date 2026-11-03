# -*- coding: utf-8 -*-
"""把 tokenize_cn.py 里的停用词表/场景词表**机械导出**成 Java 常量类。

为什么要生成而不是手抄：SINGLE_STOP 有 200+ 个汉字，手抄必错，
错了会静默地把商品词当功能字丢掉（召回直接掉一截，还很难查）。
生成一次，以后 Python 侧改词表只要重跑本脚本。

    python gen_java_lexicon.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tokenize_cn as TK  # noqa: E402

OUT = os.path.abspath(os.path.join(
    HERE, os.pardir, u"选品检索服务-java", "src", "main", "java",
    "com", "shop", "search", "core", "CnLexicon.java"))


def jarr(items):
    return ", ".join('"%s"' % s for s in items)


def main():
    stops = sorted(TK.STOPWORDS)
    singles = sorted(TK.SINGLE_STOP)
    scene = sorted(TK.SCENE)
    L = []
    L.append("package com.shop.search.core;")
    L.append("")
    L.append("/* 本文件由 选品引擎评测/gen_java_lexicon.py 自动生成，不要手改。 */")
    L.append("public final class CnLexicon {")
    L.append("")
    L.append("    /** 多字停用词：命中就整段丢弃（推荐/想买/的话 ...） */")
    L.append("    public static final String[] STOPWORDS = {")
    L.append("        " + jarr(stops))
    L.append("    };")
    L.append("")
    L.append("    /** 单字停用表：单字兜底时这些字直接丢（不/多/上 ...），共 %d 字 */" % len(singles))
    L.append("    public static final String[] SINGLE_STOP = {")
    L.append("        " + jarr(singles))
    L.append("    };")
    L.append("")
    L.append("    /** 场景词：只说明\"在哪用\"，不说明\"是什么\"，降权 0.35 */")
    L.append("    public static final String[] SCENE = {")
    L.append("        " + jarr(scene))
    L.append("    };")
    L.append("")
    L.append("    private CnLexicon() {")
    L.append("    }")
    L.append("}")
    d = os.path.dirname(OUT)
    if not os.path.isdir(d):
        os.makedirs(d)
    io.open(OUT, "w", encoding="utf-8").write(chr(10).join(L) + chr(10))
    print(u"已生成 %s" % OUT)
    print(u"  多字停用词 %d 个 / 单字停用表 %d 字 / 场景词 %d 个"
          % (len(stops), len(singles), len(scene)))


if __name__ == "__main__":
    main()
