"""三个项目脚本共用的小工具：路径、代码片段切分、导出给网页的数据。"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "site" / "data"   # 原始脏数据 CSV（供下载练习）和预计算结果 JS 放在一起
OUT = DATA
SEED = 42


def snippets(path):
    """按 `# %% step:名字` 把脚本切成代码片段；裸 `# %%` 开始的块不展示（导出、断言等杂务）。
    这样网页上展示的代码就是真正跑出结果的代码。"""
    out, name = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"# %% step:(\w+)", line)
        if m:
            name = m.group(1)
            out.setdefault(name, [])
        elif line.startswith("# %%"):
            name = None
        elif name:
            out[name].append(line)
    return {k: "\n".join(v).strip("\n") for k, v in out.items()}


def r(x, n=4):
    """数组/数值四舍五入后转成 JSON 友好的 Python 对象，减小导出体积。"""
    a = np.round(np.asarray(x, dtype=float), n)
    return a.tolist() if a.ndim else float(a)


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def to_js(var, obj, filename):
    """写成 `window.VAR = {...}` 的 .js 文件：双击打开 html（file://）也能加载，不需要服务器。"""
    OUT.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, default=_default, separators=(",", ":"))
    (OUT / filename).write_text(f"window.{var} = {text};\n", encoding="utf-8")


def sample_rows(df, idx):
    """取若干行给网页表格展示，缺失值转成 None。"""
    sub = df.loc[idx]
    return {
        "cols": list(sub.columns),
        "rows": [[None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
                  for v in row] for row in sub.itertuples(index=False)],
    }


def hist(values, bins=30):
    counts, edges = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return {"counts": counts.tolist(), "edges": r(edges, 3)}
