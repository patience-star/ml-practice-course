"""一键重建：生成数据 → 用真实 scikit-learn 跑完三个项目 → 导出网页数据。

    python build.py

每个项目脚本末尾都有断言：网页上讲的结论（比如"泄漏版分数虚高"）若不被真实结果支持，构建直接失败。
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ML = Path(__file__).resolve().parent / "ml"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

for script in ["make_data.py", "regression.py", "classification.py", "clustering.py"]:
    t = time.time()
    subprocess.run([sys.executable, script], cwd=ML, env=ENV, check=True)
    print(f"[OK] {script}  {time.time() - t:.0f}s")
print("全部完成：双击打开 site/index.html")
