# 机器学习实战课

三个贴合真实业务的 scikit-learn 实战项目，做成一个可交互的教学网站：

| 项目 | 任务 | 业务场景 | 重点 |
|---|---|---|---|
| A | 回归 | 二手房一键估价 | 时间切分、log 目标、TargetEncoder、模型过期 |
| B | 分类 | 会员流失预警 | 准确率陷阱、目标泄漏、调参选型、用利润定阈值 |
| C | 聚类 | 电商客户分群 | 订单→客户特征、缩放与异常、选 K、用未来验证 |

三个项目走同一条 8 步流程（参考 scikit-learn 官方 Getting Started / User Guide / Common pitfalls），
**数据泄漏**是贯穿全课的主线：14 个漏点，每个都有"泄漏版 vs 正确版"的真实分数对比。

## 打开

双击 `site/index.html` 即可（离线可用，无第三方依赖）。

## 重新生成

```bash
pip install scikit-learn pandas numpy joblib
python build.py
```

`build.py` 依次运行 `ml/` 下的四个脚本：生成仿真脏数据 → 用真实 sklearn 跑完三个项目 → 把结果写成 `site/data/*.js`。
每个项目脚本末尾都有断言：网页上讲的教学结论（例如"泄漏版分数虚高""直接上线会系统性高估"）
如果不被真实结果支持，构建会失败。

## 目录

```
build.py            一键重建
ml/
  make_data.py      三份带"坑"的仿真业务数据（固定随机种子）
  regression.py     项目 A 完整流程
  classification.py 项目 B 完整流程
  clustering.py     项目 C 完整流程
  common.py         导出工具
site/
  index.html        总览：8 步流程图、泄漏博物馆、选型小测
  regression.html / classification.html / clustering.html
  assets/           style.css（含暗色模式）、app.js（图表、测验、找茬等组件）
  data/             原始脏数据 CSV（供下载练习）+ 预计算结果 JS
models/             训练好的整条 Pipeline（joblib）
```

## 怎么改内容

- **改代码**：项目脚本用 `# %% step:名字` 分段，网页里的 `<div data-code="CLF.code.名字">` 会显示同名代码段。
  裸 `# %%` 开头的块是导出、断言等幕后代码，不在网页上显示。改完运行 `python build.py`。
- **改数字**：正文里的数字写成 `<b data-v="REG.test_mae" data-f="wan"></b>`，从数据文件自动填充，不要手写。
  格式见 `site/assets/app.js` 里的 `FMT`。
- **改文字 / 加一步**：直接编辑对应的 html。每个 `<section class="step">` 是一步，左侧导轨自动生成；
  含 `<aside class="leak" data-id="...">` 的步骤会自动标上漏点。新增漏点记得同时加到 `index.html` 的 `LEAKS` 列表。
- **改数据里的"坑"**：在 `ml/make_data.py` 里调整，再相应更新脚本里的断言。

学习进度保存在浏览器 localStorage 里，清除浏览器数据会重置。
