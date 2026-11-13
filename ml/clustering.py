"""项目 C · 聚类：星海优选电商客户分群。

由 build.py 调用，也可单独运行：python clustering.py
`# %% step:xxx` 块会原样展示在网页上；裸 `# %%` 块是幕后工作（导出数据、断言教学结论）。
"""
# %%
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import DATA, SEED, r, sample_rows, snippets, to_js

EX = {}

# %% step:load
orders = pd.read_csv(DATA / "orders.csv", encoding="utf-8-sig", parse_dates=["下单时间"])
orders.shape
orders["客户ID"].isna().mean()     # 游客订单占比
orders["下单时间"].max()           # 怎么会有 2025 年的订单？
orders["数量"].describe()          # 负数？几百件？

# %%
EX["n_orders"], EX["n_ids"] = len(orders), int(orders["客户ID"].nunique())
EX["date_min"], EX["date_max"] = str(orders["下单时间"].min().date()), str(orders["下单时间"].max().date())
raw_csv = pd.read_csv(DATA / "orders.csv", encoding="utf-8-sig")   # 原样字符串，用于展示
o = orders


def first(mask):
    return o.index[mask][0]


dupm = o.duplicated(keep=False)
dup_a = o.index[dupm & o["客户ID"].str.startswith("U0", na=False)][0]
dup_b = o.index[(o == o.loc[dup_a]).all(axis=1) & (o.index != dup_a)][0]
ret_id = o.loc[o["数量"] < 0, "订单号"].iloc[0]
ok = o["客户ID"].str.startswith("U0", na=False) & o["数量"].between(1, 3) & (o["单价"] > 0) \
     & (o["下单时间"] <= "2024-12-31") & ~dupm & ~o["订单号"].duplicated(keep=False)
normal = list(o.index[ok][:3])
p = {"guest": first(o["客户ID"].isna()), "ret": first(o["数量"] < 0),
     "ret_orig": first((o["订单号"] == ret_id) & (o["数量"] > 0)), "gift": first(o["单价"] == 0),
     "test": first(o["客户ID"].str.startswith("TEST", na=False)), "bulk": first(o["数量"] >= 50),
     "future": first(o["下单时间"] > "2024-12-31 23:59")}
order = [normal[0], p["ret_orig"], p["guest"], dup_a, p["gift"], normal[1], p["ret"], p["test"], p["future"],
         dup_b, p["bulk"], normal[2]]
EX["sample"] = sample_rows(raw_csv, order)
pos = order.index
EX["flags"] = [
    [pos(p["guest"]), "客户ID", "游客下单：没有客户 ID，无法归属到人"],
    [pos(p["ret"]), "数量", "负数量 = 退货，要和上面同一订单号的原单冲抵"],
    [pos(p["gift"]), "单价", "单价 0：赠品，不算真实购买"],
    [pos(p["test"]), "客户ID", "TEST 开头：内部测试账号"],
    [pos(p["bulk"]), "数量", "一单几百件：批发商，是 B 端客户，混进来会扭曲 C 端分群"],
    [pos(p["future"]), "下单时间", "快照日（2024-12-31）之后的订单：来自'未来'，算特征时要排除"],
    [pos(dup_a), "订单号", "和下面某行完全相同：系统重复写入"],
    [pos(dup_b), "订单号", "重复记录"],
]
EX["flag_cols"] = []
EX["clean_log"] = [
    ["完全重复的行", int(o.duplicated().sum())],
    ["游客订单（无客户 ID）", int(o["客户ID"].isna().sum())],
    ["测试账号订单", int(o["客户ID"].str.startswith("TEST", na=False).sum())],
    ["快照日之后的订单", int((o["下单时间"] > "2024-12-31 23:59").sum())],
    ["赠品行（单价 0）", int((o["单价"] == 0).sum())],
    ["退货行（负数量）", int((o["数量"] < 0).sum())],
    ["批发商账号", int(o.loc[o["数量"] >= 50, "客户ID"].nunique())],
]

# %% step:clean
SNAPSHOT = pd.Timestamp("2024-12-31 23:59")   # 站在哪一天看客户：只能用这之前的数据


def clean(orders, snapshot=SNAPSHOT, drop_outliers=True):
    """订单级清洗，返回"一行一单"的干净订单表。"""
    o = orders.drop_duplicates()                                        # ① 系统重复写入
    o = o[o["客户ID"].notna()]                                           # ② 游客订单无法归属到人
    o = o[o["下单时间"] <= snapshot]                                     # ③ 快照日之后 = 来自"未来"
    o = o[o["单价"] > 0]                                                # ④ 赠品不是购买
    if drop_outliers:
        o = o[~o["客户ID"].str.startswith("TEST")]                      # ⑤ 内部测试账号
        o = o[~o["客户ID"].isin(o.loc[o["数量"] >= 50, "客户ID"])]       # ⑥ 批发商走 B 端，不进 C 端分群
    o = o.assign(金额=o["数量"] * o["单价"])
    od = o.groupby("订单号").agg(客户ID=("客户ID", "first"), 时间=("下单时间", "min"),
                               金额=("金额", "sum"), 用券=("用券", "max"))
    return od[od["金额"] > 0]                                           # ⑦ 退货冲抵后 ≤0 的整单剔除


# %% step:features
def customer_features(od, snapshot=SNAPSHOT):
    """订单表 → 客户表：聚类的"样本"是客户，不是订单。经典 RFM 再加三个业务特征。"""
    g = od.groupby("客户ID")
    return pd.DataFrame({
        "最近购买距今天数": (snapshot - g["时间"].max()).dt.days,   # R：Recency
        "购买次数": g.size(),                                      # F：Frequency
        "总消费": g["金额"].sum().round(2),                         # M：Monetary
        "客单价": g["金额"].mean().round(2),
        "首购距今天数": (snapshot - g["时间"].min()).dt.days,        # 老客还是新客
        "用券率": g["用券"].mean().round(3),                        # 对促销有多敏感
    })


od = clean(orders)
feat = customer_features(od)

# %%
EX["n_customers"] = len(feat)
cid = feat.index[(feat["购买次数"] == 6)][0]
ex_orders = od[od["客户ID"] == cid].sort_values("时间").reset_index()
ex_orders["时间"] = ex_orders["时间"].dt.strftime("%Y-%m-%d")
EX["agg_orders"] = sample_rows(ex_orders, ex_orders.index)
EX["agg_row"] = [cid] + [r(v, 3) for v in feat.loc[cid]]
EX["feat_cols"] = list(feat.columns)
EX["feat_desc"] = {c: [r(feat[c].min(), 2), r(feat[c].median(), 2), r(feat[c].max(), 2)] for c in feat.columns}

# %% step:prep
SKEWED = ["购买次数", "总消费", "客单价"]       # 长尾分布：先取 log，免得少数大客户主宰"距离"
pre = make_pipeline(
    ColumnTransformer([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), SKEWED)],
                      remainder="passthrough", verbose_feature_names_out=False),
    StandardScaler(),                            # 统一量纲：天数、次数、金额、比例放到同一把尺子上
)
Z = pre.fit_transform(feat)

# %% step:scale_demo
# ❌ 原始特征直接聚类：总消费的数值动辄上万，"距离"几乎只由它决定
raw_labels = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(feat)
# ✅ log + 标准化之后再聚类
scaled_labels = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(Z)


# %%
def profile(f, labels):
    g = f.assign(群=labels).groupby("群")
    out = g.mean().round(2)
    out.insert(0, "人数", g.size())
    return out


def table(prof):
    return {"cols": list(prof.columns), "rows": [[int(i)] + [r(v, 2) for v in row] for i, row in prof.iterrows()]}


EX["scale_demo"] = {"raw": table(profile(feat, raw_labels)), "scaled": table(profile(feat, scaled_labels))}
sizes_raw, sizes_scaled = np.bincount(raw_labels), np.bincount(scaled_labels)

# %% step:choose_k
ks = list(range(2, 9))
fits = {k: KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z) for k in ks}
inertia = [fits[k].inertia_ for k in ks]                         # 肘部法：簇内距离平方和，越小越紧
silhouette = [silhouette_score(Z, fits[k].labels_) for k in ks]  # 轮廓系数：-1~1，越大分得越开

# %%
EX["ks"], EX["inertia"], EX["silhouette"] = ks, r(inertia, 0), r(silhouette, 3)
EX["profiles"] = {k: table(profile(feat, fits[k].labels_)) for k in ks}
K = 5
EX["K"] = K
EX["sil_best_k"] = ks[int(np.argmax(silhouette))]

# %% step:stability
# 稳定性检验：随机抽 80% 客户重新聚类，看分群结果和全量版本是否一致（ARI=1 完全一致，0 等于瞎分）
rng = np.random.RandomState(SEED)
stability = []
for s in range(10):
    idx = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
    km = KMeans(n_clusters=5, n_init=10, random_state=s).fit(Z[idx])
    stability.append(adjusted_rand_score(fits[5].labels_, km.predict(Z)))

# %%
EX["stability"] = r(stability, 3)


# %% step:name
def name_clusters(prof):
    """按业务规则命名：每条规则认领"最鲜明"的那个群，剩下的是普通活跃客户。"""
    rules = [("沉睡客户", "最近购买距今天数", max), ("新客", "首购距今天数", min),
             ("高价值客户", "总消费", max), ("促销敏感型", "用券率", max)]
    names, left = {}, set(prof.index)
    for name, col, pick in rules:
        c = pick(left, key=lambda i: prof.loc[i, col])
        names[c] = name
        left.remove(c)
    return names | {c: "普通活跃" for c in left}


segments = feat.assign(群=fits[5].labels_)
names = name_clusters(profile(feat, fits[5].labels_))
segments["人群"] = segments["群"].map(names)

# %% step:validate
# 用"未来"验证分群有没有业务意义：快照日之后的一个季度，各群有多少人回来复购？
q1 = clean(orders, snapshot=orders["下单时间"].max())
q1_buyers = q1.loc[q1["时间"] > SNAPSHOT, "客户ID"].unique()
segments["次季度复购"] = segments.index.isin(q1_buyers)
repurchase = segments.groupby("人群")["次季度复购"].mean()

# %%
EX["names"] = {int(k): v for k, v in names.items()}
EX["repurchase"] = {k: r(v, 3) for k, v in repurchase.items()}
EX["segment_sizes"] = {k: int(v) for k, v in segments["人群"].value_counts().items()}

# %% step:time_leak
# ❌ 导数时没卡快照日：把 2025 年一季度的订单也算进了特征
end = orders["下单时间"].max()
feat_leak = customer_features(clean(orders, snapshot=end), snapshot=end)
labels_leak = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(clone(pre).fit_transform(feat_leak))
leak_seg = feat_leak.assign(人群=pd.Series(labels_leak, feat_leak.index).map(
    name_clusters(profile(feat_leak, labels_leak))))
leak_seg["次季度复购"] = leak_seg.index.isin(q1_buyers)
repurchase_leak = leak_seg.groupby("人群")["次季度复购"].mean()
# "沉睡客户一个都没回来"——这是假象：凡是回来了的，都因为"偷看了未来"被分进了活跃群

# %%
EX["repurchase_leak"] = {k: r(v, 3) for k, v in repurchase_leak.items()}

# %% step:outlier_demo
# ❌ 不剔除批发商和测试账号
feat_out = customer_features(clean(orders, drop_outliers=False))
labels_out = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(clone(pre).fit_transform(feat_out))
prof_out = profile(feat_out, labels_out)
vip_out = prof_out.loc[[c for c, n in name_clusters(prof_out).items() if n == "高价值客户"][0]]
# 14 个异常账号混进"高价值客户"群，把这个群的平均消费抬高了好几倍 —— 预算按它算就全错了

# %%
prof_ok = profile(feat, fits[5].labels_)
vip_ok = prof_ok.loc[[c for c, n in names.items() if n == "高价值客户"][0]]
EX["outlier_demo"] = {"with": [int(vip_out["人数"]), r(vip_out["总消费"], 0), r(vip_out["客单价"], 0)],
                      "without": [int(vip_ok["人数"]), r(vip_ok["总消费"], 0), r(vip_ok["客单价"], 0)],
                      "n_outliers": int(len(feat_out) - len(feat))}

# 可视化：PCA 压到二维，只用于画图（聚类本身在 6 维空间里做）
pca = PCA(n_components=2, random_state=SEED).fit(Z)
P = pca.transform(Z)
pick = np.random.RandomState(SEED).choice(len(Z), 800, replace=False)
EX["pca"] = {"xy": r(P[pick], 3), "var": r(pca.explained_variance_ratio_, 3),
             "labels": {k: fits[k].labels_[pick].tolist() for k in ks}, "raw": raw_labels[pick].tolist()}

# 新客户分群：导出预处理参数和质心，网页 JS 实时判断属于哪个群
ct, sc = pre[0], pre[1]
EX["sim"] = {"order": list(ct.get_feature_names_out()), "log": SKEWED, "mean": r(sc.mean_, 5),
             "scale": r(sc.scale_, 5), "centers": r(fits[5].cluster_centers_, 5)}


def sim_assign(row, s=EX["sim"]):
    """与网页 JS 一致的手算版。"""
    z = [((np.log1p(row[c]) if c in s["log"] else row[c]) - m) / sd for c, m, sd in zip(s["order"], s["mean"], s["scale"])]
    return int(np.argmin(((np.array(s["centers"]) - z) ** 2).sum(axis=1)))


EX["code"] = snippets(__file__)
print("clustering:", {"n_customers": EX["n_customers"], "silhouette": EX["silhouette"], "sil_best_k": EX["sil_best_k"],
                      "sizes_raw": sizes_raw.tolist(), "sizes_scaled": sizes_scaled.tolist(),
                      "outlier_demo": EX["outlier_demo"], "stability": EX["stability"], "names": EX["names"],
                      "repurchase": EX["repurchase"], "repurchase_leak": EX["repurchase_leak"],
                      "pca_var": EX["pca"]["var"]})
# 断言：网页里讲的每个结论都必须被真实结果支持，否则构建失败
assert sizes_raw.min() < 0.5 * sizes_scaled.min(), "不缩放时应出现极小的群"
assert EX["outlier_demo"]["with"][1] > 3 * EX["outlier_demo"]["without"][1], "异常账号应明显扭曲高价值群"
assert np.mean(stability) > 0.8
assert EX["sil_best_k"] == K == 5, "页面说轮廓系数和业务都指向 K=5"
assert repurchase_leak["沉睡客户"] < 0.02 < repurchase["沉睡客户"], "时间泄漏应制造'沉睡客户全不回来'的假象"
assert repurchase["高价值客户"] > repurchase["普通活跃"] > repurchase["沉睡客户"]
assert all(sim_assign(row) == lab for (_, row), lab in zip(feat.head(100).iterrows(), fits[5].labels_[:100]))
to_js("CLU", EX, "clustering.js")
