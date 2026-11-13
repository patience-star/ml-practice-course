"""项目 A · 回归：星海市二手房"一键估价"。

由 build.py 调用，也可单独运行：python regression.py
`# %% step:xxx` 块会原样展示在网页上；裸 `# %%` 块是幕后工作（导出数据、断言教学结论）。
"""
# %%
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, TimeSeriesSplit, cross_val_score, train_test_split, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeRegressor

from common import DATA, ROOT, SEED, hist, r, sample_rows, snippets, to_js

EX = {}

# %% step:load
raw = pd.read_csv(DATA / "house.csv", encoding="utf-8-sig")
raw.describe()                  # 面积最大 9999？建成年份最小 0？
raw["区域"].value_counts()       # 同一个区出现了好几种写法
raw["成交价"].skew()             # 偏度 > 1：少数豪宅把均值拉得很高

# %%
EX["n_raw"] = len(raw)
EX["dtypes"] = [[c, str(t), int(raw[c].isna().sum())] for c, t in raw.dtypes.items()]
EX["area_max"], EX["year_min"] = r(raw["面积"].max()), int(raw["建成年份"].min())
EX["district_counts"] = {k: int(v) for k, v in raw["区域"].value_counts().items()}
EX["skew"], EX["skew_log"] = r(raw["成交价"].skew(), 2), r(np.log(raw["成交价"]).skew(), 2)
EX["hist_raw"], EX["hist_log"] = hist(raw["成交价"], 40), hist(np.log(raw["成交价"]), 40)
EX["price_median"], EX["price_mean"] = r(raw["成交价"].median(), 1), r(raw["成交价"].mean(), 1)
month = pd.to_datetime(raw["成交日期"]).dt.to_period("M")
trend = raw.groupby(month)["单价"].median()   # 单价不能当特征，但做探索分析正好
EX["trend"] = {"month": [str(m) for m in trend.index], "unit": r(trend.to_numpy() / 10000, 3)}


def first(mask):
    return raw.index[mask][0]


content = raw.drop(columns="房源编号")
dup = raw.index[content.duplicated(keep=False)]
dup_a = dup[0]
dup_b = raw.index[(content == content.loc[dup_a]).all(axis=1) & (raw.index != dup_a)][0]
ok = (raw["面积"] < 300) & raw["建成年份"].between(1950, 2025) & raw["区域"].str.match(r"^\S+区$") \
     & ~content.duplicated(keep=False)
normal = list(raw.index[ok & raw["距地铁米"].notna()][:3])
p = {"feet": first(raw["面积"].between(300, 3100)), "9999": first(raw["面积"] == 9999),
     "nan": first(raw["面积"].isna()), "year0": first(raw["建成年份"] == 0),
     "future": first(raw["建成年份"] > 2025), "district": first(~raw["区域"].str.match(r"^\S+区$")),
     "subway": first(ok & raw["距地铁米"].isna())}
order = [normal[0], p["feet"], dup_a, p["year0"], normal[1], p["district"], p["subway"], p["9999"], dup_b,
         p["future"], normal[2], p["nan"]]
EX["sample"] = sample_rows(raw, order)
pos = order.index
EX["flags"] = [
    [pos(p["feet"]), "面积", "单位错误：中介按平方英尺录入（÷10.764 才是平方米）"],
    [pos(p["9999"]), "面积", "9999 是系统占位值，不是真面积"],
    [pos(p["nan"]), "面积", "缺失值"],
    [pos(p["year0"]), "建成年份", "0 年？录入缺失被写成了 0"],
    [pos(p["future"]), "建成年份", "未来年份：不可能"],
    [pos(p["district"]), "区域", "写法不统一：少了“区”字或多了空格，模型会当成另一个区"],
    [pos(dup_a), "房源编号", "和下面某行除编号外一模一样：同一套房被两个中介重复挂牌"],
    [pos(dup_b), "房源编号", "重复挂牌：编号不同，内容相同"],
    [pos(p["subway"]), "距地铁米", "陷阱！这不是脏数据：附近根本没有地铁。缺失本身就是信息", "ok"],
]
EX["flag_cols"] = [["单价", "数据泄漏！单价 = 成交价 ÷ 面积，成交后才知道"],
                   ["契税", "数据泄漏！契税按成交价计算，成交后才知道"]]

# %% step:clean
LEAK = ["单价", "契税"]   # 成交之后才有，而且由成交价直接算出 —— 估价时不可能知道


def clean(df, dedup=True):
    """规则型清洗：不从数据里"学"任何参数，所以切分前做没问题，上线时对新数据原样复用。"""
    if dedup:
        df = df.drop_duplicates(subset=df.columns.drop("房源编号"))       # ① 编号不同、内容一样 = 重复挂牌
    df["区域"] = df["区域"].str.strip().str.removesuffix("区") + "区"      # ② "东湖"/" 东湖区" → "东湖区"
    feet = df["面积"].between(300, 3100)
    df["面积"] = df["面积"].mask(feet, (df["面积"] / 10.764).round(1))      # ③ 平方英尺 → 平方米
    df["面积"] = df["面积"].where(df["面积"] <= 300)                        # ④ 9999 之类的占位值 → 缺失
    df["建成年份"] = df["建成年份"].where(df["建成年份"].between(1950, 2025))  # ⑤ 0 年、未来年份 → 缺失
    df["成交日期"] = pd.to_datetime(df["成交日期"])
    df["成交月"] = (df["成交日期"].dt.year - 2022) * 12 + df["成交日期"].dt.month - 1   # 时间趋势
    # "距地铁米"缺失 = 附近没地铁，不是错误：交给 SimpleImputer(add_indicator=True) 变成一个特征
    return df.drop(columns=LEAK + ["房源编号"])


df = clean(raw).sort_values("成交日期")

# %%
EX["clean_log"] = [
    ["重复挂牌（编号不同、内容相同）", int(content.duplicated().sum())],
    ["区域写法不统一", int((~raw["区域"].str.match(r"^\S+区$")).sum())],
    ["面积按平方英尺录入", int(raw["面积"].between(300, 3100).sum())],
    ["面积为占位值 9999", int((raw["面积"] == 9999).sum())],
    ["面积缺失", int(raw["面积"].isna().sum())],
    ["建成年份为 0 或未来", int((~raw["建成年份"].between(1950, 2025)).sum())],
]
EX["n_clean"] = len(df)

# %% step:split
train = df[df["成交日期"] < "2024-07-01"]    # 用过去训练……
test = df[df["成交日期"] >= "2024-07-01"]    # ……预测"未来"：最后半年当测试集，封存到最后
X_train, y_train = train.drop(columns="成交价"), train["成交价"]
X_test, y_test = test.drop(columns="成交价"), test["成交价"]

# %%
EX["n_train"], EX["n_test"] = len(train), len(test)

# %% step:prep
num = ["室", "厅", "总楼层", "建成年份", "距地铁米", "成交月"]
cat = ["区域", "楼层", "装修", "学区房"]
pre = ColumnTransformer([
    ("area", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log), StandardScaler()),
     ["面积"]),
    ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    # 800 个小区：独热太稀疏 → 目标编码（训练时内部交叉拟合，样本看不到自己的答案）
    ("comm", TargetEncoder(cv=KFold(5, shuffle=True, random_state=SEED)), ["小区"]),
])


def priced(model):
    """预处理 + 模型，并在 log(价格) 上训练：房价是"乘法"结构（面积 × 单价 × 各种系数），取 log 变加法。"""
    return TransformedTargetRegressor(make_pipeline(pre, model), func=np.log, inverse_func=np.exp)


# %% step:model
cv = TimeSeriesSplit(n_splits=5)   # 交叉验证也按时间：每一折都用更早的数据预测更晚的
models = {
    "Dummy·猜中位数": DummyRegressor(strategy="median"),
    "Ridge·原始价格": make_pipeline(pre, Ridge()),
    "Ridge·log 价格": priced(Ridge()),
    "随机森林": priced(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=SEED)),
    "HistGB 梯度提升": priced(HistGradientBoostingRegressor(random_state=SEED)),
}
arena = {name: -cross_val_score(m, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
         for name, m in models.items()}

# %%
EX["arena"] = [[k, r(v, 1)] for k, v in arena.items()]

# %% step:depth
depths = list(range(1, 21))
train_mae, val_mae = validation_curve(
    priced(DecisionTreeRegressor(random_state=SEED)), X_train, y_train,
    param_name="regressor__decisiontreeregressor__max_depth", param_range=depths,
    cv=cv, scoring="neg_mean_absolute_error")

# %%
EX["depth"] = {"d": depths, "train": r(-train_mae.mean(axis=1), 1), "val": r(-val_mae.mean(axis=1), 1)}
EX["best_depth"] = depths[int(np.argmax(val_mae.mean(axis=1)))]

# %% step:tune
search = GridSearchCV(
    priced(HistGradientBoostingRegressor(random_state=SEED)),
    param_grid={"regressor__histgradientboostingregressor__learning_rate": [0.03, 0.06, 0.1, 0.2],
                "regressor__histgradientboostingregressor__max_leaf_nodes": [7, 15, 31, 63]},
    cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
search.fit(X_train, y_train)    # 只用训练期数据；测试集依然封存
best = search.best_estimator_

# %%
LR_GRID, LEAF_GRID = [0.03, 0.06, 0.1, 0.2], [7, 15, 31, 63]
key = "regressor__histgradientboostingregressor__"
cvr = {(p[key + "learning_rate"], p[key + "max_leaf_nodes"]): -s
       for p, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])}
EX["grid"] = {"lr": LR_GRID, "leaf": LEAF_GRID, "mae": [[r(cvr[(a, b)], 1) for b in LEAF_GRID] for a in LR_GRID]}
EX["best_params"] = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
EX["best_cv_mae"] = r(-search.best_score_, 1)

# %% step:leak_demo
# 泄漏版：把"单价"加回特征。log(成交价) = log(单价) + log(面积)，连线性模型都能一眼"学会"
leaky = TransformedTargetRegressor(
    make_pipeline(ColumnTransformer(pre.transformers + [("leak", FunctionTransformer(np.log), ["单价"])]),
                  Ridge()), func=np.log, inverse_func=np.exp)
r2_leak = cross_val_score(leaky, X_train.join(raw["单价"]), y_train, cv=cv, scoring="r2").mean()
r2_clean = cross_val_score(priced(Ridge()), X_train, y_train, cv=cv, scoring="r2").mean()

# %%
EX["r2_leak"], EX["r2_clean"] = r(r2_leak), r(r2_clean)

# %% step:dup_demo
# 对照实验：不去重 + 随机切分
dirty = clean(raw, dedup=False)
Xa, Xb, ya, yb = train_test_split(dirty.drop(columns="成交价"), dirty["成交价"], test_size=0.2, random_state=SEED)
rf = priced(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)).fit(Xa, ya)
err = np.abs(rf.predict(Xb) - yb)


def fingerprint(X):
    return pd.util.hash_pandas_object(X, index=False)   # 每行内容算一个指纹


twin = fingerprint(Xb).isin(fingerprint(Xa))   # 测试集里这套房，训练集里有一模一样的"双胞胎"

# %%
EX["dup_twin_share"] = r(twin.mean())
EX["dup_mae_twin"], EX["dup_mae_other"] = r(err[twin].mean(), 1), r(err[~twin].mean(), 1)

# %% step:time_demo
# 对照实验都在训练期内部做：2023 年底前当"过去"，2024 上半年当"未来"，真正的测试集继续封存
past = X_train["成交日期"] < "2024-01-01"
hgb = priced(HistGradientBoostingRegressor(random_state=SEED))

# ❌ 随机切分：训练数据里混着"未来"的成交，等于提前偷看了房价走势
Xa, Xb, ya, yb = train_test_split(X_train, y_train, test_size=(~past).mean(), random_state=SEED)
pred_random = hgb.fit(Xa, ya).predict(Xb)
mape_random, bias_random = mean_absolute_percentage_error(yb, pred_random), (pred_random / yb - 1).mean()

# ✅ 按时间切分：只用过去预测未来，和上线后的真实处境一致
pred_time = hgb.fit(X_train[past], y_train[past]).predict(X_train[~past])
mape_time, bias_time = mean_absolute_percentage_error(y_train[~past], pred_time), (pred_time / y_train[~past] - 1).mean()

# %%
EX["mape_random"], EX["mape_time"] = r(mape_random), r(mape_time)
EX["bias_random"], EX["bias_time"] = r(bias_random), r(bias_time)

# %% step:te_demo
# ❌ 朴素目标编码：用"所有数据"（含要预测的那部分）算每个小区的平均成交价，当作特征
naive_pre = ColumnTransformer(pre.transformers[:3] + [("comm", "passthrough", ["小区均价"])])
naive = TransformedTargetRegressor(make_pipeline(naive_pre, Ridge()), func=np.log, inverse_func=np.exp)
comm_mean = np.log(y_train).groupby(X_train["小区"]).mean()


def add_comm_mean(X):
    return X.assign(小区均价=X["小区"].map(comm_mean).fillna(comm_mean.mean()))   # 没见过的小区用全市均值


with_mean = add_comm_mean(X_train)
naive.fit(with_mean[past], y_train[past])
mae_naive = mean_absolute_error(y_train[~past], naive.predict(with_mean[~past]))

# ✅ TargetEncoder 放进 Pipeline：只用训练部分编码，且训练时内部交叉拟合，样本看不到自己的答案
honest = priced(Ridge()).fit(X_train[past], y_train[past])
mae_honest = mean_absolute_error(y_train[~past], honest.predict(X_train[~past]))

# 上线检验：两种做法都用整个训练期重训，去预测 2025 年一季度的真实成交，看谁在"说谎"
new = clean(pd.read_csv(DATA / "house_2025q1.csv", encoding="utf-8-sig"))
X_new, y_new = new.drop(columns="成交价"), new["成交价"]
naive.fit(with_mean, y_train)
honest.fit(X_train, y_train)
mae_naive_new = mean_absolute_error(y_new, naive.predict(add_comm_mean(X_new)))
mae_honest_new = mean_absolute_error(y_new, honest.predict(X_new))

# %%
EX["te"] = {"naive": r(mae_naive, 1), "honest": r(mae_honest, 1),
            "naive_new": r(mae_naive_new, 1), "honest_new": r(mae_honest_new, 1)}

# %% step:final
pred = best.predict(X_test)                              # 测试集：只在这里打开一次
mae = mean_absolute_error(y_test, pred)
within10 = (np.abs(pred / y_test - 1) <= 0.10).mean()   # 业务验收：误差 ≤10% 的房源占比

# 和"现行做法"比：经纪人常用的 区域历史单价中位数 × 面积
unit = (y_train / X_train["面积"]).groupby(X_train["区域"]).median()
rule = X_test["区域"].map(unit) * X_test["面积"].fillna(X_train["面积"].median())

# %%
EX["test_mae"], EX["within10"], EX["test_r2"] = r(mae, 1), r(within10), r(r2_score(y_test, pred))
EX["y_test"], EX["pred"], EX["rule"] = r(y_test, 1), r(pred, 1), r(rule, 1)
EX["rule_mae"], EX["rule_within10"] = r(mean_absolute_error(y_test, rule), 1), r((np.abs(rule / y_test - 1) <= 0.1).mean())
EX["district_test"] = X_test["区域"].tolist()
EX["test_median_price"] = r(y_test.median(), 1)

# %% step:explain
imp = permutation_importance(best, X_test, y_test, scoring="neg_mean_absolute_error",
                             n_repeats=10, random_state=SEED)
# 打乱某一列后 MAE 涨得越多，说明模型越依赖它

# %%
cols = ["面积"] + num + cat + ["小区"]
EX["importance"] = sorted([[c, r(imp.importances_mean[list(X_test.columns).index(c)], 1)] for c in cols],
                          key=lambda t: -t[1])

# 估价器：把"Ridge·log 价格"拆开导出，网页 JS 用同一套公式实时估价
lin = priced(Ridge()).fit(X_train, y_train)
ct, ridge = lin.regressor_[0], lin.regressor_[-1]
a_imp, _, a_sc = ct.named_transformers_["area"]
n_imp, n_sc = ct.named_transformers_["num"]
ohe, te = ct.named_transformers_["cat"], ct.named_transformers_["comm"]
Z = ct.transform(X_train)
info = X_train.groupby("小区").agg(区域=("区域", "first"), 地铁=("距地铁米", "first"), 学区=("学区房", "first"))
EX["sim"] = {
    "area": [r(a_imp.statistics_[0], 3), r(a_sc.mean_[0], 5), r(a_sc.scale_[0], 5)],
    "num": num, "median": r(n_imp.statistics_, 3), "ind": n_imp.indicator_.features_.tolist(),
    "mean": r(n_sc.mean_, 5), "scale": r(n_sc.scale_, 5),
    "cat": cat, "cats": [list(c) for c in ohe.categories_],
    "comm": dict(zip(te.categories_[0], r(te.encodings_[0], 5))), "comm_default": r(te.target_mean_, 5),
    "coef": r(ridge.coef_, 6), "intercept": r(ridge.intercept_, 6), "zmean": r(Z.mean(axis=0), 6),
    "info": {k: [v["区域"], None if pd.isna(v["地铁"]) else v["地铁"], v["学区"]] for k, v in info.iterrows()},
    "month": int(X_train["成交月"].max()),
}
EX["lin_test_mae"] = r(mean_absolute_error(y_test, lin.predict(X_test)), 1)


def sim_predict(row, s=EX["sim"]):
    """与网页 JS 实现一致的手算版，用来核对导出参数没错。"""
    area = s["area"][0] if pd.isna(row["面积"]) else row["面积"]
    z = [(np.log(area) - s["area"][1]) / s["area"][2]]
    vals = [s["median"][i] if pd.isna(row[c]) else row[c] for i, c in enumerate(s["num"])]
    vals += [float(pd.isna(row[s["num"][i]])) for i in s["ind"]]
    z += [(v - m) / sc for v, m, sc in zip(vals, s["mean"], s["scale"])]
    z += [float(row[c] == v) for c, vs in zip(s["cat"], s["cats"]) for v in vs]
    z += [s["comm"].get(row["小区"], s["comm_default"])]
    return np.exp(s["intercept"] + np.dot(s["coef"], z))


# %% step:deploy
(ROOT / "models").mkdir(exist_ok=True)
# 上线后：2025 年一季度的新成交，走同一个 clean()、同一条 Pipeline
new_pred = best.predict(X_new)
new_within10 = (np.abs(new_pred / y_new - 1) <= 0.10).mean()

# 验收通过后，上线前用"全部"历史数据（含测试期）重训：离现在最近的半年行情最有价值
final = clone(best).fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
refit_pred = final.predict(X_new)
refit_within10 = (np.abs(refit_pred / y_new - 1) <= 0.10).mean()
joblib.dump(final, ROOT / "models" / "house_model.joblib")   # 整条流水线：清洗后的原始列进，价格出

# %%
EX["new_within10"], EX["refit_within10"], EX["n_new"] = r(new_within10), r(refit_within10), len(X_new)
EX["new_bias"], EX["refit_bias"] = r((new_pred / y_new - 1).mean()), r((refit_pred / y_new - 1).mean())
EX["code"] = snippets(__file__)
A = dict(EX["arena"])
print("regression:", {k: EX[k] for k in ["arena", "best_depth", "best_params", "best_cv_mae", "r2_leak", "r2_clean",
                                          "dup_twin_share", "dup_mae_twin", "dup_mae_other", "mape_random",
                                          "mape_time", "bias_random", "bias_time", "te", "test_mae", "within10", "test_r2", "lin_test_mae",
                                          "rule_mae", "rule_within10", "new_within10", "refit_within10", "new_bias", "refit_bias", "skew", "skew_log"]})
# 断言：网页里讲的每个结论都必须被真实结果支持，否则构建失败
assert A["Ridge·log 价格"] < A["Ridge·原始价格"], "log 目标应更好"
assert A["HistGB 梯度提升"] < A["Dummy·猜中位数"] * 0.5
assert EX["r2_leak"] > 0.98 and EX["r2_clean"] < 0.95, (EX["r2_leak"], EX["r2_clean"])
assert EX["dup_mae_twin"] < 0.7 * EX["dup_mae_other"], (EX["dup_mae_twin"], EX["dup_mae_other"])
assert EX["mape_time"] > EX["mape_random"] and EX["bias_time"] > abs(EX["bias_random"]) + 0.03,     (EX["mape_time"], EX["mape_random"], EX["bias_time"], EX["bias_random"])
assert EX["te"]["naive"] < EX["te"]["honest"] and EX["te"]["naive_new"] > EX["te"]["honest_new"], EX["te"]
assert 3 <= EX["best_depth"] <= 15
assert EX["test_mae"] < EX["best_cv_mae"] and EX["new_bias"] > 0.1  # 页面上"测试集比交叉验证低""直接上线会高估"的说法
assert EX["within10"] > 1.5 * EX["rule_within10"] and EX["refit_within10"] > EX["new_within10"] + 0.1
assert EX["skew"] > 1 and abs(EX["skew_log"]) < 0.5
assert np.allclose([sim_predict(row) for _, row in X_test.head(50).iterrows()], lin.predict(X_test.head(50)),
                   rtol=1e-3)
to_js("REG", EX, "regression.js")
