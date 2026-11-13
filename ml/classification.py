"""项目 B · 分类：星海通信会员流失预警。

由 build.py 调用，也可单独运行：python classification.py
`# %% step:xxx` 块会原样展示在网页上；裸 `# %%` 块是幕后工作（导出数据、断言教学结论）。
"""
# %%
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DATA, ROOT, SEED, hist, r, sample_rows, snippets, to_js

EX = {}

# %% step:load
raw = pd.read_csv(DATA / "churn.csv", encoding="utf-8-sig")
raw.shape                          # 多少行、多少列
raw.dtypes                         # 每列被读成了什么类型 —— "累计消费"居然是 str
raw.isna().sum()                   # 每列缺多少
(raw["是否流失"] == "是").mean()    # 流失率：类别是否不平衡？

# %%
PAY = {"auto_debit", "bank_card", "alipay", "cash"}
EX["n_raw"] = len(raw)
EX["dtypes"] = [[c, str(t), int(raw[c].isna().sum())] for c, t in raw.dtypes.items()]
EX["churn_rate"] = r((raw["是否流失"] == "是").mean())
churned = raw["是否流失"] == "是"
EX["by_contract"] = {k: r(v) for k, v in churned.groupby(raw["合约类型"]).mean().items()}
bins = pd.cut(raw["在网月数"], [-1, 5, 11, 23, 47, 72], labels=["0-5月", "6-11月", "1-2年", "2-4年", "4年+"])
EX["by_tenure"] = {str(k): r(v) for k, v in churned.groupby(bins, observed=True).mean().items()}


def first(mask):
    return raw.index[mask][0]


dup_id = raw.loc[raw.duplicated(keep=False), "客户ID"].iloc[0]
pair = list(raw.index[raw["客户ID"] == dup_id][:2])
normal = list(raw.index[raw["客户ID"].str.startswith("C00") & ~raw.duplicated(keep=False)
                        & raw["年龄"].between(18, 100) & raw["支付方式"].isin(PAY)
                        & raw["近30天使用时长"].notna() & (raw["累计消费"] != " ")][:3])
problems = {
    "blank": first(raw["累计消费"] == " "), "age": first(raw["年龄"] > 100),
    "pay": first(~raw["支付方式"].isin(PAY)), "usage": first(raw["近30天使用时长"].isna()),
}
order = [normal[0], problems["blank"], pair[0], normal[1], problems["age"], problems["pay"], normal[2], pair[1],
         problems["usage"]]
EX["sample"] = sample_rows(raw, order)
pos = order.index
EX["flags"] = [
    [pos(problems["blank"]), "累计消费", "空格字符串：在网 0 个月的新用户，系统写了个空格，整列因此变成文本"],
    [pos(problems["age"]), "年龄", "150 岁？录入错误，当缺失处理"],
    [pos(problems["pay"]), "支付方式", "同一种支付方式的另一种写法（大小写/空格）"],
    [pos(problems["usage"]), "近30天使用时长", "缺失值：埋点没采到"],
    [pos(pair[0]), "客户ID", "和下面某行一模一样：系统重复写入"],
    [pos(pair[1]), "客户ID", "重复记录：会让模型把同一个人学两遍"],
]
EX["flag_cols"] = [["挽留专员通话次数", "数据泄漏！客户提交注销申请后挽留专员才会打电话——预测时点根本拿不到"]]
EX["clean_log"] = [
    ["完全重复的记录", int(raw.duplicated().sum())],
    ["支付方式写法不统一", int((raw["支付方式"] != raw["支付方式"].str.strip().str.lower()).sum())],
    ["累计消费是空格字符串", int((raw["累计消费"].str.strip() == "").sum())],
    ["年龄不在 18–100 之间", int((~raw["年龄"].between(18, 100)).sum())],
    ["近30天使用时长缺失", int(raw["近30天使用时长"].isna().sum())],
]

# %% step:clean
LEAK = ["挽留专员通话次数"]   # 客户提交注销申请后才产生 —— 预测时点根本拿不到
DROP = ["客户ID", "性别"]     # ID 不是特征；性别与流失无关且涉及公平性，不进模型


def clean(df):
    """只做"规则型"清洗：不从数据里学任何参数，所以可以在切分前做，上线时对新数据原样复用。"""
    df = df.drop_duplicates()                                                  # ① 重复写入
    df["支付方式"] = df["支付方式"].str.strip().str.lower()                     # ② 同值多写法
    df["累计消费"] = pd.to_numeric(df["累计消费"], errors="coerce").fillna(0)    # ③ 新用户的空格 = 0 元
    df["年龄"] = df["年龄"].where(df["年龄"].between(18, 100))                   # ④ 不可能的年龄 → 缺失
    y = (df.pop("是否流失") == "是").astype(int)
    return df.drop(columns=LEAK + DROP), y


X, y = clean(raw)

# %% step:split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)   # stratify：训练/测试两边流失率一致
# 从这一行起，X_test 被"封存"——直到最后验收才打开一次

# %%
EX["n_clean"], EX["n_train"], EX["n_test"] = len(X), len(X_train), len(X_test)
EX["rate_train"], EX["rate_test"] = r(y_train.mean()), r(y_test.mean())
no_strat = [train_test_split(y, test_size=0.2, random_state=s)[1].mean() for s in range(30)]
EX["no_strat_range"] = [r(min(no_strat)), r(max(no_strat))]

# %% step:prep
num = ["年龄", "在网月数", "月费", "累计消费", "近30天投诉次数", "近30天使用时长"]
cat = ["合约类型", "宽带类型", "家庭套餐", "支付方式"]
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
])
# 中位数、均值、方差、类别表……都只从训练数据里学。
# 把 pre 和模型串进同一个 Pipeline，交叉验证的每一折也只会用该折的训练部分去 fit。

# %% step:noise
rng = np.random.RandomState(SEED)
X_noise = rng.standard_normal((200, 10_000))   # 200 个试点用户 × 1 万个埋点特征，全是随机数
y_noise = rng.randint(0, 2, 200)               # 标签也是抛硬币：任何模型的真实准确率都该是 50%

# ❌ 先在全部数据上挑出"最相关"的 25 个特征，再做交叉验证
X_sel = SelectKBest(k=25).fit_transform(X_noise, y_noise)
wrong = cross_val_score(LogisticRegression(), X_sel, y_noise, cv=5).mean()

# ✅ 把挑特征放进 Pipeline：每一折只用该折的训练部分去挑
right = cross_val_score(make_pipeline(SelectKBest(k=25), LogisticRegression()),
                        X_noise, y_noise, cv=5).mean()

# %%
EX["noise_wrong"], EX["noise_right"] = r(wrong), r(right)

# %% step:model
models = {
    "Dummy·全猜不流失": DummyClassifier(strategy="most_frequent"),
    "逻辑回归": make_pipeline(pre, LogisticRegression(max_iter=1000)),
    "随机森林": make_pipeline(pre, RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "HistGB 梯度提升": make_pipeline(pre, HistGradientBoostingClassifier(random_state=SEED)),
}
arena = {name: cross_validate(m, X_train, y_train, cv=5, scoring=["accuracy", "recall", "roc_auc"])
         for name, m in models.items()}

# %%
EX["arena"] = [[name, r(s["test_accuracy"].mean()), r(s["test_recall"].mean()), r(s["test_roc_auc"].mean())]
               for name, s in arena.items()]
EX["dummy_acc"] = EX["arena"][0][1]

# %% step:leak_demo
# 泄漏版：把"挽留专员通话次数"加回特征里
X_leak = X.join(raw[LEAK])
pre_leak = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num + LEAK),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
])
leaky = make_pipeline(pre_leak, HistGradientBoostingClassifier(random_state=SEED))
auc_leak = cross_val_score(leaky, X_leak.loc[X_train.index], y_train, cv=5, scoring="roc_auc").mean()

# 上线那一刻：还没人提交注销申请，所有人的通话次数都是 0
new_raw = pd.read_csv(DATA / "churn_next_quarter.csv", encoding="utf-8-sig")
X_new, y_new = clean(new_raw)
leaky.fit(X_leak.loc[X_train.index], y_train)
auc_leak_online = roc_auc_score(y_new, leaky.predict_proba(X_new.assign(挽留专员通话次数=0))[:, 1])

# %%
EX["auc_leak"], EX["auc_leak_online"] = r(auc_leak), r(auc_leak_online)
EX["auc_clean"] = EX["arena"][3][3]
EX["auc_clean_online"] = r(roc_auc_score(y_new, clone(models["HistGB 梯度提升"]).fit(X_train, y_train).predict_proba(X_new)[:, 1]))

# %% step:weight
weights = {cw: cross_validate(make_pipeline(pre, LogisticRegression(max_iter=1000, class_weight=cw)),
                              X_train, y_train, cv=5, scoring=["precision", "recall"])
           for cw in [None, "balanced"]}

# %%
EX["weight"] = {str(cw): [r(s["test_precision"].mean()), r(s["test_recall"].mean())] for cw, s in weights.items()}

# %% step:tune
search = GridSearchCV(
    make_pipeline(pre, HistGradientBoostingClassifier(random_state=SEED)),
    param_grid={"histgradientboostingclassifier__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "histgradientboostingclassifier__max_leaf_nodes": [4, 8, 16, 31]},
    scoring="roc_auc", cv=5, n_jobs=-1)
search.fit(X_train, y_train)          # 只用训练集做交叉验证调参；测试集依然没拆封

# %%
LR_GRID, LEAF_GRID = [0.02, 0.05, 0.1, 0.2], [4, 8, 16, 31]
cv = {(p["histgradientboostingclassifier__learning_rate"], p["histgradientboostingclassifier__max_leaf_nodes"]): s
      for p, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])}
EX["grid"] = {"lr": LR_GRID, "leaf": LEAF_GRID, "auc": [[r(cv[(a, b)]) for b in LEAF_GRID] for a in LR_GRID]}
EX["best_params"] = {k.split("__")[1]: v for k, v in search.best_params_.items()}
EX["best_cv_auc"] = r(search.best_score_)
EX["lr_cv_auc"] = EX["arena"][1][3]

# %% step:choose
# 调参后的 HistGB 和逻辑回归 AUC 几乎打平 → 选更简单、可解释、上线成本更低的逻辑回归
best = models["逻辑回归"].fit(X_train, y_train)

# %% step:test_leak
# ❌ 反面教材：团队试了 50 个版本（换特征、换参数），都拿同一份 200 人的"小测试集"打分，挑最高的汇报
rng = np.random.RandomState(SEED)
preds = []
for _ in range(50):
    feats = [str(f) for f in rng.choice(num + cat, size=rng.randint(5, 11), replace=False)]
    version = make_pipeline(
        ColumnTransformer([("num", SimpleImputer(strategy="median"), [f for f in feats if f in num]),
                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            [f for f in feats if f in cat])]),
        HistGradientBoostingClassifier(learning_rate=rng.choice([0.02, 0.05, 0.1, 0.2, 0.3]),
                                       max_leaf_nodes=int(rng.choice([4, 8, 16, 31, 63])), random_state=SEED))
    preds.append(version.fit(X_train, y_train).predict_proba(X_new)[:, 1])


def auc_on(p, idx):
    return roc_auc_score(y_new.iloc[idx], p[idx])


# 把"抽 200 人当小测试集 → 选出冠军 → 看冠军在其余新客户上的真实分"重复 200 次
gaps = []
for _ in range(200):
    idx = rng.permutation(len(y_new))
    small, fresh = idx[:200], idx[200:]
    champ = max(preds, key=lambda p: auc_on(p, small))
    gaps.append(auc_on(champ, small) - auc_on(champ, fresh))   # 汇报分 − 真实分

# %%
gaps = np.array(gaps)
EX["gap_mean"], EX["gap_pos"] = r(gaps.mean()), r((gaps > 0).mean())
EX["gap_hist"] = hist(gaps, 20)

# %% step:final
proba = best.predict_proba(X_test)[:, 1]   # 测试集：只在这里打开一次
test_auc = roc_auc_score(y_test, proba)

# %%
EX["test_auc"] = r(test_auc)
EX["y_test"] = y_test.tolist()
EX["proba"] = r(proba, 3)
GRID = np.linspace(0, 1, 101)
curves = {}
for name, p in [("逻辑回归（最终）", proba), ("HistGB（调参后）", search.predict_proba(X_test)[:, 1])]:
    fpr, tpr, _ = roc_curve(y_test, p)
    prec, rec, _ = precision_recall_curve(y_test, p)
    curves[name] = {"roc": r(np.interp(GRID, fpr, tpr), 3),
                    "pr": r(np.interp(GRID, rec[::-1], prec[::-1]), 3),
                    "auc": r(roc_auc_score(y_test, p))}
EX["curves"] = curves
EX["grid01"] = r(GRID, 2)

# %% step:explain
imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED)
# 打乱某一列后 AUC 掉得越多，说明模型越依赖它

# %%
EX["importance"] = sorted([[c, r(m), r(s)] for c, m, s in zip(X_test.columns, imp.importances_mean,
                                                              imp.importances_std)], key=lambda t: -t[1])

# 客户模拟器：把逻辑回归"拆开"导出，网页 JS 用同一个公式实时算概率
imputer, scaler = best[0].named_transformers_["num"]
ohe, lr = best[0].named_transformers_["cat"], best[-1]
coef = lr.coef_[0]
Z = best[0].transform(X_train)
EX["sim"] = {
    "num": num, "median": r(imputer.statistics_), "mean": r(scaler.mean_), "scale": r(scaler.scale_),
    "cat": cat, "cats": [list(c) for c in ohe.categories_],
    "coef": r(coef, 5), "intercept": r(lr.intercept_[0], 5), "zmean": r(Z.mean(axis=0), 5),
    "ranges": {c: [r(X_train[c].min()), r(X_train[c].max()), r(X_train[c].median())] for c in num},
}


def sim_predict(row, s=EX["sim"]):
    """与网页 app 中 JS 实现一致的手算版，用来核对导出参数没错。"""
    z = [((s["median"][i] if pd.isna(row[c]) else row[c]) - s["mean"][i]) / s["scale"][i] for i, c in enumerate(s["num"])]
    z += [float(row[c] == v) for c, vs in zip(s["cat"], s["cats"]) for v in vs]
    return 1 / (1 + np.exp(-(s["intercept"] + np.dot(s["coef"], z))))


# %% step:deploy
(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(best, ROOT / "models" / "churn_model.joblib")   # 存的是整条 Pipeline：原始列进，概率出

# 上线后的新数据：走同一个 clean()、同一个 Pipeline —— 杜绝"训练一套口径、线上另一套"
new_auc = roc_auc_score(y_new, best.predict_proba(X_new)[:, 1])

# %%
EX["new_auc"] = r(new_auc)
EX["n_new"] = len(X_new)
EX["code"] = snippets(__file__)

print("classification:", {k: EX[k] for k in ["dummy_acc", "arena", "auc_clean", "auc_leak", "auc_leak_online",
                                         "noise_wrong", "noise_right", "weight", "gap_mean", "gap_pos",
                                         "test_auc", "new_auc", "best_params", "best_cv_auc"]})
# 断言：网页里讲的每个结论都必须被真实结果支持，否则构建失败
assert 0.78 < EX["dummy_acc"] < 0.85, EX["dummy_acc"]
assert EX["arena"][0][2] == 0 and EX["arena"][0][3] == 0.5
assert EX["auc_leak"] > 0.97 and 0.78 < EX["auc_clean"] < 0.92, (EX["auc_leak"], EX["auc_clean"])
assert EX["auc_leak_online"] < EX["new_auc"] - 0.05, (EX["auc_leak_online"], EX["new_auc"])
assert EX["noise_wrong"] > 0.65 and EX["noise_right"] < 0.6, (EX["noise_wrong"], EX["noise_right"])
assert EX["weight"]["balanced"][1] > EX["weight"]["None"][1] + 0.2
assert EX["gap_mean"] > 0.01 and EX["gap_pos"] > 0.65, (EX["gap_mean"], EX["gap_pos"])
assert abs(EX["best_cv_auc"] - EX["lr_cv_auc"]) < 0.01  # "打平"的说法成立
assert abs(EX["test_auc"] - EX["new_auc"]) < 0.03
assert np.allclose([sim_predict(row) for _, row in X_test.head(50).iterrows()],
                   best.predict_proba(X_test.head(50))[:, 1], atol=1e-3)
to_js("CLF", EX, "classification.js")
