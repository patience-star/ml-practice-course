"""生成三份"像真的一样脏"的仿真业务数据。

规律按真实业务设定（合约越短越易流失、地铁越近越贵……），再故意埋入真实数据里常见的问题：
缺失、录入错误、单位错误、写法不一、重复、类别不平衡、以及最阴险的——数据泄漏字段。
每份数据另生成一份"上线后的新数据"，用来检验模型上线后的真实表现。
"""
import numpy as np
import pandas as pd

from common import DATA, SEED

CSV = dict(index=False, encoding="utf-8-sig")  # utf-8-sig：Excel 直接打开中文不乱码


def _dirty_variants(rng, s, frac, variants):
    """随机把 frac 比例的取值替换成"同义不同写法"。"""
    s = s.copy()
    hit = rng.random(len(s)) < frac
    s[hit] = [variants(v, rng) for v in s[hit]]
    return s


# ---------------------------------------------------------------- 回归：二手房
DISTRICTS = {"高新区": 4.2, "东湖区": 3.6, "老城区": 3.2, "南城区": 2.5, "北岸区": 2.1, "西山区": 1.6}  # 基准单价 万/㎡
PREFIX = list("锦翠阳书星枫金滨云紫融华天碧御龙泰嘉和明")
SUFFIX = ["花园", "家园", "苑", "公寓", "府", "新村", "名邸", "雅居", "湾", "城", "庭", "郡", "里", "台", "园"]


def _communities(rng, k=800):  # 真实城市小区成千上万，每个小区成交没几套 → 高基数
    names = rng.permutation([p + q + t for p in PREFIX for q in SUFFIX for t in ["", "一期", "二期", "三期"]])[:k]
    district = rng.choice(list(DISTRICTS), k, p=[.15, .2, .15, .2, .15, .15])
    year = np.where(district == "老城区", rng.integers(1990, 2008, k), rng.integers(1998, 2022, k))
    return pd.DataFrame({
        "小区": names, "区域": district, "效应": rng.lognormal(0, 0.12, k), "年份": year,
        "地铁": np.where(rng.random(k) < 0.2, np.nan, rng.uniform(200, 3000, k).round(-1)),
        "学区": rng.random(k) < 0.25,
    })


def _price_index(month):
    """房价指数：2023 年中见顶后一路下行（制造时间趋势，演示"时间穿越"泄漏）。"""
    month = np.asarray(month, dtype=float)
    return np.where(month <= 18, 1 + 0.08 * month / 18, 1.08 - 0.016 * (month - 18))


def _houses(rng, comm, n, start, end):
    c = comm.iloc[rng.integers(0, len(comm), n)].reset_index(drop=True)
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    date = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days + 1, n), unit="D")
    month = (date.year - 2022) * 12 + date.month - 1
    area = np.clip(rng.lognormal(np.log(88), 0.33, n), 32, 280).round(1)
    rooms = np.clip(np.round(area / 32 + rng.normal(0, 0.5, n)), 1, 5).astype(int)
    halls = np.where((area > 70) ^ (rng.random(n) < 0.15), 2, 1)
    level = rng.choice(["低", "中", "高"], n)
    deco = rng.choice(["毛坯", "简装", "精装"], n, p=[.2, .45, .35])
    year = np.clip(c["年份"] + rng.integers(-2, 3, n), 1990, 2023)
    subway = c["地铁"].to_numpy()
    unit = (c["区域"].map(DISTRICTS) * c["效应"] * _price_index(month)
            * pd.Series(deco).map({"毛坯": 0.95, "简装": 1.0, "精装": 1.08})
            * pd.Series(level).map({"低": 0.97, "中": 1.02, "高": 1.0})
            * np.exp(-0.008 * (2025 - year))
            * np.where(np.isnan(subway), 0.78, np.exp(-0.00008 * np.nan_to_num(subway)))
            * np.where(c["学区"], 1.15, 1.0)
            * np.exp(rng.normal(0, 0.05, n)))
    price = (area * unit).round(1)
    return pd.DataFrame({
        "成交日期": date.strftime("%Y-%m-%d"), "区域": c["区域"], "小区": c["小区"], "面积": area,
        "室": rooms, "厅": halls, "楼层": level, "总楼层": rng.choice([6, 7, 11, 18, 26, 33], n),
        "建成年份": year, "装修": deco, "距地铁米": subway, "学区房": np.where(c["学区"], "是", "否"),
        # ↓ 两个泄漏字段：成交之后才知道，由成交价反推
        "单价": (price * 10000 / area).round(), "契税": (price * np.where(area <= 90, 0.01, 0.015)).round(2),
        "成交价": price,
    })


def make_house(rng):
    comm = _communities(rng)
    df = _houses(rng, comm, 4000, "2022-01-01", "2024-12-31")
    n = len(df)
    pick = lambda frac: rng.random(n) < frac
    df["区域"] = _dirty_variants(rng, df["区域"], 0.06,
                               lambda v, g: g.choice([v[:-1], " " + v, v + " "]))
    df.loc[pick(0.03), "面积"] = np.nan
    feet = pick(0.01)
    df.loc[feet, "面积"] = (df.loc[feet, "面积"] * 10.764).round(1)  # 中介按平方英尺录入
    df.loc[pick(0.003), "面积"] = 9999
    df.loc[pick(0.01), "建成年份"] = 0
    fut = pick(0.005)
    df.loc[fut, "建成年份"] = rng.integers(2030, 2036, fut.sum())
    dup = df.sample(frac=0.12, random_state=SEED)  # 同一套房被两个中介重复挂牌
    df = pd.concat([df, dup]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "房源编号", [f"H{i:05d}" for i in rng.permutation(len(df))])
    df.to_csv(DATA / "house.csv", **CSV)

    new = _houses(rng, comm, 400, "2025-01-01", "2025-03-31")
    new.insert(0, "房源编号", [f"N{i:05d}" for i in range(len(new))])
    new.to_csv(DATA / "house_2025q1.csv", **CSV)


# ---------------------------------------------------------------- 分类：会员流失
def _customers(rng, n, b0=None):
    contract = rng.choice(["月付", "一年", "两年"], n, p=[.55, .25, .20])
    tenure = np.select([contract == "月付", contract == "一年"],
                       [np.clip(rng.exponential(18, n), 0, 72), rng.uniform(6, 72, n)],
                       rng.uniform(12, 72, n)).astype(int)
    broadband = rng.choice(["光纤", "普通", "无"], n, p=[.45, .35, .20])
    monthly = (30 + pd.Series(broadband).map({"光纤": 60, "普通": 35, "无": 0}) + rng.uniform(0, 60, n)).round(1)
    family = rng.random(n) < 0.3
    pay = rng.choice(["auto_debit", "bank_card", "alipay", "cash"], n, p=[.3, .2, .35, .15])
    age = np.clip(rng.normal(40, 12, n), 18, 80).astype(int)
    complaints = rng.poisson(0.35 + 0.5 * (broadband == "光纤"))
    usage = np.clip(rng.normal(45, 15, n), 0, None).round(1)
    logit = (1.4 * (contract == "月付") - 1.1 * (contract == "两年") - 0.03 * tenure
             + 1.0 * (tenure < 6)  # 新用户"磨合期"最容易走：非线性，线性模型抓不全
             + 0.015 * (monthly - 80) + 0.4 * complaints * (1 + (broadband == "光纤"))
             - 0.04 * (usage - 45)
             + 0.35 * (broadband == "光纤") - 0.5 * family - 0.4 * (pay == "auto_debit")
             + 0.4 * (pay == "cash") + 0.015 * (40 - age))
    if b0 is None:  # 二分法找截距，使流失率 ≈ 18%
        lo, hi = -10.0, 10.0
        for _ in range(60):
            b0 = (lo + hi) / 2
            lo, hi = (b0, hi) if np.mean(1 / (1 + np.exp(-(logit + b0)))) < 0.18 else (lo, b0)
    churn = rng.random(n) < 1 / (1 + np.exp(-(logit + b0)))
    total = monthly * tenure * rng.uniform(0.95, 1.05, n)
    # ↓ 泄漏字段：客户提交注销申请后，挽留专员才会打电话
    calls = np.where(churn, np.where(rng.random(n) < 0.92, rng.integers(1, 5, n), 0),
                     (rng.random(n) < 0.04).astype(int))
    df = pd.DataFrame({
        "性别": rng.choice(["男", "女"], n), "年龄": age, "在网月数": tenure, "合约类型": contract,
        "宽带类型": broadband, "月费": monthly,
        "累计消费": [" " if t == 0 else f"{v:.1f}" for t, v in zip(tenure, total)],  # 新用户是空格字符串
        "家庭套餐": np.where(family, "是", "否"), "支付方式": pay,
        "近30天投诉次数": complaints, "近30天使用时长": usage,
        "挽留专员通话次数": calls, "是否流失": np.where(churn, "是", "否"),
    })
    return df, b0


def make_churn(rng):
    df, b0 = _customers(rng, 5000)
    n = len(df)
    pick = lambda frac: rng.random(n) < frac
    df.loc[pick(0.006), "年龄"] = 0
    df.loc[pick(0.004), "年龄"] = 150
    df["支付方式"] = _dirty_variants(rng, df["支付方式"], 0.05,
                                 lambda v, g: g.choice([v.capitalize(), v.upper(), " " + v]))
    df.loc[pick(0.04), "近30天使用时长"] = np.nan
    df.insert(0, "客户ID", [f"C{i:05d}" for i in range(n)])
    df = pd.concat([df, df.sample(frac=0.02, random_state=SEED)])  # 系统导数时重复写入
    df.sample(frac=1, random_state=SEED).to_csv(DATA / "churn.csv", **CSV)

    new, _ = _customers(rng, 2000, b0)
    new.insert(0, "客户ID", [f"N{i:05d}" for i in range(len(new))])
    new.to_csv(DATA / "churn_next_quarter.csv", **CSV)


# ---------------------------------------------------------------- 聚类：订单流水
SNAPSHOT = pd.Timestamp("2024-12-31")
# 人群：占比, 首购距快照天数范围, 最后活跃距快照天数范围, 平均下单间隔天, 客单价中位, 用券概率, 次季度复购概率
SEGMENTS = {
    "高价值": (.10, (300, 730), (0, 20), 12, 480, .15, .85),
    "普通活跃": (.35, (200, 730), (0, 60), 35, 160, .25, .55),
    "新客": (.20, (5, 75), (0, 5), 30, 130, .45, .35),
    "沉睡": (.25, (400, 730), (180, 500), 45, 140, .20, .08),
    "促销型": (.10, (200, 730), (0, 40), 25, 70, .85, .45),
}
CATS = ["服饰", "美妆", "食品", "家居", "数码", "母婴"]


def make_orders(rng, n_cust=2000):
    rows = []

    def order(cid, day, basket, coupon, qty=None):
        qty = qty if qty is not None else int(rng.integers(1, 4))
        rows.append([cid, day, rng.choice(CATS), qty, round(basket / qty, 2), int(rng.random() < coupon),
                     rng.choice(["App", "小程序", "网页"], p=[.6, .3, .1])])

    names = list(SEGMENTS)
    seg = rng.choice(names, n_cust, p=[SEGMENTS[s][0] for s in names])
    for i, s in enumerate(seg):
        _, first, last, gap, basket, coupon, q1 = SEGMENTS[s]
        cid = f"U{i:05d}"
        t, end = -rng.uniform(*first), -rng.uniform(*last)
        order(cid, t, rng.lognormal(np.log(basket), 0.4), coupon)
        while (t := t + rng.exponential(gap)) < min(end, 0):
            order(cid, t, rng.lognormal(np.log(basket), 0.4), coupon)
        if rng.random() < q1:  # 快照日之后（2025Q1）的订单
            for _ in range(1 + rng.poisson(1)):
                order(cid, rng.uniform(1, 90), rng.lognormal(np.log(basket), 0.4), coupon)
    for i in range(6):  # 批发商：极端大客户
        for _ in range(150):
            order(f"U9{i:04d}", -rng.uniform(0, 700), 0, 0.05, qty=int(rng.integers(50, 300)))
            rows[-1][4] = round(rng.uniform(40, 200), 2)
    for i in range(8):  # 测试账号：内部压测下的单
        for _ in range(300):
            order(f"TEST{i:03d}", -rng.uniform(0, 700), 0.5, 0)
    for _ in range(1200):  # 游客下单，没有客户 ID
        order(None, -rng.uniform(0, 730), rng.lognormal(np.log(120), 0.5), 0.3)

    df = pd.DataFrame(rows, columns=["客户ID", "天", "类目", "数量", "单价", "用券", "渠道"])
    df["下单时间"] = (SNAPSHOT + pd.to_timedelta(df.pop("天"), unit="D")).dt.strftime("%Y-%m-%d %H:%M")
    df.insert(0, "订单号", [f"O{i:06d}" for i in rng.permutation(len(df))])
    gift = rng.random(len(df)) < 0.01
    df.loc[gift, "单价"] = 0.0  # 赠品
    ret = df[(rng.random(len(df)) < 0.03) & ~gift].copy()
    ret["数量"] = -ret["数量"]  # 退货：同一订单号的负数量行
    df = pd.concat([df, ret, df.sample(frac=0.02, random_state=SEED)])  # 再加 2% 重复写入
    df = df.sample(frac=1, random_state=SEED)
    df[["订单号", "客户ID", "下单时间", "类目", "数量", "单价", "用券", "渠道"]].to_csv(DATA / "orders.csv", **CSV)


def main():
    DATA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    make_house(rng)
    make_churn(rng)
    make_orders(rng)


if __name__ == "__main__":
    main()
