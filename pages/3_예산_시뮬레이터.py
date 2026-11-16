from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="예산 시뮬레이터", page_icon="🧮", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "data" / "marketing_telemetry.csv"
EMAIL = "Email / CRM"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["day"])


@st.cache_data
def fit_models(df: pd.DataFrame) -> pd.DataFrame:
    """채널마다 log(ROAS) = a + b × log(하루 비용) 직선(선형 회귀)을 맞춘다."""
    d = df[(df["channel"] != EMAIL) & (df["revenue_usd"] > 0) & (df["spend_usd"] >= 1)].copy()
    d["log_spend"] = np.log(d["spend_usd"])
    d["log_roas"] = np.log(d["revenue_usd"] / d["spend_usd"])

    rows = []
    for ch, g in d.groupby("channel"):
        b, a = np.polyfit(g["log_spend"], g["log_roas"], 1)
        r2 = np.corrcoef(g["log_spend"], g["log_roas"])[0, 1] ** 2
        rows.append({"channel": ch, "a": a, "b": b, "r2": r2, "n": len(g)})
    models = pd.DataFrame(rows).set_index("channel")

    all_rows = df[df["channel"] != EMAIL]
    totals = all_rows.groupby("channel").agg(budget=("spend_usd", "sum"), revenue=("revenue_usd", "sum"))
    models = models.join(totals)
    models["roas"] = models["revenue"] / models["budget"]
    models["marginal_roas"] = models["roas"] * (1 + models["b"])  # 지금 1달러를 더 쓸 때 늘어나는 매출
    return models


def revenue_after(models: pd.DataFrame, budgets: pd.Series) -> pd.Series:
    """예산이 바뀌면 매출은 현재 매출 × (새 예산 ÷ 현재 예산)^(1+b) 로 바뀐다."""
    ratio = budgets / models["budget"]
    return models["revenue"] * ratio.clip(lower=0) ** (1 + models["b"])


def recommend(models: pd.DataFrame, low: float, high: float, steps: int = 2000) -> pd.Series:
    """각 채널을 현재의 low~high배 안에서, 1달러당 늘어나는 매출이 큰 곳부터 채운다."""
    total = models["budget"].sum()
    lo = models["budget"] * low
    hi = models["budget"] * high
    budgets = lo.copy()
    step = (total - lo.sum()) / steps
    if step <= 0:
        return models["budget"].copy()
    for _ in range(steps):
        marginal = models["revenue"] * (1 + models["b"]) * budgets.clip(lower=1e-9) ** models["b"] / (
            models["budget"] ** (1 + models["b"])
        )
        marginal = marginal.where(budgets + step <= hi, other=-np.inf)
        best = marginal.idxmax()
        if not np.isfinite(marginal[best]):
            break
        budgets[best] += step
    return budgets


if not DATA_PATH.exists():
    st.error(
        "데이터 파일을 찾을 수 없습니다. 깃허브 저장소에 `data/marketing_telemetry.csv` 파일이 "
        "올라가 있는지, 폴더 이름(`data`)과 파일 이름의 철자·대소문자가 같은지 확인하세요."
    )
    st.stop()

df = load_data()
models = fit_models(df)
channels = list(models.index)

st.title("🧮 예산 시뮬레이터")
st.write(
    "예산을 한 채널에서 다른 채널로 옮기면 매출이 얼마나 달라질까요? "
    "돈을 더 쓸수록 효율이 떨어지는 효과를 **선형 회귀**로 계산해서 반영했습니다."
)

# =====================================================================
# 1. 선형 회귀로 효율 떨어지는 정도 구하기
# =====================================================================
st.header("1. 돈을 더 쓰면 효율이 얼마나 떨어질까?")

st.markdown(
    """
    채널마다 캠페인-일 데이터로 아래 직선을 맞췄습니다. (비용과 ROAS에 로그를 씌우면 점들이 직선에 가까워집니다.)

    **log(ROAS) = a + b × log(하루 비용)**

    - **b가 −0.12**라면, 하루 비용이 2배가 될 때 ROAS는 약 8% 줄어듭니다.
    - 그때 매출은 비용만큼 2배가 되지 않고 약 1.9배가 됩니다. (매출 = 비용 × ROAS)
    - 그래서 예산을 옮기면 **새 매출 = 현재 매출 × (새 예산 ÷ 현재 예산)^(1 + b)** 로 계산합니다.
    """
)

table = models.reset_index()[["channel", "b", "r2", "roas", "marginal_roas", "budget", "n"]]
table.columns = ["채널", "기울기 b", "설명력 R²", "현재 ROAS", "추가 1달러당 매출", "광고비", "데이터 수"]
st.dataframe(
    table.sort_values("현재 ROAS", ascending=False),
    hide_index=True,
    use_container_width=True,
    column_config={
        "기울기 b": st.column_config.NumberColumn(format="%.3f"),
        "설명력 R²": st.column_config.NumberColumn(format="%.3f"),
        "현재 ROAS": st.column_config.NumberColumn(format="%.2f"),
        "추가 1달러당 매출": st.column_config.NumberColumn(format="$%.2f"),
        "광고비": st.column_config.NumberColumn(format="$%d"),
        "데이터 수": st.column_config.NumberColumn(format="%d"),
    },
)
st.caption(
    "이메일/CRM은 제외했습니다. 설명력 R²가 낮은 것은 캠페인마다 전환율과 주문 금액이 제각각이라 "
    "점들이 넓게 퍼져 있기 때문입니다. 기울기가 모든 채널에서 −0.11 ~ −0.14로 비슷하게 나온다는 점이 중요합니다."
)

with st.expander("회귀 직선 그래프 보기"):
    pick = st.selectbox("채널 선택", channels, index=channels.index("Meta Ads") if "Meta Ads" in channels else 0)
    d = df[(df["channel"] == pick) & (df["revenue_usd"] > 0) & (df["spend_usd"] >= 1)].copy()
    d["roas_row"] = d["revenue_usd"] / d["spend_usd"]
    sample = d.sample(min(len(d), 2000), random_state=0)
    m = models.loc[pick]
    xs = np.linspace(np.log(d["spend_usd"].min()), np.log(d["spend_usd"].max()), 100)
    line_x = np.exp(xs)
    line_y = np.exp(m["a"] + m["b"] * xs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample["spend_usd"],
            y=sample["roas_row"],
            mode="markers",
            name="캠페인-일 (2,000개만 표시)",
            marker=dict(size=4, opacity=0.35),
        )
    )
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="회귀 직선", line=dict(width=3)))
    fig.update_xaxes(type="log", title="하루 비용 (달러, 로그 눈금)")
    fig.update_yaxes(type="log", title="ROAS (로그 눈금)")
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=40, b=10), title=f"{pick}: 비용이 커질수록 ROAS가 내려가는 경향")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================================
# 2. 예산 옮기기
# =====================================================================
st.header("2. 예산을 옮겨 보기")

default_from = channels.index("Meta Ads") if "Meta Ads" in channels else 0
default_to = channels.index("Affiliate") if "Affiliate" in channels else 1

c1, c2, c3 = st.columns(3)
src = c1.selectbox("예산을 줄일 채널", channels, index=default_from)
dst = c2.selectbox("예산을 늘릴 채널", channels, index=default_to)
pct = c3.slider("줄이는 비율 (%)", 0, 100, 20, step=5)

if src == dst:
    st.warning("줄일 채널과 늘릴 채널을 서로 다르게 골라 주세요.")
    st.stop()

now_budget = models["budget"].copy()
new_budget = now_budget.copy()
moved = now_budget[src] * pct / 100
new_budget[src] -= moved
new_budget[dst] += moved

now_rev = models["revenue"]
new_rev = revenue_after(models, new_budget)

total_now, total_new = now_rev.sum(), new_rev.sum()
change = (total_new / total_now - 1) * 100

m1, m2, m3 = st.columns(3)
m1.metric("옮기는 돈", f"${moved / 1e6:,.2f}M")
m2.metric("예상 매출 (현재 → 조정 후)", f"${total_new / 1e6:,.1f}M", delta=f"{(total_new - total_now) / 1e6:+,.1f}M")
m3.metric("매출 증감률", f"{change:+.1f}%")

verb = "늘어난다" if change >= 0 else "줄어든다"
st.success(
    f"**{src} 예산의 {pct}%를 {dst}로 옮기면, 같은 비용으로 매출이 약 {abs(change):.1f}% {verb}.**"
)

compare = pd.DataFrame(
    {
        "채널": channels * 2,
        "구분": ["현재"] * len(channels) + ["조정 후"] * len(channels),
        "광고비(백만 달러)": list(now_budget / 1e6) + list(new_budget / 1e6),
        "예상 매출(백만 달러)": list(now_rev / 1e6) + list(new_rev / 1e6),
    }
)
left, right = st.columns(2)
with left:
    fig = px.bar(compare, x="광고비(백만 달러)", y="채널", color="구분", barmode="group", orientation="h", title="채널별 광고비")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(compare, x="예상 매출(백만 달러)", y="채널", color="구분", barmode="group", orientation="h", title="채널별 예상 매출")
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================================
# 3. 추천 배분
# =====================================================================
st.header("3. 컴퓨터가 추천하는 배분")
st.write("총 광고비는 그대로 두고, 추가 1달러당 매출이 큰 채널부터 예산을 채웁니다. 한 채널을 너무 많이 바꾸지 않도록 한도를 정할 수 있습니다.")

r1, r2 = st.columns(2)
low = r1.slider("채널당 최소 예산 (현재의 몇 배)", 0.3, 1.0, 0.5, step=0.1)
high = r2.slider("채널당 최대 예산 (현재의 몇 배)", 1.0, 3.0, 1.5, step=0.1)

rec_budget = recommend(models, low, high)
rec_rev = revenue_after(models, rec_budget)
rec_change = (rec_rev.sum() / total_now - 1) * 100

st.metric("추천대로 배분했을 때 예상 매출 증감률", f"{rec_change:+.1f}%", delta=f"{(rec_rev.sum() - total_now) / 1e6:+,.1f}M 달러")

rec_table = pd.DataFrame(
    {
        "채널": channels,
        "현재 광고비": now_budget.values,
        "추천 광고비": rec_budget.values,
        "예산 변화(%)": ((rec_budget / now_budget - 1) * 100).values,
        "예상 매출 변화(%)": ((rec_rev / now_rev - 1) * 100).values,
    }
).sort_values("예산 변화(%)", ascending=False)
st.dataframe(
    rec_table,
    hide_index=True,
    use_container_width=True,
    column_config={
        "현재 광고비": st.column_config.NumberColumn(format="$%d"),
        "추천 광고비": st.column_config.NumberColumn(format="$%d"),
        "예산 변화(%)": st.column_config.NumberColumn(format="%+.1f"),
        "예상 매출 변화(%)": st.column_config.NumberColumn(format="%+.1f"),
    },
)

st.caption(
    "이 계산은 단순한 추정입니다. 비용이 지금의 몇 배로 늘어난 적이 없는 구간은 데이터가 말해 주지 못하고, "
    "채널마다 실제로 받을 수 있는 광고 물량에는 한계가 있습니다. 결과는 방향을 보는 참고용으로 읽어 주세요."
)
