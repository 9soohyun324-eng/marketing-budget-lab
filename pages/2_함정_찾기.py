from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="함정 찾기", page_icon="⚠️", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "data" / "marketing_telemetry.csv"
EMAIL = "Email / CRM"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["day"])


def roas_of(d: pd.DataFrame) -> float:
    return d["revenue_usd"].sum() / d["spend_usd"].sum()


def cac_of(d: pd.DataFrame) -> float:
    return d["spend_usd"].sum() / d["orders"].sum()


if not DATA_PATH.exists():
    st.error(
        "데이터 파일을 찾을 수 없습니다. 깃허브 저장소에 `data/marketing_telemetry.csv` 파일이 "
        "올라가 있는지, 폴더 이름(`data`)과 파일 이름의 철자·대소문자가 같은지 확인하세요."
    )
    st.stop()

df = load_data()
no_email = df[df["channel"] != EMAIL]
email = df[df["channel"] == EMAIL]

st.title("⚠️ 함정 찾기")
st.write("평균 숫자만 보면 잘못된 결론을 내리기 쉽습니다. 이 데이터에 숨어 있는 함정 세 가지를 확인해 보세요.")

# 2·3번 함정에 쓸 데이터 범위
st.sidebar.header("설정")
use_email = st.sidebar.checkbox(
    "함정 2·3에 이메일/CRM 포함하기",
    value=False,
    help="함정 1은 항상 이메일 포함·제외를 모두 비교합니다.",
)
base = df if use_email else no_email

# =====================================================================
# 함정 1. 이메일
# =====================================================================
st.header("함정 1. 이메일/CRM이 평균을 부풀린다")

c1, c2, c3 = st.columns(3)
c1.metric("전체 ROAS (이메일 포함)", f"{roas_of(df):.2f}")
c2.metric("전체 ROAS (이메일 제외)", f"{roas_of(no_email):.2f}", delta=f"{roas_of(no_email) - roas_of(df):.2f}")
c3.metric("이메일 ROAS", f"{roas_of(email):,.0f}")

spend_share = email["spend_usd"].sum() / df["spend_usd"].sum() * 100
rev_share = email["revenue_usd"].sum() / df["revenue_usd"].sum() * 100

share = pd.DataFrame(
    {
        "구분": ["광고비 비중", "매출 비중"],
        "이메일/CRM": [spend_share, rev_share],
        "나머지 9개 채널": [100 - spend_share, 100 - rev_share],
    }
).melt(id_vars="구분", var_name="채널 묶음", value_name="비중(%)")

fig = px.bar(
    share,
    x="비중(%)",
    y="구분",
    color="채널 묶음",
    orientation="h",
    text=share["비중(%)"].map(lambda v: f"{v:.1f}%"),
    title="이메일/CRM의 광고비 비중 vs 매출 비중",
)
fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10), yaxis_title=None, barmode="stack")
st.plotly_chart(fig, use_container_width=True)

st.info(
    f"이메일/CRM은 광고비가 전체의 {spend_share:.1f}%뿐인데 매출은 {rev_share:.1f}%를 차지합니다. "
    "이메일은 이미 우리 고객에게 보내는 거라 돈이 거의 안 들기 때문입니다. "
    "그래서 광고 예산을 나눌 때는 이메일을 빼고 비교해야 합니다."
)

st.divider()

# =====================================================================
# 함정 2. 소재 클릭률
# =====================================================================
st.header("함정 2. 클릭률이 높다고 매출이 높은 건 아니다")

g = base.groupby("creative_format").agg(
    클릭=("clicks", "sum"),
    노출=("impressions", "sum"),
    주문=("orders", "sum"),
    비용=("spend_usd", "sum"),
    매출=("revenue_usd", "sum"),
)
g["CTR(%)"] = g["클릭"] / g["노출"] * 100
g["전환율(%)"] = g["주문"] / g["클릭"] * 100
g["ROAS"] = g["매출"] / g["비용"]
g = g.sort_values("CTR(%)", ascending=False)

metrics = ["CTR(%)", "전환율(%)", "ROAS"]
index_df = (g[metrics] / g[metrics].mean() * 100).reset_index().melt(
    id_vars="creative_format", var_name="지표", value_name="평균을 100으로 한 값"
)

left, right = st.columns(2)
with left:
    t = g.reset_index()
    fig = px.bar(
        t,
        x="creative_format",
        y="CTR(%)",
        text=t["CTR(%)"].map(lambda v: f"{v:.2f}%"),
        title="소재별 클릭률 (CTR)",
        labels={"creative_format": "소재"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(
        t,
        x="creative_format",
        y="ROAS",
        text=t["ROAS"].map(lambda v: f"{v:.2f}"),
        title="소재별 ROAS",
        labels={"creative_format": "소재"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), yaxis_range=[0, t["ROAS"].max() * 1.25])
    st.plotly_chart(fig, use_container_width=True)

fig = px.bar(
    index_df,
    x="creative_format",
    y="평균을 100으로 한 값",
    color="지표",
    barmode="group",
    title="평균을 100으로 맞춰서 비교하기 (클릭률만 들쭉날쭉한지 확인)",
    labels={"creative_format": "소재"},
)
fig.add_hline(y=100, line_dash="dot")
fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

ctr_spread = g["CTR(%)"].max() / g["CTR(%)"].min()
roas_spread = g["ROAS"].max() / g["ROAS"].min()
st.info(
    f"가장 높은 클릭률은 가장 낮은 클릭률의 약 {ctr_spread:.1f}배인데, "
    f"ROAS는 가장 높은 것과 낮은 것이 약 {roas_spread:.2f}배밖에 차이 나지 않습니다. "
    "클릭은 많이 받아도 그 클릭이 주문으로 이어지는 비율(전환율)은 소재와 상관없이 비슷하기 때문입니다. "
    "소재를 고를 때 클릭률만 보면 안 되는 이유입니다."
)

with st.expander("소재별 숫자 표 보기"):
    st.dataframe(
        g[["비용", "매출", "CTR(%)", "전환율(%)", "ROAS"]].reset_index(),
        hide_index=True,
        use_container_width=True,
        column_config={
            "비용": st.column_config.NumberColumn(format="$%d"),
            "매출": st.column_config.NumberColumn(format="$%d"),
            "CTR(%)": st.column_config.NumberColumn(format="%.2f"),
            "전환율(%)": st.column_config.NumberColumn(format="%.2f"),
            "ROAS": st.column_config.NumberColumn(format="%.2f"),
        },
    )

st.divider()

# =====================================================================
# 함정 3. 비용 규모
# =====================================================================
st.header("함정 3. 돈을 많이 쓸수록 효율이 떨어진다")

b = base.copy()
b["구간"] = pd.qcut(b["spend_usd"], 8, duplicates="drop")
agg = b.groupby("구간", observed=True).agg(
    비용=("spend_usd", "sum"),
    매출=("revenue_usd", "sum"),
    최소=("spend_usd", "min"),
    최대=("spend_usd", "max"),
    중앙CPC=("cpc_usd", "median"),
    캠페인일=("spend_usd", "size"),
)
agg["ROAS"] = agg["매출"] / agg["비용"]
agg = agg.reset_index(drop=True)
agg["하루 비용 구간"] = [f"${lo:,.0f}~${hi:,.0f}" for lo, hi in zip(agg["최소"], agg["최대"])]
order = list(agg["하루 비용 구간"])

left, right = st.columns(2)
with left:
    fig = px.line(
        agg,
        x="하루 비용 구간",
        y="ROAS",
        markers=True,
        text=agg["ROAS"].map(lambda v: f"{v:.2f}"),
        title="캠페인 하루 비용 구간별 ROAS",
        category_orders={"하루 비용 구간": order},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), yaxis_range=[0, agg["ROAS"].max() * 1.2])
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(
        agg,
        x="하루 비용 구간",
        y="중앙CPC",
        text=agg["중앙CPC"].map(lambda v: f"${v:.2f}"),
        title="구간별 클릭 한 번당 비용 (CPC 중앙값)",
        category_orders={"하루 비용 구간": order},
        labels={"중앙CPC": "CPC(달러)"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

first, last = agg.iloc[0], agg.iloc[-1]
st.info(
    f"하루 비용이 가장 작은 구간의 ROAS는 {first['ROAS']:.2f}, 가장 큰 구간은 {last['ROAS']:.2f}입니다. "
    "비용이 큰 캠페인일수록 클릭 한 번당 비용(CPC)도 함께 올라가는 모습이 보입니다. "
    "그래서 예산을 한 곳에 몰아주면 같은 돈으로 얻는 매출이 줄어듭니다. "
    "다음 쪽의 예산 시뮬레이터에서 이 효과를 반영해 예산을 나눠 봅니다."
)
st.caption("구간은 캠페인-일 10만 건 중 이메일 설정에 맞는 데이터를 비용 순서로 8등분한 것입니다.")
