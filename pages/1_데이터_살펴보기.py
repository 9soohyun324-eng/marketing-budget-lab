from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="데이터 살펴보기", page_icon="🔍", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "data" / "marketing_telemetry.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["day"])


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """합계로 다시 계산한 지표 (원본 roas 열은 쓰지 않음)."""
    g = df.groupby(by).agg(
        비용=("spend_usd", "sum"),
        매출=("revenue_usd", "sum"),
        주문=("orders", "sum"),
        클릭=("clicks", "sum"),
        노출=("impressions", "sum"),
        캠페인일=("campaign_day_id", "count"),
    )
    g["ROAS"] = g["매출"] / g["비용"]
    g["CAC"] = g["비용"] / g["주문"]
    g["CTR(%)"] = g["클릭"] / g["노출"] * 100
    g["전환율(%)"] = g["주문"] / g["클릭"] * 100
    return g.sort_values("ROAS", ascending=False).reset_index()


def show_table(table: pd.DataFrame, key_col: str) -> None:
    cols = [key_col, "비용", "매출", "ROAS", "CAC", "CTR(%)", "전환율(%)", "캠페인일"]
    st.dataframe(
        table[cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "비용": st.column_config.NumberColumn(format="$%d"),
            "매출": st.column_config.NumberColumn(format="$%d"),
            "ROAS": st.column_config.NumberColumn(format="%.2f"),
            "CAC": st.column_config.NumberColumn(format="$%.2f"),
            "CTR(%)": st.column_config.NumberColumn(format="%.2f"),
            "전환율(%)": st.column_config.NumberColumn(format="%.2f"),
            "캠페인일": st.column_config.NumberColumn(format="%d"),
        },
    )


def bar(table: pd.DataFrame, key_col: str, value_col: str, title: str, fmt: str) -> None:
    t = table.sort_values(value_col)
    fig = px.bar(
        t,
        x=value_col,
        y=key_col,
        orientation="h",
        text=t[value_col].map(lambda v: fmt.format(v)),
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, margin=dict(l=10, r=40, t=50, b=10), yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


if not DATA_PATH.exists():
    st.error(
        "데이터 파일을 찾을 수 없습니다. 깃허브 저장소에 `data/marketing_telemetry.csv` 파일이 "
        "올라가 있는지, 폴더 이름(`data`)과 파일 이름의 철자·대소문자가 같은지 확인하세요."
    )
    st.stop()

df = load_data()

st.title("🔍 데이터 살펴보기")
st.write("지금 어느 채널, 어느 타깃, 어떤 소재가 잘 되고 있는지 확인합니다. 왼쪽 필터로 범위를 좁혀 보세요.")

# ---------- 사이드바 필터 ----------
st.sidebar.header("필터")

all_channels = sorted(df["channel"].unique())
default_channels = [c for c in all_channels if c != "Email / CRM"]
channels = st.sidebar.multiselect(
    "채널",
    all_channels,
    default=default_channels,
    help="이메일/CRM은 기본값에서 뺐습니다. 비용이 아주 적은데 ROAS가 커서 평균을 왜곡합니다.",
)

all_audiences = sorted(df["audience"].unique())
audiences = st.sidebar.multiselect("타깃", all_audiences, default=all_audiences)

min_day, max_day = df["day"].min().date(), df["day"].max().date()
picked = st.sidebar.date_input("기간", value=(min_day, max_day), min_value=min_day, max_value=max_day)
if isinstance(picked, (list, tuple)):
    if len(picked) == 2:
        start, end = picked
    else:  # 시작 날짜만 고른 상태
        start, end = picked[0], max_day
else:
    start, end = picked, max_day

view = df[
    df["channel"].isin(channels)
    & df["audience"].isin(audiences)
    & (df["day"].dt.date >= start)
    & (df["day"].dt.date <= end)
]

if view.empty:
    st.warning("선택한 조건에 맞는 데이터가 없습니다. 필터를 넓혀 보세요.")
    st.stop()

# ---------- 핵심 지표 ----------
spend = view["spend_usd"].sum()
revenue = view["revenue_usd"].sum()
orders = view["orders"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("캠페인-일 수", f"{len(view):,}")
c2.metric("광고비", f"${spend / 1e6:,.2f}M")
c3.metric("ROAS", f"{revenue / spend:.2f}")
c4.metric("CAC", f"${spend / orders:,.2f}")

st.caption("ROAS = 매출 합계 ÷ 비용 합계, CAC = 비용 합계 ÷ 주문 합계")

# ---------- 탭 ----------
tab_channel, tab_audience, tab_creative, tab_mix = st.tabs(
    ["채널별", "타깃별", "소재별", "채널 × 타깃"]
)

with tab_channel:
    t = summarize(view, "channel")
    left, right = st.columns(2)
    with left:
        bar(t, "channel", "ROAS", "채널별 ROAS (높을수록 효율적)", "{:.2f}")
    with right:
        bar(t, "channel", "CAC", "채널별 CAC (낮을수록 효율적, 달러)", "${:.2f}")
    show_table(t, "channel")

with tab_audience:
    t = summarize(view, "audience")
    left, right = st.columns(2)
    with left:
        bar(t, "audience", "ROAS", "타깃별 ROAS", "{:.2f}")
    with right:
        bar(t, "audience", "CAC", "타깃별 CAC (달러)", "${:.2f}")
    show_table(t, "audience")

with tab_creative:
    t = summarize(view, "creative_format")
    left, right = st.columns(2)
    with left:
        bar(t, "creative_format", "CTR(%)", "소재별 CTR (클릭률)", "{:.2f}%")
    with right:
        bar(t, "creative_format", "ROAS", "소재별 ROAS", "{:.2f}")
    show_table(t, "creative_format")
    st.info("클릭률은 소재마다 크게 다르지만, ROAS는 거의 비슷한지 비교해 보세요.")

with tab_mix:
    p = view.pivot_table(
        index="channel", columns="audience", values=["revenue_usd", "spend_usd"], aggfunc="sum"
    )
    roas = (p["revenue_usd"] / p["spend_usd"]).round(2)
    fig = px.imshow(
        roas,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="타깃", y="채널", color="ROAS"),
        title="채널 × 타깃 ROAS (진할수록 효율적)",
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("칸이 비어 있으면 선택한 조건에 해당 조합의 데이터가 없다는 뜻입니다.")
