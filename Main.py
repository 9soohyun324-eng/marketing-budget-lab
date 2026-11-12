from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="광고 예산 재배분 실습실", page_icon="📊", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "marketing_telemetry.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["day"])
    return df


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """합계로 다시 계산한 ROAS·CAC (원본 roas 열은 비용이 1달러 미만인 행에서 어긋나므로 쓰지 않음)."""
    g = df.groupby(by).agg(
        비용=("spend_usd", "sum"),
        매출=("revenue_usd", "sum"),
        주문=("orders", "sum"),
        캠페인일=("campaign_day_id", "count"),
    )
    g["ROAS"] = g["매출"] / g["비용"]
    g["CAC"] = g["비용"] / g["주문"]
    return g.sort_values("ROAS", ascending=False).reset_index()


df = load_data()

st.title("📊 광고 예산 재배분 실습실")
st.caption("데이터로 문제해결하기 프로젝트 · 이커머스 마케팅 가상 데이터 기반")

st.header("우리가 풀 문제")
st.markdown(
    """
    쇼핑몰 마케팅 팀장이 되어 보세요. 다음 분기 광고 예산을 **채널과 타깃에 어떻게 나눠야** 같은 돈으로 더 많은 매출을 올릴 수 있을까요?

    1. **데이터 살펴보기** — 지금 어느 채널과 타깃이 잘 되고 있는지 확인합니다.
    2. **함정 찾기** — 평균만 보면 틀리는 이유를 확인합니다.
    3. **예산 시뮬레이터** — 예산을 옮기면 예상 매출이 얼마나 달라지는지 계산합니다.
    """
)

st.divider()

st.header("데이터 한눈에 보기")
include_email = st.toggle(
    "이메일/CRM 채널 포함하기",
    value=False,
    help="이메일은 비용이 전체의 0.3%인데 ROAS가 약 589배라 전체 평균을 크게 부풀립니다.",
)
view = df if include_email else df[df["channel"] != "Email / CRM"]

spend = view["spend_usd"].sum()
revenue = view["revenue_usd"].sum()
orders = view["orders"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("캠페인-일 수", f"{len(view):,}")
c2.metric("총 광고비", f"${spend / 1e6:,.1f}M")
c3.metric("전체 ROAS (매출÷비용)", f"{revenue / spend:.2f}")
c4.metric("전체 CAC (비용÷주문)", f"${spend / orders:,.2f}")

by_channel = summarize(view, "channel")
fig = px.bar(
    by_channel.sort_values("ROAS"),
    x="ROAS",
    y="channel",
    orientation="h",
    text=by_channel.sort_values("ROAS")["ROAS"].round(2),
    labels={"channel": "채널", "ROAS": "ROAS (매출 ÷ 비용)"},
    title="채널별 ROAS" + ("" if include_email else " (이메일 제외)"),
)
fig.update_traces(textposition="outside")
fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.info(
    "토글을 켜고 꺼 보세요. 이메일을 포함하면 전체 ROAS가 약 6.35, 제외하면 약 4.61입니다. "
    "이 차이가 '함정 찾기' 쪽에서 다룰 첫 번째 주제입니다."
)

st.divider()

with st.expander("데이터 출처와 한계"):
    st.markdown(
        """
        - **출처:** E-commerce Growth & Retail Media Benchmark 2026 (dianatofficial), 라이선스 CC BY-SA 4.0
        - **기간:** 2026-03-01 ~ 2026-09-26, 10만 개의 캠페인-일 기록
        - **주의:** 실제 광고 성과가 아니라 **시뮬레이션으로 만든 가상 데이터**입니다. 결과는 학습용 가설로만 해석해야 합니다.
        - **데이터 처리:** 원본의 `roas` 열은 비용이 1달러 미만인 277행에서 값이 어긋나므로, 이 앱에서는 합계로 ROAS를 다시 계산합니다. `week` 열은 날짜와 맞지 않아 사용하지 않습니다.
        """
    )

with st.expander("원본 데이터 미리보기"):
    st.dataframe(df.head(100), use_container_width=True)
