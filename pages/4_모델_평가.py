from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="모델 평가", page_icon="📏", layout="wide")

DATA_PATH = Path(__file__).parent.parent / "data" / "marketing_telemetry.csv"
EMAIL = "Email / CRM"
CATS = ["channel", "audience", "creative_format"]
NAMES_KR = {"channel": "채널", "audience": "타깃", "creative_format": "소재"}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["day"])


@st.cache_data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """이메일 제외, 매출이 0이거나 비용이 1달러 미만인 행 제외 후 로그 변수를 만든다."""
    d = df[(df["channel"] != EMAIL) & (df["revenue_usd"] > 0) & (df["spend_usd"] >= 1)].copy()
    d["y"] = np.log(d["revenue_usd"] / d["spend_usd"])  # 종속 변수: log(ROAS)
    d["log_spend"] = np.log(d["spend_usd"])  # 독립 변수: log(하루 비용)
    return d


def reference_levels(d: pd.DataFrame) -> dict:
    """범주형 변수는 가나다·알파벳 순 첫 번째 값을 기준으로 두고 나머지를 0/1 열로 만든다."""
    return {c: sorted(d[c].unique()) for c in CATS}


def design(frame: pd.DataFrame, feats: list, levels: dict):
    cols = [np.ones(len(frame))]
    names = ["절편"]
    if "log_spend" in feats:
        cols.append(frame["log_spend"].to_numpy())
        names.append("log(하루 비용)")
    for c in CATS:
        if c in feats:
            for lv in levels[c][1:]:
                cols.append((frame[c] == lv).astype(float).to_numpy())
                names.append(f"{NAMES_KR[c]}: {lv}")
    return np.column_stack(cols), names


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    r2 = 1 - (err**2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    return {"R²": r2, "MAE": np.abs(err).mean(), "RMSE": np.sqrt((err**2).mean())}


@st.cache_data
def run_all(d: pd.DataFrame):
    levels = reference_levels(d)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(d))
    cut = int(len(d) * 0.8)
    train, test = d.iloc[order[:cut]], d.iloc[order[cut:]]

    sets = {
        "비용만": ["log_spend"],
        "비용 + 채널": ["log_spend", "channel"],
        "비용 + 타깃": ["log_spend", "audience"],
        "비용 + 채널 + 타깃 (우리 모델)": ["log_spend", "channel", "audience"],
        "비용 + 채널 + 타깃 + 소재": ["log_spend", "channel", "audience", "creative_format"],
    }
    rows, final = [], None
    for name, feats in sets.items():
        Xtr, names = design(train, feats, levels)
        Xte, _ = design(test, feats, levels)
        beta = np.linalg.lstsq(Xtr, train["y"].to_numpy(), rcond=None)[0]
        m_tr = metrics(train["y"].to_numpy(), Xtr @ beta)
        m_te = metrics(test["y"].to_numpy(), Xte @ beta)
        rows.append({"모델": name, "훈련 R²": m_tr["R²"], "테스트 R²": m_te["R²"], "테스트 MAE": m_te["MAE"], "테스트 RMSE": m_te["RMSE"]})
        if "우리 모델" in name:
            final = {"beta": beta, "names": names, "test_pred": Xte @ beta, "feats": feats}
    base_pred = np.full(len(test), train["y"].mean())
    base = metrics(test["y"].to_numpy(), base_pred)
    return levels, train, test, pd.DataFrame(rows), final, base


if not DATA_PATH.exists():
    st.error(
        "데이터 파일을 찾을 수 없습니다. 깃허브 저장소에 `data/marketing_telemetry.csv` 파일이 "
        "올라가 있는지, 폴더 이름(`data`)과 파일 이름의 철자·대소문자가 같은지 확인하세요."
    )
    st.stop()

df = load_data()
d = prepare(df)
levels, train, test, compare, final, base = run_all(d)
me = compare[compare["모델"].str.contains("우리 모델")].iloc[0]

st.title("📏 모델 평가")
st.write("광고 캠페인의 ROAS를 맞히는 선형 회귀 모델이 얼마나 잘 맞는지 확인합니다.")

# =====================================================================
# 1. 모델 설명
# =====================================================================
st.header("1. 어떤 모델인가?")
st.markdown(
    f"""
    - **모델 종류:** 다중 선형 회귀 (숫자를 맞히는 회귀 모델)
    - **종속 변수(맞히려는 값):** log(ROAS)  ← ROAS = 매출 ÷ 비용
    - **독립 변수(맞히는 데 쓰는 값):** log(하루 비용), 채널, 타깃
    - **사용한 데이터:** 이메일/CRM을 뺀 캠페인-일 {len(d):,}개
      (매출이 0이거나 하루 비용이 1달러 미만인 행은 로그를 구할 수 없거나 값이 불안정해서 뺐습니다)
    - **평가 방법:** 데이터를 훈련용 80%({len(train):,}개)와 테스트용 20%({len(test):,}개)로 무작위로 나누고,
      훈련용으로 모델을 만든 뒤 **처음 보는 테스트 데이터**로 성능을 확인합니다.
    """
)

# =====================================================================
# 2. 성능 지표
# =====================================================================
st.header("2. 성능은 어느 정도인가?")

c1, c2, c3, c4 = st.columns(4)
c1.metric("테스트 R² (설명력)", f"{me['테스트 R²']:.3f}")
c2.metric("테스트 MAE (평균 오차)", f"{me['테스트 MAE']:.3f}")
c3.metric("테스트 RMSE", f"{me['테스트 RMSE']:.3f}")
c4.metric("기준선 R² (평균으로 찍기)", f"{base['R²']:.3f}")

ratio = float(np.exp(me["테스트 MAE"]))
st.info(
    f"R²가 {me['테스트 R²']:.2f}라는 것은 캠페인마다 ROAS가 들쭉날쭉한 이유 중 약 {me['테스트 R²'] * 100:.0f}%를 "
    "비용·채널·타깃으로 설명한다는 뜻입니다. 훈련 R²와 테스트 R²가 거의 같아서 훈련 데이터에만 맞춘 모델(과적합)은 아닙니다. "
    f"평균 오차(MAE)는 로그 값으로 {me['테스트 MAE']:.2f}라서, 실제 ROAS가 예측값의 약 {1 / ratio:.2f}배 ~ {ratio:.2f}배 범위에 "
    "들어온다고 읽으면 됩니다. 정확한 한 개 값을 맞히기에는 부족하고, 어떤 방향으로 효율이 달라지는지 보는 용도에 맞는 성능입니다."
)

st.subheader("어떤 변수가 도움이 될까? (변수를 바꿔 가며 비교)")
show = compare.copy()
fig = px.bar(
    show,
    x="테스트 R²",
    y="모델",
    orientation="h",
    text=show["테스트 R²"].map(lambda v: f"{v:.3f}"),
    title="독립 변수 조합별 테스트 R² (높을수록 잘 맞음)",
)
fig.update_traces(textposition="outside")
fig.update_layout(height=340, margin=dict(l=10, r=40, t=50, b=10), yaxis_title=None, xaxis_range=[0, show["테스트 R²"].max() * 1.25])
st.plotly_chart(fig, use_container_width=True)
st.dataframe(
    show,
    hide_index=True,
    use_container_width=True,
    column_config={
        "훈련 R²": st.column_config.NumberColumn(format="%.3f"),
        "테스트 R²": st.column_config.NumberColumn(format="%.3f"),
        "테스트 MAE": st.column_config.NumberColumn(format="%.3f"),
        "테스트 RMSE": st.column_config.NumberColumn(format="%.3f"),
    },
)
st.caption("채널과 타깃을 넣으면 설명력이 크게 오르고, 소재를 더 넣어도 거의 오르지 않습니다. 그래서 소재는 독립 변수에서 뺐습니다.")

# =====================================================================
# 3. 예측 vs 실제
# =====================================================================
st.header("3. 예측값과 실제값 비교")

test_y = test["y"].to_numpy()
pred = final["test_pred"]
idx = np.random.default_rng(0).choice(len(test_y), size=min(3000, len(test_y)), replace=False)

left, right = st.columns(2)
with left:
    lo, hi = float(min(test_y.min(), pred.min())), float(max(test_y.max(), pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred[idx], y=test_y[idx], mode="markers", name="테스트 데이터(3,000개만 표시)", marker=dict(size=4, opacity=0.3)))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="완벽하게 맞힌 선", line=dict(width=3)))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        title="예측 log(ROAS) vs 실제 log(ROAS)",
        xaxis_title="예측값",
        yaxis_title="실제값",
    )
    st.plotly_chart(fig, use_container_width=True)
with right:
    resid = test_y - pred
    fig = px.histogram(x=resid, nbins=60, title="오차(실제 − 예측) 분포", labels={"x": "오차"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="개수")
    st.plotly_chart(fig, use_container_width=True)

st.caption("점이 선 근처에 몰릴수록 잘 맞는 모델입니다. 오차가 0을 중심으로 좌우 대칭이면 한쪽으로 치우쳐 틀리지는 않는다는 뜻입니다.")

# =====================================================================
# 4. 계수 해석
# =====================================================================
st.header("4. 모델이 알아낸 것 (계수 해석)")

beta, names = final["beta"], final["names"]
coef = pd.DataFrame({"항목": names, "계수": beta})
slope = beta[names.index("log(하루 비용)")]
coef["ROAS 변화(%)"] = [
    (np.exp(b) - 1) * 100 if n not in ("절편", "log(하루 비용)") else np.nan for n, b in zip(names, beta)
]

st.markdown(
    f"""
    - **하루 비용 기울기 b = {slope:.3f}** → 하루 비용이 2배가 되면 ROAS는 약 {(2**slope - 1) * 100:.1f}% 달라집니다.
      (예산 시뮬레이터의 채널별 기울기 −0.11 ~ −0.14와 같은 수준입니다.)
    - 채널·타깃 항목은 **기준 그룹보다 ROAS가 몇 % 높고 낮은지**를 보여 줍니다.
      기준 그룹은 채널 **{levels['channel'][0]}**, 타깃 **{levels['audience'][0]}**입니다.
    """
)
st.dataframe(
    coef,
    hide_index=True,
    use_container_width=True,
    column_config={
        "계수": st.column_config.NumberColumn(format="%.3f"),
        "ROAS 변화(%)": st.column_config.NumberColumn(format="%+.0f"),
    },
)

# =====================================================================
# 5. 직접 예측해 보기
# =====================================================================
st.header("5. 직접 예측해 보기")
p1, p2, p3 = st.columns(3)
ch = p1.selectbox("채널", levels["channel"])
au = p2.selectbox("타깃", levels["audience"])
sp = p3.number_input("캠페인 하루 비용 (달러)", min_value=1.0, max_value=30000.0, value=300.0, step=50.0)

row = pd.DataFrame({"channel": [ch], "audience": [au], "log_spend": [np.log(sp)]})
X_row, _ = design(row, final["feats"], levels)
pred_roas = float(np.exp(X_row @ beta)[0])
double = float(np.exp(slope * np.log(2)))

q1, q2 = st.columns(2)
q1.metric("예측 ROAS", f"{pred_roas:.2f}")
q2.metric("비용을 2배로 늘리면", f"{pred_roas * double:.2f}", delta=f"{(double - 1) * 100:.1f}%")
st.caption("예측값은 비슷한 조건의 캠페인들이 보통 내는 ROAS(중앙값에 가까운 값)입니다. 실제 캠페인은 이보다 위아래로 크게 흔들릴 수 있습니다.")
