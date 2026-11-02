"""뇌졸중 데이터 탐색 페이지: 분포, 그룹 비교, 결측값, 범주별 빈도."""

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/ehdrjs925/data/refs/heads/main/stroke.csv"
REQUIRED_COLUMNS = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
]
GROUP_LABELS = {0: "미경험 (0)", 1: "경험 (1)"}
GROUP_ORDER = [GROUP_LABELS[0], GROUP_LABELS[1]]

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | 탐색",
    page_icon="🧠",
    layout="wide",
)

st.title("🔎 뇌졸중 데이터 탐색")
st.write(
    "나이와 평균 혈당의 분포, 뇌졸중 경험 여부에 따른 차이, "
    "고혈압·심장병 여부에 따른 뇌졸중 비율 등을 확인합니다."
)


@st.cache_data(ttl=3600)
def load_data():
    """첫 화면과 동일한 원본 CSV를 UTF-8로 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    original_df = load_data()
except Exception as error:
    st.error("CSV 데이터를 읽지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in original_df.columns]
if missing_columns:
    st.error("필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

# 분석에 필요한 열만 숫자형으로 변환합니다. 원본 CSV 파일은 변경하지 않습니다.
df = original_df.copy()
for col in ["age", "avg_glucose_level", "stroke", "hypertension", "heart_disease", "bmi"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

valid_stroke = df[df["stroke"].isin([0, 1])].copy()


# 1. 히스토그램 ------------------------------------------------------------
st.subheader("1. 나이와 평균 혈당의 분포")
hist_left, hist_right = st.columns(2)

with hist_left:
    fig_age = px.histogram(
        df.dropna(subset=["age"]),
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이(세)"},
    )
    fig_age.update_layout(
        xaxis_title="나이(세)",
        yaxis_title="사람 수(명)",
        bargap=0.05,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_age, use_container_width=True)

with hist_right:
    fig_glucose = px.histogram(
        df.dropna(subset=["avg_glucose_level"]),
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수(명)",
        bargap=0.05,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_glucose, use_container_width=True)


# 2. 뇌졸중 경험 여부별 상자그림과 평균값 ----------------------------------
