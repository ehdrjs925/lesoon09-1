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
st.subheader("2. 뇌졸중 경험 여부에 따른 나이·평균 혈당 비교")
comparison = valid_stroke.copy()
comparison["그룹"] = comparison["stroke"].map(GROUP_LABELS)
box_left, box_right = st.columns(2)

with box_left:
    fig_age_box = px.box(
        comparison.dropna(subset=["age"]),
        x="그룹",
        y="age",
        category_orders={"그룹": GROUP_ORDER},
        title="뇌졸중 경험 여부별 나이",
        labels={"age": "나이(세)"},
        points="outliers",
    )
    fig_age_box.update_layout(
        xaxis_title="뇌졸중 경험 여부",
        yaxis_title="나이(세)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_right:
    fig_glucose_box = px.box(
        comparison.dropna(subset=["avg_glucose_level"]),
        x="그룹",
        y="avg_glucose_level",
        category_orders={"그룹": GROUP_ORDER},
        title="뇌졸중 경험 여부별 평균 혈당",
        labels={"avg_glucose_level": "평균 혈당"},
        points="outliers",
    )
    fig_glucose_box.update_layout(
        xaxis_title="뇌졸중 경험 여부",
        yaxis_title="평균 혈당",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

mean_rows = []
for stroke_value in [0, 1]:
    group = valid_stroke[valid_stroke["stroke"] == stroke_value]
    mean_rows.append(
        {
            "뇌졸중 경험 여부": GROUP_LABELS[stroke_value],
            "사람 수(명)": len(group),
            "평균 나이(세)": round(group["age"].mean(), 2),
            "평균 혈당": round(group["avg_glucose_level"].mean(), 2),
        }
    )
st.markdown("**두 그룹의 평균값**")
st.dataframe(pd.DataFrame(mean_rows), hide_index=True, use_container_width=True)
st.caption("평균값은 해당 열에 값이 있는 사람을 기준으로 계산합니다.")


# 3. 고혈압/심장병 여부별 뇌졸중 비율 ---------------------------------------
st.subheader("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def condition_rate_chart(column, name):
    """해당 상태가 0/1인 두 집단의 뇌졸중 경험 비율(%)을 계산합니다."""
    data = valid_stroke[valid_stroke[column].isin([0, 1])]
    rows = []
    for condition_value in [0, 1]:
        group = data[data[column] == condition_value]
        total = len(group)
        if not total:
            continue
        stroke_count = int((group["stroke"] == 1).sum())
        rows.append(
            {
                "구분": "없음 (0)" if condition_value == 0 else "있음 (1)",
                "뇌졸중 비율(%)": stroke_count / total * 100,
                "사람 수(명)": total,
                "뇌졸중 경험자(명)": stroke_count,
            }
        )

    if not rows:
        st.info(f"{name} 여부를 비교할 수 있는 데이터가 없습니다.")
        return

    summary = pd.DataFrame(rows)
    fig = px.bar(
        summary,
        x="구분",
        y="뇌졸중 비율(%)",
        title=f"{name} 여부별 뇌졸중 비율",
        text="뇌졸중 비율(%)",
        hover_data=["사람 수(명)", "뇌졸중 경험자(명)"],
        category_orders={"구분": ["없음 (0)", "있음 (1)"]},
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        xaxis_title=f"{name} 여부",
        yaxis_title="뇌졸중 비율(%)",
        yaxis=dict(range=[0, min(100, max(5, summary["뇌졸중 비율(%)"].max() * 1.35))]),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


bar_left, bar_right = st.columns(2)
with bar_left:
    condition_rate_chart("hypertension", "고혈압")
with bar_right:
    condition_rate_chart("heart_disease", "심장병")
st.caption("각 비율 = 해당 그룹의 stroke=1인 사람 수 ÷ 해당 그룹의 사람 수 × 100")


# 4. BMI 결측값 집단과 전체 뇌졸중 비율 -----------------------------------
st.subheader("4. BMI가 비어 있는 사람과 전체의 뇌졸중 비율")
bmi_missing = df[df["bmi"].isna()]


def stroke_rate(data):
    """stroke가 0/1로 확인된 행을 분모로 사용합니다."""
    eligible = data[data["stroke"].isin([0, 1])]
    total = len(eligible)
    stroke_count = int((eligible["stroke"] == 1).sum())
    return stroke_count, (stroke_count / total * 100 if total else None)


missing_stroke_count, missing_stroke_rate = stroke_rate(bmi_missing)
all_stroke_count, all_stroke_rate = stroke_rate(df)
bmi_summary = pd.DataFrame(
    [
        {
            "BMI 빈 값 사람 수(명)": len(bmi_missing),
            "그중 뇌졸중 경험자(명)": missing_stroke_count,
            "BMI 빈 값 집단의 뇌졸중 비율": (
                f"{missing_stroke_rate:.2f}%" if missing_stroke_rate is not None else "계산 불가"
            ),
            "전체 사람 수(명)": len(df),
            "전체 뇌졸중 비율": (
                f"{all_stroke_rate:.2f}%" if all_stroke_rate is not None else "계산 불가"
            ),
        }
    ]
)
st.dataframe(bmi_summary, hide_index=True, use_container_width=True)
st.caption("BMI의 빈 값은 결측값(NaN)을 뜻합니다. 뇌졸중 비율의 분모는 stroke 값이 0 또는 1인 사람 수입니다.")


# 5. 흡연 상태의 범주별 사람 수 -------------------------------------------
st.subheader("5. 흡연 상태별 사람 수")
smoking_counts = (
    df["smoking_status"]
    .fillna("(빈 값)")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태(smoking_status)")
    .reset_index(name="사람 수(명)")
)
st.dataframe(smoking_counts, hide_index=True, use_container_width=True)
st.caption("'Unknown'은 원본 데이터에 기록된 범주이며, 빈 값(NaN)과 구분합니다.")

st.divider()
st.caption("이 화면의 통계는 표본 데이터의 관찰 결과입니다. 집단 차이만으로 원인이나 개인의 질병 위험을 확정할 수 없습니다.")
