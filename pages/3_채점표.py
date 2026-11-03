"""분류 모델 페이지와 같은 조건으로 학습한 모델의 혼동행렬과 오류 사례를 확인합니다."""

import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/ehdrjs925/data/refs/heads/main/stroke.csv"
FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
FEATURES = list(FEATURE_NAMES)
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
LOGISTIC_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASELINE_NAME = "훈련용에서 많은 쪽으로만 답하는 모델"

st.set_page_config(page_title="뇌졸중 예측 실습실 | 채점표", page_icon="🧠", layout="wide")
st.title("🧠 뇌졸중 예측 실습실 · 채점표")
st.write(
    "**실제 뇌졸중(1)을 양성**으로 두고, 테스트 데이터에서 각 모델이 "
    "누구를 맞혔고 누구를 놓쳤는지 확인합니다."
)


@st.cache_data(ttl=3600)
def load_data():
    """앞 페이지와 같은 GitHub CSV 파일을 UTF-8로 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.subheader("1. 학습에 사용할 속성 고르기")
selected = st.multiselect(
    "입력 속성 (두 개 이상 선택)",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda column: FEATURE_NAMES[column],
    help="분류 모델 페이지와 같은 속성을 선택하면 채점 결과를 그대로 비교할 수 있습니다.",
)
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개 이상 골라야 채점할 수 있습니다.")
    st.stop()

try:
    raw = load_data()
except Exception as error:
    st.error("데이터를 읽지 못했습니다. CSV 주소와 인터넷 연결을 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

# 원래의 '분류 모델' 페이지와 동일한 정렬, 분할, 전처리, 학습을 사용합니다.
# 오류 사례 표에 필요한 항목은 선택 여부와 관계없이 별도로 보존합니다.
list_columns = ["id", "stroke", "age", "avg_glucose_level", "hypertension", "heart_disease"]
required_columns = list(dict.fromkeys(list_columns + selected))
missing_columns = [column for column in required_columns if column not in raw.columns]
if missing_columns:
    st.error("CSV에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

data = raw[required_columns].copy()
for column in required_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

if data["id"].isna().any() or data["id"].duplicated().any():
    st.error("id에 빈 값 또는 중복 값이 있어 정렬할 수 없습니다.")
    st.stop()
if data["stroke"].isna().any() or not data["stroke"].isin([0, 1]).all():
    st.error("stroke 열은 모든 행에 0 또는 1로 기록되어 있어야 합니다.")
    st.stop()

# bmi가 선택된 경우에만 bmi 빈 값을 뒤에서 훈련용 중앙값으로 채웁니다.
other_missing = [feature for feature in selected if feature != "bmi" and data[feature].isna().any()]
if other_missing:
    st.error("선택한 열에 예상하지 못한 빈 값이 있습니다: " + ", ".join(other_missing))
    st.stop()

ordered = data.sort_values("id", kind="stable").reset_index(drop=True)
test_mask = ordered.index % 10 < 3
train_data = ordered.loc[~test_mask].copy()
test_data = ordered.loc[test_mask].copy()

X_train = train_data[selected].copy()
X_test = test_data[selected].copy()
y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

if "bmi" in selected:
    train_bmi_median = X_train["bmi"].median()
    if pd.isna(train_bmi_median):
        st.error("훈련 데이터의 BMI가 모두 비어 있어 중앙값을 계산할 수 없습니다.")
        st.stop()
    X_train["bmi"] = X_train["bmi"].fillna(train_bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(train_bmi_median)
else:
    train_bmi_median = None

if y_train.nunique() != 2:
    st.error("훈련 데이터에 뇌졸중(1)과 아님(0)이 모두 있어야 분류 모델을 학습할 수 있습니다.")
    st.stop()

# 표준화의 평균·표준편차는 오직 훈련 데이터에서 계산합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic = LogisticRegression(max_iter=2000, random_state=42)
logistic.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

predictions = {
    LOGISTIC_NAME: logistic.predict(X_test_scaled),
    TREE_NAME: tree_model.predict(X_test),
    BASELINE_NAME: baseline.predict(X_test),
}

st.caption(
    f"id 오름차순 정렬 → 10명마다 앞 3명 테스트 / 뒤 7명 훈련 "
    f"| 훈련 {len(train_data):,}명 · 테스트 {len(test_data):,}명"
)
if "bmi" in selected:
    st.caption(
        f"BMI의 빈 값은 훈련 데이터로 구한 중앙값 {train_bmi_median:.2f}로 채웠습니다. "
        "테스트 데이터에서 중앙값을 구하지 않았습니다."
    )


def count_results(actual, predicted):
    """행=실제(0,1), 열=예측(0,1) 순서에서 TN,FP,FN,TP를 얻습니다."""
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp = (int(number) for number in matrix[0])
    fn, tp = (int(number) for number in matrix[1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def make_confusion_table(counts):
    """가운데 2×2 칸에 약자와 인원수를 쓰고 합계 행·열을 붙입니다."""
    tn, fp, fn, tp = (counts[key] for key in ("TN", "FP", "FN", "TP"))
    return pd.DataFrame(
        [
            [f"{tn:,}명 (TN)", f"{fp:,}명 (FP)", f"{tn + fp:,}명"],
            [f"{fn:,}명 (FN)", f"{tp:,}명 (TP)", f"{fn + tp:,}명"],
            [f"{tn + fn:,}명", f"{fp + tp:,}명", f"{tn + fp + fn + tp:,}명"],
        ],
        index=["실제 아님(0)", "실제 뇌졸중(1)", "합계"],
        columns=["예측 아님(0)", "예측 뇌졸중(1)", "합계"],
    )


def make_metrics(counts):
    """혼동행렬 네 칸의 사람 수만 사용해 지표를 직접 계산합니다."""
    tn, fp, fn, tp = (counts[key] for key in ("TN", "FP", "FN", "TP"))
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if tp + fn else None
    predicted_positive = tp + fp
    if predicted_positive == 0:
        # 실제 지시: 뇌졸중이라고 예측한 사람이 없으면 정밀도·F1은 계산 불가.
        precision = None
        f1 = None
    else:
        precision = tp / predicted_positive
        if recall is None:
            f1 = None
        elif precision + recall == 0:
            # 양성을 예측했지만 정답 양성이 전혀 없으면 F1을 0으로 표시.
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def show_percentage(value):
    return "계산할 수 없음" if value is None else f"{value:.2%}"


results = {name: count_results(y_test, predicted) for name, predicted in predictions.items()}

st.subheader("2. 테스트 데이터 혼동행렬")
st.write(
    f"**테스트용 사람 수: {len(y_test):,}명 | "
    f"그중 실제 뇌졸중인 사람 수: {int((y_test == 1).sum()):,}명**"
)
st.caption("행은 실제 상태, 열은 모델의 예측입니다. 뇌졸중(1)을 양성으로 해석합니다.")

left, right = st.columns(2)
with left:
    st.markdown(f"**{LOGISTIC_NAME}**")
    st.table(make_confusion_table(results[LOGISTIC_NAME]))
with right:
    st.markdown(f"**{TREE_NAME}**")
    st.table(make_confusion_table(results[TREE_NAME]))

st.caption(
    "TP: 실제 뇌졸중을 뇌졸중으로 예측 · TN: 실제 아님을 아님으로 예측 · "
    "FN: 실제 뇌졸중을 아님으로 예측 · FP: 실제 아님을 뇌졸중으로 예측"
)

st.subheader("3. 정확도·재현율·정밀도·F1 비교")
metric_rows = []
for name in (LOGISTIC_NAME, TREE_NAME, BASELINE_NAME):
    scores = make_metrics(results[name])
    metric_rows.append(
        {
            "모델": name,
            "정확도": show_percentage(scores[0]),
            "재현율": show_percentage(scores[1]),
            "정밀도": show_percentage(scores[2]),
            "F1": show_percentage(scores[3]),
        }
    )
st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
st.caption(
    "정확도 = (TP+TN) / 전체 · 재현율 = TP / (TP+FN) · "
    "정밀도 = TP / (TP+FP) · F1 = 2×정밀도×재현율 / (정밀도+재현율). "
    "양성 예측이 0명이면 정밀도와 F1은 '계산할 수 없음'으로 표시합니다."
)

st.subheader("4. 누가 잘못 분류되었을까?")
chosen_name = st.selectbox(
    "오류 목록을 확인할 모델",
    options=[LOGISTIC_NAME, TREE_NAME],
)

# 전체 테스트 집합에서 사례를 고르므로 앞 20명만 보여 주더라도 전체 개수는 바뀌지 않습니다.
details = test_data[["id", "age", "avg_glucose_level", "hypertension", "heart_disease", "stroke"]].copy()
details["예측"] = predictions[chosen_name]
missed = details.loc[(details["stroke"] == 1) & (details["예측"] == 0)].copy()
false_alerts = details.loc[(details["stroke"] == 0) & (details["예측"] == 1)].copy()


def display_errors(frame):
    """연령 내림차순, 연령이 같으면 id 오름차순으로 첫 20명만 표시합니다."""
    frame = frame.sort_values(["age", "id"], ascending=[False, True]).head(20).copy()
    frame = frame.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )
    for column in ("고혈압 여부", "심장병 여부"):
        frame[column] = frame[column].map({0: "없음(0)", 1: "있음(1)"}).fillna("미기록")
    return frame[["번호", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"]]


col_missed, col_false = st.columns(2)
with col_missed:
    st.markdown(f"**놓친 환자 (FN): 전체 {len(missed):,}명**")
    st.caption("실제로 뇌졸중(1)이지만 아님(0)으로 예측한 사람")
    if missed.empty:
        st.info("해당하는 사람이 없습니다.")
    else:
        st.dataframe(display_errors(missed), hide_index=True, use_container_width=True)
with col_false:
    st.markdown(f"**헛짚은 사람 (FP): 전체 {len(false_alerts):,}명**")
    st.caption("실제로 아님(0)이지만 뇌졸중(1)으로 예측한 사람")
    if false_alerts.empty:
        st.info("해당하는 사람이 없습니다.")
    else:
        st.dataframe(display_errors(false_alerts), hide_index=True, use_container_width=True)

current_counts = results[chosen_name]
if len(missed) == current_counts["FN"] and len(false_alerts) == current_counts["FP"]:
    st.write(
        f"**전체 목록의 인원수는 혼동행렬과 같습니다:** "
        f"놓친 환자 {len(missed):,}명 = FN {current_counts['FN']:,}명, "
        f"헛짚은 사람 {len(false_alerts):,}명 = FP {current_counts['FP']:,}명."
    )
else:
    st.error("오류 목록과 혼동행렬의 인원수가 일치하지 않습니다. 데이터를 다시 확인해 주세요.")
st.caption("오류가 20명을 넘는 목록은 나이가 많은 순서대로 앞 20명만 표시합니다.")
st.divider()
st.caption("이 결과는 데이터 모델링 학습용이며, 실제 개인의 뇌졸중 여부를 진단하는 데 사용할 수 없습니다.")
