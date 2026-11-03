"""뇌졸중 분류 모델 비교: 분할·전처리·평가·결정 경계·의사결정트리."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
BINARY_FEATURES = {"hypertension", "heart_disease"}
CLASS_NAMES = {0: "아님 (0)", 1: "뇌졸중 (1)"}
POINT_COLORS = {0: "#2563A6", 1: "#D04A35"}
REGION_COLORS = {0: "rgba(60, 120, 200, 0.13)", 1: "rgba(230, 90, 90, 0.13)"}

st.set_page_config(page_title="뇌졸중 예측 실습실 | 분류 모델", page_icon="🧠", layout="wide")
st.title("🧠 뇌졸중 예측 실습실 · 분류 모델")
st.write(
    "뇌졸중(1)을 **양성**, 뇌졸중 아님(0)을 **음성**으로 정하고, "
    "같은 훈련·테스트 데이터로 세 가지 분류 방식을 비교합니다."
)


@st.cache_data(ttl=3600)
def load_data():
    """앞의 두 페이지와 같은 CSV를 UTF-8로 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.subheader("1. 학습에 사용할 속성 고르기")
selected = st.multiselect(
    "입력 속성 (두 개 이상 선택)",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda feature: FEATURE_NAMES[feature],
    help="체질량지수를 선택한 경우에만, 빈 값을 훈련 데이터에서 계산한 중앙값으로 채웁니다.",
)
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개를 고르면 모델과 그래프가 표시됩니다.")
    st.stop()

try:
    raw = load_data()
except Exception as error:
    st.error("데이터를 읽지 못했습니다. CSV 주소와 인터넷 연결을 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

required_columns = ["id", "stroke"] + selected
missing_columns = [column for column in required_columns if column not in raw.columns]
if missing_columns:
    st.error("CSV에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

# 0) ID 기준으로 먼저 정렬합니다. 그룹마다 앞의 세 행은 테스트, 나머지는 훈련.
data = raw[required_columns].copy()
for column in required_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

if data["id"].isna().any() or data["id"].duplicated().any():
    st.error("id에 빈 값 또는 중복 값이 있어 정렬할 수 없습니다.")
    st.stop()
if data["stroke"].isna().any() or not data["stroke"].isin([0, 1]).all():
    st.error("stroke 열은 모든 행에 0 또는 1로 기록되어 있어야 합니다.")
    st.stop()

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

# 1) BMI를 선택한 경우에만 훈련 데이터로 중앙값을 정합니다.
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

# 2) 크기 맞추기(표준화)는 훈련 데이터로만 학습합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic = LogisticRegression(max_iter=2000, random_state=42)
logistic.fit(X_train_scaled, y_train)

tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

models = [
    ("로지스틱 회귀(확률로 답하는 모델)", logistic, X_train_scaled, X_test_scaled),
    ("의사결정트리(질문으로 답하는 모델)", tree_model, X_train, X_test),
    ("입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델", baseline, X_train, X_test),
]

st.caption(
    f"id 오름차순 정렬 → 10명마다 앞 3명 테스트 / 뒤 7명 훈련  "
    f"|  훈련 {len(train_data):,}명 · 테스트 {len(test_data):,}명"
)
if "bmi" in selected:
    st.caption(
        f"BMI 결측값은 **훈련 데이터에서 계산한 중앙값 {train_bmi_median:.2f}**로 "
        "훈련·테스트 양쪽에 채웠습니다. 테스트 데이터로 중앙값을 계산하지 않았습니다."
    )

# 3) 정확도 카드: 큰 숫자는 테스트 정확도, 아래는 훈련/테스트 정확도.
st.subheader("2. 세 가지 모델의 정확도 비교")
metric_columns = st.columns(3)
for display_column, (model_name, model, train_x, test_x) in zip(metric_columns, models):
    train_accuracy = accuracy_score(y_train, model.predict(train_x))
    test_accuracy = accuracy_score(y_test, model.predict(test_x))
    with display_column:
        with st.container(border=True):
            st.markdown(f"**{model_name}**")
            st.metric("테스트 데이터 정확도", f"{test_accuracy:.2%}")
            left_caption, right_caption = st.columns(2)
            left_caption.caption(f"훈련: **{train_accuracy:.2%}**")
            right_caption.caption(f"테스트: **{test_accuracy:.2%}**")

st.caption(
    "정확도 = 맞힌 사람 수 ÷ 전체 사람 수. 뇌졸중(1)이 상대적으로 적은 데이터에서는 "
    "정확도만으로 양성 예측 성능을 충분히 판단할 수 없습니다."
)

# 4) 실제 테스트 점 + 로지스틱 회귀 0.5 선 + 트리 분할 영역.
st.subheader("3. 테스트 데이터와 두 모델의 분류 경계")
left_axis, right_axis = st.columns(2)
with left_axis:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected,
        index=0,
        format_func=lambda feature: FEATURE_NAMES[feature],
    )
with right_axis:
    y_candidates = [feature for feature in selected if feature != x_feature]
    y_feature = st.selectbox(
        "세로축 속성",
        options=y_candidates,
        index=0,
        format_func=lambda feature: FEATURE_NAMES[feature],
    )


def display_range(values, feature):
    """테스트 점을 모두 포함하는 가로/세로 영역을 정합니다."""
    if feature in BINARY_FEATURES:
        return -0.15, 1.15
    minimum, maximum = float(values.min()), float(values.max())
    pad = max((maximum - minimum) * 0.05, 0.1)
    return minimum - pad, maximum + pad


x_min, x_max = display_range(X_test[x_feature], x_feature)
y_min, y_max = display_range(X_test[y_feature], y_feature)

# 그래프의 두 축을 제외한 모든 선택 속성은 보정된 테스트 데이터의 중앙값에 고정합니다.
held_features = [feature for feature in selected if feature not in (x_feature, y_feature)]
held_values = {feature: float(X_test[feature].median()) for feature in held_features}


def grid_values(start, end, points=61):
    return [start + (end - start) * i / (points - 1) for i in range(points)]


xs = grid_values(x_min, x_max)
ys = grid_values(y_min, y_max)
grid_rows = []
for y in ys:
    for x in xs:
        row = held_values.copy()
        row[x_feature] = x
        row[y_feature] = y
        grid_rows.append(row)
grid = pd.DataFrame(grid_rows, columns=selected)

# 위에서 채점한 바로 그 모델/표준화기로 0.5 확률을 계산합니다.
grid_scaled = scaler.transform(grid)
positive_class_index = list(logistic.classes_).index(1)
probs_flat = logistic.predict_proba(grid_scaled)[:, positive_class_index]
probs = probs_flat.reshape(len(ys), len(xs))

# 나무의 마디마다 실제로 도착한 훈련 데이터 수와 양성 수를 구합니다.
# (sklearn 버전에 따라 tree_.value는 비율/건수로 달라질 수 있어 직접 집계합니다.)
tree_structure = tree_model.tree_
paths = tree_model.decision_path(X_train)
node_counts = {}
node_positives = {}
for node in range(tree_structure.node_count):
    arriving = paths[:, node].nonzero()[0]
    node_counts[node] = len(arriving)
    node_positives[node] = int(y_train.iloc[arriving].sum())


def majority_prediction(node):
    """해당 마지막 마디에서 다수인 답(동점이면 0)을 돌려줍니다."""
    positives = node_positives[node]
    negatives = node_counts[node] - positives
    return int(positives > negatives)


fig = go.Figure()


def paint_tree_regions(node, x0, x1, y0, y1):
    """고정 속성을 반영하며 나무의 정확한 직사각형 분할을 칠합니다."""
    if x0 >= x1 or y0 >= y1:
        return
    left_child = int(tree_structure.children_left[node])
    right_child = int(tree_structure.children_right[node])
    if left_child == right_child:
        answer = majority_prediction(node)
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=REGION_COLORS[answer],
            line=dict(color="rgba(70,80,95,0.28)", width=1),
            layer="below",
        )
        return

    split_feature = selected[int(tree_structure.feature[node])]
    threshold = float(tree_structure.threshold[node])
    if split_feature == x_feature:
        paint_tree_regions(left_child, x0, min(x1, threshold), y0, y1)
        paint_tree_regions(right_child, max(x0, threshold), x1, y0, y1)
    elif split_feature == y_feature:
        paint_tree_regions(left_child, x0, x1, y0, min(y1, threshold))
        paint_tree_regions(right_child, x0, x1, max(y0, threshold), y1)
    elif held_values[split_feature] <= threshold:
        paint_tree_regions(left_child, x0, x1, y0, y1)
    else:
        paint_tree_regions(right_child, x0, x1, y0, y1)


paint_tree_regions(0, x_min, x_max, y_min, y_max)


def contour_segments(x_values, y_values, probabilities, level=0.5):
    """인접 격자에서 0.5를 가로지르는 지점을 선분으로 연결합니다."""
    line_x, line_y = [], []
    for row_idx in range(len(y_values) - 1):
        for col_idx in range(len(x_values) - 1):
            # 왼쪽 아래, 오른쪽 아래, 오른쪽 위, 왼쪽 위 순서
            corners = [
                (x_values[col_idx], y_values[row_idx], probabilities[row_idx, col_idx]),
                (x_values[col_idx + 1], y_values[row_idx], probabilities[row_idx, col_idx + 1]),
                (x_values[col_idx + 1], y_values[row_idx + 1], probabilities[row_idx + 1, col_idx + 1]),
                (x_values[col_idx], y_values[row_idx + 1], probabilities[row_idx + 1, col_idx]),
            ]
            crossings = []
            for edge_idx in range(4):
                a, b = corners[edge_idx], corners[(edge_idx + 1) % 4]
                if (a[2] < level <= b[2]) or (b[2] < level <= a[2]):
                    fraction = (level - a[2]) / (b[2] - a[2])
                    crossings.append(
                        (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))
                    )
            for idx in range(0, len(crossings) - 1, 2):
                first, second = crossings[idx], crossings[idx + 1]
                line_x.extend([first[0], second[0], None])
                line_y.extend([first[1], second[1], None])
    return line_x, line_y


boundary_x, boundary_y = contour_segments(xs, ys, probs)
if boundary_x:
    fig.add_trace(
        go.Scatter(
            x=boundary_x, y=boundary_y,
            mode="lines", name="로지스틱 회귀: 0.5 경계",
            line=dict(color="#252525", width=2.8),
            hoverinfo="skip",
        )
    )

for class_value in [0, 1]:
    mask = y_test == class_value
    fig.add_trace(
        go.Scatter(
            x=X_test.loc[mask, x_feature],
            y=X_test.loc[mask, y_feature],
            mode="markers",
            name=f"실제 {CLASS_NAMES[class_value]}",
            marker=dict(
       size=7, color=POINT_COLORS[class_value], opacity=0.65,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
                f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
                f"실제: {CLASS_NAMES[class_value]}<extra></extra>"
            ),
        )
    )

fig.update_layout(
    template="plotly_white",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    xaxis=dict(range=[x_min, x_max]),
    yaxis=dict(range=[y_min, y_max]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=20, r=20, t=80, b=20),
    height=540,
)
for feature, axis_name in [(x_feature, "xaxis"), (y_feature, "yaxis")]:
    if feature in BINARY_FEATURES:
        fig.update_layout(**{axis_name: dict(tickmode="array", tickvals=[0, 1], ticktext=["0 (없음)", "1 (있음)"])})

st.plotly_chart(fig, use_container_width=True)
st.caption(
    "점의 색은 실제 뇌졸중 여부를 나타냅니다. 배경의 옅은 파란 칸은 의사결정트리가 "
    "'아님(0)'으로, 옅은 붉은 칸은 '뇌졸중(1)'으로 답하는 영역입니다. "
    "검은 선은 로지스틱 회귀가 뇌졸중 확률 0.5로 나누는 경계입니다."
)

if held_values:
    held_texts = []
    for feature, value in held_values.items():
        formatted = f"{value:.0f}" if feature in BINARY_FEATURES else f"{value:.2f}"
        held_texts.append(f"{FEATURE_NAMES[feature]} = {formatted}")
    st.write("**그림에서 고정한 속성(테스트 데이터 중앙값):** " + ", ".join(held_texts))
else:
    st.write("**그림에서 고정한 속성:** 없음 (선택한 두 속성을 모두 축으로 사용)")

if not boundary_x:
    st.info(
        "**로지스틱 회귀의 0.5 경계선은 현재 그림 범위 밖에 있습니다.** "
        "고정한 다른 속성의 값에 따라, 이 두 축의 평면에 경계선이 나타나지 않을 수도 있습니다."
    )
if x_feature in BINARY_FEATURES or y_feature in BINARY_FEATURES:
    st.caption("고혈압·심장병은 실제로 0 또는 1인 속성입니다. 두 값 사이의 선은 모델을 설명하기 위한 시각화입니다.")

# 5) 실제 학습된 트리를 DOT 문자로 나타냅니다.
st.subheader("4. 의사결정트리의 질문과 답")
st.write("각 마디에는 그곳에 도착한 **훈련 사람 수**, **실제 뇌졸중인 사람 수**, **뇌졸중 비율**을 표시합니다.")


def dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


dot_lines = [
    "digraph StrokeTree {",
    'graph [rankdir=TB, bgcolor="transparent", nodesep=0.35, ranksep=0.55];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=11, margin="0.15,0.12"];',
    'edge [fontname="sans-serif", fontsize=11, color="#65717f"];',
]

leaf_total = 0
leaf_no_count = 0
actually_asked = set()
