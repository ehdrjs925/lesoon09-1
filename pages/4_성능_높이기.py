"""뇌졸중 분류 모델에서 클래스 가중치를 맞추기 전후의 성능과 경계를 비교합니다.

앞선 '채점표'와 동일하게 id순 정렬, 10명 중 앞 3명 테스트 분리,
훈련 데이터 기준 BMI 중앙값 대체와 로지스틱 회귀용 표준화를 사용합니다.
"""

from math import atan2, sqrt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
BINARY_FEATURES = {"hypertension", "heart_disease"}
LOGISTIC_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BEFORE_COLOR = "#2077B4"
AFTER_COLOR = "#E58A20"
POINT_COLORS = {0: "#3569B0", 1: "#CC3A4B"}

st.set_page_config(
    page_title="뇌졸중 예측 실습실 | 성능 높이기",
    page_icon="🧠",
    layout="wide",
)
st.title("🧠 뇌졸중 예측 실습실 · 성능 높이기")
st.write(
    "**뇌졸중(1)을 양성**으로 두고, 드문 뇌졸중 사례에 가중치를 맞추기 "
    "**전과 후**의 예측 결과를 같은 테스트 데이터에서 비교합니다."
)


@st.cache_data(ttl=3600)
def load_data():
    """다른 페이지와 똑같은 CSV 주소를 UTF-8로 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


# 속성 선택 부분은 '분류 모델' 페이지와 같고, 기본값에는 BMI를 넣지 않습니다.
st.subheader("1. 학습에 사용할 속성 고르기")
selected = st.multiselect(
    "입력 속성 (두 개 이상 선택)",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda feature: FEATURE_NAMES[feature],
    help="BMI를 고르면 빈 값을 훈련용 중앙값으로 채웁니다.",
)
if len(selected) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개 이상 고르면 결과를 비교할 수 있습니다.")
    st.stop()

try:
    raw = load_data()
except Exception as error:
    st.error("데이터를 읽지 못했습니다. CSV 주소와 인터넷 연결을 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

# 채점표와 동일한 행 선택 / 분할 / 결측값 처리.
required_columns = ["id", "stroke"] + selected
missing_columns = [name for name in required_columns if name not in raw.columns]
if missing_columns:
    st.error("CSV에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

data = raw[required_columns].copy()
for name in required_columns:
    data[name] = pd.to_numeric(data[name], errors="coerce")

if data["id"].isna().any() or data["id"].duplicated().any():
    st.error("id에 빈 값 또는 중복 값이 있어 정렬할 수 없습니다.")
    st.stop()
if data["stroke"].isna().any() or not data["stroke"].isin([0, 1]).all():
    st.error("stroke 열은 모두 0(아님) 또는 1(뇌졸중)이어야 합니다.")
    st.stop()
other_missing = [name for name in selected if name != "bmi" and data[name].isna().any()]
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
    bmi_median = X_train["bmi"].median()
    if pd.isna(bmi_median):
        st.error("훈련용 BMI가 모두 비어 있어 중앙값을 계산할 수 없습니다.")
        st.stop()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
else:
    bmi_median = None

if y_train.nunique() != 2:
    st.error("훈련 데이터에는 뇌졸중(1)과 아님(0)이 모두 있어야 합니다.")
    st.stop()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 가중치 외에는 동일한 하이퍼파라미터를 사용합니다.
logistic_before = LogisticRegression(max_iter=2000, random_state=42)
logistic_after = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)

tree_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"
)
tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)

st.caption(
    f"id 오름차순 정렬 → 10명 중 앞 3명 테스트 / 뒤 7명 훈련 "
    f"| 훈련 {len(y_train):,}명 · 테스트 {len(y_test):,}명 "
    f"(실제 뇌졸중 {int(y_test.sum()):,}명)"
)
if bmi_median is not None:
    st.caption(
        f"선택한 BMI의 결측값은 훈련용 중앙값 **{bmi_median:.2f}**로 "
        "훈련·테스트 데이터에 동일하게 채웠습니다."
    )
st.caption(
    "'가중치를 맞춘 뒤'는 두 모델에 class_weight='balanced'를 적용한 결과입니다. "
    "테스트 데이터의 정답이나 평균·표준편차는 학습에 사용하지 않았습니다."
)


def count_results(actual, predicted):
    """행=실제 0,1 / 열=예측 0,1인 혼동행렬에서 네 칸을 계산합니다."""
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp = (int(value) for value in matrix[0])
    fn, tp = (int(value) for value in matrix[1])
    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def evaluate(counts):
    """이전 페이지의 정의와 같은 정확도·재현율·정밀도·F1을 구합니다."""
    tn, fp, fn, tp = (counts[key] for key in ("TN", "FP", "FN", "TP"))
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if tp + fn else None
    if tp + fp == 0:
        precision, f1 = None, None
    else:
        precision = tp / (tp + fp)
        if recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def show_percent(value):
    return "계산할 수 없음" if value is None else f"{value:.2%}"


def comparison_table(before, after):
    """혼동행렬 네 칸 + 지표 네 개를 한 표의 전/후 열에서 비교합니다."""
    metric_before, metric_after = evaluate(before), evaluate(after)
    rows = []
    descriptions = {
        "TN": "TN · 실제 아님 → 예측 아님",
        "FP": "FP · 실제 아님 → 예측 뇌졸중",
        "FN": "FN · 실제 뇌졸중 → 예측 아님",
        "TP": "TP · 실제 뇌졸중 → 예측 뇌졸중",
    }
    for key in ("TN", "FP", "FN", "TP"):
        rows.append({
            "평가 항목": descriptions[key],
            "가중치 전": f"{before[key]:,}명",
            "가중치 후": f"{after[key]:,}명",
                    })
    for metric in ("정확도", "재현율", "정밀도", "F1"):
        rows.append({
            "평가 항목": metric,
            "가중치 전": show_percent(metric_before[metric]),
            "가중치 후": show_percent(metric_after[metric]),
        })
    rows.append({
        "평가 항목": "인원 변화 (찾아냄 / 놓침 / 헛짚음)",
        "가중치 전": f"{before['TP']:,} / {before['FN']:,} / {before['FP']:,}명",
        "가중치 후": f"{after['TP']:,} / {after['FN']:,} / {after['FP']:,}명",
    })
    return pd.DataFrame(rows)


results = {
    LOGISTIC_NAME: (
        count_results(y_test, logistic_before.predict(X_test_scaled)),
        count_results(y_test, logistic_after.predict(X_test_scaled)),
    ),
    TREE_NAME: (
        count_results(y_test, tree_before.predict(X_test)),
        count_results(y_test, tree_after.predict(X_test)),
    ),
}

st.subheader("2. 가중치를 맞추기 전과 맞춘 뒤의 성능")
st.write(
    f"**테스트용 {len(y_test):,}명 중 실제 뇌졸중 {int(y_test.sum()):,}명** "
    "| TP=찾아낸 환자 · FN=놓친 환자 · FP=헛짚은 사람"
)
left, right = st.columns(2)
for column, name in zip((left, right), (LOGISTIC_NAME, TREE_NAME)):
    before, after = results[name]
    with column:
        st.markdown(f"#### {name}")
        st.dataframe(comparison_table(before, after), hide_index=True, use_container_width=True)
        st.write(
            f"**찾아낸 환자 {before['TP']:,}명 → {after['TP']:,}명 · "
            f"놓친 환자 {before['FN']:,}명 → {after['FN']:,}명 · "
            f"헛짚은 사람 {before['FP']:,}명 → {after['FP']:,}명**"
        )
st.caption(
    "정확도=(TP+TN)/전체 · 재현율=TP/(TP+FN) · "
    "정밀도=TP/(TP+FP) · F1=2×정밀도×재현율/(정밀도+재현율). "
    "양성 예측이 0명이면 정밀도와 F1은 계산할 수 없음으로 표시합니다."
)

# 두 로지스틱 회귀의 정확히 같은 모델 계수로 0.5 경계선을 그립니다.
# 표준화 식을 원래 값의 선형식으로 바꾸면 축과 눈금이 원래 단위를 유지합니다.
def original_coefficients(model):
    coefficients = {
        feature: float(model.coef_[0][index]) / float(scaler.scale_[index])
        for index, feature in enumerate(selected)
    }
    bias = float(model.intercept_[0]) - sum(
        coefficients[feature] * float(scaler.mean_[index])
        for index, feature in enumerate(selected)
    )
    return bias, coefficients


before_bias, before_coef = original_coefficients(logistic_before)
after_bias, after_coef = original_coefficients(logistic_after)


def axis_range(feature):
    """모든 테스트 점을 포함하는 표시 영역을 정합니다."""
    if feature in BINARY_FEATURES:
        return (-0.15, 1.15)
    low, high = float(X_test[feature].min()), float(X_test[feature].max())
    padding = max((high - low) * 0.05, 0.1)
    return (low - padding, high + padding)


def fix_other_features(axes):
    """그림의 축이 아닌 선택 속성은 보정된 테스트 데이터 중앙값에 고정합니다."""
    return {
        feature: float(X_test[feature].median())
        for feature in selected
        if feature not in axes
    }


def fixed_caption(fixed):
    if not fixed:
        return "고정한 속성 없음 (선택한 속성을 모두 축으로 사용)."
    return "축이 아닌 속성은 테스트 데이터의 중앙값으로 고정: " + ", ".join(
        f"{FEATURE_NAMES[feature]}={value:.2f}"
        for feature, value in fixed.items()
    )


def plane_parts(bias, coefficients, fixed):
    return bias + sum(coefficients[feature] * value for feature, value in fixed.items())


def intersect_line_rectangle(axis_x, axis_y, xr, yr, bias, coefficients, fixed):
    """원래 단위에서 p=0.5인 직선과 그래프 사각형의 교점을 계산합니다."""
    a = float(coefficients[axis_x])
    b = float(coefficients[axis_y])
    c = plane_parts(bias, coefficients, fixed)
 eps = 1e-12
    if abs(a) < eps and abs(b) < eps:
        return [], ("all" if abs(c) < eps else "outside")
    points = []
    xmin, xmax = xr
    ymin, ymax = yr

    def insert(x, y):
        if xmin - 1e-8 <= x <= xmax + 1e-8 and ymin - 1e-8 <= y <= ymax + 1e-8:
            x = min(max(x, xmin), xmax)
            y = min(max(y, ymin), ymax)
            if not any(abs(x - px) < 1e-7 and abs(y - py) < 1e-7 for px, py in points):
                points.append((x, y))

    if abs(b) >= eps:
        insert(xmin, -(a * xmin + c) / b)
        insert(xmax, -(a * xmax + c) / b)
    if abs(a) >= eps:
        insert(-(b * ymin + c) / a, ymin)
        insert(-(b * ymax + c) / a, ymax)
    if len(points) < 2:
        return points, "outside"
    # 경계선이 사각형 모서리를 따라 있는 경우까지 처리합니다.
    endpoints = max(
        ((p, q) for p in points for q in points if p != q),
        key=lambda pair: (pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2,
    )
    return list(endpoints), "visible"


st.subheader("3. 2차원 산점도와 로지스틱 회귀의 0.5 경계선")
col_x, col_y = st.columns(2)
with col_x:
    x_axis = st.selectbox(
        "가로축", options=selected, index=0,
        format_func=lambda feature: FEATURE_NAMES[feature], key="performance_x",
    )
with col_y:
    y_choices = [feature for feature in selected if feature != x_axis]
    y_axis = st.selectbox(
        "세로축", options=y_choices, index=0,
        format_func=lambda feature: FEATURE_NAMES[feature], key="performance_y",
    )

x_limits = axis_range(x_axis)
y_limits = axis_range(y_axis)
fixed_2d = fix_other_features([x_axis, y_axis])
fig2 = go.Figure()
for stroke_value in (0, 1):
    mask = y_test == stroke_value
    subset = X_test.loc[mask]
    fig2.add_trace(go.Scatter(
        x=subset[x_axis], y=subset[y_axis], mode="markers",
        name="실제 아님(0)" if stroke_value == 0 else "실제 뇌졸중(1)",
        marker=dict(color=POINT_COLORS[stroke_value], size=6, opacity=0.58),
        customdata=test_data.loc[mask, "id"],
        hovertemplate=(
            "번호: %{customdata}<br>" + FEATURE_NAMES[x_axis] + ": %{x:.2f}<br>"
            + FEATURE_NAMES[y_axis] + ": %{y:.2f}<extra>%{fullData.name}</extra>"
        ),
    ))

outside_2d = []
for label, bias, coef, color, dash in (
    ("가중치 전 (p=0.5)", before_bias, before_coef, BEFORE_COLOR, "solid"),
    ("가중치 후 (p=0.5)", after_bias, after_coef, AFTER_COLOR, "dash"),
):
    endpoints, status = intersect_line_rectangle(
        x_axis, y_axis, x_limits, y_limits, bias, coef, fixed_2d
    )
    if status == "visible":
        fig2.add_trace(go.Scatter(
            x=[point[0] for point in endpoints],
            y=[point[1] for point in endpoints],
            mode="lines", name=label,
            line=dict(color=color, width=3.5, dash=dash),
            hoverinfo="skip",
        ))
    elif status == "all":
        outside_2d.append(f"{label}: 두 축의 모든 점이 정확히 0.5인 특수한 경우입니다.")
    else:
        outside_2d.append(f"{label}: 0.5 경계선이 현재 그림 범위 밖에 있습니다.")
fig2.update_layout(
    height=560, margin=dict(l=10, r=10, t=25, b=10),
    xaxis=dict(title=FEATURE_NAMES[x_axis], range=x_limits),
    yaxis=dict(title=FEATURE_NAMES[y_axis], range=y_limits),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(fixed_caption(fixed_2d))
for message in outside_2d:
    st.info(message)


# 3차원 0.5 평면은 선형식과 그래프 상자의 12개 모서리의 교점으로 정확하게 잘라 그립니다.
# 이렇게 그리면 어떤 축의 기울기가 0이어도 수직 평면을 표현할 수 있습니다.
def vector_dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def vector_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def unit_vector(vector):
    length = sqrt(vector_dot(vector, vector))
    return tuple(component / length for component in vector) if length else None


def intersect_plane_box(axes, ranges, bias, coefficients, fixed):
    """p=0.5 평면과 삼차원 표시 범위의 교차 다각형 꼭짓점을 반환합니다."""
    normal = [float(coefficients[axis]) for axis in axes]
    c = plane_parts(bias, coefficients, fixed)
    vertices = [
        (a, b, d)
        for a in ranges[0] for b in ranges[1] for d in ranges[2]
    ]
    edges = []
    for i, p in enumerate(vertices):
        for q in vertices[i + 1:]:
            if sum(p[k] != q[k] for k in range(3)) == 1:
                edges.append((p, q))
    found = []
    for p, q in edges:
        fp = vector_dot(normal, p) + c
        fq = vector_dot(normal, q) + c
        if abs(fp) <= 1e-10:
            found.append(p)
        if abs(fq) <= 1e-10:
            found.append(q)
        if fp * fq < 0:
            t = fp / (fp - fq)
            found.append(tuple(p[k] + t * (q[k] - p[k]) for k in range(3)))
    unique = []
    for point in found:
        if not any(sum((point[k] - other[k]) ** 2 for k in range(3)) < 1e-12 for other in unique):
            unique.append(point)
    if len(unique) < 3:
        return []
    normal_unit = unit_vector(normal)
    if normal_unit is None:
        return []
    reference = (0, 0, 1) if abs(normal_unit[2]) < 0.9 else (0, 1, 0)
    basis_u = unit_vector(vector_cross(normal_unit, reference))
    basis_v = vector_cross(normal_unit, basis_u)
    center = tuple(sum(p[k] for p in unique) / len(unique) for k in range(3))
    return sorted(
        unique,
        key=lambda p: atan2(
            vector_dot(tuple(p[k] - center[k] for k in range(3)), basis_v),
            vector_dot(tuple(p[k] - center[k] for k in range(3)), basis_u),
        ),
    )


if len(selected) >= 3:
    st.subheader("4. 회전해서 보는 3차원 산점도와 두 개의 0.5 평면")
    col3_x, col3_y, col3_z = st.columns(3)
    with col3_x:
        axis3_x = st.selectbox(
            "3차원 가로축", options=selected, index=0,
            format_func=lambda feature: FEATURE_NAMES[feature], key="performance_3x",
        )
    with col3_y:
        choices3_y = [feature for feature in selected if feature != axis3_x]
        axis3_y = st.selectbox(
            "3차원 세로축", options=choices3_y, index=0,
            format_func=lambda feature: FEATURE_NAMES[feature], key="performance_3y",
        )
    with col3_z:
        choices3_z = [feature for feature in selected if feature not in (axis3_x, axis3_y)]
        axis3_z = st.selectbox(
            "3차원 높이축", options=choices3_z, index=0,
            format_func=lambda feature: FEATURE_NAMES[feature], key="performance_3z",
        )

    axes3 = [axis3_x, axis3_y, axis3_z]
    fixed_3d = fix_other_features(axes3)
    ranges3 = [axis_range(feature) for feature in axes3]
    fig3 = go.Figure()
    for stroke_value in (0, 1):
            mask = y_test == stroke_value
        subset = X_test.loc[mask]
        fig3.add_trace(go.Scatter3d(
            x=subset[axis3_x], y=subset[axis3_y], z=subset[axis3_z],
            mode="markers",
            name="실제 아님(0)" if stroke_value == 0 else "실제 뇌졸중(1)",
            marker=dict(color=POINT_COLORS[stroke_value], size=3, opacity=0.5),
            customdata=test_data.loc[mask, "id"],
            hovertemplate=(
                "번호: %{customdata}<br>" + FEATURE_NAMES[axis3_x] + ": %{x:.2f}<br>"
                + FEATURE_NAMES[axis3_y] + ": %{y:.2f}<br>"
                + FEATURE_NAMES[axis3_z] + ": %{z:.2f}<extra>%{fullData.name}</extra>"
            ),
        ))

    outside_3d = []
    for label, bias, coef, color in (
        ("가중치 전 (p=0.5)", before_bias, before_coef, BEFORE_COLOR),
        ("가중치 후 (p=0.5)", after_bias, after_coef, AFTER_COLOR),
    ):
        polygon = intersect_plane_box(axes3, ranges3, bias, coef, fixed_3d)
        if not polygon:
            outside_3d.append(f"{label}: 0.5 평면이 현재 3차원 그림 범위 밖에 있거나 면적 없이 접합니다.")
            continue
        # 다각형을 부채꼴 삼각형으로 채워 투명한 실제 평면 조각을 만듭니다.
        fig3.add_trace(go.Mesh3d(
            x=[p[0] for p in polygon],
            y=[p[1] for p in polygon],
            z=[p[2] for p in polygon],
            i=[0] * (len(polygon) - 2),
            j=list(range(1, len(polygon) - 1)),
            k=list(range(2, len(polygon))),
            color=color, opacity=0.32, flatshading=True,
            name=label, showlegend=True,
            hoverinfo="name",
        ))
        rim = polygon + polygon[:1]
        fig3.add_trace(go.Scatter3d(
            x=[p[0] for p in rim], y=[p[1] for p in rim], z=[p[2] for p in rim],
            mode="lines", line=dict(color=color, width=4),
            showlegend=False, hoverinfo="skip",
        ))

    fig3.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=25, b=0),
        scene=dict(
            xaxis=dict(title=FEATURE_NAMES[axis3_x], range=ranges3[0]),
            yaxis=dict(title=FEATURE_NAMES[axis3_y], range=ranges3[1]),
            zaxis=dict(title=FEATURE_NAMES[axis3_z], range=ranges3[2]),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(fixed_caption(fixed_3d))
    for message in outside_3d:
        st.info(message)
    if any(feature in BINARY_FEATURES for feature in axes3):
        st.caption("고혈압·심장병은 실제로 0 또는 1만 가지며, 그 사이 공간의 평면은 수학적 시각화입니다.")
else:
    st.caption("입력 속성을 세 개 이상 고르면 회전할 수 있는 3차원 산점도와 0.5 평면도 표시합니다.")
