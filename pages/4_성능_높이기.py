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
