
import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/ehdrjs925/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "데이터_소개_기록.json"

st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide")
st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 관련 데이터의 구성과 결측값을 살펴보고, 교재를 참고하여 각 열의 의미와 출처를 기록하는 공간입니다.")


@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_record():
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if not isinstance(loaded, dict):
            return {"meanings": {}, "source": ""}
        return {
            "meanings": loaded.get("meanings", {}) if isinstance(loaded.get("meanings", {}), dict) else {},
            "source": str(loaded.get("source", "")),
        }
    except (OSError, json.JSONDecodeError):
        st.warning("이전에 저장한 기록을 읽지 못했습니다. 새 기록을 작성할 수 있습니다.")
        return {"meanings": {}, "source": ""}


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 GitHub CSV 주소를 확인해 주세요.")
    st.caption(f"오류 정보: {error}")
    st.stop()

if "stroke" not in df.columns:
    st.error("CSV 파일에 'stroke' 열이 없습니다. 데이터 주소를 확인해 주세요.")
    st.stop()

people = len(df)
stroke_count = int((pd.to_numeric(df["stroke"], errors="coerce") == 1).sum())
stroke_rate = stroke_count / people * 100 if people else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 사람 수", f"{people:,}명")
c2.metric("열 개수", f"{len(df.columns):,}개")
c3.metric("stroke = 1인 사람 수", f"{stroke_count:,}명")
c4.metric("stroke = 1 비율", f"{stroke_rate:.2f}%")

st.subheader("열 정보와 우리말 뜻")
st.caption("'우리말 뜻' 칸을 직접 수정한 뒤 아래의 '기록 저장' 버튼을 누르세요. 빈 값 개수는 CSV의 실제 결측값을 기준으로 계산됩니다.")

if "record" not in st.session_state:
    st.session_state.record = load_record()

record = st.session_state.record
meanings = record["meanings"]
info = pd.DataFrame({
    "열 이름": df.columns.tolist(),
    "우리말 뜻": [str(meanings.get(col, "")) for col in df.columns],
    "값의 종류": [str(df[col].dtype) for col in df.columns],
    "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns],
})

with st.form("record_form"):
    edited = st.data_editor(
        info,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 작성하세요."),
            "값의 종류": st.column_config.TextColumn("값의 종류"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
        },
        key="meaning_editor",
    )

    st.subheader("데이터 미리보기 (처음 5행)")
    st.dataframe(df.head(5), hide_index=True, use_container_width=True)

    st.subheader("데이터 출처 기록")
    source = st.text_area("교재를 참고하여 데이터 출처를 직접 적어 주세요.", value=record["source"], height=110)
    submitted = st.form_submit_button("기록 저장", type="primary")

if submitted:
    new_record = {
        "meanings": {
            str(row["열 이름"]): "" if pd.isna(row["우리말 뜻"]) else str(row["우리말 뜻"])
            for _, row in edited.iterrows()
        },
        "source": source,
    }
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        with RECORD_FILE.open("w", encoding="utf-8") as file:
            json.dump(new_record, file, ensure_ascii=False, indent=2)
        st.session_state.record = new_record
        st.success(f"저장 완료: {RECORD_FILE}")
    except OSError as error:
        st.error(f"파일을 저장하지 못했습니다. 저장 경로의 쓰기 권한을 확인해 주세요: {error}")

st.caption("참고: 파일 저장 내용은 현재 앱이 실행되는 환경에 보관됩니다. 일부 클라우드 배포 환경에서는 재시작·재배포 시 삭제될 수 있습니다.")
