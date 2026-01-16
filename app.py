import streamlit as st
from datetime import datetime
import google.generativeai as genai

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="🤖 AI 개인 비서",
    page_icon="🤖",
    layout="wide"
)

# ==================== CSS 스타일링 ====================
st.markdown("""
<style>
    .schedule-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 5px;
        font-size: 0.9rem;
    }
    .todo-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        background: #ff6b6b;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== API 키 설정 ====================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 추가해주세요.")
    st.stop()

# ==================== 세션 상태 초기화 ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "schedules" not in st.session_state:
    st.session_state.schedules = []

if "todos" not in st.session_state:
    st.session_state.todos = []

if "health_records" not in st.session_state:
    st.session_state.health_records = []

# ==================== Gemini 모델 초기화 ====================
@st.cache_resource
def get_model():
    return genai.GenerativeModel("gemini-2.5-flash")

# ==================== 컨텍스트 생성 ====================
def get_context():
    """사용자 데이터를 컨텍스트로 변환"""
    current_date = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

    # 일정 정보
    schedule_text = "📅 등록된 일정:\n"
    if st.session_state.schedules:
        for s in st.session_state.schedules:
            schedule_text += f"  - {s['date']} {s['time']}: {s['title']}"
            if s.get('description'):
                schedule_text += f" ({s['description']})"
            schedule_text += "\n"
    else:
        schedule_text += "  - 없음\n"

    # 할 일 정보
    todo_text = "✅ 할 일 목록:\n"
    if st.session_state.todos:
        for t in st.session_state.todos:
            status = "완료" if t.get('done') else "진행중"
            todo_text += f"  - [{status}] {t['task']} (우선순위: {t.get('priority', '보통')})\n"
    else:
        todo_text += "  - 없음\n"

    # 건강 기록
    health_text = "💪 건강 기록:\n"
    if st.session_state.health_records:
        for h in st.session_state.health_records[-5:]:  # 최근 5개만
            health_text += f"  - {h['date']}: {h['type']} = {h['value']}\n"
    else:
        health_text += "  - 없음\n"

    context = f"""현재 시간: {current_date}

{schedule_text}
{todo_text}
{health_text}
"""
    return context

# ==================== 대화 기록 포맷 ====================
def get_chat_history():
    """최근 대화 기록 반환 (최근 10개)"""
    history = []
    for msg in st.session_state.messages[-10:]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})
    return history

# ==================== AI 응답 생성 ====================
def get_ai_response(user_input):
    """Gemini API로 응답 생성"""

    system_prompt = f"""당신은 친절하고 유능한 AI 개인 비서입니다.

## 사용자 정보:
{get_context()}

## 역할:
1. 일정 관리: 등록된 일정 확인 및 안내
2. 할 일 관리: 할 일 목록 확인 및 우선순위 추천
3. 건강 관리: 건강 기록 확인 및 일반적인 건강 팁 제공
4. 일반 대화: 친근하고 도움이 되는 대화

## 규칙:
- 모르는 것은 솔직하게 "잘 모르겠습니다"라고 답변
- 일정/할일 추가는 "왼쪽 사이드바에서 추가할 수 있습니다"라고 안내
- 의료 조언은 일반적인 정보만 제공하고 전문가 상담 권장
- 항상 한국어로 답변
- 친근하고 도움이 되는 어조 유지
"""

    try:
        model = get_model()
        chat = model.start_chat(history=get_chat_history())

        response = chat.send_message(
            f"{system_prompt}\n\n사용자: {user_input}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
        )
        return response.text
    except Exception as e:
        return f"죄송합니다. 오류가 발생했습니다: {str(e)}"

# ==================== 사이드바 UI ====================
with st.sidebar:
    st.title("📋 비서 관리 패널")

    # 일정 관리 섹션
    st.subheader("📅 일정 추가")
    with st.form("schedule_form", clear_on_submit=True):
        schedule_date = st.date_input("날짜", datetime.now())
        schedule_time = st.time_input("시간")
        schedule_title = st.text_input("일정 제목")
        schedule_desc = st.text_area("설명 (선택)", height=68)

        if st.form_submit_button("✅ 일정 추가", use_container_width=True):
            if schedule_title:
                st.session_state.schedules.append({
                    "date": schedule_date.strftime("%Y-%m-%d"),
                    "time": schedule_time.strftime("%H:%M"),
                    "title": schedule_title,
                    "description": schedule_desc
                })
                st.success("일정이 추가되었습니다!")
                st.rerun()

    # 현재 일정 표시
    if st.session_state.schedules:
        st.markdown("**📆 등록된 일정:**")
        for i, s in enumerate(st.session_state.schedules):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"<div class='schedule-item'>📌 {s['date']} {s['time']}<br>{s['title']}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_schedule_{i}"):
                    st.session_state.schedules.pop(i)
                    st.rerun()

    st.divider()

    # 할 일 관리 섹션
    st.subheader("✅ 할 일 추가")
    with st.form("todo_form", clear_on_submit=True):
        todo_task = st.text_input("할 일")
        todo_priority = st.selectbox("우선순위", ["높음", "보통", "낮음"])

        if st.form_submit_button("➕ 할 일 추가", use_container_width=True):
            if todo_task:
                st.session_state.todos.append({
                    "task": todo_task,
                    "priority": todo_priority,
                    "done": False
                })
                st.success("할 일이 추가되었습니다!")
                st.rerun()

    # 현재 할 일 표시
    if st.session_state.todos:
        st.markdown("**📝 할 일 목록:**")
        for i, t in enumerate(st.session_state.todos):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status = "✅" if t['done'] else "⏳"
                st.markdown(f"{status} {t['task']}")
            with col2:
                if st.button("✔️", key=f"done_todo_{i}"):
                    st.session_state.todos[i]['done'] = not st.session_state.todos[i]['done']
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"del_todo_{i}"):
                    st.session_state.todos.pop(i)
                    st.rerun()

    st.divider()

    # 건강 기록 섹션
    st.subheader("💪 건강 기록")
    with st.form("health_form", clear_on_submit=True):
        health_type = st.selectbox("기록 유형", ["체중(kg)", "수면(시간)", "걸음수", "물(L)", "운동(분)"])
        health_value = st.number_input("값", min_value=0.0, step=0.1)

        if st.form_submit_button("📊 기록 추가", use_container_width=True):
            st.session_state.health_records.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "type": health_type,
                "value": health_value
            })
            st.success("건강 기록이 추가되었습니다!")
            st.rerun()

    st.divider()

    # 초기화 버튼
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==================== 메인 채팅 UI ====================
st.title("🤖 AI 개인 비서")
st.caption("일정 관리, 할 일 추천, 건강 관리를 도와드립니다!")

# 채팅 히스토리 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("무엇을 도와드릴까요?"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = get_ai_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# 초기 안내 메시지
if not st.session_state.messages:
    st.info("""
    👋 안녕하세요! AI 개인 비서입니다.

    **도움 드릴 수 있는 것들:**
    - 📅 일정 관리 (왼쪽 사이드바에서 추가)
    - ✅ 할 일 관리 및 추천
    - 💪 건강 관리 팁
    - 💬 일반적인 질문과 대화

    무엇이든 물어보세요!
    """)
