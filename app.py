import streamlit as st
from datetime import datetime, timedelta
import json

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="🤖 AI 개인 비서",
    page_icon="🤖",
    layout="wide"
)

# ==================== CSS 스타일링 ====================
st.markdown("""
<style>
    .stChat message {
        padding: 1rem;
        border-radius: 10px;
    }
    .sidebar-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .schedule-item {
        padding: 0.5rem;
        margin: 0.3rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 5px;
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

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=10
    )

# ==================== LLM 및 임베딩 초기화 ====================
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True
    )

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )

# ==================== RAG 문서 생성 ====================
def create_knowledge_documents():
    """비서 역할에 필요한 기본 지식 문서 생성"""

    current_date = datetime.now().strftime("%Y년 %m월 %d일")

    # 사용자 데이터를 문서로 변환
    schedule_text = "## 현재 등록된 일정:\n"
    if st.session_state.schedules:
        for s in st.session_state.schedules:
            schedule_text += f"- {s['date']} {s['time']}: {s['title']} ({s.get('description', '')})\n"
    else:
        schedule_text += "- 등록된 일정이 없습니다.\n"

    todo_text = "## 현재 할 일 목록:\n"
    if st.session_state.todos:
        for t in st.session_state.todos:
            status = "✅ 완료" if t.get('done') else "⏳ 진행중"
            todo_text += f"- [{status}] {t['task']} (우선순위: {t.get('priority', '보통')})\n"
    else:
        todo_text += "- 등록된 할 일이 없습니다.\n"

    health_text = "## 건강 기록:\n"
    if st.session_state.health_records:
        for h in st.session_state.health_records:
            health_text += f"- {h['date']}: {h['type']} - {h['value']}\n"
    else:
        health_text += "- 등록된 건강 기록이 없습니다.\n"

    documents = [
        Document(
            page_content=f"""
            # AI 비서 기본 정보
            오늘 날짜: {current_date}

            당신은 사용자의 개인 비서입니다. 다음과 같은 역할을 수행합니다:
            1. 일정 관리: 일정 추가, 조회, 알림
            2. 할 일 관리: 할 일 추가, 완료 처리, 우선순위 관리
            3. 건강 관리: 건강 기록 추적, 건강 팁 제공
            4. 일반 대화: 친근하고 도움이 되는 대화

            모르는 정보에 대해서는 솔직하게 "잘 모르겠습니다"라고 답변합니다.
            """,
            metadata={"source": "system_info"}
        ),
        Document(
            page_content=schedule_text,
            metadata={"source": "user_schedules"}
        ),
        Document(
            page_content=todo_text,
            metadata={"source": "user_todos"}
        ),
        Document(
            page_content=health_text,
            metadata={"source": "user_health"}
        ),
        Document(
            page_content="""
            # 건강 관리 팁

            ## 일반 건강 권장사항:
            - 하루 물 섭취량: 2L 이상
            - 권장 수면 시간: 7-9시간
            - 하루 걸음 수 목표: 10,000보
            - 스트레칭: 1시간마다 5분

            ## 식사 권장:
            - 아침: 단백질과 복합 탄수화물
            - 점심: 균형 잡힌 식단
            - 저녁: 가벼운 식사, 취침 3시간 전 완료

            ## 운동 권장:
            - 유산소: 주 3-5회, 30분 이상
            - 근력운동: 주 2-3회
            - 스트레칭: 매일
            """,
            metadata={"source": "health_tips"}
        ),
        Document(
            page_content="""
            # 생산성 팁

            ## 시간 관리:
            - 포모도로 기법: 25분 집중 + 5분 휴식
            - 중요한 일은 아침에 처리
            - 2분 규칙: 2분 내 완료 가능한 일은 즉시 처리

            ## 우선순위 설정:
            - 긴급하고 중요한 일: 즉시 처리
            - 중요하지만 긴급하지 않은 일: 계획 수립
            - 긴급하지만 중요하지 않은 일: 위임 고려
            - 둘 다 아닌 일: 제거 고려
            """,
            metadata={"source": "productivity_tips"}
        )
    ]

    return documents

def update_vectorstore():
    """벡터 스토어 업데이트"""
    documents = create_knowledge_documents()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)

# ==================== 대화 체인 설정 ====================
def get_conversation_chain():
    """대화 체인 생성"""

    if st.session_state.vectorstore is None:
        update_vectorstore()

    llm = get_llm()

    prompt_template = """당신은 친절하고 유능한 AI 개인 비서입니다.
사용자의 일정, 할 일, 건강 관리를 도와주세요.

다음 맥락 정보를 참고하여 답변하세요:
{context}

이전 대화 기록:
{chat_history}

중요 규칙:
1. 모르는 것에 대해서는 솔직하게 "잘 모르겠습니다"라고 답변하세요.
2. 사용자가 일정이나 할 일을 추가하고 싶어하면, 사이드바에서 추가할 수 있다고 안내하세요.
3. 건강 관련 조언은 일반적인 정보만 제공하고, 의료 전문가 상담을 권장하세요.
4. 친근하고 도움이 되는 어조를 유지하세요.
5. 한국어로 답변하세요.

사용자 질문: {question}

답변:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )

    return chain

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
                update_vectorstore()
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
                    update_vectorstore()
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
                update_vectorstore()
                st.success("할 일이 추가되었습니다!")
                st.rerun()

    # 현재 할 일 표시
    if st.session_state.todos:
        st.markdown("**📝 할 일 목록:**")
        for i, t in enumerate(st.session_state.todos):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                status = "✅" if t['done'] else "⏳"
                st.markdown(f"{status} {t['task']} ({t['priority']})")
            with col2:
                if st.button("✔️", key=f"done_todo_{i}"):
                    st.session_state.todos[i]['done'] = not st.session_state.todos[i]['done']
                    update_vectorstore()
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"del_todo_{i}"):
                    st.session_state.todos.pop(i)
                    update_vectorstore()
                    st.rerun()

    st.divider()

    # 건강 기록 섹션
    st.subheader("💪 건강 기록")
    with st.form("health_form", clear_on_submit=True):
        health_type = st.selectbox("기록 유형", ["체중(kg)", "수면시간(시간)", "걸음수", "물섭취(L)", "운동(분)"])
        health_value = st.number_input("값", min_value=0.0, step=0.1)

        if st.form_submit_button("📊 기록 추가", use_container_width=True):
            st.session_state.health_records.append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "type": health_type,
                "value": health_value
            })
            update_vectorstore()
            st.success("건강 기록이 추가되었습니다!")
            st.rerun()

    st.divider()

    # 초기화 버튼
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
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
            try:
                chain = get_conversation_chain()
                response = chain.invoke({"question": prompt})
                answer = response["answer"]

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

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
