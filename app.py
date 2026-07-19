"""
AI-Powered Adaptive Tutor - Streamlit App
Chat-based tutoring, quizzes, and a progress dashboard that adapts to student pace
"""

import streamlit as st
import uuid
import time
import json
import re
import traceback
from typing import Dict, List

from langchain_groq import ChatGroq

# Import agent utilities
try:
    from agent import (
        create_agent,
        chat as agent_chat,
        reset_chat_history,
        score_quiz_answers,
        save_quiz_result,
        track_learning_progress,
        get_student_dashboard,
    )
except Exception as imp_err:
    create_agent = None
    agent_chat = None
    reset_chat_history = None
    score_quiz_answers = None
    save_quiz_result = None
    track_learning_progress = None
    get_student_dashboard = None
    IMPORT_ERROR = imp_err
else:
    IMPORT_ERROR = None

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Sage - Adaptive AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Session State Initialization
# =============================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "student_name" not in st.session_state:
    st.session_state.student_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

if "current_level" not in st.session_state:
    st.session_state.current_level = "Beginner"

if "current_subject" not in st.session_state:
    st.session_state.current_subject = ""

if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""

if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None

if "quiz_result" not in st.session_state:
    st.session_state.quiz_result = None

if "dashboard_data" not in st.session_state:
    st.session_state.dashboard_data = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None

# =============================================================================
# Theming (Dark / Light Mode)
# =============================================================================

def inject_theme_css(dark: bool):
    if dark:
        bg = "linear-gradient(135deg, #0a0e27 0%, #131a3a 100%)"
        text_color = "#e8edf4"
        card_bg = "rgba(255, 255, 255, 0.05)"
        card_border = "rgba(255, 255, 255, 0.1)"
        input_bg = "rgba(255, 255, 255, 0.08)"
        input_text = "#8ec7ff"
        muted_text = "#b8c2d8"
        accent1 = "#2563eb"
        accent2 = "#7c3aed"
    else:
        bg = "linear-gradient(135deg, #fff7ed 0%, #eef2ff 50%, #f0fdfa 100%)"
        text_color = "#1e1b2e"
        card_bg = "#ffffff"
        card_border = "rgba(124, 58, 237, 0.15)"
        input_bg = "#ffffff"
        input_text = "#1e1b2e"
        muted_text = "#5b5470"
        accent1 = "#7c3aed"
        accent2 = "#db2777"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {{ font-size: 17px; }}

        * {{ font-family: 'Sora', sans-serif; }}

        .stApp {{
            background: {bg};
            color: {text_color};
        }}

        .stApp, .stApp p, .stApp span, .stApp div, .stApp label,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {{
            color: {text_color} !important;
            font-size: 1.05rem;
        }}

        [data-testid="stMarkdownContainer"] h3 {{ font-size: 1.5rem !important; font-weight: 700 !important; }}
        [data-testid="stMarkdownContainer"] h4 {{ font-size: 1.25rem !important; font-weight: 700 !important; }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
        .stTabs [data-baseweb="tab"] {{
            color: {muted_text} !important;
            font-weight: 600;
            font-size: 1.05rem;
        }}
        .stTabs [aria-selected="true"] {{
            color: {accent1} !important;
            border-bottom-color: {accent1} !important;
        }}

        .stTextInput label, .stTextArea label, .stSelectbox label,
        .stRadio label, [data-testid="stWidgetLabel"] p {{
            color: {text_color} !important;
            font-weight: 600;
            font-size: 1rem;
        }}

        .stRadio [data-testid="stMarkdownContainer"] p {{
            color: {text_color} !important;
            font-size: 1.02rem;
        }}

        .header-container {{
            background: linear-gradient(135deg, {accent1} 0%, {accent2} 100%);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 40px rgba(124, 58, 237, 0.35);
        }}

        .main-title {{
            font-size: 2.8rem;
            font-weight: 800;
            color: white !important;
            margin: 0;
            letter-spacing: -0.5px;
        }}

        .subtitle {{
            font-size: 1.15rem;
            color: rgba(255, 255, 255, 0.95) !important;
            margin-top: 0.5rem;
            font-weight: 400;
        }}

        .card {{
            background: {card_bg};
            border: 1px solid {card_border};
            border-radius: 16px;
            padding: 1.6rem;
            margin: 1rem 0;
            box-shadow: {"none" if dark else "0 4px 20px rgba(124, 58, 237, 0.08)"};
            color: {text_color} !important;
        }}

        .card h4, .card p, .card li, .card ul, .card ol {{
            color: {text_color} !important;
            font-size: 1.05rem;
            line-height: 1.6;
        }}

        .level-card {{
            background: linear-gradient(135deg, {accent1} 0%, {accent2} 100%);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(124, 58, 237, 0.4);
        }}

        .level-text {{
            font-size: 2.4rem;
            font-weight: 800;
            color: white !important;
            font-family: 'JetBrains Mono', monospace;
        }}

        .level-label {{
            font-size: 1.05rem;
            color: rgba(255, 255, 255, 0.9) !important;
            margin-top: 0.3rem;
            font-weight: 500;
        }}

        .stat-tag {{
            display: inline-block;
            padding: 0.5rem 1rem;
            margin: 0.3rem;
            border-radius: 20px;
            font-size: 0.95rem;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
        }}

        .tag-strong {{ background: rgba(16, 185, 129, 0.15); color: #059669 !important; border: 1.5px solid rgba(5, 150, 105, 0.4); }}
        .tag-weak {{ background: rgba(239, 68, 68, 0.15); color: #dc2626 !important; border: 1.5px solid rgba(220, 38, 38, 0.4); }}
        .tag-average {{ background: rgba(245, 158, 11, 0.15); color: #d97706 !important; border: 1.5px solid rgba(217, 119, 6, 0.4); }}

        .chat-message {{
            background: {card_bg};
            border-radius: 12px;
            padding: 1.1rem;
            margin: 0.8rem 0;
            border-left: 4px solid {accent1};
            box-shadow: {"none" if dark else "0 2px 12px rgba(124, 58, 237, 0.08)"};
            color: {text_color} !important;
            font-size: 1.05rem;
            line-height: 1.6;
        }}

        .chat-message-user {{ border-left-color: {accent2}; }}
        .chat-message-ai {{ border-left-color: #059669; }}

        .streak-box {{
            background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
            border-radius: 16px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 6px 20px rgba(245, 158, 11, 0.35);
        }}

        .streak-number {{
            font-size: 2.6rem;
            font-weight: 800;
            color: white !important;
            font-family: 'JetBrains Mono', monospace;
        }}

        .progress-container {{
            background: {"rgba(255,255,255,0.08)" if dark else "#f1f0f7"};
            border-radius: 10px;
            height: 24px;
            margin: 0.4rem 0;
            overflow: hidden;
        }}

        .progress-bar {{
            background: linear-gradient(90deg, {accent1} 0%, {accent2} 100%);
            height: 100%;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white !important;
            font-weight: 700;
            font-size: 0.85rem;
        }}

        .stButton > button {{
            background: linear-gradient(135deg, {accent1} 0%, {accent2} 100%);
            color: white !important;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.9rem;
            font-weight: 700;
            font-size: 1rem;
            box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.55);
        }}

        .stButton > button p {{ color: white !important; font-size: 1rem; }}

        .stTextArea textarea, .stTextInput input {{
            background: {input_bg} !important;
            border: 1.5px solid {card_border} !important;
            border-radius: 12px;
            color: {input_text} !important;
            font-family: 'JetBrains Mono', monospace;
            font-size: 1rem !important;
        }}

        .info-box {{
            background: {"rgba(124, 58, 237, 0.12)" if dark else "rgba(124, 58, 237, 0.08)"};
            border: 1.5px solid rgba(124, 58, 237, 0.3);
            border-radius: 12px;
            padding: 1.1rem;
            margin: 1rem 0;
            color: {text_color} !important;
            font-size: 1.05rem;
        }}

        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, {card_border}, transparent);
            margin: 2rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Agent Initialization
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_agent_executor_cached():
    if create_agent is None:
        raise RuntimeError(f"Agent import failed: {IMPORT_ERROR}")
    return create_agent()


def ensure_agent_ready():
    if st.session_state.agent_ready and st.session_state.agent_executor:
        return st.session_state.agent_executor

    agent_exec = get_agent_executor_cached()
    st.session_state.agent_executor = agent_exec
    st.session_state.agent_ready = True
    return agent_exec


@st.cache_resource(show_spinner=False)
def get_quiz_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


def generate_quiz_json(subject: str, topic: str, level: str, num_questions: int = 6) -> List[Dict]:
    llm = get_quiz_llm()
    prompt = f"""You are an expert quiz generator for an adaptive tutoring system.
Subject: {subject}
Topic: {topic}
Student level: {level}

Generate exactly {num_questions} quiz questions on this topic, suited to a {level} student.
Include a mix of these types: "mcq", "true_false", "fill_blank", "short_answer".

Respond ONLY with a JSON array, nothing else, in this exact format:
[
  {{"id": "q1", "type": "mcq", "question": "text", "options": ["a","b","c","d"], "correct_answer": "a"}},
  {{"id": "q2", "type": "true_false", "question": "text", "options": ["True","False"], "correct_answer": "True"}},
  {{"id": "q3", "type": "fill_blank", "question": "text with ____", "options": [], "correct_answer": "word"}},
  {{"id": "q4", "type": "short_answer", "question": "text", "options": [], "correct_answer": "key points"}}
]"""
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    try:
        data = _extract_json(text)
        if isinstance(data, dict):
            data = data.get("questions", [])
        return data
    except Exception:
        return []


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    st.markdown("""
        <div class="header-container">
            <h1 class="main-title">🎓 Sage — Adaptive AI Tutor</h1>
            <p class="subtitle">Learns your pace, adjusts difficulty, and helps you actually understand</p>
        </div>
    """, unsafe_allow_html=True)


def render_level_card(level: str):
    st.markdown(f"""
        <div class="level-card">
            <div class="level-text">{level}</div>
            <div class="level-label">Current Level</div>
        </div>
    """, unsafe_allow_html=True)


def render_streak_box(current_streak: int, longest_streak: int):
    st.markdown(f"""
        <div class="streak-box">
            <div class="streak-number">🔥 {current_streak}</div>
            <div class="level-label">Day Streak (Best: {longest_streak})</div>
        </div>
    """, unsafe_allow_html=True)


def render_progress_bar(label: str, value: float, max_value: float = 100):
    percentage = min(100, (value / max_value) * 100) if max_value else 0
    st.markdown(f"""
        <div style="margin: 0.8rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="font-weight: 600;">{label}</span>
                <span style="color: #7c3aed; font-weight: 600;">{value:.1f}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {percentage}%;">{percentage:.1f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_topic_tags(title: str, topics: list, tag_class: str):
    if not topics:
        return
    st.markdown(f"**{title}**")
    tags_html = "".join([
        f'<span class="stat-tag {tag_class}">{t.get("topic", "")} ({t.get("avg_accuracy", 0)}%)</span>'
        for t in topics
    ])
    st.markdown(f'<div style="margin: 0.5rem 0 1rem 0;">{tags_html}</div>', unsafe_allow_html=True)


def render_chat_bubble(role: str, content: str):
    css_class = "chat-message-user" if role == "user" else "chat-message-ai"
    speaker = "You" if role == "user" else "🎓 Sage"
    st.markdown(f"""
        <div class="chat-message {css_class}">
            <strong>{speaker}:</strong><br>{content}
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main App
# =============================================================================

def main():
    inject_theme_css(st.session_state.dark_mode)
    render_header()

    top_col1, top_col2, top_col3 = st.columns([2, 2, 1])
    with top_col1:
        st.session_state.student_name = st.text_input(
            "Student name", value=st.session_state.student_name, placeholder="Enter your name to begin"
        )
    with top_col3:
        toggle_label = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
        st.session_state.dark_mode = st.toggle(toggle_label, value=st.session_state.dark_mode)

    if not st.session_state.agent_ready:
        with st.spinner("🚀 Waking up your tutor..."):
            try:
                ensure_agent_ready()
            except Exception as e:
                st.error(f"❌ Failed to initialize agent: {str(e)}")
                st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["🎓 Tutor Chat", "📝 Quiz", "📊 Dashboard", "ℹ️ About"])

    with tab1:
        if not st.session_state.student_name:
            st.markdown('<div class="info-box">👋 Enter your name above to start your tutoring session.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">Tell Sage your class, subject, and topic to begin. Sage will ask 3 quick questions to find your level, then teach at your pace.</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns([1, 4])
            with col_a:
                if st.button("🔄 New Session", use_container_width=True):
                    st.session_state.messages = []
                    if reset_chat_history:
                        reset_chat_history()
                    st.rerun()

            for msg in st.session_state.messages:
                render_chat_bubble(msg["role"], msg["content"])

            with st.form("tutor_chat_form", clear_on_submit=True):
                user_input = st.text_area(
                    "Message Sage",
                    height=90,
                    placeholder="e.g. My name is Riya, I'm in class 10, I want to study Physics, topic: Newton's Laws",
                    label_visibility="collapsed"
                )
                submit = st.form_submit_button("Send 📤", use_container_width=True)

            if submit and user_input.strip():
                st.session_state.messages.append({"role": "user", "content": user_input})

                with st.spinner("🎓 Sage is thinking..."):
                    try:
                        agent_exec = st.session_state.agent_executor or ensure_agent_ready()
                        contextual_input = (
                            f"[Student name: {st.session_state.student_name}] {user_input}"
                        )
                        response = agent_chat(contextual_input, agent_exec)
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        level_match = re.search(r"\b(Beginner|Intermediate|Advanced)\b", response)
                        if level_match:
                            st.session_state.current_level = level_match.group(1)

                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.session_state.last_error = traceback.format_exc()

    with tab2:
        if not st.session_state.student_name:
            st.markdown('<div class="info-box">👋 Enter your name above first.</div>', unsafe_allow_html=True)
        else:
            st.markdown("### 📝 Generate a Practice Quiz")

            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                quiz_subject = st.text_input("Subject", value=st.session_state.current_subject, key="quiz_subject")
            with qc2:
                quiz_topic = st.text_input("Topic", value=st.session_state.current_topic, key="quiz_topic")
            with qc3:
                quiz_level = st.selectbox(
                    "Level", ["Beginner", "Intermediate", "Advanced"],
                    index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.current_level)
                )

            if st.button("✨ Generate Quiz"):
                if not quiz_subject or not quiz_topic:
                    st.warning("⚠️ Please enter both subject and topic.")
                else:
                    with st.spinner("🧠 Building your quiz..."):
                        questions = generate_quiz_json(quiz_subject, quiz_topic, quiz_level, 6)
                        if questions:
                            st.session_state.current_quiz = {
                                "subject": quiz_subject,
                                "topic": quiz_topic,
                                "level": quiz_level,
                                "questions": questions
                            }
                            st.session_state.current_subject = quiz_subject
                            st.session_state.current_topic = quiz_topic
                            st.session_state.quiz_result = None
                            st.success("✅ Quiz ready!")
                        else:
                            st.error("❌ Could not generate quiz. Please try again.")

            if st.session_state.current_quiz:
                quiz = st.session_state.current_quiz
                st.markdown("---")
                st.markdown(f"### {quiz['subject']} — {quiz['topic']} ({quiz['level']})")

                with st.form("quiz_form"):
                    student_answers = {}
                    for q in quiz["questions"]:
                        st.markdown(f"**{q['question']}**")
                        if q["type"] == "mcq" and q.get("options"):
                            student_answers[q["id"]] = st.radio(
                                "Choose one", q["options"], key=f"q_{q['id']}", label_visibility="collapsed"
                            )
                        elif q["type"] == "true_false":
                            student_answers[q["id"]] = st.radio(
                                "True or False", ["True", "False"], key=f"q_{q['id']}", label_visibility="collapsed"
                            )
                        else:
                            student_answers[q["id"]] = st.text_input(
                                "Your answer", key=f"q_{q['id']}", label_visibility="collapsed"
                            )
                        st.markdown("")

                    quiz_submit = st.form_submit_button("✅ Submit Quiz", use_container_width=True)

                if quiz_submit:
                    with st.spinner("📊 Grading your quiz..."):
                        try:
                            grading = score_quiz_answers.invoke({
                                "quiz_json": json.dumps(quiz["questions"]),
                                "student_answers_json": json.dumps(student_answers)
                            })
                            grading_data = json.loads(grading)

                            save_quiz_result.invoke({
                                "student_name": st.session_state.student_name,
                                "subject": quiz["subject"],
                                "topic": quiz["topic"],
                                "level": quiz["level"],
                                "score": grading_data["score"],
                                "total": grading_data["total"]
                            })

                            track_learning_progress.invoke({
                                "student_name": st.session_state.student_name,
                                "subject": quiz["subject"],
                                "topic": quiz["topic"],
                                "level": quiz["level"],
                                "accuracy": grading_data["accuracy"]
                            })

                            st.session_state.quiz_result = grading_data
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Grading failed: {str(e)}")

            if st.session_state.quiz_result:
                result = st.session_state.quiz_result
                st.markdown("---")
                st.markdown("## 📊 Quiz Result")

                rc1, rc2 = st.columns([1, 2])
                with rc1:
                    render_level_card(f"{result['score']}/{result['total']}")
                with rc2:
                    render_progress_bar("Accuracy", result["accuracy"])

                st.markdown("### Answer Breakdown")
                for ans in result["graded_answers"]:
                    icon = "✅" if ans["is_correct"] else "❌"
                    st.markdown(f"""
                        <div class="card">
                            {icon} <strong>{ans['type'].replace('_', ' ').title()}</strong><br>
                            Your answer: {ans['student_answer'] or '(blank)'}<br>
                            Correct answer: {ans['correct_answer']}
                        </div>
                    """, unsafe_allow_html=True)

    with tab3:
        if not st.session_state.student_name:
            st.markdown('<div class="info-box">👋 Enter your name above first.</div>', unsafe_allow_html=True)
        else:
            if st.button("🔄 Refresh Dashboard"):
                st.session_state.dashboard_data = None

            if st.session_state.dashboard_data is None:
                with st.spinner("📊 Loading dashboard..."):
                    try:
                        raw = get_student_dashboard.invoke({"student_name": st.session_state.student_name})
                        st.session_state.dashboard_data = json.loads(raw)
                    except Exception as e:
                        st.error(f"❌ Could not load dashboard: {str(e)}")
                        st.session_state.dashboard_data = {}

            dash = st.session_state.dashboard_data or {}

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                render_level_card(dash.get("level", "Beginner"))
            with dc2:
                render_streak_box(
                    dash.get("streak", {}).get("current_streak", 0),
                    dash.get("streak", {}).get("longest_streak", 0)
                )
            with dc3:
                st.markdown('<div class="card" style="text-align:center;">', unsafe_allow_html=True)
                st.markdown(f'<div class="level-text" style="color:#059669 !important;">{dash.get("overall_accuracy", 0)}%</div>', unsafe_allow_html=True)
                st.markdown('<div class="level-label" style="color:inherit !important;">Overall Accuracy</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📈 Topic Progress")
            for p in dash.get("progress", []):
                render_progress_bar(f"{p['subject']} — {p['topic']}", p["avg_accuracy"])

            st.markdown("---")
            cold1, cold2 = st.columns(2)
            with cold1:
                render_topic_tags("💪 Strong Topics", dash.get("strong_topics", []), "tag-strong")
            with cold2:
                render_topic_tags("⚠️ Weak Topics", dash.get("weak_topics", []), "tag-weak")

            st.markdown("---")
            st.markdown("### 🗂️ Recent Quiz History")
            history = dash.get("quiz_history", [])
            if not history:
                st.markdown('<div class="info-box">No quizzes taken yet. Head to the Quiz tab!</div>', unsafe_allow_html=True)
            else:
                for h in reversed(history):
                    st.markdown(f"""
                        <div class="card">
                            <strong>{h['subject']} — {h['topic']}</strong> ({h['level']})<br>
                            Score: {h['score']}/{h['total']} &nbsp;|&nbsp; Accuracy: {h['accuracy']}%
                        </div>
                    """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### About Sage")
        st.markdown("""
        <div class="card">
        <h4>🎯 What Sage Does</h4>
        <p>Sage is an adaptive AI tutor that finds your current knowledge level through a
        short diagnostic, then teaches at exactly your pace — simplifying when you're stuck,
        and leveling up when you're ready.</p>
        <ul>
            <li><strong>Diagnostic Assessment:</strong> 3 quick questions to detect Beginner / Intermediate / Advanced level</li>
            <li><strong>Adaptive Explanations:</strong> Re-explains with a new example if you don't understand</li>
            <li><strong>Auto-Generated Quizzes:</strong> MCQ, True/False, Fill-in-the-blank, and Short Answer</li>
            <li><strong>Progress Dashboard:</strong> Accuracy, streaks, weak topics, strong topics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h4>🚀 How to Use</h4>
        <ol>
            <li>Enter your name at the top</li>
            <li>Go to <strong>Tutor Chat</strong> and tell Sage your class, subject, and topic</li>
            <li>Answer the 3 diagnostic questions</li>
            <li>Learn at your level — say "Yes" or "No" after each explanation</li>
            <li>Head to <strong>Quiz</strong> to test yourself, and check <strong>Dashboard</strong> for progress</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h4>⚙️ Powered By</h4>
        <ul>
            <li><strong>Groq LLM:</strong> Fast tutoring responses</li>
            <li><strong>LangChain:</strong> Tool-calling agent framework</li>
            <li><strong>Tavily Search:</strong> Real-world example research</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()