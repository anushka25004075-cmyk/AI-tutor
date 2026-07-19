"""
AI-Powered Adaptive Tutoring Agent
Adjusts explanations and difficulty to a student's pace using LangChain with Groq LLM
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import sys
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global chat history
chat_history = []

# Local JSON "database" file for student progress (no external DB in this build)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_data.json")

# =============================================================================
# STEP 0: Local Persistence Helpers
# =============================================================================

def _load_data() -> Dict:
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_data(data: Dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _get_student_record(student_name: str) -> Dict:
    data = _load_data()
    key = student_name.strip().lower()
    if key not in data:
        data[key] = {
            "name": student_name,
            "level": "Beginner",
            "progress": {},          # {"subject|topic": {"attempts":n,"avg_accuracy":f,"status":s}}
            "quiz_results": [],
            "chat_log": [],
            "streak": {"current_streak": 0, "longest_streak": 0, "last_active_date": None},
        }
        _save_data(data)
    return data[key]


def _update_student_record(student_name: str, record: Dict) -> None:
    data = _load_data()
    key = student_name.strip().lower()
    data[key] = record
    _save_data(data)


def _classify_status(accuracy: float) -> str:
    if accuracy >= 75:
        return "strong"
    elif accuracy >= 40:
        return "average"
    else:
        return "weak"


def _record_streak(record: Dict) -> int:
    today = datetime.utcnow().date()
    last_active = record["streak"].get("last_active_date")
    last_date = datetime.fromisoformat(last_active).date() if last_active else None

    if last_date == today:
        return record["streak"]["current_streak"]

    if last_date == today - timedelta(days=1):
        record["streak"]["current_streak"] += 1
    else:
        record["streak"]["current_streak"] = 1

    record["streak"]["longest_streak"] = max(
        record["streak"]["current_streak"], record["streak"]["longest_streak"]
    )
    record["streak"]["last_active_date"] = today.isoformat()
    return record["streak"]["current_streak"]


# =============================================================================
# STEP 1: Define Custom Tools for Adaptive Tutoring
# =============================================================================

@tool
def assess_diagnostic_level(diagnostic_answers: str, subject: str, topic: str) -> str:
    """
    Analyze a student's 3 diagnostic answers and suggest a knowledge level.

    Args:
        diagnostic_answers: JSON array string of the student's 3 diagnostic answers, e.g. '["ans1","ans2","ans3"]'
        subject: The subject being studied
        topic: The topic being studied

    Returns:
        JSON string with suggested level (Beginner/Intermediate/Advanced) and a score breakdown
    """
    try:
        try:
            answers = json.loads(diagnostic_answers)
            if not isinstance(answers, list):
                answers = [str(answers)]
        except Exception:
            answers = [a.strip() for a in diagnostic_answers.split("|") if a.strip()]

        technical_terms = {
            "algorithm", "function", "variable", "equation", "theorem", "hypothesis",
            "derivative", "integral", "compound", "molecule", "reaction", "vector",
            "matrix", "protocol", "syntax", "recursion", "complexity", "coefficient",
            "photosynthesis", "mitochondria", "ecosystem", "democracy", "constitution",
            "inflation", "gdp", "polynomial", "probability", "genome", "catalyst"
        }

        total_length = sum(len(a.split()) for a in answers)
        avg_length = total_length / len(answers) if answers else 0

        technical_hits = sum(
            1 for a in answers for w in re.findall(r"[a-zA-Z]+", a.lower()) if w in technical_terms
        )

        vague_markers = ["i don't know", "not sure", "no idea", "idk", "maybe", "i think so"]
        vague_count = sum(1 for a in answers if any(m in a.lower() for m in vague_markers))

        length_score = min(40, avg_length * 3)
        technical_score = min(40, technical_hits * 15)
        vague_penalty = vague_count * 15

        raw_score = max(0, length_score + technical_score - vague_penalty)

        if raw_score >= 55:
            suggested_level = "Advanced"
        elif raw_score >= 25:
            suggested_level = "Intermediate"
        else:
            suggested_level = "Beginner"

        result = {
            "suggested_level": suggested_level,
            "raw_score": round(raw_score, 1),
            "avg_answer_length_words": round(avg_length, 1),
            "technical_terms_used": technical_hits,
            "vague_answers": vague_count,
            "subject": subject,
            "topic": topic
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Level assessment failed: {str(e)}"})


@tool
def score_quiz_answers(quiz_json: str, student_answers_json: str) -> str:
    """
    Grade a student's quiz submission against the correct answers.

    Args:
        quiz_json: JSON string of the quiz questions list, each with id, type, question, correct_answer
        student_answers_json: JSON string mapping question id -> student's answer

    Returns:
        JSON string with score, total, accuracy, and per-question grading detail
    """
    try:
        questions = json.loads(quiz_json)
        student_answers = json.loads(student_answers_json)

        if isinstance(questions, dict):
            questions = questions.get("questions", [])

        score = 0
        total = len(questions)
        graded = []

        for q in questions:
            qid = q.get("id")
            qtype = q.get("type", "short_answer")
            correct = str(q.get("correct_answer", "")).strip().lower()
            student_ans = str(student_answers.get(qid, "")).strip()

            is_correct = False

            if qtype in ("mcq", "true_false", "fill_blank"):
                is_correct = student_ans.lower() == correct
            else:
                correct_words = set(re.findall(r"[a-zA-Z]+", correct.lower()))
                student_words = set(re.findall(r"[a-zA-Z]+", student_ans.lower()))
                overlap = correct_words.intersection(student_words)
                is_correct = len(overlap) >= max(1, len(correct_words) // 2) and len(student_ans) > 0

            if is_correct:
                score += 1

            graded.append({
                "id": qid,
                "type": qtype,
                "student_answer": student_ans,
                "correct_answer": q.get("correct_answer"),
                "is_correct": is_correct
            })

        accuracy = round((score / total) * 100, 1) if total > 0 else 0

        result = {
            "score": score,
            "total": total,
            "accuracy": accuracy,
            "graded_answers": graded
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Quiz scoring failed: {str(e)}"})


@tool
def track_learning_progress(student_name: str, subject: str, topic: str, level: str, accuracy: float) -> str:
    """
    Save a learning attempt to the student's persistent progress record and update their streak.

    Args:
        student_name: The student's name (used as their record key)
        subject: The subject studied
        topic: The topic studied
        level: The student's current level (Beginner/Intermediate/Advanced)
        accuracy: Accuracy percentage (0-100) for this attempt

    Returns:
        JSON string confirming the updated progress, status classification, and streak
    """
    try:
        record = _get_student_record(student_name)
        record["level"] = level

        key = f"{subject.strip().lower()}|{topic.strip().lower()}"
        topic_progress = record["progress"].get(key, {"attempts": 0, "avg_accuracy": 0, "status": "average"})

        attempts = topic_progress["attempts"] + 1
        avg_accuracy = ((topic_progress["avg_accuracy"] * topic_progress["attempts"]) + accuracy) / attempts
        status = _classify_status(avg_accuracy)

        record["progress"][key] = {
            "subject": subject,
            "topic": topic,
            "attempts": attempts,
            "avg_accuracy": round(avg_accuracy, 1),
            "status": status
        }

        streak = _record_streak(record)
        _update_student_record(student_name, record)

        return json.dumps({
            "subject": subject,
            "topic": topic,
            "attempts": attempts,
            "avg_accuracy": round(avg_accuracy, 1),
            "status": status,
            "current_streak": streak
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Progress tracking failed: {str(e)}"})


@tool
def get_student_dashboard(student_name: str) -> str:
    """
    Retrieve a student's full dashboard summary: level, accuracy, weak/strong topics, streak, quiz history.

    Args:
        student_name: The student's name

    Returns:
        JSON string with the dashboard summary
    """
    try:
        record = _get_student_record(student_name)
        progress_list = list(record["progress"].values())

        weak_topics = [p for p in progress_list if p["status"] == "weak"]
        strong_topics = [p for p in progress_list if p["status"] == "strong"]

        total_attempts = sum(p["attempts"] for p in progress_list)
        overall_accuracy = (
            round(sum(p["avg_accuracy"] * p["attempts"] for p in progress_list) / total_attempts, 1)
            if total_attempts > 0 else 0
        )

        result = {
            "name": record["name"],
            "level": record["level"],
            "overall_accuracy": overall_accuracy,
            "streak": record["streak"],
            "progress": progress_list,
            "weak_topics": weak_topics,
            "strong_topics": strong_topics,
            "quiz_history": record["quiz_results"][-10:]
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Dashboard fetch failed: {str(e)}"})


@tool
def save_quiz_result(student_name: str, subject: str, topic: str, level: str, score: int, total: int) -> str:
    """
    Save a completed quiz result to the student's persistent record.

    Args:
        student_name: The student's name
        subject: The subject of the quiz
        topic: The topic of the quiz
        level: The level the quiz was generated at
        score: Number of correct answers
        total: Total number of questions

    Returns:
        JSON string confirming the saved result
    """
    try:
        record = _get_student_record(student_name)
        accuracy = round((score / total) * 100, 1) if total > 0 else 0

        entry = {
            "subject": subject,
            "topic": topic,
            "level": level,
            "score": score,
            "total": total,
            "accuracy": accuracy,
            "timestamp": datetime.utcnow().isoformat()
        }
        record["quiz_results"].append(entry)
        _update_student_record(student_name, record)

        return json.dumps({"message": "Quiz result saved", "entry": entry}, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Saving quiz result failed: {str(e)}"})


# =============================================================================
# STEP 2: Create the Agent
# =============================================================================

def create_agent():
    """Initialize and return the agent executor for adaptive tutoring."""

    # Initialize the Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=2048,
        timeout=60,
        max_retries=2,
    )

    # Initialize Tavily search tool for real-world examples / current context
    tavily_tool = TavilySearchResults(
        max_results=3,
        search_depth="basic",
        include_answer=True,
        include_raw_content=False
    )

    # Define all tools
    tools = [
        assess_diagnostic_level,
        score_quiz_answers,
        track_learning_progress,
        get_student_dashboard,
        save_quiz_result,
        tavily_tool
    ]

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Sage, an expert AI Tutor that adapts perfectly to each student's pace.

Your job in a NEW conversation, in order:
1. Ask the student's name.
2. Ask their class/grade.
3. Ask which subject they want to study.
4. Ask which topic within that subject.
5. Ask exactly 3 short diagnostic questions about that topic, ONE at a time, waiting for
   each answer before asking the next.
6. Once you have all 3 answers, call assess_diagnostic_level to get a suggested level,
   then tell the student their detected level (Beginner, Intermediate, or Advanced).

Then, for each explanation:
- Beginner: explain with very easy words, short sentences, and simple everyday examples.
- Intermediate: explain with moderate depth, some terminology, and practical examples.
- Advanced: explain with detailed technical depth, precise terminology, and real-world advanced examples.

After every explanation, ask exactly: "Did you understand? (Yes/No)"
- If the student says No: re-explain the SAME concept using a COMPLETELY DIFFERENT example.
- If the student says Yes: congratulate them briefly, increase the difficulty one notch
  (Beginner -> Intermediate -> Advanced), and explain the next relevant sub-concept at
  the new level.

Whenever the student's understanding changes or a learning attempt happens, call
track_learning_progress to persist it. When asked to grade a quiz, use score_quiz_answers
and then save_quiz_result. When asked about progress, accuracy, weak topics, strong topics,
or streak, call get_student_dashboard and present the results in friendly, readable text
(not raw JSON). Use tavily_search_results_json only when a real-world / current example
would meaningfully help the explanation.

Be warm, encouraging, and patient. Never dump raw JSON at the student — always translate
tool output into natural, friendly language."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=90
    )

    return agent_executor


# =============================================================================
# STEP 3: Chat Function with Memory
# =============================================================================

def chat(user_input: str, agent_executor):
    """
    Process user input and maintain chat history.

    Args:
        user_input: The user's message
        agent_executor: The agent executor instance

    Returns:
        The agent's response
    """
    global chat_history

    try:
        if chat_history is None:
            chat_history = []

        # Format chat history
        formatted_history = []
        for msg in chat_history:
            if isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role == "human":
                    formatted_history.append(HumanMessage(content=content))
                elif role == "assistant" and content:
                    if isinstance(content, str):
                        formatted_history.append(AIMessage(content=content))

        if agent_executor is None:
            agent_executor = create_agent()

        # Prepare input
        input_data = {
            "input": user_input,
            "chat_history": formatted_history or []
        }

        # Run agent
        try:
            response = agent_executor.invoke(input_data)

            if response is None:
                output = "No response was generated. Please try again."
            elif isinstance(response, dict):
                output = response.get('output', '')
                if not output:
                    output = "I didn't get a proper response. Could you rephrase?"
            elif hasattr(response, 'output') and response.output is not None:
                output = str(response.output)
            else:
                output = str(response) if response is not None else "No response was generated."

            if not output or not isinstance(output, str):
                output = "I'm having trouble understanding. Could you rephrase?"

        except Exception as e:
            output = f"I encountered an error: {str(e)}. Please try again."

        # Update chat history
        if output and output != 'No response generated':
            chat_history.append(("human", user_input))
            chat_history.append(("assistant", output))

        # Keep last 30 messages
        if len(chat_history) > 30:
            chat_history = chat_history[-30:]

        return output if output else "I'm not sure how to respond. Could you rephrase?"

    except Exception as e:
        error_msg = f"Error in chat function: {str(e)}"
        print(error_msg)
        return "I'm sorry, I encountered an error. Please try again."


def reset_chat_history():
    """Reset the global chat history (used when starting a new tutoring session)."""
    global chat_history
    chat_history = []


# =============================================================================
# STEP 4: Main Execution (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI Adaptive Tutor - Agent Test")
    print("=" * 70)
    print("\n📋 Loading API keys from .env file...")

    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not found!")
        print("\nCreate a .env file with:")
        print("GROQ_API_KEY=gsk-your-key-here")
        print("TAVILY_API_KEY=tvly-your-key-here")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY"):
        print("\n⚠️  TAVILY_API_KEY not found!")
        sys.exit(1)

    print("✅ API keys loaded!")
    print("\n🤖 Initializing agent...")

    try:
        agent_executor = create_agent()
        print("✅ Agent ready!\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {str(e)}")
        sys.exit(1)

    print("Test the agent (type 'quit' to exit):\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break

        try:
            response = chat(user_input, agent_executor)
            print(f"\n🤖 Sage: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")