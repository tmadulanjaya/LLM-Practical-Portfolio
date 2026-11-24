"""
PUSL 3205 - Lab 6, Parts B + C   (fixed version)
Agent with:
  - Short-term memory : recent conversation turns (kept in a Python list)
  - Long-term memory  : facts about the user stored in SQLite (memory.db)
  - Working memory    : messages + tool results for the current question
Tools: calculator, search_student_memory, search_university_knowledge (RAG)

Needs agent.py and rag_tool.py in the same folder, and rag_index.pkl built.
Run:  python mem_rag_agent.py
"""
import re
import json
import sqlite3
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_ollama import ChatOllama
from gcal_tools import check_schedule, create_calendar_event

from agent import open_calculator                 # Part A calculator tool
from rag_tool import search_university_knowledge  # Part C RAG tool

DB_FILE = "memory.db"
MAX_HISTORY = 10                                  # short-term memory size

llm = ChatOllama(model="llama3.2:3b", temperature=0)


# ---------------- Long-term memory (SQLite) ----------------

def init_db():
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS memories (
                           id INTEGER PRIMARY KEY AUTOINCREMENT,
                           fact TEXT UNIQUE,
                           created TEXT)""")


def save_memory(fact: str) -> bool:
    try:
        with sqlite3.connect(DB_FILE) as con:
            con.execute("INSERT INTO memories (fact, created) VALUES (?, ?)",
                        (fact.strip(), datetime.now().isoformat(timespec="seconds")))
        return True
    except sqlite3.IntegrityError:                # already stored
        return False


STOP = {"what", "is", "my", "the", "are", "you", "do", "i", "me", "a", "an",
        "of", "to", "in", "and", "about", "tell", "know", "remember", "did",
        "who", "which", "am", "your", "can", "hi", "im", "it", "for", "where",
        "from", "study", "like"}


@tool
def search_student_memory(query: str) -> str:
    """Search stored information about the user (name, degree, interests,
    preferences, hometown, goals). Use this for questions like
    'what is my name?', 'where am I from?' or 'what do I study?'."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower())
             if w not in STOP and len(w) > 2]
    # map question words onto how the facts are written
    q = query.lower()
    if "where" in q or "from" in q:
        words.append("from")
    if "study" in q or "degree" in q or "course" in q:
        words.append("studies")
    if "like" in q or "interest" in q or "hobby" in q:
        words.append("likes")
    if not words:
        return "No matching memories found."
    with sqlite3.connect(DB_FILE) as con:
        rows = con.execute("SELECT fact FROM memories").fetchall()
    hits = [r[0] for r in rows if any(w in r[0].lower() for w in words)]
    return "\n".join(hits) if hits else "No matching memories found."


# Reliable pattern matching first, LLM only as a fallback
APOS = "['\u2019]"
PATTERNS = [
    (r"\bmy name is ([A-Za-z]+)", "The user's name is {}."),
    (r"\bcall me ([A-Za-z]+)", "The user's name is {}."),
    (r"\bi" + APOS + r"?m called ([A-Za-z]+)", "The user's name is {}."),
    (r"\bi am called ([A-Za-z]+)", "The user's name is {}."),
    (r"\bi (?:study|am studying|" + APOS + r"m studying) ([A-Za-z &]+?)(?:[.,!]|$| and )",
     "The user studies {}."),
    (r"\bi (?:like|love|enjoy) ([A-Za-z &]+?)(?:[.,!]|$| and )", "The user likes {}."),
    (r"\bi(?:" + APOS + r"m| am) from ([A-Za-z ]+?)(?:[.,!]|$)", "The user is from {}."),
]


def extract_memories(user_input: str):
    facts = []
    for pattern, template in PATTERNS:
        m = re.search(pattern, user_input, re.I)
        if m:
            value = m.group(1).strip()
            if "name" in template or "from" in template:
                value = value.title()
            facts.append(template.format(value))
    if facts or "?" in user_input:
        return facts

    # fallback: ask the LLM for other kinds of facts
    prompt = (
        "Extract lasting personal facts about the USER from the message. "
        "Write each as a short sentence like \"The user's name is Alex.\" "
        "Reply with ONLY a JSON array of strings, or [] if none.\n\n"
        f"Message: {user_input}"
    )
    text = llm.invoke(prompt).content
    m = re.search(r"\[.*?\]", text, re.S)
    if not m:
        return []
    try:
        return [f for f in json.loads(m.group(0))
                if isinstance(f, str) and f.strip()]
    except json.JSONDecodeError:
        return []


# ---------------- Agent ----------------

TOOLS = [open_calculator, search_student_memory, search_university_knowledge,
         check_schedule, create_calendar_event]
TOOL_MAP = {t.name: t for t in TOOLS}
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are an intelligent student assistant.
- To see events or availability, call check_schedule (date: today, tomorrow or YYYY-MM-DD).
- To add an event, call create_calendar_event with title, date, start_time (HH:MM, 24-hour).
- For ANY calculation, call open_calculator.
- For questions about the user (their name, hometown, degree, interests),
  call search_student_memory and answer from its result. Never guess.
- For questions about the university, courses, modules or policies, call
  search_university_knowledge and answer ONLY from its result.
- When the user tells you something about themselves, just acknowledge it
  warmly. Do NOT call any tool for that.
- For greetings and small talk, answer directly without tools.
Be brief and friendly."""

history = []                                      # short-term memory


def run_agent(user_input: str, max_steps: int = 5) -> str:
    # 1. long-term memory: detect and save new facts
    facts = extract_memories(user_input)
    new = [f for f in facts if save_memory(f)]
    if new:
        print("\n[NEW MEMORIES DETECTED]")
        for f in new:
            print(f"[SAVED] {f}")
    else:
        print("\n[MEMORY] No new information to save.")

    # 2. working memory: system prompt + short-term history + this question
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + \
               [HumanMessage(content=user_input)]

    answer = "Stopped: too many steps."
    if facts and "?" not in user_input:
        # The user just told us something: acknowledge it, no tools needed.
        answer = llm.invoke(messages).content
    else:
        for _ in range(max_steps):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                answer = response.content
                break
            for call in response.tool_calls:
                print(f"\n[TOOL SELECTED] {call['name']}")
                print(f"[TOOL ARGUMENTS] {call['args']}")
                try:
                    result = TOOL_MAP[call["name"]].invoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
                print(f"[TOOL RESULT] {str(result)[:600]}")
                messages.append(ToolMessage(content=str(result),
                                            tool_call_id=call["id"]))

    # 3. short-term memory: keep the last few turns
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=answer))
    del history[:-MAX_HISTORY]
    print("\n[SHORT-TERM MEMORY] Conversation updated.")
    return answer


if __name__ == "__main__":
    init_db()
    print("=" * 36)
    print(" INTELLIGENT STUDENT AI AGENT")
    print(" SHORT-TERM MEMORY")
    print(" LONG-TERM SQLITE MEMORY")
    print(" WORKING MEMORY")
    print("=" * 36)
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print(f"\nAgent: {run_agent(q)}")
