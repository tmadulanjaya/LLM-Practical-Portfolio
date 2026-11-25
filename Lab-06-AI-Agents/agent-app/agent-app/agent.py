import os
import re
import time
import platform
import subprocess
from datetime import datetime

import psutil
import pyautogui
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage, SystemMessage, ToolMessage,
)
from langchain_ollama import ChatOllama


# 5. Create tools ---------------------------------------------------------

@tool
def open_calculator(expression: str) -> str:
    """Open the Windows Calculator and calculate a maths expression.
    Use this for ANY arithmetic. Pass the full expression, e.g. '25*4'.
    Use * for multiplication, / for division."""
    expression = expression.replace("×", "*").replace("x", "*").replace("÷", "/")
    if not re.fullmatch(r"[0-9+\-*/().\s%]+", expression):
        return "Invalid expression."
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # chars validated above
    except Exception as e:
        return f"Could not evaluate: {e}"

    subprocess.Popen("calc.exe")
    time.sleep(2)  # wait for the window to open
    pyautogui.write(expression.replace(" ", ""), interval=0.15)
    pyautogui.press("enter")
    return f"Entered {expression} into Calculator. Result: {result}"


@tool
def get_date_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M:%S %p")


@tool
def open_notepad(text: str) -> str:
    """Open Windows Notepad and write the given text into it."""
    subprocess.Popen("notepad.exe")
    time.sleep(2)
    pyautogui.write(text, interval=0.05)
    return f"Opened Notepad and wrote: {text}"


@tool
def search_files(directory: str, query: str) -> str:
    """Search a directory (recursively) for files by name or extension.
    query can be a name fragment like 'report' or an extension like '.pdf' or 'pdf'."""
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    q = query.lower().lstrip("*")
    matches = []
    for root, _, files in os.walk(directory):
        for f in files:
            if q in f.lower():
                matches.append(os.path.join(root, f))
    if not matches:
        return "No matching files found."
    shown = matches[:30]
    extra = f"\n...and {len(matches) - 30} more" if len(matches) > 30 else ""
    return f"Found {len(matches)} file(s):\n" + "\n".join(shown) + extra


@tool
def create_file(path: str, content: str) -> str:
    """Create a text file at the given full path with the given content."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"File created: {path}"
    except Exception as e:
        return f"Failed to create file: {e}"


@tool
def system_info() -> str:
    """Get operating system, processor and memory information."""
    mem = psutil.virtual_memory()
    return (
        f"OS: {platform.system()} {platform.release()} ({platform.version()})\n"
        f"Processor: {platform.processor()}\n"
        f"CPU cores: {psutil.cpu_count(logical=False)} physical, "
        f"{psutil.cpu_count()} logical\n"
        f"RAM: {mem.total / 1024**3:.1f} GB total, "
        f"{mem.available / 1024**3:.1f} GB available"
    )


TOOLS = [open_calculator, get_date_time, open_notepad,
         search_files, create_file, system_info]
TOOL_MAP = {t.name: t for t in TOOLS}

# 4. Initialize LLM  /  6. Register tools with LLM
llm = ChatOllama(model="llama3.2:3b", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = """You are a helpful desktop assistant with tools.
- For ANY calculation, you MUST call open_calculator. Never calculate yourself.
- For date/time questions, call get_date_time.
- To write text in Notepad, call open_notepad.
- To find files, call search_files with the directory and name/extension.
- To create a file, call create_file with the full path and the content.
- For computer specifications, call system_info.
After a tool returns, answer the user briefly using the tool result."""


# 7. Agent execution loop -------------------------------------------------

def run_agent(user_input: str, max_steps: int = 5) -> str:
    messages = [SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input)]

    for _ in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:          # model gave a final answer
            return response.content

        for call in response.tool_calls:
            print(f"\n[TOOL SELECTED] {call['name']}")
            print(f"[TOOL ARGUMENTS] {call['args']}")
            try:
                result = TOOL_MAP[call["name"]].invoke(call["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            print(f"[TOOL RESULT] {result}")
            messages.append(ToolMessage(content=str(result),
                                        tool_call_id=call["id"]))
    return "Stopped: too many steps."


# 8. Run and test ---------------------------------------------------------

if __name__ == "__main__":
    print("Tool-Using Agent (type 'exit' to quit)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            print(f"\nAgent: {run_agent(q)}")
