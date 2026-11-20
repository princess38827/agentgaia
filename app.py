import os
import re
import time
import requests
import pandas as pd
from pypdf import PdfReader
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    InferenceClientModel,
    LiteLLMModel,
    tool,
)

SCORING_API = "https://agents-course-unit4-scoring.hf.space"
HF_USERNAME = os.getenv("HF_USERNAME", "princess38827")
AGENT_CODE_URL = os.getenv(
    "AGENT_CODE_URL",
    "https://github.com/princess38827/agentgaia/tree/main",
)
FILES_DIR = "./gaia_task_files"
os.makedirs(FILES_DIR, exist_ok=True)


@tool
def read_task_file(file_path: str) -> str:
    """
    Reads and extracts text or tabular data from a downloaded GAIA task file.
    Supports .txt, .md, .py, .json, .csv, .xlsx, .xls, and .pdf files.

    Args:
        file_path: Local path to the downloaded attachment file.
    """
    if not file_path or not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".txt", ".md", ".py", ".json", ".xml", ".html", ".log"]:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()[:15000]
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return f"Columns: {list(df.columns)}\nShape: {df.shape}\n\n{df.to_string()}"[:15000]
        elif ext in [".xlsx", ".xls"]:
            excel_data = pd.read_excel(file_path, sheet_name=None)
            output = [f"--- Sheet: {name} ---\n{df.to_string()}" for name, df in excel_data.items()]
            return "\n\n".join(output)[:15000]
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            pages = [f"--- Page {i+1} ---\n{page.extract_text() or ''}" for i, page in enumerate(reader.pages)]
            return "\n\n".join(pages)[:15000]
        else:
            return f"Unsupported extension '{ext}'. Inspect directly with Python at: {file_path}"
    except Exception as e:
        return f"Failed to read {file_path}: {str(e)}"


def build_agent() -> CodeAgent:
    if os.getenv("GEMINI_API_KEY"):
        model = LiteLLMModel(model_id="gemini/gemini-2.5-flash")
    elif os.getenv("OPENROUTER_API_KEY"):
        model = LiteLLMModel(model_id="openrouter/qwen/qwen-2.5-coder-32b-instruct")
    else:
        model = InferenceClientModel(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=os.getenv("HF_TOKEN"),
        )

    return CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), read_task_file],
        model=model,
        additional_authorized_imports=[
            "math",
            "statistics",
            "datetime",
            "json",
            "re",
            "csv",
            "os",
            "pandas",
            "openpyxl",
        ],
        max_steps=10,
    )


def download_task_file(task_id: str, file_name: str | None) -> str | None:
    if not file_name:
        return None
    local_path = os.path.join(FILES_DIR, f"{task_id}_{file_name}")
    if os.path.exists(local_path):
        return local_path
    resp = requests.get(f"{SCORING_API}/files/{task_id}", timeout=30)
    if resp.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return local_path
    return None


def clean_final_answer(raw_output: str) -> str:
    text = str(raw_output).strip()
    text = re.sub(r"^(final answer\s*[:\-]\s*|answer\s*[:\-]\s*)", "", text, flags=re.IGNORECASE).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    if text.endswith(".") and "\n" not in text and len(text.split()) <= 6:
        text = text[:-1].strip()
    return text


SYSTEM_FORMAT_RULES = """
You are solving a GAIA benchmark question evaluated by STRICT EXACT MATCH.
Rules for your final_answer():
1. Return ONLY the exact answer value — no explanations, no full sentences, no labels like "Final Answer:".
2. If the answer is a number, do NOT include commas, currency symbols, or units unless explicitly requested in the question.
3. If the answer is a list, separate items with a comma and a single space (e.g., "apple, banana, cherry").
4. If an attached file path is provided below, inspect it using `read_task_file` or Python code first.
"""


def run_and_submit():
    print("Fetching Unit 4 GAIA questions...")
    q_resp = requests.get(f"{SCORING_API}/questions", timeout=30)
    q_resp.raise_for_status()
    questions = q_resp.json()
    print(f"Loaded {len(questions)} questions.\n")

    agent = build_agent()
    answers_payload = []

    for idx, item in enumerate(questions, start=1):
        task_id = item["task_id"]
        question_text = item["question"]
        file_name = item.get("file_name")

        local_file = download_task_file(task_id, file_name)
        file_note = f"\nAttached file saved locally at: {local_file}" if local_file else ""

        prompt = f"{SYSTEM_FORMAT_RULES}\nQuestion:\n{question_text}{file_note}"
        print(f"[{idx}/{len(questions)}] Running task {task_id}...")

        try:
            raw_result = agent.run(prompt)
            cleaned = clean_final_answer(raw_result)
        except Exception as e:
            print(f"  Error on {task_id}: {e}")
            cleaned = ""

        print(f"  -> Submitted Answer: {cleaned}\n")
        answers_payload.append({"task_id": task_id, "submitted_answer": cleaned})
        time.sleep(1.5)

    submission = {
        "username": HF_USERNAME,
        "agent_code": AGENT_CODE_URL,
        "answers": answers_payload,
    }

    print("Submitting answers to GAIA leaderboard...")
    sub_resp = requests.post(f"{SCORING_API}/submit", json=submission, timeout=60)
    print(f"Status Code: {sub_resp.status_code}")
    print("Response:", sub_resp.json())


if __name__ == "__main__":
    run_and_submit()
