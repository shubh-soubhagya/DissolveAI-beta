import os
import shutil
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
from groq import Groq
import time

# --- Local Imports ---
from clone import clone_repo
from file_contents import extract_files_to_csv
from issues import extract_issues
from repo_summarizer import summarize_repository
from models.gemini_models_rag import (
    load_env_and_configure as load_gemini_env,
    build_vector_index,
    create_prompt,
    retrieve_relevant_files,
    MODEL_NAME as GEMINI_MODEL_NAME,
)
from models.groq_models_rag import (
    load_env_and_configure as load_groq_env,
    MODEL_NAME as GROQ_MODEL_NAME,
)

# --- Configuration ---
TEMP_DATA_DIR = "data"
FILES_CSV = os.path.join(TEMP_DATA_DIR, "repo_files_data.csv")
ISSUES_CSV = os.path.join(TEMP_DATA_DIR, "repo_issues.csv")
INDEX_PATH = os.path.join(TEMP_DATA_DIR, "repo_index.pkl")
CLONE_DIR = "cloned_repo"

# Create temp directory
os.makedirs(TEMP_DATA_DIR, exist_ok=True)

# --- Configuration & App Setup ---
try:
    load_gemini_env()
    print("✅ Gemini API key configured.")
except ValueError as e:
    print(f"⚠️ Gemini API key not found: {e}")

try:
    groq_client = load_groq_env()
    print("✅ Groq API key configured.")
except ValueError as e:
    print(f"⚠️ Groq API key not found: {e}")

app = FastAPI()

# --- Global State ---
gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
chat_sessions = {}


# --- Pydantic Models ---
class RepoRequest(BaseModel):
    """Request model for processing a new repository."""
    url: str
    model: str = "gemini"


class AiRequest(BaseModel):
    """Request model for asking the AI a question."""
    issue_id: int
    prompt: str
    model: str = "gemini"


# --- Cleanup Helper (ROBUST VERSION) ---
def remove_readonly(func, path, exc):
    """Error handler for shutil.rmtree to handle readonly files on Windows."""
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        func(path)
    else:
        raise


def cleanup_temp_data():
    """Removes temporary data files with retry logic for Windows lock issues."""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            if os.path.exists(CLONE_DIR):
                print(f"🗑️ Cleaning up {CLONE_DIR}...")
                shutil.rmtree(CLONE_DIR, onerror=remove_readonly)

            if os.path.exists(TEMP_DATA_DIR):
                print(f"🗑️ Cleaning up {TEMP_DATA_DIR}...")
                shutil.rmtree(TEMP_DATA_DIR, onerror=remove_readonly)

            os.makedirs(TEMP_DATA_DIR, exist_ok=True)
            print("✅ Cleanup complete")
            return

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"⚠️ Cleanup failed (attempt {retry_count}/{max_retries}), retrying in 1 second: {e}")
                time.sleep(1)
            else:
                print(f"❌ Cleanup failed after {max_retries} attempts: {e}")
                if not os.path.exists(TEMP_DATA_DIR):
                    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
                return


def load_issue_by_id(csv_path: str, issue_id: int) -> dict | None:
    """Loads a specific issue from the CSV using its GitHub issue number (id)."""
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    if 'number' not in df.columns:
        return None

    df = df.fillna("")

    try:
        df['number'] = pd.to_numeric(df['number'], errors='coerce')
    except Exception:
        pass

    issue_df = df[df['number'] == issue_id]

    if issue_df.empty:
        return None

    record = issue_df.to_dict(orient="records")[0]

    for k, v in record.items():
        if v is None:
            record[k] = ""
        elif isinstance(v, float) and pd.isna(v):
            record[k] = ""
        elif not isinstance(v, (str, int, bool, list, dict)):
            record[k] = str(v)

    return record


# --- API Endpoint 1: Process Repository ---
@app.post("/process-repo")
async def process_repo(request: RepoRequest):
    """Clones a repo, extracts files/issues, builds the vector index, and generates summary."""
    print(f"🚀 Processing repo with {request.model} model: {request.url}")

    if request.model not in ["gemini", "groq"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be 'gemini' or 'groq'"
        )

    chat_sessions.clear()
    cleanup_temp_data()

    token = os.getenv("GITHUB_TOKEN")

    try:
        print(f"Cloning {request.url}...")
        clone_repo(request.url, CLONE_DIR)

        repo_name = request.url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_path = os.path.join(CLONE_DIR, repo_name)

        print("📂 Extracting repository files...")
        extract_files_to_csv(repo_path, FILES_CSV)

        print("🐞 Extracting repository issues...")
        if not token:
            print("⚠️ GITHUB_TOKEN not set. May be rate-limited.")
        extract_issues(request.url, output_file=ISSUES_CSV, token=token)

        print("🧠 Building vector index...")
        try:
            build_vector_index()
        except Exception as e_index:
            print(f"⚠️ Warning: failed to build vector index: {e_index}")

        # NEW: Generate repository summary
        print("📖 Generating repository summary...")
        summary = ""
        try:
            summary = summarize_repository(model=request.model)
        except Exception as e_summary:
            print(f"⚠️ Warning: failed to generate summary: {e_summary}")
            summary = "Summary generation failed. Please check API keys."

        print("✅ Processing complete.")
        if not os.path.exists(ISSUES_CSV):
            raise FileNotFoundError(f"Issues CSV not created at {ISSUES_CSV}")

        df_issues = pd.read_csv(ISSUES_CSV)

        if 'number' not in df_issues.columns:
            raise KeyError("Issues CSV must contain a 'number' column.")

        df_issues = df_issues.fillna("")
        df_issues['id'] = df_issues['number']

        issues_df_subset = df_issues[['id', 'title', 'body']].copy()

        for col in ['id', 'title', 'body']:
            issues_df_subset[col] = issues_df_subset[col].apply(
                lambda v: "" if v is None else (str(v) if not isinstance(v, (str, int, bool, list, dict)) else v)
            )

        issues_list = issues_df_subset.to_dict(orient="records")

        safe_payload = jsonable_encoder({
            "issues": issues_list,
            "summary": summary,
            "model": request.model
        })

        return JSONResponse(content=safe_payload)

    except Exception as e:
        print(f"❌ Error during repo processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}"
        )


# --- API Endpoint 2: Chat with AI ---
@app.post("/ask-ai")
async def ask_ai(request: AiRequest):
    """Handles chat using selected model (Gemini or Groq)."""
    print(f"🤖 Chat request (model: {request.model}) for issue {request.issue_id}")

    if request.model not in ["gemini", "groq"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be 'gemini' or 'groq'"
        )

    try:
        session_key = f"{request.issue_id}_{request.model}"

        if session_key not in chat_sessions:
            print(f"Creating new chat session for issue {request.issue_id} ({request.model})...")

            issue = load_issue_by_id(ISSUES_CSV, request.issue_id)
            if issue is None:
                raise HTTPException(status_code=404, detail="Selected issue not found.")

            print("Retrieving relevant files...")
            repo_context = retrieve_relevant_files(issue.get("body", ""))
            system_prompt = create_prompt(issue, repo_context)

            chat = gemini_model.start_chat(
                history=[{"role": "user", "parts": system_prompt}]
            )
            chat_sessions[session_key] = chat

        chat = chat_sessions[session_key]

        print(f"Sending prompt to {request.model}...")
        response = chat.send_message(request.prompt)
        response_text = response.text

        print(f"✅ Got response from {request.model}")
        return {"response": response_text}

    except Exception as e:
        print(f"❌ Error during AI chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred with the AI model: {str(e)}"
        )


# --- API Endpoint 3: Get Available Models ---
@app.get("/models")
async def get_models():
    """Returns available models."""
    return {
        "models": [
            {"name": "gemini", "display": "Google Gemini"},
            {"name": "groq", "display": "Groq"}
        ]
    }


# --- API Endpoint 4: Cleanup ---
@app.post("/cleanup")
async def cleanup():
    """Cleanup temporary data."""
    cleanup_temp_data()
    chat_sessions.clear()
    return {"message": "Cleanup complete"}


# --- API Endpoint 5: Serve Frontend ---
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
async def get_index():
    """Serves the main index.html file."""
    return FileResponse("frontend/index.html")


# --- Run the Server ---
if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000")
    if not os.path.exists("frontend/index.html"):
        print("WARNING: 'frontend/index.html' not found.")

    uvicorn.run(app, host="127.0.0.1", port=8000)