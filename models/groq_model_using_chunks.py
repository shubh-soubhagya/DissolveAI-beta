# groq_utils.py
import os
import pickle
import time
from collections import deque
from typing import Tuple

import numpy as np
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss

# CONFIGURATION
MODEL_NAME = "openai/gpt-oss-120b"
FILES_CSV = r"data/repo_files_data.csv"
ISSUES_CSV = r"data/repo_issues.csv"
CUSTOM_MODEL_PATH = r"sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = r"embeddings/repo_index.pkl"

CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 40
EMBED_BATCH_SIZE = 64

TOP_K = 5


MAX_INPUT_TOKENS = 6000
RESERVED_TOKENS = 400
TOKEN_ENCODING = "gpt2"

TPM_LIMIT = 8000
TPM_WINDOW_SECONDS = 60


# TOKENIZER HELPERS
def _get_tokenizer(enc_name: str = TOKEN_ENCODING):
    return tiktoken.get_encoding(enc_name)


def count_tokens(text: str, enc_name: str = TOKEN_ENCODING) -> int:
    enc = _get_tokenizer(enc_name)
    return len(enc.encode(text))


# ENV / GROQ CLIENT
def load_env_and_configure() -> Groq:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY not found in .env file.")
    client = Groq(api_key=GROQ_API_KEY)
    print("✅ GROQ client configured.")
    return client


# TOKEN RATE LIMITER
class TokenRateLimiter:
    def __init__(self, token_limit=TPM_LIMIT, window_seconds=TPM_WINDOW_SECONDS):
        self.token_limit = token_limit
        self.window = window_seconds
        self.events = deque()
        self.total = 0

    def allow(self, tokens: int) -> bool:
        now = time.time()
        while self.events and now - self.events[0][0] > self.window:
            tstamp, tk = self.events.popleft()
            self.total -= tk
        if self.total + tokens > self.token_limit:
            return False
        self.events.append((now, tokens))
        self.total += tokens
        return True


# CHUNKING LOGIC
def chunk_text_by_tokens(text: str, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS):
    enc = _get_tokenizer()
    token_ids = enc.encode(text)

    if len(token_ids) == 0:
        return []

    step = max_tokens - overlap if (max_tokens - overlap) > 0 else max_tokens
    chunks = []
    start = 0
    while start < len(token_ids):
        chunk_ids = token_ids[start:start + max_tokens]
        chunks.append(enc.decode(chunk_ids))
        start += step
    return chunks


# BUILD VECTOR INDEX (CHUNKED)
def build_vector_index(chunk_max_tokens: int = CHUNK_MAX_TOKENS):
    print("🧠 Building chunked vector index...")

    if not os.path.exists(FILES_CSV):
        raise FileNotFoundError(f"Files CSV not found at {FILES_CSV}")

    df = pd.read_csv(FILES_CSV, encoding="utf-8")
    if df.empty:
        raise ValueError("repo_files_data.csv is empty.")

    records = []

    for _, row in df.iterrows():
        file_name = row.get("file_name", "")
        file_path = row.get("file_path", "")
        file_ext = row.get("file_extension", "")
        content = str(row.get("file_content", "") or "")

        if not content.strip():
            records.append({
                "file_name": file_name,
                "file_path": file_path,
                "file_extension": file_ext,
                "chunk_id": 0,
                "chunk_text": ""
            })
            continue

        chunks = chunk_text_by_tokens(content, max_tokens=chunk_max_tokens, overlap=CHUNK_OVERLAP_TOKENS)

        for cid, ctext in enumerate(chunks):
            records.append({
                "file_name": file_name,
                "file_path": file_path,
                "file_extension": file_ext,
                "chunk_id": cid,
                "chunk_text": ctext
            })

    chunks_df = pd.DataFrame(records)
    if chunks_df.empty:
        raise ValueError("No chunks created.")

    print(f"📦 Total chunks: {len(chunks_df)} — embedding...")

    embed_model = SentenceTransformer(CUSTOM_MODEL_PATH)
    texts = chunks_df["chunk_text"].tolist()

    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        emb = embed_model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(emb.astype("float32"))

    embeddings = np.vstack(all_embeddings)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump((index, chunks_df), f)

    print(f"✅ Saved FAISS index to {INDEX_PATH}")


# RETRIEVAL (TOP-K CHUNKS)
def retrieve_relevant_files(query: str, top_k: int = TOP_K) -> str:
    """
    Defensive retrieval of top-k chunks for a query.
    - Rebuilds index if index file missing or corrupt.
    - Ensures necessary columns exist with safe defaults.
    - Never raises KeyError for missing metadata like 'chunk_id'.
    """
    # Ensure index exists; attempt to build if missing
    if not os.path.exists(INDEX_PATH):
        print("⚠️ Index file not found; building index now...")
        try:
            build_vector_index()
        except Exception as e:
            print("❌ Failed to build index:", e)
            return ""

    # Load the index and chunks_df safely
    try:
        with open(INDEX_PATH, "rb") as f:
            index, chunks_df = pickle.load(f)
    except Exception as e:
        print("❌ Failed to load index pickle:", e)
        # try a rebuild once
        try:
            build_vector_index()
            with open(INDEX_PATH, "rb") as f:
                index, chunks_df = pickle.load(f)
        except Exception as e2:
            print("❌ Rebuild failed:", e2)
            return ""

    # Validate chunks_df
    if not isinstance(chunks_df, pd.DataFrame) or chunks_df.empty:
        print("⚠️ Chunks DataFrame is empty or invalid — no repository context available.")
        return ""

    # Ensure expected columns exist; add safe defaults if missing
    required_cols = {
        "file_name": "",
        "file_path": "",
        "file_extension": "",
        "chunk_id": 0,
        "chunk_text": ""
    }
    for col, default in required_cols.items():
        if col not in chunks_df.columns:
            print(f"⚠️ chunks_df missing column '{col}' — adding default values.")
            chunks_df[col] = default

    # Prepare embedding for query
    try:
        embed_model = SentenceTransformer(CUSTOM_MODEL_PATH)
        query_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    except Exception as e:
        print("❌ Failed to embed query:", e)
        return ""

    # If index empty, return no context
    if getattr(index, "ntotal", 0) == 0:
        print("⚠️ FAISS index has zero vectors. No repo context to return.")
        return ""

    # Run search
    try:
        faiss.normalize_L2(query_emb)
        k = min(top_k, int(index.ntotal))
        if k <= 0:
            return ""
        D, I = index.search(query_emb, k)
        indices = [int(i) for i in I[0] if i != -1]
    except Exception as e:
        print("❌ FAISS search failed:", e)
        return ""

    # Validate indices
    indices = [i for i in indices if 0 <= i < len(chunks_df)]
    if not indices:
        print("⚠️ FAISS returned no valid chunk indices.")
        return ""

    # Build repo_context safely
    enc = _get_tokenizer()
    parts = []
    for idx in indices:
        rec = chunks_df.iloc[idx].to_dict()
        fname = rec.get("file_name") or ""
        fpath = rec.get("file_path") or ""
        fext = rec.get("file_extension") or ""
        chunk_id = rec.get("chunk_id") if rec.get("chunk_id") is not None else 0
        chunk_text = str(rec.get("chunk_text") or "")

        # Safety: trim long chunk_text by tokens
        try:
            token_ids = enc.encode(chunk_text)
            if len(token_ids) > CHUNK_MAX_TOKENS:
                chunk_text = enc.decode(token_ids[:CHUNK_MAX_TOKENS])
        except Exception:
            # fallback: character truncate
            chunk_text = chunk_text[:2000]

        snippet = (
            f"\n\n### File: {fname} ({fpath}) [chunk {chunk_id}]\n"
            f"```{fext}\n{chunk_text}\n```\n"
        )
        parts.append(snippet)

    return "".join(parts)


# PROMPT CREATION
def create_prompt(issue: dict, repo_context: str) -> Tuple[str, int]:
    issue_title = issue.get("title", "Untitled Issue")
    issue_body = str(issue.get("body", ""))

    if count_tokens(issue_body) > 800:
        enc = _get_tokenizer()
        issue_body = enc.decode(enc.encode(issue_body)[:800])

    header = (
        "You are Desolve AI — an expert AI developer assistant.\n\n"
        f"Issue: {issue_title}\n\n{issue_body}\n\n"
        "Relevant repository chunks:\n"
    )

    footer = (
        "\n\nYour task:\n"
        "- Analyze the files and issue.\n"
        "- Suggest exact code fixes.\n"
        "- Provide reasoning.\n"
        "- Keep formatting clean.\n"
        "- Dont generate Tables"
    )

    enc = _get_tokenizer()
    allowed_ctx = max(0, MAX_INPUT_TOKENS - RESERVED_TOKENS - count_tokens(header) - count_tokens(footer))

    ctx_ids = enc.encode(repo_context)
    if len(ctx_ids) > allowed_ctx:
        repo_context = enc.decode(ctx_ids[:allowed_ctx])

    prompt = header + repo_context + footer
    return prompt, count_tokens(prompt)


# ISSUE LOADER
def load_issue(issue_csv: str = ISSUES_CSV, row_index: int = 0) -> dict:
    if not os.path.exists(issue_csv):
        raise FileNotFoundError("Issues CSV not found.")
    df = pd.read_csv(issue_csv)
    if row_index >= len(df):
        raise IndexError("Row index out of range.")
    return df.to_dict(orient="records")[row_index]


# EXAMPLE CALL
def ask_issue_with_groq(issue_row_index: int = 0):
    issue = load_issue(ISSUES_CSV, issue_row_index)
    repo_context = retrieve_relevant_files(issue.get("body", ""), top_k=TOP_K)

    prompt, token_count = create_prompt(issue, repo_context)
    print("Prompt tokens:", token_count)

    limiter = TokenRateLimiter()
    if not limiter.allow(token_count):
        raise RuntimeError("TPM limit exceeded — slow down.")

    client = load_env_and_configure()

    try:
        response = client.chat.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1024,
        )
        return response
    except Exception as e:
        print("❌ Groq error:", e)
        raise


if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        build_vector_index()

    issue = load_issue(ISSUES_CSV, 0)
    ctx = retrieve_relevant_files(issue["body"], top_k=TOP_K)
    prompt, tokens = create_prompt(issue, ctx)
    print(prompt[:1500], "\n\nTokens:", tokens)
