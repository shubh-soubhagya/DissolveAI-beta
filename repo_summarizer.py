import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from groq import Groq

# CONFIGURATION
FILES_CSV = r"data/repo_files_data.csv"
ISSUES_CSV = r"data/repo_issues.csv"
SUMMARY_TARGET_LENGTH = 500  # words


# ============================================
# LOAD ENVIRONMENT & CONFIGURE CLIENTS
# ============================================
def load_clients():
    """Load and configure both Gemini and Groq clients."""
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    gemini_client = None
    groq_client = None

    if gemini_key:
        genai.configure(api_key=gemini_key)
        gemini_client = genai.GenerativeModel("gemini-2.5-flash-lite")
        print("✅ Gemini client configured.")
    else:
        print("⚠️ Gemini API key not found.")

    if groq_key:
        groq_client = Groq(api_key=groq_key)
        print("✅ Groq client configured.")
    else:
        print("⚠️ Groq API key not found.")

    return gemini_client, groq_client


# ============================================
# EXTRACT REPOSITORY INFORMATION
# ============================================
def extract_repo_info() -> dict:
    """Extracts key information from CSV files about the repository."""
    info = {
        "total_files": 0,
        "file_types": {},
        "total_issues": 0,
        "key_files": [],
        "programming_languages": set(),
        "file_sample": ""
    }

    # Extract file information
    if os.path.exists(FILES_CSV):
        df_files = pd.read_csv(FILES_CSV, encoding="utf-8")
        info["total_files"] = len(df_files)

        # Count file extensions
        extensions = df_files["file_extension"].value_counts()
        info["file_types"] = extensions.to_dict()

        # Identify key files (README, setup, config, etc.)
        key_keywords = ["readme", "setup", "config", "main", "requirements", "dockerfile", "package.json"]
        key_files = df_files[df_files["file_name"].str.lower().str.contains("|".join(key_keywords), na=False)]
        info["key_files"] = key_files["file_name"].tolist()[:10]

        # Detect programming languages by extension
        lang_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".java": "Java",
            ".cpp": "C++",
            ".c": "C",
            ".rb": "Ruby",
            ".go": "Go",
            ".rs": "Rust",
            ".php": "PHP",
            ".cs": "C#",
            ".swift": "Swift",
            ".kt": "Kotlin",
        }

        for ext, lang in lang_map.items():
            if ext in info["file_types"]:
                info["programming_languages"].add(lang)

        # Sample file content for context
        if len(df_files) > 0:
            readme_files = df_files[df_files["file_name"].str.lower().str.contains("readme", na=False)]
            if len(readme_files) > 0:
                info["file_sample"] = readme_files.iloc[0]["file_content"][:1000]
            else:
                sample_row = df_files.iloc[0]
                if pd.notna(sample_row["file_content"]) and len(str(sample_row["file_content"])) > 100:
                    info["file_sample"] = str(sample_row["file_content"])[:1000]

    # Extract issues information
    if os.path.exists(ISSUES_CSV):
        df_issues = pd.read_csv(ISSUES_CSV, encoding="utf-8")
        info["total_issues"] = len(df_issues)

        # Sample issue titles
        info["sample_issues"] = df_issues["title"].head(5).tolist() if "title" in df_issues.columns else []

    return info


# ============================================
# BUILD CONTEXT FOR SUMMARIZATION
# ============================================
def build_summary_context(repo_info: dict) -> str:
    """Builds a structured context string for the AI to summarize."""
    languages = ", ".join(sorted(repo_info["programming_languages"])) if repo_info[
        "programming_languages"] else "Unknown"

    file_types_str = ", ".join(
        [f"{ext} ({count})" for ext, count in sorted(repo_info["file_types"].items())[:10]]
    )

    key_files_str = ", ".join(repo_info["key_files"]) if repo_info["key_files"] else "None identified"

    sample_issues = "\n".join([f"- {issue}" for issue in repo_info["sample_issues"][:5]])

    context = f"""
Repository Analysis Data:
- Total Files: {repo_info['total_files']}
- File Types: {file_types_str}
- Primary Languages: {languages}
- Total Issues: {repo_info['total_issues']}
- Key Files Identified: {key_files_str}
- Sample Issues:
{sample_issues}

Sample File Content:
{repo_info['file_sample'][:500]}
"""
    return context


# ============================================
# GENERATE SUMMARY WITH GEMINI
# ============================================
def generate_summary_gemini(gemini_client, context: str) -> str:
    """Generates a repository summary using Gemini."""
    prompt = f"""Based on the following repository analysis, provide a comprehensive yet concise summary of approximately 500 words. 
Include:
1. Project Overview - What is this project about?
2. Technology Stack - What languages and frameworks are used?
3. Structure - How is the codebase organized?
4. Key Components - What are the main modules/files?
5. Purpose & Functionality - What does this project do?
6. Issues/Problems - What are users reporting/working on?

Analysis Data:
{context}

Provide a well-structured, professional summary in exactly 500 words (±10 words):"""

    try:
        response = gemini_client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini summarization failed: {e}")
        return None


# ============================================
# GENERATE SUMMARY WITH GROQ
# ============================================
def generate_summary_groq(groq_client, context: str) -> str:
    """Generates a repository summary using Groq."""
    prompt = f"""Based on the following repository analysis, provide a comprehensive yet concise summary of approximately 500 words. 
Include:
1. Project Overview - What is this project about?
2. Technology Stack - What languages and frameworks are used?
3. Structure - How is the codebase organized?
4. Key Components - What are the main modules/files?
5. Purpose & Functionality - What does this project do?
6. Issues/Problems - What are users reporting/working on?

Analysis Data:
{context}

Provide a well-structured, professional summary in exactly 500 words (±10 words):"""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system",
                 "content": "You are an expert software engineer who provides clear, concise technical summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq summarization failed: {e}")
        return None


# ============================================
# MAIN SUMMARIZATION FUNCTION
# ============================================
def summarize_repository(model: str = "gemini") -> str:
    """Main function to summarize a repository using the specified model."""
    print(f"🔍 Analyzing repository for summarization (using {model})...")

    # Load clients
    gemini_client, groq_client = load_clients()

    # Validate model and client availability
    if model == "gemini" and not gemini_client:
        print("⚠️ Gemini client not available. Falling back to Groq...")
        model = "groq"

    if model == "groq" and not groq_client:
        print("⚠️ Groq client not available. Falling back to Gemini...")
        model = "gemini"

    if not gemini_client and not groq_client:
        raise RuntimeError("❌ No AI clients available. Please check API keys.")

    # Extract repository information
    repo_info = extract_repo_info()
    print(f"✅ Extracted repo info: {repo_info['total_files']} files, {repo_info['total_issues']} issues")

    # Build context
    context = build_summary_context(repo_info)
    print("📝 Building summarization context...")

    # Generate summary
    print(f"🤖 Generating summary with {model}...")

    if model == "gemini":
        summary = generate_summary_gemini(gemini_client, context)
    else:
        summary = generate_summary_groq(groq_client, context)

    if not summary:
        raise RuntimeError(f"❌ Failed to generate summary with {model}")

    print(f"✅ Summary generated ({len(summary.split())} words)")
    return summary


# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    try:
        summary = summarize_repository(model="gemini")
        print("\n" + "=" * 80)
        print("REPOSITORY SUMMARY")
        print("=" * 80)
        print(summary)
    except Exception as e:
        print(f"Error: {e}")