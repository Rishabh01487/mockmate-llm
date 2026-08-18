"""
app_ui.py
=========

Streamlit UI for the Mockmate-LLM interview assistant. Talks to the FastAPI
backend defined in `app_api.py`. Designed to be embeddable as an iframe inside
the Mockmate interview platform, or run standalone as a developer preview tool.

Features
--------
- Problem input form (title, difficulty, description, examples, constraints, language)
- Optional: paste a problem URL to auto-fetch (uses /parse endpoint — TBD)
- "Generate Solution" button — calls POST /generate
- Displays code + Big-O complexity with syntax highlighting
- Copy-to-clipboard button for each block
- "Explain this solution" toggle — calls POST /explain
- Latency / token-usage display per request
- Full conversation history panel (left sidebar) — like CoderPad

Run
---
    streamlit run app_ui.py --server.port 8501 --server.address 0.0.0.0

Env vars
--------
    API_BASE_URL    default: http://localhost:8000
    API_KEY         optional bearer token for the backend
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


# =========================================================================
# Config
# =========================================================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")

LANGUAGES = ["python", "java", "cpp", "javascript", "c", "go", "rust"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

# Ace-editor language map (for streamlit-ace)
ACE_LANG_MAP = {
    "python": "python",
    "java": "java",
    "cpp": "c_cpp",
    "javascript": "javascript",
    "c": "c_cpp",
    "go": "golang",
    "rust": "rust",
}


# =========================================================================
# API client
# =========================================================================
def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def api_health() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE_URL}/ready", timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def api_generate(problem: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{API_BASE_URL}/generate",
                      json=problem, headers=_headers(), timeout=120)
    r.raise_for_status()
    return r.json()


def api_explain(code: str, language: str, title: str) -> Dict[str, Any]:
    r = requests.post(f"{API_BASE_URL}/explain",
                      json={"code": code, "language": language,
                            "problem_title": title},
                      headers=_headers(), timeout=120)
    r.raise_for_status()
    return r.json()


# =========================================================================
# Page setup
# =========================================================================
st.set_page_config(
    page_title="Mockmate-LLM Interview Assistant",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for a CoderPad-like feel
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0; max-width: 1400px; }
    .complexity-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .pill-time    { background: #e3f2fd; color: #1565c0; }
    .pill-space   { background: #f3e5f5; color: #7b1fa2; }
    .pill-latency { background: #fff3e0; color: #e65100; }
    .pill-tokens  { background: #e8f5e9; color: #2e7d32; }
    .stButton>button { border-radius: 8px; }
    .problem-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================
# Session state
# =========================================================================
@dataclass
class Turn:
    role: str  # "user" | "assistant"
    title: str
    language: str
    difficulty: str
    code: str = ""
    time_complexity: str = ""
    space_complexity: str = ""
    elapsed_ms: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    explanation: str = ""
    error: str = ""


def _init_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history: List[Turn] = []
    if "busy" not in st.session_state:
        st.session_state.busy = False


# =========================================================================
# Components
# =========================================================================
def render_header() -> None:
    st.title("Mockmate-LLM · Interview Assistant")
    st.caption(
        "Fine-tuned DeepSeek-Coder 6.7B + QLoRA · generates code + Big-O "
        "complexity for LeetCode-style interview problems."
    )

    health = api_health()
    if health is None:
        st.error(
            f"Backend is not reachable at `{API_BASE_URL}`. "
            "Start it with: `uvicorn app_api:app --port 8000`"
        )
    else:
        adapter = health.get("adapter", "(none)") or "(none)"
        st.success(
            f"Backend ready — base: `{health.get('base_model','?')}`, "
            f"adapter: `{adapter}`"
        )


def render_problem_form() -> Optional[Dict[str, Any]]:
    """Render the problem-input card. Returns the problem dict on submit, else None."""
    with st.container(border=True):
        st.subheader("Problem")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            title = st.text_input("Title", value="Two Sum",
                                   placeholder="e.g., Longest Palindromic Substring")
        with col2:
            difficulty = st.selectbox("Difficulty", DIFFICULTIES, index=0)
        with col3:
            language = st.selectbox("Language", LANGUAGES, index=0)

        description = st.text_area(
            "Description", height=140,
            value="Given an array of integers `nums` and an integer `target`, "
                  "return indices of the two numbers such that they add up to target.\n\n"
                  "You may assume that each input has exactly one solution, "
                  "and you may not use the same element twice.",
        )
        examples = st.text_area(
            "Examples", height=80,
            value="Input: nums = [2,7,11,15], target = 9\nOutput: [0,1]\nExplanation: nums[0] + nums[1] == 9.",
        )
        constraints = st.text_area(
            "Constraints", height=60,
            value="2 <= nums.length <= 10^4\n-10^9 <= nums[i] <= 10^9",
        )

        submitted = st.button("Generate Solution", type="primary",
                              use_container_width=True,
                              disabled=st.session_state.busy)
        if submitted:
            if not title.strip() or not description.strip():
                st.error("Title and Description are required.")
                return None
            return {
                "title": title.strip(),
                "difficulty": difficulty,
                "description": description.strip(),
                "examples": examples.strip(),
                "constraints": constraints.strip(),
                "language": language,
            }
    return None


def render_solution(turn: Turn, idx: int) -> None:
    with st.container(border=True):
        # Header line
        st.markdown(
            f"**{turn.title}** · `{turn.difficulty}` · `{turn.language}`",
        )

        # Complexity pills
        st.markdown(
            f'<span class="complexity-pill pill-time">'
            f'⏱ Time: <code>{turn.time_complexity}</code></span>'
            f'<span class="complexity-pill pill-space">'
            f'💾 Space: <code>{turn.space_complexity}</code></span>'
            f'<span class="complexity-pill pill-latency">'
            f'⚡ {turn.elapsed_ms:.0f}ms</span>'
            f'<span class="complexity-pill pill-tokens">'
            f'🔤 {turn.prompt_tokens}/{turn.generated_tokens} tok</span>',
            unsafe_allow_html=True,
        )

        # Code block with copy button
        st.code(turn.code, language=turn.language)
        cpy1, cpy2, cpy3 = st.columns([1, 1, 6])
        with cpy1:
            st.button("📋 Copy code", key=f"copy_code_{idx}",
                      on_click=lambda: st.session_state.update(
                          _clipboard=turn.code))
        with cpy2:
            if st.button("Explain", key=f"explain_{idx}",
                         disabled=st.session_state.busy):
                _explain(turn, idx)

        if turn.explanation:
            with st.expander("Step-by-step explanation", expanded=True):
                st.markdown(turn.explanation)

        if turn.error:
            st.error(turn.error)


def _explain(turn: Turn, idx: int) -> None:
    st.session_state.busy = True
    try:
        out = api_explain(turn.code, turn.language, turn.title)
        turn.explanation = out.get("explanation", "")
    except Exception as e:
        turn.error = f"Explain failed: {e}"
    finally:
        st.session_state.busy = False


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Conversation")
        if not st.session_state.history:
            st.caption("No problems yet — submit one to get started.")
        for i, t in enumerate(st.session_state.history):
            label = f"{'🧑' if t.role == 'user' else '🤖'} #{i+1} · {t.title}"
            with st.expander(label):
                st.caption(f"{t.difficulty} · {t.language}")
                if t.code:
                    st.code(t.code[:200] + ("..." if len(t.code) > 200 else ""),
                            language=t.language)

        st.divider()
        if st.button("🧹 Clear history"):
            st.session_state.history = []
            st.rerun()

        st.caption(f"Backend: `{API_BASE_URL}`")
        st.caption("Built for [Mockmate](https://github.com/Rishabh01487/Mockmate-interview-platform)")


# =========================================================================
# Main flow
# =========================================================================
def main() -> None:
    _init_state()
    render_header()
    render_sidebar()

    problem = render_problem_form()
    if problem is None:
        # Still render any prior turns below the form
        for i, t in enumerate(st.session_state.history):
            render_solution(t, i)
        return

    # Append a user-side turn first
    st.session_state.history.append(Turn(
        role="user", title=problem["title"], language=problem["language"],
        difficulty=problem["difficulty"],
    ))

    # Generate
    st.session_state.busy = True
    with st.spinner(f"Generating {problem['language']} solution ..."):
        try:
            t0 = time.perf_counter()
            resp = api_generate(problem)
            sol = resp.get("solution", {})
            turn = Turn(
                role="assistant", title=problem["title"],
                language=problem["language"], difficulty=problem["difficulty"],
                code=sol.get("code", ""), time_complexity=sol.get("time_complexity", ""),
                space_complexity=sol.get("space_complexity", ""),
                elapsed_ms=sol.get("elapsed_ms", 0.0) or (time.perf_counter() - t0) * 1000,
                prompt_tokens=sol.get("prompt_tokens", 0),
                generated_tokens=sol.get("generated_tokens", 0),
            )
        except requests.HTTPError as e:
            turn = Turn(
                role="assistant", title=problem["title"],
                language=problem["language"], difficulty=problem["difficulty"],
                error=f"Backend returned {e.response.status_code}: "
                      f"{e.response.text[:200]}",
            )
        except Exception as e:
            turn = Turn(
                role="assistant", title=problem["title"],
                language=problem["language"], difficulty=problem["difficulty"],
                error=f"Request failed: {e}",
            )
        finally:
            st.session_state.busy = False

    st.session_state.history.append(turn)
    st.rerun()


if __name__ == "__main__":
    main()
