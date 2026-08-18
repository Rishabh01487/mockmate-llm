"""
prepare_data.py
================

Converts `leetcode_dataset.jsonl` into a ChatML-formatted, HuggingFace-ready
dataset suitable for QLoRA fine-tuning of `deepseek-ai/deepseek-coder-6.7b-base`.

Key responsibilities
--------------------
1. Load & validate the raw JSONL.
2. Synthesize Time/Space complexity labels per solution (heuristic + LLM hook).
3. Render the EXACT ChatML template requested by the interview pipeline:
       <|im_start|>system ...
       <|im_start|>user ...
       <|im_start|>assistant ```{language}\n{solution}\n```\n**Time Complexity:** ...\n**Space Complexity:** ...<|im_end|>
4. Train/val split (stratified by difficulty + language).
5. Save to disk as both raw JSONL (for inspection) and HuggingFace `datasets.Dataset`.

Usage
-----
    python prepare_data.py \
        --input  ./leetcode_dataset.jsonl \
        --out_dir ./data_processed \
        --val_size 0.05 \
        --seed 42

The output directory will contain:
    data_processed/
      train.jsonl
      val.jsonl
      chatml_dataset/         # HuggingFace arrow cache
        dataset_info.json
        ...
    stats.json                # row counts, distribution, sample preview
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# --- HuggingFace datasets (lazy import so the script can --dry-run fast) ---
try:
    from datasets import Dataset, DatasetDict
    import pyarrow as pa
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False


# =========================================================================
# 1. Constants
# =========================================================================
SUPPORTED_LANGS = {"python", "java", "cpp", "javascript", "c", "go", "rust"}

SYSTEM_PROMPT = (
    "You are an expert programmer in a technical interview. "
    "Provide a clean, optimal solution, followed by its Time and Space complexity."
)

# The exact user-side template (matches the spec in the task description).
USER_TEMPLATE = """Solve the following problem in {language}:

**{title}** ({difficulty})
{description}

Examples:
{examples}

Constraints:
{constraints}

Write the {language} solution:"""

# The assistant-side template. Note: we always close the code fence AND emit
# the Big-O block, then the ChatML end token. This is what the model learns
# to reproduce at inference time.
ASSISTANT_TEMPLATE = """```{language}
{solution}
```

**Time Complexity:** {time_complexity}
**Space Complexity:** {space_complexity}"""


# =========================================================================
# 2. Heuristic Big-O complexity estimator
# =========================================================================
# A reasonable, language-agnostic starting point. You can swap this out for an
# LLM-based labeler (see `label_complexity_with_llm` hook below).
#
# Heuristics:
#   - `sorted(` / `.sort(` / `PriorityQueue` / `TreeMap`  -> O(n log n)
#   - nested for-loops (depth=2)                            -> O(n^2)
#   - nested for-loops (depth=3)                            -> O(n^3)
#   - recursive binary split / `bisect` / `lower_bound`     -> O(log n) per op
#   - hashmap / set lookup inside loop                      -> O(n) total
#   - DP table fills nxm                                     -> O(n*m)
#   - backtracking / DFS permutations                       -> O(2^n) or O(n!)
# -------------------------------------------------------------------------

def _normalize_solution_for_analysis(code: str) -> str:
    """Strip comments and strings to reduce false-positive matches."""
    # Remove block comments /* */ (Java/JS/C++)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # Remove line comments // ...
    code = re.sub(r"//[^\n]*", "", code)
    # Remove Python-style comments # ...
    code = re.sub(r"(?m)^\s*#[^\n]*", "", code)
    # Remove string literals (rough — good enough for heuristic)
    code = re.sub(r'"(\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(\\.|[^'\\])*'", "''", code)
    return code


def _max_loop_depth(code: str, lang: str) -> int:
    """Estimate the maximum nesting depth of for/while loops.

    For C-like languages, we walk the source character-by-character while
    maintaining a brace depth, and record the brace depth *at the moment*
    each `for(`/`while(` is encountered. The outermost brace depth inside a
    function body is treated as depth 1, so a single loop is depth 1, a
    nested loop is depth 2, etc. This avoids counting class+function braces
    as loop nesting.
    """
    if lang == "python":
        loop_re = re.compile(r"^\s*(for|while)\b", re.MULTILINE)
        lines = code.splitlines()
        depth_per_line: List[int] = []
        stack: List[int] = []
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)
            while stack and stack[-1] >= indent:
                stack.pop()
            if loop_re.match(stripped):
                stack.append(indent)
            depth_per_line.append(len(stack))
        return max(depth_per_line) if depth_per_line else 0
    else:
        # Find the first `{` that opens the function body, then walk forward.
        loop_re = re.compile(r"\b(for|while)\s*\(")
        # Skip everything before the first '{' so we don't get confused by class braces.
        first_brace = code.find("{")
        if first_brace < 0:
            return 0
        body = code[first_brace + 1:]
        depth = 1  # we're now inside the function body
        loop_depths: List[int] = []
        i = 0
        while i < len(body):
            ch = body[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth = max(1, depth - 1)
            elif ch in "fw":
                # Try to match a loop keyword at this position
                if loop_re.match(body, i):
                    loop_depths.append(depth)
                    i += 4  # skip 'for(' / 'whi'
                    continue
            i += 1
        if not loop_depths:
            return 0
        # The deepest loop encountered, minus the function-body offset (1)
        return max(1, max(loop_depths) - 1)


def _has_sort(code: str) -> bool:
    """Detect any sort/sorted call across Python/Java/JS/C++."""
    return bool(re.search(
        r"\bsorted\(|\.sort\(|\bsort\(|PriorityQueue|TreeMap|TreeSet|collections\.OrderedDict|\.orderBy\(",
        code,
    ))


def _has_two_pointer(code: str) -> bool:
    """Detect a classic two-pointer pattern: declaration of `l, r`
    (or `left, right`) followed by a `while l < r` contraction.
    Works across C++/Java/JavaScript (parens, ends with `{`) and Python
    (no parens, ends with `:`).
    """
    # Find any `l, r` declaration first
    decl_re = re.compile(
        r"(?:int\s+|let\s+|const\s+|var\s+)?(l|left)\s*[,=].*?(r|right)\b",
        re.DOTALL,
    )
    if not decl_re.search(code):
        return False
    # Then check for the `while X < Y` contraction using the same names
    return bool(re.search(
        r"while\s*\(?\s*(?:l|left)\s*<\s*(?:r|right)\b",
        code,
    ))


def _has_expand_around_center(code: str) -> bool:
    """Detect the classic palindrome expand-around-center helper:
        while (a >= 0 && b < ...) { ... }      // C-like
        while a >= 0 and b < len(s) and ...:    # Python
    This pattern runs O(n) per call but is called O(n) times by the outer
    loop, giving O(n^2) total.
    """
    return bool(re.search(
        r"while\s*\(?\s*(?:\w+)\s*>=\s*0\b.*?(?:&&|\band\b)\s*\w+\s*<\s*(?:\w+\.)?(?:length|size|len)\b",
        code,
    ))


def heuristic_complexity(solution: str, language: str) -> Tuple[str, str]:
    """Return (time_complexity, space_complexity) as Big-O strings."""
    code = _normalize_solution_for_analysis(solution)
    n_loops = _max_loop_depth(code, language)

    # --- Time complexity ---
    # Exponential patterns first (most specific)
    if re.search(r"\bpermute|permutations|itertools\.permutations|next_permutation\b", code, re.IGNORECASE):
        time_c = "O(n!)"
    elif re.search(r"\bbacktrack|dfs\(.*path|recursion.*recursion|combinations\(|subsets\b", code, re.IGNORECASE):
        time_c = "O(2^n)"
    # DP / memo tables
    elif re.search(r"\bdp\[|memo\[|f\[i\]\[j\]\b", code):
        time_c = "O(n * m)" if re.search(r"\[\d*\]\s*\[", code) else "O(n^2)"
    # Two-pointer after sort (3Sum / Container With Most Water / etc.)
    # Sort is O(n log n), two-pointer scan is O(n), so total = O(n^2) regardless of any
    # dedup while-loops inside the inner pointer.
    elif _has_sort(code) and _has_two_pointer(code) and n_loops >= 2:
        time_c = "O(n^2)"
    # Expand-around-center (Longest Palindrome) — O(n^2)
    elif _has_expand_around_center(code):
        time_c = "O(n^2)"
    # Sorting alone
    elif _has_sort(code):
        time_c = "O(n log n)"
    # Nested loops
    elif n_loops >= 3:
        time_c = "O(n^3)"
    elif n_loops == 2:
        time_c = "O(n^2)"
    elif n_loops == 1:
        time_c = "O(n)"
    # Binary search
    elif re.search(r"\bbisect|binary_search|lower_bound|upper_bound|>> 1\b", code):
        time_c = "O(log n)"
    else:
        time_c = "O(n)"

    # --- Space complexity ---
    if re.search(r"\bdp\[|memo\[|f\[i\]\[j\]\]", code):
        space_c = "O(n * m)" if re.search(r"\[\d*\]\s*\[", code) else "O(n^2)"
    elif re.search(r"\brecursion|backtrack|dfs\(", code, re.IGNORECASE):
        space_c = "O(n)"   # call stack
    elif re.search(r"\bset\(|dict\(|HashMap|HashSet|unordered_set|unordered_map|defaultdict", code):
        space_c = "O(n)"
    elif n_loops >= 2 and re.search(r"\bappend\(|\.push_back\(|\.add\(", code):
        space_c = "O(n^2)"
    else:
        space_c = "O(1)"

    return time_c, space_c


# =========================================================================
# 3. Optional LLM-based complexity labeler (HOOK)
# =========================================================================
# If you want higher-quality complexity labels than the heuristic above,
# plug in any chat-completion LLM here. Set the env var `COMPLEXITY_LLM_PROVIDER`
# to one of: "openai" | "anthropic" | "zai" and provide the relevant API key.
#
# The function receives (solution, language, problem_title) and must return
# (time_complexity, space_complexity). On any error, it falls back to the
# heuristic.
# -------------------------------------------------------------------------
def label_complexity_with_llm(
    solution: str, language: str, title: str
) -> Tuple[str, str]:
    provider = os.getenv("COMPLEXITY_LLM_PROVIDER", "").lower()
    if not provider:
        return heuristic_complexity(solution, language)

    try:
        if provider == "zai":
            # Uses z-ai CLI (installed system-wide as /usr/local/bin/z-ai).
            # The CLI prints a JSON envelope to stdout containing the model's
            # response; we parse `choices[0].message.content` and strip any
            # markdown code fence around it before JSON-parsing.
            prompt = (
                f"Analyze this {language} solution to '{title}' and respond with ONLY valid JSON "
                f"(no markdown, no commentary) in this exact shape: "
                f'{{"time": "O(...)", "space": "O(...)"}}\n\n'
                f"```\n{solution}\n```"
            )
            result = subprocess.run(
                ["z-ai", "chat", "--prompt", prompt],
                capture_output=True, text=True, timeout=30, check=True,
            )
            # The CLI emits "🚀 Initializing..." / "🚀 Sending..." status lines
            # to stdout, followed by a JSON envelope. Find the first `{` and
            # parse from there.
            stdout = result.stdout
            json_start = stdout.find("{")
            if json_start < 0:
                raise ValueError("No JSON in z-ai CLI output: " + stdout[:200])
            envelope = json.loads(stdout[json_start:])
            content = envelope["choices"][0]["message"]["content"].strip()
            # Strip markdown fence if the model added one anyway.
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*\n", "", content)
                content = re.sub(r"\n```\s*$", "", content)
            data = json.loads(content)
            return data["time"], data["space"]
        # Add openai / anthropic clients here if desired.
    except Exception as e:  # pragma: no cover
        print(f"[LLM labeler] fallback to heuristic: {e}", file=sys.stderr)
    return heuristic_complexity(solution, language)


# =========================================================================
# 4. ChatML formatting
# =========================================================================
def render_chatml(
    *, title: str, difficulty: str, description: str,
    examples: str, constraints: str, language: str, solution: str,
    time_complexity: str, space_complexity: str,
) -> str:
    """Render the full ChatML string with the exact template from the spec."""
    user_text = USER_TEMPLATE.format(
        language=language, title=title, difficulty=difficulty,
        description=description.strip(), examples=examples.strip(),
        constraints=constraints.strip(),
    )
    assistant_text = ASSISTANT_TEMPLATE.format(
        language=language, solution=solution.rstrip(),
        time_complexity=time_complexity, space_complexity=space_complexity,
    )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_text}<|im_end|>"
    )


# =========================================================================
# 5. Validation
# =========================================================================
def validate_row(row: Dict[str, Any], idx: int) -> Optional[str]:
    """Return None if valid, else an error message."""
    required = ["problem_id", "title", "difficulty", "description",
                "examples", "constraints", "language", "solution"]
    for k in required:
        if k not in row:
            return f"row {idx}: missing key '{k}'"
        val = row[k]
        if not isinstance(val, str) and k != "problem_id":
            return f"row {idx}: key '{k}' must be string, got {type(val).__name__}"
        if k != "problem_id" and not val.strip():
            return f"row {idx}: key '{k}' is empty"
    if row["language"].lower() not in SUPPORTED_LANGS:
        return (f"row {idx}: unsupported language '{row['language']}'. "
                f"Supported: {sorted(SUPPORTED_LANGS)}")
    return None


# =========================================================================
# 6. Main pipeline
# =========================================================================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error on line {i+1}: {e}")
    return rows


def build_examples(rows: List[Dict[str, Any]], use_llm: bool) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    skipped = 0
    for row in tqdm(rows, desc="Formatting ChatML"):
        err = validate_row(row, len(examples))
        if err:
            print(f"[skip] {err}", file=sys.stderr)
            skipped += 1
            continue
        lang = row["language"].lower()
        sol = row["solution"]
        if use_llm:
            tc, sc = label_complexity_with_llm(sol, lang, row["title"])
        else:
            tc, sc = heuristic_complexity(sol, lang)
        examples.append({
            "problem_id": row["problem_id"],
            "title": row["title"],
            "difficulty": row["difficulty"],
            "language": lang,
            "time_complexity": tc,
            "space_complexity": sc,
            "chatml_text": render_chatml(
                title=row["title"], difficulty=row["difficulty"],
                description=row["description"], examples=row["examples"],
                constraints=row["constraints"], language=lang, solution=sol,
                time_complexity=tc, space_complexity=sc,
            ),
        })
    print(f"[data] Built {len(examples)} examples, skipped {skipped} malformed rows.")
    return examples


def stratified_split(
    examples: List[Dict[str, Any]], val_size: float, seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratify by (difficulty, language) so the val set is representative."""
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        buckets[(ex["difficulty"], ex["language"])].append(ex)
    train, val = [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_size)) if len(items) > 1 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_hf_dataset(
    train: List[Dict[str, Any]], val: List[Dict[str, Any]], out_dir: Path
) -> None:
    if not HF_AVAILABLE:
        print("[warn] datasets/pyarrow not installed — skipping HF cache.")
        return
    cols = ["problem_id", "title", "difficulty", "language",
            "time_complexity", "space_complexity", "chatml_text"]
    ds = DatasetDict({
        "train": Dataset.from_list([{c: r[c] for c in cols} for r in train]),
        "val":   Dataset.from_list([{c: r[c] for c in cols} for r in val]),
    })
    ds.save_to_disk(str(out_dir))
    print(f"[hf] Saved DatasetDict to {out_dir}  (train={len(train)}, val={len(val)})")


def write_stats(
    train: List[Dict[str, Any]], val: List[Dict[str, Any]],
    out_path: Path, sample: Optional[Dict[str, Any]] = None,
) -> None:
    diff_dist = Counter(r["difficulty"] for r in train + val)
    lang_dist = Counter(r["language"] for r in train + val)
    tc_dist = Counter(r["time_complexity"] for r in train + val)
    stats = {
        "train_count": len(train),
        "val_count": len(val),
        "difficulty_distribution": dict(diff_dist),
        "language_distribution": dict(lang_dist),
        "time_complexity_distribution": dict(tc_dist),
        "sample": sample,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[stats] Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare LeetCode dataset for QLoRA training.")
    p.add_argument("--input", type=Path, default=Path("./leetcode_dataset.jsonl"),
                   help="Path to leetcode_dataset.jsonl")
    p.add_argument("--out_dir", type=Path, default=Path("./data_processed"),
                   help="Output directory for processed dataset")
    p.add_argument("--val_size", type=float, default=0.05,
                   help="Validation fraction (stratified)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_llm_complexity", action="store_true",
                   help="Use LLM-based complexity labeler instead of heuristic")
    p.add_argument("--max_rows", type=int, default=0,
                   help="Cap total rows (0 = no cap). Useful for smoke tests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    rows = load_jsonl(args.input)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
        print(f"[data] Capped to {len(rows)} rows (smoke test mode).")
    print(f"[data] Loaded {len(rows)} rows from {args.input}")

    examples = build_examples(rows, use_llm=args.use_llm_complexity)
    if not examples:
        raise RuntimeError("No valid examples after formatting — check input schema.")

    train, val = stratified_split(examples, args.val_size, args.seed)
    print(f"[split] train={len(train)} val={len(val)}")

    save_jsonl(train, args.out_dir / "train.jsonl")
    save_jsonl(val, args.out_dir / "val.jsonl")
    save_hf_dataset(train, val, args.out_dir / "chatml_dataset")
    write_stats(train, val, args.out_dir / "stats.json",
                sample=train[0] if train else None)

    print("\n=== PREVIEW (first train example) ===")
    print(train[0]["chatml_text"][:2000])
    print("\n... [truncated]")


if __name__ == "__main__":
    main()
