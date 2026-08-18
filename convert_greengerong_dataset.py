"""
convert_greengerong_dataset.py
==============================

Converts the `greengerong/leetcode` HuggingFace dataset (2,360 problems × 4
languages) into the JSONL schema expected by `prepare_data.py`.

Input schema (greengerong/leetcode, JSONL):
    {
      "id": 1,
      "slug": "two-sum",
      "title": "Two Sum",
      "difficulty": "Easy",
      "content": "<markdown with description + examples + constraints>",
      "java":     "```java\n<solution code>\n``` <explanation>",
      "c++":      "```cpp\n<solution code>\n``` <explanation>",
      "python":   "```python\n<solution code>\n``` <explanation>",
      "javascript":"```javascript\n<solution code>\n``` <explanation>"
    }

Output schema (one row per problem-language pair):
    {
      "problem_id": 1,
      "title": "Two Sum",
      "difficulty": "Easy",
      "description": "...",
      "examples": "...",
      "constraints": "...",
      "language": "python",
      "solution": "class Solution: ..."
    }

Run:
    python convert_greengerong_dataset.py \
        --input /tmp/greengerong.jsonl \
        --out ./leetcode_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple


# =========================================================================
# Markdown parsing helpers
# =========================================================================
EXAMPLE_RE = re.compile(
    r"\*\*Example\s*\d+:?\*\*[^\n]*\n(.*?)(?=\*\*Example\s*\d+:?\*\*|\*\*Constraints:?\*\*|$)",
    re.DOTALL | re.IGNORECASE,
)
CONSTRAINTS_RE = re.compile(
    r"\*\*Constraints?:?\*\*\s*\n(.*?)$",
    re.DOTALL | re.IGNORECASE,
)
# Strip HTML tags from content
HTML_TAG_RE = re.compile(r"<[^>]+>")
# Strip markdown backticks/inline code already present — we keep code spans
# inline because they're useful context (e.g., `nums`, `target`)
CODE_FENCE_RE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)


def clean_text(text: str) -> str:
    """Normalize whitespace and strip HTML from a content blob."""
    # Remove HTML tags
    text = HTML_TAG_RE.sub("", text)
    # Normalize newlines
    text = re.sub(r"\r\n", "\n", text)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_content(content: str) -> Tuple[str, str, str]:
    """Split the markdown `content` into (description, examples, constraints)."""
    content = clean_text(content)

    # Find all example blocks
    example_matches = EXAMPLE_RE.findall(content)
    examples_text = "\n\n".join(
        f"Example {i+1}:\n{ex.strip()}"
        for i, ex in enumerate(example_matches)
    ) if example_matches else ""

    # Find constraints
    cm = CONSTRAINTS_RE.search(content)
    constraints_text = cm.group(1).strip() if cm else ""

    # Description = everything before the first "**Example"
    desc_end_match = re.search(r"\*\*Example\s*\d+:?\*\*", content, re.IGNORECASE)
    if desc_end_match:
        description = content[:desc_end_match.start()].strip()
    else:
        # No examples — take everything before constraints
        cm2 = CONSTRAINTS_RE.search(content)
        description = content[:cm2.start()].strip() if cm2 else content

    # Remove the "**Constraints:**" header from description if it leaked
    description = re.sub(r"\*\*Constraints?:?\*\*\s*$", "", description).strip()

    return description, examples_text, constraints_text


def extract_code_from_solution_block(block: str) -> Optional[str]:
    """Extract the first ```lang\n<code>\n``` block from a solution field.
    Returns just the code, no fence, no surrounding explanation.
    """
    if not block or not block.strip():
        return None
    m = CODE_FENCE_RE.search(block)
    if m:
        return m.group(2).strip()
    # No fence — return the whole thing stripped
    return block.strip()


# =========================================================================
# Main converter
# =========================================================================
LANG_FIELDS = [("python", "python"), ("java", "java"),
               ("cpp", "c++"), ("javascript", "javascript")]


def convert(input_path: Path, out_path: Path) -> dict:
    """Convert greengerong JSONL to our schema. Returns stats dict."""
    out_rows = []
    seen_problems = set()
    problems_with_all_4_langs = 0
    lang_counts = {l: 0 for l, _ in LANG_FIELDS}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("id")
            if pid is None:
                continue
            seen_problems.add(pid)

            title = (r.get("title") or "").strip()
            difficulty = (r.get("difficulty") or "Medium").strip().capitalize()
            content = r.get("content") or ""
            description, examples, constraints = split_content(content)

            # Track whether all 4 languages have non-empty solutions
            have_all = True
            for lang_name, field_name in LANG_FIELDS:
                raw = r.get(field_name) or ""
                code = extract_code_from_solution_block(raw)
                if not code or len(code) < 10:
                    have_all = False
                    continue
                out_rows.append({
                    "problem_id": int(pid),
                    "title": title,
                    "difficulty": difficulty,
                    "description": description,
                    "examples": examples,
                    "constraints": constraints,
                    "language": lang_name,
                    "solution": code,
                })
                lang_counts[lang_name] += 1
            if have_all:
                problems_with_all_4_langs += 1

    # Write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "total_problems_in_source": len(seen_problems),
        "total_rows_emitted": len(out_rows),
        "problems_with_all_4_languages": problems_with_all_4_langs,
        "rows_per_language": lang_counts,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("/tmp/greengerong.jsonl"),
                   help="Path to greengerong/leetcode leetcode-train.jsonl")
    p.add_argument("--out", type=Path, default=Path("./leetcode_dataset.jsonl"),
                   help="Output JSONL path")
    args = p.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    print(f"[convert] Reading {args.input} ...")
    stats = convert(args.input, args.out)
    print(f"[convert] Wrote {stats['total_rows_emitted']} rows to {args.out}")
    print(f"[convert]   Source problems:        {stats['total_problems_in_source']}")
    print(f"[convert]   Problems with all 4 langs: {stats['problems_with_all_4_languages']}")
    print(f"[convert]   Rows per language:")
    for lang, count in stats["rows_per_language"].items():
        print(f"     {lang:12} {count}")

    # Print a preview of the first row
    print("\n=== PREVIEW (first row) ===")
    with args.out.open() as f:
        first = json.loads(f.readline())
        for k, v in first.items():
            val = str(v)
            if len(val) > 300:
                val = val[:300] + "..."
            print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
