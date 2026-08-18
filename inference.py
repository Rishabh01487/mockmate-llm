"""
inference.py
============

Loads the QLoRA fine-tuned model and exposes a single high-level function:
    `generate_solution(problem: InterviewProblem) -> InterviewSolution`

Designed to be:
- importable by `app_api.py` (FastAPI) and `app_ui.py` (Streamlit).
- runnable directly from CLI for smoke tests:
    python inference.py --adapter ./checkpoints/deepseek-leetcode-qlora \\
        --title "Two Sum" --difficulty Easy --language python \\
        --description "Given an array..." --examples "..." --constraints "..."
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import time
from dataclasses import dataclass, asdict, field
from threading import Lock
from typing import Optional

try:
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        GenerationConfig,
    )
    _CUDA_AVAILABLE = True
except ImportError:
    # torch / transformers / peft not installed — only MOCK_MODE will work.
    torch = None
    PeftModel = None
    _CUDA_AVAILABLE = False


# Decorator that no-ops if torch is unavailable (for MOCK_MODE)
def _maybe_inference_mode(func):
    if torch is not None:
        return torch.inference_mode()(func)
    return func


# =========================================================================
# Dataclasses — wire format for API + UI
# =========================================================================
@dataclass
class InterviewProblem:
    title: str
    difficulty: str
    description: str
    examples: str
    constraints: str
    language: str  # python|java|cpp|javascript|c|go|rust


@dataclass
class InterviewSolution:
    code: str
    language: str
    time_complexity: str
    space_complexity: str
    raw_response: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_ms: float


# =========================================================================
# Constants — must match prepare_data.py & train_qlora.py exactly
# =========================================================================
SYSTEM_PROMPT = (
    "You are an expert programmer in a technical interview. "
    "Provide a clean, optimal solution, followed by its Time and Space complexity."
)
USER_TEMPLATE = """Solve the following problem in {language}:

**{title}** ({difficulty})
{description}

Examples:
{examples}

Constraints:
{constraints}

Write the {language} solution:"""
CHATML_SPECIAL_TOKENS = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]}
SUPPORTED_LANGS = {"python", "java", "cpp", "javascript", "c", "go", "rust"}


# =========================================================================
# Model manager — singleton loader
# =========================================================================
class ModelManager:
    """Singleton that loads base + LoRA adapter exactly once per process."""

    _instance: Optional["ModelManager"] = None
    _lock: Lock = Lock()
    _mock_mode: bool = os.getenv("MOCK_MODE", "0") == "1"

    def __init__(self, base_model: str, adapter_path: Optional[str]):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        if self._mock_mode:
            self._init_mock()
        else:
            self._load()

    def _init_mock(self) -> None:
        """Initialise the manager in MOCK_MODE — no CUDA, no real weights.
        Returns deterministic canned responses so the API/UI can be smoke-tested
        end-to-end before training is finished.
        """
        print("[inference] MOCK_MODE=1 — skipping model load; using canned responses.")
        self.generation_config = None  # not used in mock

    @classmethod
    def get(cls,
            base_model: Optional[str] = None,
            adapter_path: Optional[str] = None) -> "ModelManager":
        # Resolve from env if not provided
        base_model = base_model or os.getenv(
            "BASE_MODEL", "deepseek-ai/deepseek-coder-6.7b-base")
        adapter_path = adapter_path or os.getenv("ADAPTER_PATH")
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(base_model, adapter_path)
            return cls._instance

    def _load(self) -> None:
        print(f"[inference] Loading tokenizer: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True)
        self.tokenizer.add_special_tokens(CHATML_SPECIAL_TOKENS)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # for batched generation

        # Quantization config — match training
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        print(f"[inference] Loading base model: {self.base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Disable cache during inference batching is fine; we keep it on for speed.
        self.model.config.use_cache = True

        if self.adapter_path:
            print(f"[inference] Loading LoRA adapter: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model = self.model.merge_and_unload()
            print("[inference] LoRA adapter merged into base weights.")
        else:
            print("[inference] No adapter path provided — using base model only.")

        self.model.eval()
        self.generation_config = GenerationConfig(
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "1024")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            top_p=float(os.getenv("TOP_P", "0.95")),
            top_k=int(os.getenv("TOP_K", "50")),
            repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.05")),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
        print(f"[inference] Ready. Generation config: {self.generation_config}")

    @_maybe_inference_mode
    def generate_raw(self, prompt: str) -> tuple[str, int, int]:
        """Generate raw text from a ChatML prompt. Returns (text, in_tokens, out_tokens)."""
        if self._mock_mode:
            return self._mock_generate_raw(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=2048).to(self.model.device)
        in_tokens = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
        )
        # Strip the prompt portion
        gen_ids = out[0, in_tokens:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
        out_tokens = gen_ids.shape[0]
        return text, in_tokens, out_tokens

    # ---------------------------------------------------------------------
    # MOCK implementation — used when MOCK_MODE=1 is set in the environment.
    # Returns deterministic canned responses so the full API + UI stack can be
    # smoke-tested end-to-end before training is finished.
    # ---------------------------------------------------------------------
    _MOCK_SOLUTIONS = {
        "python": 'class Solution:\n    def solve(self, *args, **kwargs):\n        # TODO: implement\n        raise NotImplementedError()',
        "java":   'class Solution {\n    public Object solve(Object... args) {\n        // TODO: implement\n        throw new UnsupportedOperationException();\n    }\n}',
        "cpp":    'class Solution {\npublic:\n    // TODO: implement\n};',
        "javascript": '/**\n * TODO: implement\n */\nvar solve = function(...args) {\n    throw new Error("not implemented");\n};',
        "c":     '// TODO: implement\nint solve(int* nums, int n) {\n    return 0;\n}',
        "go":    'package main\n\n// TODO: implement\nfunc Solve(nums []int) int {\n    return 0\n}',
        "rust":  '// TODO: implement\npub fn solve(nums: Vec<i32>) -> i32 {\n    0\n}',
    }

    def _mock_generate_raw(self, prompt: str) -> tuple[str, int, int]:
        """Deterministic fake response. Detects language from the prompt and
        returns a stub solution + Big-O placeholder."""
        # Extract language from "<|im_start|>user\nSolve the following problem in {language}:"
        m = re.search(r"in (python|java|cpp|javascript|c|go|rust):", prompt)
        lang = m.group(1) if m else "python"
        code = self._MOCK_SOLUTIONS.get(lang, self._MOCK_SOLUTIONS["python"])
        response = (
            f"```{lang}\n{code}\n```\n\n"
            f"**Time Complexity:** O(1)\n"
            f"**Space Complexity:** O(1)<|im_end|>"
        )
        # Fake token counts (heuristic: 1 token per ~4 chars)
        in_tok = max(1, len(prompt) // 4)
        out_tok = max(1, len(response) // 4)
        return response, in_tok, out_tok

    def generate_solution(self, problem: InterviewProblem) -> InterviewSolution:
        prompt = self._build_prompt(problem)
        t0 = time.perf_counter()
        raw, in_tok, out_tok = self.generate_raw(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        code, tc, sc = parse_response(raw, problem.language)
        return InterviewSolution(
            code=code, language=problem.language,
            time_complexity=tc, space_complexity=sc,
            raw_response=raw, prompt_tokens=in_tok,
            generated_tokens=out_tok, elapsed_ms=elapsed_ms,
        )

    def _build_prompt(self, p: InterviewProblem) -> str:
        user_text = USER_TEMPLATE.format(
            language=p.language, title=p.title, difficulty=p.difficulty,
            description=p.description.strip(),
            examples=p.examples.strip(),
            constraints=p.constraints.strip(),
        )
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


# =========================================================================
# Response parser — extract code + Big-O from the model output
# =========================================================================
# Accepts:
#   ```python\n<code>\n```\n**Time Complexity:** O(n)\n**Space Complexity:** O(1)
# OR (more permissive):
#   <code>\nTime: O(n)\nSpace: O(1)
# -------------------------------------------------------------------------
CODE_FENCE_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+\-]+)?\s*\n(.*?)```",
    re.DOTALL,
)
TIME_RE = re.compile(
    r"(?:\*\*)?Time\s*Complexity(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*([OΩΘoωθ]\s*\([^)]*\))",
    re.IGNORECASE,
)
SPACE_RE = re.compile(
    r"(?:\*\*)?Space\s*Complexity(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*([OΩΘoωθ]\s*\([^)]*\))",
    re.IGNORECASE,
)


def parse_response(raw: str, language: str) -> tuple[str, str, str]:
    """Extract (code, time_complexity, space_complexity) from model output."""
    # Trim at the ChatML end token (the model may continue past it)
    if "<|im_end|>" in raw:
        raw = raw.split("<|im_end|>")[0]

    # 1) Extract code block
    m = CODE_FENCE_RE.search(raw)
    if m:
        code = m.group(1).strip()
    else:
        # No fence — assume the entire (pre-complexity) section is code
        code = re.split(
            r"(?:\*\*)?Time\s*Complexity", raw, flags=re.IGNORECASE)[0].strip()
        # Strip a leading language tag if present (rare)
        if code.startswith(f"```{language}"):
            code = code[len(f"```{language}"):].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

    # 2) Extract Big-O
    tm = TIME_RE.search(raw)
    sm = SPACE_RE.search(raw)
    time_c = tm.group(1).replace(" ", "") if tm else "O(unknown)"
    space_c = sm.group(1).replace(" ", "") if sm else "O(unknown)"

    return code, time_c, space_c


# =========================================================================
# Smoke-test entry point
# =========================================================================
def _smoke_test(args: argparse.Namespace) -> None:
    problem = InterviewProblem(
        title=args.title, difficulty=args.difficulty,
        description=args.description, examples=args.examples,
        constraints=args.constraints, language=args.language,
    )
    if problem.language not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported language: {problem.language}")

    mgr = ModelManager.get(adapter_path=args.adapter)
    sol = mgr.generate_solution(problem)
    print("\n=== GENERATED SOLUTION ===")
    print(f"// Time Complexity : {sol.time_complexity}")
    print(f"// Space Complexity: {sol.space_complexity}")
    print(f"// Tokens in/out   : {sol.prompt_tokens}/{sol.generated_tokens}")
    print(f"// Latency (ms)    : {sol.elapsed_ms:.0f}\n")
    print(f"```{sol.language}\n{sol.code}\n```")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=str, default=None,
                   help="Path to LoRA adapter dir")
    p.add_argument("--base_model", type=str,
                   default="deepseek-ai/deepseek-coder-6.7b-base")
    p.add_argument("--title", type=str, default="Two Sum")
    p.add_argument("--difficulty", type=str, default="Easy")
    p.add_argument("--language", type=str, default="python")
    p.add_argument("--description", type=str,
                   default="Given an array of integers nums and an integer target, "
                           "return indices of the two numbers such that they add up to target.")
    p.add_argument("--examples", type=str,
                   default="Input: nums = [2,7,11,15], target = 9 -> Output: [0,1]")
    p.add_argument("--constraints", type=str,
                   default="2 <= nums.length <= 10^4; -10^9 <= nums[i] <= 10^9")
    return p.parse_args()


if __name__ == "__main__":
    _smoke_test(parse_args())
