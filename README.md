# Mockmate-LLM

Fine-tuned **DeepSeek-Coder 6.7B** (QLoRA, 4-bit NF4) on 5,000 LeetCode problems
across Python / Java / C++ / JavaScript, exposed through a **FastAPI** backend and
**Streamlit** UI for direct integration into the
[Mockmate interview platform](https://github.com/Rishabh01487/Mockmate-interview-platform).

## Pipeline Overview

```
leetcode_dataset.jsonl
        │
        ▼
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ prepare_data.py  │ ──► │ train_qlora.py   │ ──► │  LoRA adapter       │
│  (ChatML + Big-O)│     │  (QLoRA 4-bit)   │     │  checkpoints/       │
└──────────────────┘     └──────────────────┘     └─────────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │ inference.py     │
                                                 │ (singleton mgr)  │
                                                 └──────────────────┘
                                                          │
                              ┌───────────────────────────┴───────────────────────────┐
                              ▼                                                       ▼
                      ┌──────────────────┐                                  ┌──────────────────┐
                      │ app_api.py       │  ◄─── HTTP/CORS ───              │ app_ui.py        │
                      │ FastAPI :8000    │                                  │ Streamlit :8501  │
                      └──────────────────┘                                  └──────────────────┘
                              ▲
                              │  REST /generate
                              │
                    ┌──────────────────┐
                    │  Mockmate (Next.js)│
                    │  interview-platform│
                    └──────────────────┘
```

## File Manifest

| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned deps for training + API + UI (CUDA 12.1). |
| `prepare_data.py`  | JSONL → ChatML formatted HF dataset, with heuristic Big-O labels (LLM-upgradeable). |
| `train_qlora.py`   | QLoRA fine-tune deepseek-coder-6.7b-base via SFTTrainer + LoRA + 4-bit NF4. |
| `inference.py`     | Singleton `ModelManager` — loads base + adapter, generates & parses ChatML. |
| `app_api.py`       | FastAPI service: `/generate`, `/generate/batch`, `/explain`, `/health`, `/ready`. |
| `app_ui.py`        | Streamlit UI — CoderPad-like interview assistant with history + copy buttons. |
| `README.md`        | This file. |

## Quickstart

```bash
# 1. Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Prepare data (expects ./leetcode_dataset.jsonl)
python prepare_data.py --input leetcode_dataset.jsonl --out_dir ./data_processed --val_size 0.05

# 3. Smoke-test on a tiny slice first (recommended)
python prepare_data.py --max_rows 100 --out_dir ./data_processed_smoke
python train_qlora.py --data_dir ./data_processed_smoke --epochs 1 --output_dir ./ckpts_smoke

# 4. Full training (~6-9 h on a single RTX 4090)
python train_qlora.py \
    --data_dir ./data_processed \
    --base_model deepseek-ai/deepseek-coder-6.7b-base \
    --output_dir ./checkpoints/deepseek-leetcode-qlora \
    --epochs 3 --batch_size 4 --grad_accum 4 --lr 2e-4 --max_seq_len 1024

# 5. Smoke-test the adapter
python inference.py --adapter ./checkpoints/deepseek-leetcode-qlora \
    --title "Two Sum" --difficulty Easy --language python

# 6. Launch API
export ADAPTER_PATH=./checkpoints/deepseek-leetcode-qlora
uvicorn app_api:app --host 0.0.0.0 --port 8000 --workers 1

# 7. Launch UI (separate terminal)
streamlit run app_ui.py --server.port 8501
```

## Mockmate Integration

The Mockmate platform ([github.com/Rishabh01487/Mockmate-interview-platform](https://github.com/Rishabh01487/Mockmate-interview-platform))
is a Next.js app. You can integrate this LLM service in **either of two ways**:

### Option A — Iframe the Streamlit UI

Embed `https://<your-host>:8501` as an iframe inside Mockmate's interview room.
This is the lowest-code integration; the Streamlit UI handles its own state.

```tsx
// In Mockmate — e.g. app/interview/components/AICopilot.tsx
"use client";
export function AICopilot() {
  return (
    <iframe
      src={process.env.NEXT_PUBLIC_LLM_UI_URL}  // https://llm.example.com:8501
      className="w-full h-full border-0"
      title="AI Interview Copilot"
    />
  );
}
```

### Option B — Call the REST API directly from Mockmate

Recommended for tighter UX control. From Mockmate's server actions or API routes:

```ts
// app/api/interview/llm-solution/route.ts
import { NextRequest, NextResponse } from "next/server";

const LLM_API = process.env.LLM_API_URL!;           // https://llm.example.com:8000
const LLM_KEY = process.env.LLM_API_KEY!;           // server-only

export async function POST(req: NextRequest) {
  const body = await req.json();
  const r = await fetch(`${LLM_API}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LLM_KEY}`,
    },
    body: JSON.stringify({
      title: body.title,
      difficulty: body.difficulty,
      description: body.description,
      examples: body.examples ?? "",
      constraints: body.constraints ?? "",
      language: body.language ?? "python",
    }),
    cache: "no-store",
  });
  const data = await r.json();
  return NextResponse.json(data);
}
```

Then in Mockmate's editor component, fire a debounced call whenever a problem is
loaded:

```ts
// Inside the interview editor page (pseudocode)
const { solution } = await fetch("/api/interview/llm-solution", {
  method: "POST",
  body: JSON.stringify({ title, difficulty, description, examples, constraints, language }),
}).then(r => r.json()).then(d => d.solution);

editor.setValue(solution.code);
toast.success(`AI hint: ${solution.time_complexity} / ${solution.space_complexity}`);
```

### Env vars to set in Mockmate's `.env.local`

```bash
LLM_API_URL=http://localhost:8000           # or https://llm.your-host.com
LLM_API_KEY=                                # set only if API_KEY is set on the backend
NEXT_PUBLIC_LLM_UI_URL=http://localhost:8501 # for the iframe option
```

### CORS

`app_api.py` already allows the following origins by default:
- `http://localhost:5173` (Vite)
- `http://localhost:3000` (Next.js dev)
- `http://localhost:8501` (Streamlit)

For production, override with:
```bash
export ALLOWED_ORIGINS=https://mockmate.your-host.com,https://www.mockmate.your-host.com
```

## Dataset Schema

`prepare_data.py` expects one JSONL row per (problem, language) tuple:

```jsonc
{"problem_id": 1, "title": "Two Sum", "difficulty": "Easy",
 "description": "Given an array...", "examples": "Example 1: ...",
 "constraints": "...", "language": "python",
 "solution": "class Solution:\n    def twoSum(...): ..."}
```

## ChatML Template (exact)

```
<|im_start|>system
You are an expert programmer in a technical interview. Provide a clean, optimal solution, followed by its Time and Space complexity.<|im_end|>
<|im_start|>user
Solve the following problem in {language}:

**{title}** ({difficulty})
{description}

Examples:
{examples}

Constraints:
{constraints}

Write the {language} solution:<|im_end|>
<|im_start|>assistant
```{language}
{solution}
```

**Time Complexity:** {time_complexity}
**Space Complexity:** {space_complexity}<|im_end|>
```

## Memory Budget (RTX 3090 / 4090, 24 GB)

| Component | Footprint |
|-----------|-----------|
| Base weights (4-bit NF4) | ~3.5 GB |
| LoRA params (r=16, trainable) | ~80 MB |
| Activations (bs=4, seq=1024) | ~12 GB |
| AdamW states (8-bit paged) | ~1 GB |
| Buffers / fragmentation | ~2 GB |
| **Total** | **~19 GB** (safe headroom) |

If you OOM, lower `--batch_size` to 2 (or 1 with `--grad_accum 8`) and/or
`--max_seq_len` to 768.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Token indices sequence length is longer than the specified maximum sequence length` | Lower `--max_seq_len` or filter long examples in `prepare_data.py`. |
| OOM during training | Lower `--batch_size` to 2, raise `--grad_accum` to 8. |
| `flash-attn` install fails | Run without `--use_flash_attention` (default is SDPA). |
| `bitsandbytes` import error on Linux | Ensure CUDA toolkit matches the wheel (12.1). Try `pip install bitsandbytes==0.43.3 --prefer-binary`. |
| HuggingFace `401 Unauthorized` on base model download | `huggingface-cli login` with your HF token. |
| Adapter doesn't load (`PeftModel.from_pretrained` fails) | Confirm `adapter_config.json` and `adapter_model.safetensors` exist in the adapter path. |
| `/ready` returns 503 | The model hasn't finished loading. Check the FastAPI logs. |

## License

Code: MIT.
Base model weights: see `deepseek-ai/deepseek-coder-6.7b-base` (Apache 2.0).
LeetCode dataset: please review LeetCode's Terms of Use before redistributing.
