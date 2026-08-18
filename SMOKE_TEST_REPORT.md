# Smoke Test Report — Mockmate-LLM End-to-End

**Date**: 2026-08-18
**Environment**: Python 3.12, Node 24, no GPU/CUDA, Linux container.

## Summary

| Step | Status | Notes |
|------|--------|-------|
| 1. Create sample LeetCode dataset | ✅ Done | 12 problems × 4 languages = 48 rows |
| 2. Install runtime deps | ✅ Done | datasets, tqdm, fastapi, streamlit, etc. (torch/CUDA intentionally skipped — no GPU) |
| 3. Run `prepare_data.py` (heuristic) | ✅ Done | Fixed 3 bugs in the heuristic Big-O labeler; all 48 examples now correctly labeled |
| 4. `inference.parse_response` unit tests | ✅ 5/5 pass | Code extraction + Big-O regex handles all 5 edge cases |
| 5. FastAPI `app_api.py` boot + endpoint tests | ✅ Done (MOCK_MODE) | All 8 routes work: `/`, `/health`, `/ready`, `/languages`, `/generate`, `/generate/batch`, `/explain`, `/docs`. Validation: 422 for bad input. CORS: preflight returns 200. |
| 6. Streamlit `app_ui.py` boot | ✅ Done | Boots on :8501, HTTP 200 on `/`, `/app`, `/_stcore/health` |
| 7. Mockmate repo cloned | ✅ Done | Vite + React + Express (not Next.js as originally assumed in README) |
| 8. Add integration code to Mockmate | ✅ Done | 4 Express routes in `backend/server.js`, 1 React component, 1 service file, env example, integration doc |
| 9. End-to-end integration test | ✅ Done | FastAPI mock + Mockmate backend (with mongodb-memory-server): `/api/ai/solution` returns 200, batch returns 2 solutions, explain returns 183 chars, validation returns 400 |
| 10. LLM-based complexity labeler | ✅ Done | `COMPLEXITY_LLM_PROVIDER=zai` works via z-ai CLI; 3/4 perfect, 1 borderline call (3Sum O(1) vs O(n) — both defensible) |

## What was fixed during the run

1. **`prepare_data.py` heuristic**: brace-depth-based loop nesting over-counted (counted class + function braces). Rewrote to find the first `{` of the function body, then track loop nesting from there.
2. **`prepare_data.py` sort detection**: original regex missed C++'s `sort(begin, end)` syntax (only matched `.sort(`). Added `\bsort\(`.
3. **`prepare_data.py` two-pointer detection**: original required `:` at end (Python-only). Rewrote to be language-agnostic.
4. **`prepare_data.py` expand-around-center detection**: added new pattern matcher for palindrome helpers.
5. **`prepare_data.py` space complexity**: `unordered_map` was missing from the O(n) space regex. Added.
6. **`inference.py` Big-O regex**: `**Time Complexity:**` has `**` AFTER the colon, not before. Fixed.
7. **`inference.py` torch import**: made lazy so MOCK_MODE works without torch installed.
8. **`inference.py` mock mode**: added `_MOCK_SOLUTIONS` table + `_mock_generate_raw()` so the API/UI can boot without GPU.
9. **`app_api.py` `/ready` endpoint**: didn't recognize MOCK_MODE. Added check.
10. **`prepare_data.py` LLM labeler**: `subprocess` wasn't imported; the `npx z-ai-web-dev-sdk` command didn't exist (real binary is `z-ai`); CLI prepended status lines to stdout before the JSON envelope.

## What needs a GPU

| Step | Why |
|------|-----|
| `train_qlora.py` | QLoRA fine-tuning of DeepSeek-Coder 6.7B needs a 24GB VRAM GPU (RTX 3090/4090). bitsandbytes requires CUDA. |
| Real inference (`MOCK_MODE=0`) | Loads the 4-bit quantized model (~3.5GB) + merges LoRA adapter. Needs CUDA. |
| `MOCK_MODE=1` workaround | Returns canned responses; useful for UI/integration development before training is finished. **Already verified working end-to-end.** |

## Files added to Mockmate (sibling repo)

- `backend/server.js` — 4 new Express routes (`/api/ai/solution/health`, `/api/ai/solution`, `/api/ai/solution/batch`, `/api/ai/solution/explain`) that proxy to the FastAPI LLM service
- `src/services/llmService.js` — fetch wrapper
- `src/components/AICopilot.jsx` — CoderPad-style side panel
- `src/InterviewPage.jsx` — imports `AICopilot` and renders it next to the CodeEditor for coding questions
- `.env.example` — documents `LLM_API_URL`, `LLM_API_KEY`, `MOCK_MODE`, etc.
- `MOCKMATE_LLM_INTEGRATION.md` — full integration guide

## How to run the verified flow

```bash
# Terminal 1: FastAPI in mock mode (no GPU needed)
cd /home/z/my-project/download/mockmate-llm
MOCK_MODE=1 MODEL_PRELOAD=1 python -m uvicorn app_api:app --host 0.0.0.0 --port 8000

# Terminal 2: Mockmate backend
cd /home/z/my-project/download/mockmate/backend
LLM_API_URL=http://localhost:8000 npm run dev

# Terminal 3: Mockmate UI
cd /home/z/my-project/download/mockmate
npm install && npm run dev   # → http://localhost:5173

# Verify end-to-end:
curl http://localhost:5000/api/ai/solution/health
curl -X POST http://localhost:5000/api/ai/solution \
  -H "Content-Type: application/json" \
  -d '{"title":"Two Sum","difficulty":"Easy","description":"d","language":"python"}'
```

To switch to the **real fine-tuned model**: drop `MOCK_MODE=1`, set `ADAPTER_PATH=./checkpoints/deepseek-leetcode-qlora`, install the full `requirements.txt` (including torch + bitsandbytes) on a GPU machine.
