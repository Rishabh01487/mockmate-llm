# Free Training Guide — Google Colab T4

This guide walks you through fine-tuning the Mockmate-LLM model on Google
Colab's **free T4 GPU** (16 GB VRAM), which is sufficient for QLoRA on a 6.7B
parameter model with the right hyperparameters.

## TL;DR

1. Push the `mockmate-llm/` directory to a GitHub repo.
2. Open `colab_train_free.ipynb` in Google Colab.
3. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU).
4. Run all cells in order — done in ~6-9 hours.
5. Adapter saved to your Google Drive.

## Files Added

| File | Purpose |
|------|---------|
| `convert_greengerong_dataset.py` | Converts the `greengerong/leetcode` HuggingFace dataset (2,360 problems × 4 languages) to our JSONL schema. |
| `colab_train_free.ipynb` | End-to-end Colab notebook: install → login → mount Drive → download data → prepare → train → push to HF Hub. |

## The Dataset

[`greengerong/leetcode`](https://huggingface.co/datasets/greengerong/leetcode)
on Hugging Face is the source. It contains:
- 2,360 LeetCode problems with full markdown descriptions, examples, and constraints
- Reference solutions in **4 languages** per problem: Python, Java, C++, JavaScript
- Difficulty tags (Easy/Medium/Hard)

After running `convert_greengerong_dataset.py`:
- **9,438 rows** total (one per problem-language pair)
- **8,914 train / 460 val** after a 5% stratified split
- Distribution:
  - Languages: 2,343 each (perfectly balanced)
  - Difficulty: 2,156 Easy / 5,078 Medium / 2,140 Hard
  - Time complexity (heuristic): O(n) 5965, O(n²) 1574, O(n log n) 1097, O(n·m) 304, O(n³) 299, O(2ⁿ) 89, O(n!) 37, O(log n) 9

## Why Colab Free Works (Memory Math)

| Component | T4 16GB budget |
|---|---|
| DeepSeek-Coder 6.7B base weights (4-bit NF4) | ~3.5 GB |
| LoRA params (r=8, all linear layers) | ~40 MB |
| Activations (batch_size=1, seq_len=512) | ~6 GB |
| 8-bit paged AdamW optimizer states | ~1 GB |
| CUDA context + buffers | ~2 GB |
| **Total** | **~12.5 GB** ✅ |

## Hyperparameters (Tuned for T4)

| Param | Value | Note |
|---|---|---|
| `--batch_size` | 1 | T4 VRAM is tight |
| `--grad_accum` | 16 | Effective batch = 16 |
| `--max_seq_len` | 512 | Trims longest 10% of examples |
| `--lora_r` | 8 | Half the standard 16 — saves memory |
| `--lora_alpha` | 16 | Standard 2× ratio |
| `--lr` | 2e-4 | Standard QLoRA LR |
| `--epochs` | 3 | ~6-9 hours total |
| `--save_steps` | 100 | Checkpoint every ~45 min |

## What to Do If It Disconnects

Colab free tier disconnects after ~12h of inactivity or 8-12h of compute.
The notebook saves checkpoints to Google Drive every 100 steps.

To resume:
1. Re-open the notebook.
2. Re-run Steps 1-6 (install + data prep — ~10 min).
3. In Step 7, set `RESUME = 1` and re-run the training cell.

It will auto-resume from the latest checkpoint in Drive.

## After Training Completes

You'll have an adapter at:
```
/content/drive/MyDrive/mockmate-llm/checkpoints/deepseek-leetcode-qlora/
├── adapter_config.json
├── adapter_model.safetensors  (~80 MB)
├── tokenizer.json
└── ...
```

### Option A — Run FastAPI on Colab itself (for quick testing)

Add a cell to the notebook:
```python
!pip install -q fastapi uvicorn pydantic nest-asyncio
import os
os.environ['ADAPTER_PATH'] = f'{WORK_DIR}/checkpoints/deepseek-leetcode-qlora'
os.environ['MOCK_MODE'] = '0'
os.environ['MODEL_PRELOAD'] = '1'

import nest_asyncio
nest_asyncio.apply()
!uvicorn app_api:app --host 0.0.0.0 --port 8000
```

Then expose the port via **ngrok** or **cloudflared**:
```python
!pip install -q pyngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f'Public URL: {public_url}')
```

### Option B — Download adapter & run on a real GPU server

```python
# In Step 10 of the notebook, you can also download via the Files panel:
# Sidebar → Files → drive → MyDrive → mockmate-llm → right-click → Download
```

Then scp/upload to your production GPU box, set `ADAPTER_PATH`, and start the API.

## Free GPU Alternatives if Colab Disconnects Too Often

| Platform | GPU | Free quota | Notes |
|---|---|---|---|
| **Google Colab Free** | T4 16GB | 12h sessions, unlimited re-connects | Recommended default |
| **Google Colab Pro** | A100 40GB | $10/month | ~2× faster training |
| **Kaggle Notebooks** | P100 16GB / T4 ×2 | 30h/week | Sign up at kaggle.com |
| **Lightning AI Studios** | T4 16GB | 22 hours/month | studios.lightning.ai |
| **Vast.ai** | RTX 3090/4090 | $0.30/hr (~$3 total) | Not free, but cheapest |

For Kaggle, just upload the same notebook — most cells work as-is. Replace the
Google Drive mount with Kaggle's dataset output.
