# MADLAD-400 3B MT API

FastAPI backend for multilingual **machine translation** with [MADLAD-400](https://huggingface.co/google/madlad400-3b-mt) **`google/madlad400-3b-mt`**.

Submit source text + target language. Response includes `translation`.

## Model (mt)

| Setting | Value |
| --- | --- |
| Model | `google/madlad400-3b-mt` |
| Mode | Encoder–decoder T5 MT (`<2xx>` target prefix) |
| Languages | 400+; form lists common ones, any MADLAD ISO code is accepted |
| Decode | Beam search (`num_beams=4`, `do_sample=false`) |
| Long text | Split / pack paragraphs under 480 encoder tokens to keep context |
| VRAM | ~6 GB BF16 while translating; released after each job (`MT_FREE_VRAM=1`) |

MADLAD only needs the **target** language tag (`<2vi>`, `<2en>`, `<2zh>`). Source language is optional and used for Auto-detect + skip-if-same-language.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/v1/translate` | Form MT, **wait**, return `translation` |
| `POST` | `/api/v1/jobs` | Same form, return immediately (`202`) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll until `status=succeeded` and `translation` is set |
| `GET` | `/api/v1/languages` | Language list for the form |
| `POST` | `/api/v1/detect` | Detect source language |
| `POST` | `/api/v1/free-memory` | Park weights off GPU (same idea as Comfy `free_memory`) |
| `GET` | `/health` | CUDA / mock / defaults (no API key) |
| `GET` | `/docs` | Swagger |

All `/api/v1/*` routes require header `api-key` equal to `SECRET_API_KEY`. Missing or wrong key → `401`.

Example (sync — response includes `translation`):

```powershell
curl.exe -X POST "http://127.0.0.1:8002/api/v1/translate" `
  -H "api-key: change-me" `
  -F "text=Xin chào, đây là MADLAD-400. Dịch sang tiếng Anh giúp tôi." `
  -F "source_language=auto" `
  -F "target_language=en"
```

Multipart fields:

| Field | Required | Notes |
| --- | --- | --- |
| `text` | yes | Text to translate |
| `source_language` | no | `auto` (default) or ISO code (`vi`, `en`, …) |
| `target_language` | no | ISO code, default `en` |
| `num_beams` | no | `1`–`8`, default `4` |
| `length_penalty` | no | Default `1.0` |
| `max_new_tokens` | no | Per chunk, default `512` |

```json
{
  "job_id": "...",
  "status": "succeeded",
  "text": "Xin chào, đây là MADLAD-400. Dịch sang tiếng Anh giúp tôi.",
  "translation": "Hello, this is MADLAD-400. Please translate into English for me.",
  "source_language": "vi",
  "target_language": "en",
  "detected_language": "vi",
  "model": "google/madlad400-3b-mt",
  "mode": "mt",
  "num_beams": 4,
  "chunks": 1,
  "error": null
}
```

Jobs are serialized on one GPU worker so concurrent requests do not OOM.

## Setup (Windows + Python 3.12)

Same layout as the Qwen3-TTS / LTX-Video APIs.

```powershell
cd madlad400-mt-api
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env
```

### A. API contract only (no GPU)

In `.env` set `MT_MOCK=1`, then:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002
```

Open http://127.0.0.1:8002 — submit text, get a tagged placeholder translation.

### B. Real MADLAD-400 3B MT

Same CUDA 12 wheels as the working LTX / TTS setup (`cu121`). One file installs torch + transformers:

```powershell
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python scripts/download_models.py
```

If you download weights locally, set `MT_MODEL_ID` to that folder, for example `models/madlad400-3b-mt`. Otherwise the Hugging Face id is used and weights download on first load.

Port `8002` avoids colliding with TTS/LTX on `8000`. `MT_FREE_VRAM=1` unloads the 3B weights after each job so the shared 12GB card can run the other apps.

## Quality notes

- Greedy/sampling is worse than beam search for this model. Keep `do_sample` off.
- `num_beams=4` matches the MADLAD examples; raise to 5–8 on the form if a sentence feels stiff.
- Long documents are packed by paragraph/sentence under 480 tokens so the encoder is not truncated mid-sentence.
- Auto-detect uses script heuristics (Vietnamese diacritics, CJK, Hangul, Thai, …) then `langdetect`.

This is still a 3B research MT model, not Google Translate’s production stack. High-resource pairs (vi↔en, en↔fr/de/es/zh/ja/ko) are the closest.

## Project layout

```
app/            FastAPI app, job queue, MADLAD engine
static/         Vietnamese Google-Translate-style form
scripts/        weight download helper
```
