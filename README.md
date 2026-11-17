# MADLAD-400 3B MT API

FastAPI backend for multilingual **machine translation** with [MADLAD-400](https://huggingface.co/google/madlad400-3b-mt) **`google/madlad400-3b-mt`**.

Submit source text + target language. One request waits and returns `translation`.

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
| `POST` | `/api/v1/translate` | Form MT, wait, return `translation` |
| `POST` | `/api/v1/jobs` | Same as `/translate` (waits for text) |
| `GET` | `/api/v1/jobs/{job_id}` | Optional lookup of a finished job |
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

Port `8002` avoids colliding with TTS/LTX on `8000`. Keep `MT_FREE_VRAM=0` so the 3B model stays on GPU — a few lines should translate in ~1–3s after the first load. Set `1` only when this process must free the 12GB card for LTX/TTS after every job (each request then reloads ~6GB).

## Quality notes

- Greedy/sampling is worse than beam search for this model. Keep `do_sample` off.
- `num_beams=4` matches the MADLAD examples; raise to 5–8 on the form if a sentence feels stiff.
- Long documents are packed by paragraph/sentence under 480 tokens so the encoder is not truncated mid-sentence.
- Auto-detect uses script heuristics (Vietnamese diacritics, CJK, Hangul, Thai, …) then `langdetect`. Under 4 words `langdetect` is noise (`kite` → Finnish, `cape` → Romanian), so short Latin text is treated as an unreliable English guess and is never skipped as "same language".

### Single words / short phrases

MADLAD is a sentence model. With greedy decoding a lone word is often wrong: `kite` → `đi đi`, `kit` → `đi`, `cape` → `cape`. For input of at most `MT_SHORT_TEXT_MAX_WORDS` words (also each line of a one-word-per-line list) the engine:

1. generates `MT_SHORT_TEXT_CANDIDATES` beam candidates for the word, its case-flipped form and `word.`; when the source is (or is guessed as) English and the input is a lower-case plain word, also for `the word`. MADLAD has no source tag, so a bare word that exists in another language is read in that language (`cap` = Catalan "none" → `không có`; `pain` = French "bread"; `gift` = German "poison"); the article pins the English reading. A candidate's forward score is its best over `word` and `the word`;
2. scores each candidate by `log P(candidate | word)` + `MT_SHORT_TEXT_BACKWARD_WEIGHT` × `log P(word | candidate)` (back-translation: `đi` translates back to "go", not "kit", so it loses);
3. penalises degenerate repeats (`đi đi` next to `đi`, `X X X`; real reduplication like `từ từ`, `anak-anak` is left alone) and copies of a lower-case dictionary word (`cape` → `cape`, half-copies like `áo cape` less). Names, acronyms, numbers, URLs and e-mails (`Google`, `NASA`, `2024`, `a@b.com`) may be copied.

"Short" means ≤ `MT_SHORT_TEXT_MAX_WORDS` space-separated words, or ≤ 4 characters for Chinese/Japanese/Thai/Lao/Khmer/Burmese (which have no spaces). At most `MT_SHORT_TEXT_MAX_UNITS` (default 50) units per request are reranked; further lines of a long word list use plain decoding so the request stays under the timeout.

The server log prints the top candidates with their scores (`Short-text rerank ...`), which helps tune the weights. Set `MT_SHORT_TEXT_RERANK=0` to turn it off (this also stops word lists being split per line).

Smoke test on the server (API running): `python scripts/check_short_words.py --key <SECRET_API_KEY> --target vi` translates ~30 ambiguous short words and prints the results.

### Glossary (fixed translations)

`data/glossary.json` (`MT_GLOSSARY_PATH`) overrides the model for exact terms. It is re-read when the file changes, no restart needed:

```json
{
  "en->vi": { "kite": "con diều", "kit": "bộ dụng cụ", "cape": "áo choàng" },
  "vi": { "MADLAD": "MADLAD" }
}
```

A term matches only when it is the whole input, one line of a word list, or a whole paragraph (case-insensitive, trailing punctuation ignored). Use it for app vocabulary the model keeps getting wrong.

This is still a 3B research MT model, not Google Translate’s production stack. High-resource pairs (vi↔en, en↔fr/de/es/zh/ja/ko) are the closest.

## Project layout

```
app/            FastAPI app, job queue, MADLAD engine
static/         Vietnamese Google-Translate-style form
scripts/        weight download helper
```
