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
| Long text | One model input per line / sentence (CJK split at 。！？ with no space needed), decoded in padded batches |
| VRAM | ~6 GB BF16 while translating; released when the job queue is empty (`MT_FREE_VRAM=1`) |

MADLAD only needs the **target** language tag (`<2vi>`, `<2en>`, `<2zh>`). Source language is optional and used for Auto-detect + skip-if-same-language.

## API

| Method | Path | What it does |
| --- | --- | --- |
| `POST` | `/api/v1/translate` | Form MT, wait, return `translation` |
| `POST` | `/api/v1/jobs` | Same as `/translate` (waits for text) |
| `GET` | `/api/v1/jobs/{job_id}` | Optional lookup of a finished job |
| `GET` | `/api/v1/languages` | Language list for the form |
| `POST` | `/api/v1/detect` | Detect source language |
| `POST` | `/api/v1/free-memory` | Unload weights from the GPU (same idea as Comfy `free_memory`); during a job it is deferred until the job ends (`{"freed": false, "deferred": true}`) |
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
| `pivot` | no | `true` / `false`: force translating through English on / off. Omitted = server rule (see *Pivot through English*) |

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
  "pivot_language": null,
  "chunks": 1,
  "error": null
}
```

Jobs are serialized on one GPU worker so concurrent requests do not OOM. A request still queued or running after 600 s gets HTTP `202` with the job (`status` `queued` / `running`) instead of an error: poll `GET /api/v1/jobs/{job_id}`. Finished jobs are forgotten after `JOB_TTL_SECONDS`.

Every job logs one line with its phases, e.g. `Job 1a2b3c4d succeeded: ja -> vi, 230 chars, chunks=7, pivot=en, queue 0.0s, load 5.1s, translate 3.2s` (pivot jobs also log stage 1 / stage 2 times). `load` is the model reload paid when the GPU was released.

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

Port `8002` avoids colliding with TTS/LTX on `8000`. `MT_FREE_VRAM=0` keeps the 3B model on the GPU (a few lines translate in ~1–3s after the first load). With `1` (shared 12GB card) the model is unloaded as soon as no job is waiting, so back-to-back requests load it once but a request after an idle gap reloads ~6GB.

## Quality notes

- Greedy/sampling is worse than beam search for this model. Keep `do_sample` off.
- `num_beams=4` matches the MADLAD examples; raise to 5–8 on the form if a sentence feels stiff.
- Text is translated line by line and sentence by sentence (see *Segmentation and batching*): MADLAD drops sentences when several reach it as one input.
- Auto-detect uses script heuristics (Vietnamese diacritics, CJK, Hangul, Thai, …) then `langdetect`. The dominant script wins by weighted share (a Han / kana / Hangul character weighs 3 letters), so `Từ 休み trong tiếng Nhật…` is Vietnamese and `Pythonでリスト…` Japanese; mixed text (no script at 75%+) is never "reliable", so it is never skipped as same-language. Under 4 words `langdetect` is noise (`kite` → Finnish, `cape` → Romanian), so short Latin text is treated as an unreliable English guess and is never skipped as "same language".

### Segmentation and batching

MADLAD-400 is trained on sentence pairs. Sent a title line plus seven sentences as one input it keeps only some of them, or (with the line break inside) writes new text in the source language. So the engine (`app/services/segmenter.py`):

1. keeps every line separate (titles, list items, dialogue lines) and copies the line breaks, blank lines and indentation (full-width space U+3000) of the source into the output (U+2028, U+2029, VT, FF and NEL count as line breaks and come out as `
`); a sentence that an e-mail client hard-wrapped is merged back (spaced scripts: the next line starts in lower case; CJK: 2+ lines that do not end a sentence share one fixed length, so a title line or chat messages of similar length stay separate);
   A run of 2+ spaces / tabs / U+3000 between Han / Kana text also ends a piece and is copied into the output: Japanese and Chinese put no spaces inside a sentence, so it is a lost line break (`テーマ：日本の大学生活  日本の大学生活は、…` = a title, then the first sentence; sent as one input MADLAD merged them and invented content). A single space (`山田　太郎`, `東京都 港区`) is left alone;
2. translates a heading line `LABEL：content` / `LABEL: content` as two units, the label alone and the content alone. Sent whole, `テーマ：環境問題と私たちの生活` came back as `Bản mẫu:Công nghiệp và môi trường` (MADLAD read it as the wiki namespace "Template:"); the content alone translates correctly. The label gets the glossary (`ja->vi`: `テーマ` → `Chủ đề`; bare `テーマ` alone gives `Sắc thái`), else the short-text rerank, else plain decoding. If the content is a single short unit it gets the short-text rerank like a whole line (`kite` of `Word: kite`). Output: `label: content`, or `label：content` for `ja` / `zh` / `yue` when the translated label has a CJK character (`問1：`). A bullet in front of the label (`・テーマ：…`, `■注意：…`, `# Theme: …`) is copied as is and is not part of the label. A label is at most 6 CJK characters or 2 words with no sentence punctuation; times, ratios and URLs (`10:30`, `1:2`, `http://`), `Q: …` and longer clauses before a colon stay whole;
3. splits each line into sentences: `。！？` and other script stops split even without a following space; `.!?` need a space, or CJK / Hangul on both sides (`我很好.你呢`). These do not split: abbreviations with or without a space after them (`Mr.`, `Mr.田中`, `U.S.出身`, `Inc.の`, `z.B.`, `No. 5`, `Jan. 5`, `Mt. Fuji`), ordinals (`am 24. Dezember`, `1. Dünya`; 4-digit years still end a sentence), decimals, initials, list markers, and a closing quote glued to the next word (`「行こう。」と言った。`, `"Hi!"라고 말했다.`, `“我来了。”他说。`);
4. never packs CJK / Korean / Thai-like sentences together; consecutive sentences of a spaced script (`en`, `vi`, `de`, …) on one line are packed up to `MT_SEGMENT_PACK_TOKENS` (default 256) because MADLAD keeps them and uses the context. A pack is re-translated sentence by sentence only when its output misses 2+ sentences, misses one and is under 75% of the source length in tokens, is under 50% whatever its sentence count, or has a retry issue (loop, wrong script, cut off). One sentence short at full length usually means two sentences were joined in the target, so that translation is kept. `0` = one sentence per input everywhere;
5. joins the outputs with the target's spacing: no space between sentences for `ja` / `zh` / `yue`, one space otherwise.

Units are decoded in padded `generate()` calls of up to `MT_DEFAULT_BATCH_SIZE` rows, sorted by length, capped at `MT_BATCH_MAX_TOKENS` KV-cache slots (rows × beams × (source + output) tokens; ~256 KiB each in bf16, so 4096 ≈ 1 GiB). A CUDA OOM halves the batch instead of failing the job. Each row may generate `max(64, 2 × source tokens + 32)` tokens (capped by `max_new_tokens`), then every row is checked:

| Problem | Retry |
| --- | --- |
| stopped without EOS (cut off) | same decoding, `4 × source + 64` tokens |
| repetition loop (`近近近…`, a sentence repeated) with more copies than any repetition in the source (`頑張れ`×4 → `Cố lên, cố lên, cố lên, cố lên!` is not a loop) | `num_beams=4`, `no_repeat_ngram_size=4`; still looping → cut to one copy |
| wrong script (Japanese out for `vi`), or < 30% of the source length | `num_beams=4`, only when the request used fewer beams (at 4+ the retry would repeat the same search; the row is just logged) |

A retry is kept only if it is better; anything still wrong is logged (`MT output still …`).

### Pivot through English

MADLAD-400 3B is strong into and out of English but weak between two other languages. Live ja→vi, sentence by sentence, the direct output invented content (`大学生になると、自分で授業を選び、時間割を作ることができます。` → `Sau khi tốt nghiệp, sinh viên có thể tự lựa chọn ngành học…`) and replaced the subject of `私は、…必要だと思います。` with `các trường đại học`; ja→en was exact and en→vi of that English kept meaning and subject (`Tôi nghĩ là cần thiết phải…`).

So every unit decoded by the model is routed on its own (`pivot.route_unit`), from its own script, not once per request:

- **copy**: already written in the target (`vi`: Vietnamese-only letters or accented Vietnamese syllables, `Ghi chú: đây là…`; `ja`: kana; `ko`: Hangul). A Vietnamese note in a ja→vi study sheet is kept, not paraphrased through English;
- **pivot** source → en → target (`MTEngine._translate_pivot`): the unit's language (its script, else the request's trustworthy source for Latin text of a Latin-script source such as `fr`) is neither English nor the target;
- **direct**: English target, English / romaji / unknown Latin units (an English line in a Japanese document), `pivot=false`.

Stage 1 runs with all the usual checks and retries; a unit whose English is still flagged (cut off, looping, wrong script, far too short — not retried at 4 beams) is translated directly instead of carrying the loss on. Each English intermediate is segmented like an English request (sentences, packs) and decoded into the target in one batched pass. Glossary hits and the short-text rerank are unchanged (still direct). Cost: about 2× the decoding time of the pivoted units. The response field `pivot_language` is the pivot that at least one unit really went through (`null` when everything was direct, a glossary hit or a short-text rerank).

A request can force it with `pivot=true` / `pivot=false` (no environment variable: the rule is fixed in code). Live probes, about 90 sentences over 12 pairs judged by hand: 40 serious errors direct vs 9 through English; fr→vi was a tie (no serious error either way, direct kept the casual register better).

### Single words / short phrases

MADLAD is a sentence model. With greedy decoding a lone word is often wrong: `kite` → `đi đi`, `kit` → `đi`, `cape` → `cape`. For input of at most `MT_SHORT_TEXT_MAX_WORDS` words (also each line of a one-word-per-line list) the engine:

1. generates `MT_SHORT_TEXT_CANDIDATES` beam candidates for the word, its case-flipped form and `word.`; when the source is (or is guessed as) English and the input is a lower-case plain word, also for `the word`. MADLAD has no source tag, so a bare word that exists in another language is read in that language (`cap` = Catalan "none" → `không có`; `pain` = French "bread"; `gift` = German "poison"); the article pins the English reading. A candidate's forward score is its best over `word` and `the word`;
2. scores each candidate by `log P(candidate | word)` + `MT_SHORT_TEXT_BACKWARD_WEIGHT` × `log P(word | candidate)` (back-translation: `đi` translates back to "go", not "kit", so it loses);
3. penalises degenerate repeats (`đi đi` next to `đi`, `X X X`; real reduplication like `từ từ`, `anak-anak` is left alone) and copies of a lower-case dictionary word (`cape` → `cape`, half-copies like `áo cape` less). Names, acronyms, numbers, URLs and e-mails (`Google`, `NASA`, `iPhone`, `2024`, `a@b.com`) may be copied.

Before scoring, an identifier piece the model glued onto a candidate is cut off (`communication` → `giao tiếpName` → `giao tiếp`, `iPhone` → `iPhoneName` → `iPhone`): on a bare word MADLAD sometimes reads the input as a UI key (`userName`), and the per-token average score barely punishes the extra token. Inner capitals the source has (`JavaScript`), one-letter prefixes (`eBay`, Irish `an tSeirbhís`) and capitalised compounds (`ViệtNam`) are kept. The log line `Short-text ...: cut glued tails ...` shows each cut.

"Short" means ≤ `MT_SHORT_TEXT_MAX_WORDS` space-separated words, or ≤ 4 characters for Chinese/Japanese/Thai/Lao/Khmer/Burmese (which have no spaces). At most `MT_SHORT_TEXT_MAX_UNITS` (default 50) units per request are reranked; further lines of a long word list use plain decoding so the request stays under the timeout.

The server log prints the top candidates with their scores (`Short-text rerank ...`), which helps tune the weights. Set `MT_SHORT_TEXT_RERANK=0` to turn it off (lines are still translated one by one). A short line is reranked only when it is the whole input, a paragraph of its own (blank lines around it), or one line of a block where every line is short (`kite\ncape\nkit`). The short lines of a letter (`Hi John,` / `Best regards,` / `Anna` around a longer line) and short sentences inside a line (`Thanks!`, `次。`) use plain decoding. A heading label with no glossary entry (`Note` of `Note: …`) is reranked as a lone word, and so is the content of a heading line when it is one short unit (`kite` of `Word: kite`). Each text is reranked once per request: a speaker name repeated on every line of a dialogue (`田中：…`) reuses the first result.

Smoke test on the server (API running): `python scripts/check_short_words.py --key <SECRET_API_KEY> --target vi` translates ~30 ambiguous short words and prints the results.

Model-free unit tests (no GPU): `python -m pip install pytest`, then `python -m pytest tests` from the repo root.

### Glossary (fixed translations)

`data/glossary.json` (`MT_GLOSSARY_PATH`) overrides the model for exact terms. It is re-read when the file changes, no restart needed:

```json
{
  "en->vi": { "kite": "con diều", "kit": "bộ dụng cụ", "cape": "áo choàng" },
  "vi": { "MADLAD": "MADLAD" }
}
```

A term matches only when it is the whole input, a whole line, a heading label (`テーマ` of `テーマ：…`), or one translation unit of a line (a CJK sentence, or a pack of spaced-script sentences; case-insensitive, trailing punctuation ignored). Use it for app vocabulary the model keeps getting wrong.

This is still a 3B research MT model, not Google Translate’s production stack. High-resource pairs (vi↔en, en↔fr/de/es/zh/ja/ko) are the closest.

## Project layout

```
app/            FastAPI app, job queue, MADLAD engine
static/         Vietnamese Google-Translate-style form
scripts/        weight download helper
```
