from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name_vi: str
    name_en: str


# Curated MADLAD-400 ISO codes for the form. Any other 2–3 letter MADLAD tag is still accepted.
LANGUAGES: tuple[Language, ...] = (
    Language("auto", "Tự nhận diện", "Auto"),
    Language("vi", "Tiếng Việt", "Vietnamese"),
    Language("en", "Tiếng Anh", "English"),
    Language("zh", "Tiếng Trung", "Chinese"),
    Language("ja", "Tiếng Nhật", "Japanese"),
    Language("ko", "Tiếng Hàn", "Korean"),
    Language("fr", "Tiếng Pháp", "French"),
    Language("de", "Tiếng Đức", "German"),
    Language("es", "Tiếng Tây Ban Nha", "Spanish"),
    Language("pt", "Tiếng Bồ Đào Nha", "Portuguese"),
    Language("ru", "Tiếng Nga", "Russian"),
    Language("ar", "Tiếng Ả Rập", "Arabic"),
    Language("th", "Tiếng Thái", "Thai"),
    Language("id", "Tiếng Indonesia", "Indonesian"),
    Language("ms", "Tiếng Mã Lai", "Malay"),
    Language("hi", "Tiếng Hindi", "Hindi"),
    Language("it", "Tiếng Ý", "Italian"),
    Language("nl", "Tiếng Hà Lan", "Dutch"),
    Language("pl", "Tiếng Ba Lan", "Polish"),
    Language("tr", "Tiếng Thổ Nhĩ Kỳ", "Turkish"),
    Language("uk", "Tiếng Ukraina", "Ukrainian"),
    Language("cs", "Tiếng Séc", "Czech"),
    Language("ro", "Tiếng Romania", "Romanian"),
    Language("hu", "Tiếng Hungary", "Hungarian"),
    Language("sv", "Tiếng Thụy Điển", "Swedish"),
    Language("fi", "Tiếng Phần Lan", "Finnish"),
    Language("da", "Tiếng Đan Mạch", "Danish"),
    Language("no", "Tiếng Na Uy", "Norwegian"),
    Language("el", "Tiếng Hy Lạp", "Greek"),
    Language("he", "Tiếng Hebrew", "Hebrew"),
    Language("bn", "Tiếng Bengal", "Bengali"),
    Language("ta", "Tiếng Tamil", "Tamil"),
    Language("ur", "Tiếng Urdu", "Urdu"),
    Language("fa", "Tiếng Ba Tư", "Persian"),
    Language("fil", "Tiếng Philippines", "Filipino"),
    Language("tl", "Tiếng Tagalog", "Tagalog"),
    Language("my", "Tiếng Miến Điện", "Burmese"),
    Language("km", "Tiếng Khmer", "Khmer"),
    Language("lo", "Tiếng Lào", "Lao"),
)

_LANGUAGE_BY_KEY = {lang.code.lower(): lang.code for lang in LANGUAGES}

ALIASES: dict[str, str] = {
    "auto-detect": "auto",
    "detect": "auto",
    "vn": "vi",
    "vie": "vi",
    "vietnamese": "vi",
    "tieng viet": "vi",
    "eng": "en",
    "en-us": "en",
    "en-gb": "en",
    "english": "en",
    "tieng anh": "en",
    "cn": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "chinese": "zh",
    "jp": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "kr": "ko",
    "kor": "ko",
    "korean": "ko",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "ger": "de",
    "deu": "de",
    "german": "de",
    "spa": "es",
    "spanish": "es",
    "por": "pt",
    "pt-br": "pt",
    "portuguese": "pt",
    "rus": "ru",
    "russian": "ru",
    "ara": "ar",
    "arabic": "ar",
    "tha": "th",
    "thai": "th",
    "ind": "id",
    "indonesian": "id",
    "may": "ms",
    "malay": "ms",
    "hin": "hi",
    "hindi": "hi",
    "ita": "it",
    "italian": "it",
    "nld": "nl",
    "dut": "nl",
    "dutch": "nl",
    "pol": "pl",
    "polish": "pl",
    "tur": "tr",
    "turkish": "tr",
    "ukr": "uk",
    "ukrainian": "uk",
    "filipino": "fil",
    "tagalog": "tl",
    "burmese": "my",
    "khmer": "km",
    "lao": "lo",
}

_MADLAD_CODE = re.compile(r"^[a-z]{2,3}(?:[-_][a-z0-9]+)?$")
_VI_CHARS = re.compile(
    r"[ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)

_VI_ONLY_CHARS = re.compile(
    r"[ăắằẳẵặấầẩẫậếềểễệốồổỗộơớờởỡợưứừửữựđảạẻẽẹỉĩịỏọủũụỳỷỹỵ]",
    re.IGNORECASE,
)
_WORD = re.compile(r"\w+", re.UNICODE)
# Below this many words langdetect is not trusted (see detect_language_info).
MIN_RELIABLE_WORDS = 4

DEFAULT_SOURCE_LANGUAGE = "auto"
DEFAULT_TARGET_LANGUAGE = "en"

# langdetect sometimes uses these instead of MADLAD tags.
_LANGDETECT_MAP = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh": "zh",
}


def languages_payload() -> dict:
    return {
        "model": "google/madlad400-3b-mt",
        "mode": "mt",
        "prefix": "<2xx>",
        "languages": [
            {"code": lang.code, "name_vi": lang.name_vi, "name_en": lang.name_en}
            for lang in LANGUAGES
        ],
    }


def canonicalize_language(value: str | None, *, allow_auto: bool = True) -> str:
    if value is None or not str(value).strip():
        return DEFAULT_SOURCE_LANGUAGE if allow_auto else DEFAULT_TARGET_LANGUAGE
    key = str(value).strip().lower().replace("_", "-")
    key = ALIASES.get(key, key)
    if key in _LANGUAGE_BY_KEY:
        code = _LANGUAGE_BY_KEY[key]
        if code == "auto" and not allow_auto:
            raise ValueError("Ngôn ngữ đích không thể là Auto. Chọn mã ISO, ví dụ en, vi, zh.")
        return code
    primary = key.split("-")[0]
    if _MADLAD_CODE.match(primary) and primary != "auto":
        return primary
    raise ValueError(f"Language không hợp lệ: {value}")


def detect_language(text: str) -> str:
    """Best-effort source language. Script heuristics first, then langdetect."""
    return detect_language_info(text)[0]


def detect_language_info(text: str) -> tuple[str, bool]:
    """Return (language, reliable).

    langdetect is noise on one or two words ("kite" -> fi, "cape" -> ro, "dog" -> pt),
    so short Latin-script input is reported as unreliable (plain ASCII as an English
    guess, accented text as langdetect's guess). Callers must not skip translation
    (source == target) on an unreliable guess.
    """
    sample = (text or "").strip()
    if not sample:
        return DEFAULT_TARGET_LANGUAGE, False

    words = _WORD.findall(sample[:4000])
    long_enough = len(words) >= MIN_RELIABLE_WORDS

    script = _detect_script(sample[:4000])
    if script:
        if script != "vi":
            return script, True
        # á/à/é/ã… also occur in French/Spanish/Portuguese ("café", "été"): trust short
        # text only with Vietnamese-only letters or when every word is a Vietnamese syllable
        # ("nhà", "mèo", "cà phê").
        reliable = (
            long_enough
            or bool(_VI_ONLY_CHARS.search(sample))
            or (bool(words) and all(_is_vi_syllable(w) for w in words))
        )
        return script, reliable

    if not long_enough:
        # Plain-ASCII words are overwhelmingly English in practice ("kite" is not Finnish).
        if sample.isascii():
            return "en", False
        return _langdetect_guess(sample) or "en", False

    guess = _langdetect_guess(sample)
    return guess or "en", guess is not None


def _langdetect_guess(sample: str) -> str | None:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        raw = detect(sample[:4000])
        mapped = _LANGDETECT_MAP.get(raw, raw)
        return canonicalize_language(mapped, allow_auto=False)
    except Exception:
        return None


# Vietnamese syllable on accent-stripped letters: onset + vowel nucleus + coda.
_VI_SYLLABLE = re.compile(
    r"(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?"
    r"[aeiouy]{1,3}"
    r"(?:ch|ng|nh|[cmnpt])?"
)


def _is_vi_syllable(word: str) -> bool:
    import unicodedata

    base = unicodedata.normalize("NFD", word.lower().replace("đ", "d"))
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    return _VI_SYLLABLE.fullmatch(base) is not None


def _detect_script(sample: str) -> str | None:
    if re.search(r"[\uac00-\ud7a3]", sample):
        return "ko"
    if re.search(r"[\u3040-\u30ff]", sample):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", sample):
        return "zh"
    if re.search(r"[\u0e00-\u0e7f]", sample):
        return "th"
    if re.search(r"[\u0600-\u06ff]", sample):
        return "ar"
    if re.search(r"[\u0400-\u04ff]", sample):
        return "ru"
    if re.search(r"[\u0900-\u097f]", sample):
        return "hi"
    if re.search(r"[\u1780-\u17ff]", sample):
        return "km"
    if re.search(r"[\u0e80-\u0eff]", sample):
        return "lo"
    if re.search(r"[\u1000-\u109f]", sample):
        return "my"
    if _VI_CHARS.search(sample):
        return "vi"
    return None
