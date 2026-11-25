from __future__ import annotations

import pytest

from app.core.languages import detect_language_info


@pytest.mark.parametrize(
    "text, expected",
    [
        # One Japanese / Chinese word inside Vietnamese or English is not the language
        ("Từ 休み trong tiếng Nhật nghĩa là ngày nghỉ, hãy dịch giúp tôi.", ("vi", True)),
        ("Chữ 休 nghĩa là nghỉ ngơi.", ("vi", True)),
        ("He is a 先生 at our school.", ("en", True)),
        # A Vietnamese name in English is English (was: reliable vi -> skipped for target vi)
        ("I live in Hà Nội with my family.", ("en", True)),
        ("Hôm nay trời mưa to quá.", ("vi", True)),
        # Japanese with many English terms: never a reliable "en" (target en would skip it)
        ("このエラーは TypeError: Cannot read properties of undefined (reading 'map') です", ("en", False)),
        # English terms inside Japanese / Korean: still Japanese / Korean, but not reliable
        ("Pythonでリストをソートする方法", ("ja", True)),
        ("React Hooksのuseeffectについて", ("ja", False)),
        ("Samsung Galaxy 갤럭시 출시", ("ko", False)),
        ("日本の大学生活は、高校までの生活とは大きく違います。", ("ja", True)),
        ("我认为大学生应该找到平衡。", ("zh", True)),
        ("안녕하세요 반갑습니다", ("ko", True)),
    ],
)
def test_dominant_script(text, expected):
    assert detect_language_info(text) == expected


def test_vietnamese_words_in_english_are_never_a_reliable_vi():
    for text in ("We love phở and bánh mì.", "We love phở and bánh mì every weekend."):
        lang, reliable = detect_language_info(text)
        assert not (lang == "vi" and reliable), text
