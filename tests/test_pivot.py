from __future__ import annotations

import pytest

from app.services.pivot import route_unit


@pytest.mark.parametrize(
    "text, source, target, requested, expected",
    [
        # Japanese / Chinese / Korean / Vietnamese units between two non-English languages
        ("日本の大学生活は楽しいです。", "ja", "vi", None, "pivot"),
        ("日本の大学生活は楽しいです。", None, "vi", None, "pivot"),  # own script is enough
        ("我认为大学生应该找到平衡。", "zh", "vi", None, "pivot"),
        ("저는 학생입니다.", "ko", "vi", None, "pivot"),
        ("Tôi là sinh viên năm nhất.", "vi", "ja", None, "pivot"),
        ("日本の大学生活は楽しいです。", "ja", "zh", None, "pivot"),
        # Already in the target language: copied (Vietnamese needs Vietnamese-only letters)
        ("Ghi chú: đây là câu ví dụ.", "ja", "vi", None, "copy"),
        ("Ví dụ", "ja", "vi", True, "copy"),
        ("休みの日に何をしますか。", "vi", "ja", None, "copy"),
        ("저는 학생입니다.", "vi", "ko", None, "copy"),
        ("Chữ 休 nghĩa là nghỉ ngơi.", "ja", "vi", None, "copy"),
        ("Can you go to the bank", "ja", "vi", None, "direct"),  # English, not Vietnamese
        ("我认为大学生应该找到平衡。", "ja", "zh", None, "direct"),  # Han only: zh->zh, not copied
        # Latin text: English / romaji in a CJK document goes direct
        ("Please read it twice.", "ja", "vi", None, "direct"),
        ("Bonjour tout le monde.", "fr", "vi", None, "pivot"),  # Latin source: trust it
        ("Bonjour tout le monde.", None, "vi", None, "direct"),  # unknown Latin: direct
        # English on either side, or the request says no
        ("Hello world.", "en", "vi", None, "direct"),
        ("日本の大学生活は楽しいです。", "ja", "en", None, "direct"),
        ("日本の大学生活は楽しいです。", "ja", "en", True, "direct"),
        ("日本の大学生活は楽しいです。", "ja", "vi", False, "direct"),
        ("Hello world.", None, "vi", True, "pivot"),
        ("Tôi là sinh viên.", "auto", "vi", None, "copy"),  # accents + Vietnamese syllables
        # Latin source naming / resembling Vietnamese: never copied
        ("I live in Hà Nội.", "en", "vi", None, "direct"),
        ("My teacher is Mr. Nguyễn.", "en", "vi", False, "direct"),
        ("Il habite à Hà Nội depuis 2010.", "fr", "vi", None, "pivot"),
        ("Não sei.", "pt", "vi", None, "pivot"),
        ("Sí", "es", "vi", None, "pivot"),
        ("Ghi chú: đây là câu ví dụ.", "en", "vi", None, "direct"),
        ("Ghi chú", "ja", "vi", None, "copy"),
        ("Café de Paris été", "fr", "vi", None, "pivot"),  # accents, but not Vietnamese
    ],
)
def test_route_unit(text, source, target, requested, expected):
    got = route_unit(text, source_language=source, target_language=target, requested=requested)
    assert got == expected
