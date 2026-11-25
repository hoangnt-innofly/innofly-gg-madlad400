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
