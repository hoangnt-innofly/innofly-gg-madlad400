from __future__ import annotations

import pytest

from app.services.pivot import choose_pivot


@pytest.mark.parametrize(
    "source, target, requested, expected",
    [
        ("ja", "vi", None, "en"),
        ("zh", "vi", None, "en"),
        ("vi", "ja", None, "en"),
        ("ja", "zh", None, "en"),
        ("en", "vi", None, None),  # source is English
        ("ja", "en", None, None),  # target is English
        ("ja", "en", True, None),  # nothing to pivot through, even when forced
        ("vi", "vi", None, None),
        (None, "vi", None, None),  # unknown source: direct by default ...
        ("auto", "vi", None, None),
        (None, "vi", True, "en"),  # ... unless the request forces it
        ("ja", "vi", False, None),
        ("zh", "zh_Latn", None, None),
    ],
)
def test_choose_pivot(source, target, requested, expected):
    assert choose_pivot(source, target, requested=requested) == expected
