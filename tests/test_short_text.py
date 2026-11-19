"""Model-free tests for the short-text helpers and the rerank around them (no torch, no weights)."""

import logging

import pytest

from app.services.mt_engine import MTEngine
from app.services.short_text import (
    candidate_penalty,
    clean_candidates,
    contains_source,
    english_context,
    input_variants,
    is_plain_word,
    is_short_text,
    match_surface,
    normalize,
    repetition_level,
    strip_glued_tail,
)


def keys(*texts):
    return frozenset(normalize(t) for t in texts)


# --- glued identifier pieces ("giao tiếpName") -------------------------------------------


@pytest.mark.parametrize(
    ("source", "candidate", "siblings", "expected"),
    [
        # Live reports (MADLAD pieces: ['▁giao', '▁tiếp', 'Name'], ['▁iPhone', 'Name']).
        ("communication", "giao tiếpName", keys("giao tiếp"), "giao tiếp"),
        ("communication", "giao tiếpName", keys(), "giao tiếp"),  # no clean beam: accented word
        ("Communication", "Giao tiếpName", keys(), "Giao tiếp"),
        ("iPhone", "iPhoneName", keys(), "iPhone"),  # what is left is the source
        ("iPad", "iPadName", keys(), "iPad"),
        ("PlayStation", "PlayStationName", keys(), "PlayStation"),
        ("TikTok", "TikTokName", keys(), "TikTok"),
        ("you tube", "You TubeName", keys(), "You Tube"),
        ("Flower", "HoaName", keys("Hoa"), "Hoa"),  # ASCII head: another candidate
        ("communication", "communicationName", keys(), "communication"),
        # Punctuation follows the head; stacked pieces are peeled.
        ("communication.", "giao tiếpName.", keys(), "giao tiếp."),
        ("communication", "giao tiếpUserName", keys(), "giao tiếp"),
        ("iPhone", "iPhoneUserName", keys(), "iPhone"),
        ("communication", "giao tiếpNAME", keys(), "giao tiếp"),
        # Other identifier pieces and targets.
        ("information", "thông tinID", keys("thông tin"), "thông tin"),
        ("description", "mô tảText", keys(), "mô tả"),
        ("account", "Tài khoảnData", keys(), "Tài khoản"),
        ("connection", "связьName", keys(), "связь"),
        ("communication", "交流Name", keys("交流"), "交流"),
        ("communication", "소통Name", keys("소통"), "소통"),
        # The source contains "name", but not the glued word.
        ("username", "tên người dùngName", keys(), "tên người dùng"),
        ("name", "tênName", keys(), "tên"),
    ],
)
def test_strip_glued_tail_cuts_junk(source, candidate, siblings, expected):
    assert strip_glued_tail(source, candidate, siblings) == expected


@pytest.mark.parametrize(
    ("source", "candidate", "siblings"),
    [
        ("iPhone", "iPhone", keys()),
        ("iPhone", "điện thoại iPhone", keys()),
        ("iPhone", "IPhone", keys()),
        ("apple phone", "苹果iPhone", keys("苹果")),
        # The source spells the glued word that way (also across a space).
        ("youtube", "YouTube", keys("you")),
        ("javascript", "JavaScript", keys("java")),
        ("JavaScript tutorial", "hướng dẫn JavaScript", keys("hướng dẫn java")),
        ("power point", "PowerPoint", keys("power")),
        ("you tube", "YouTube", keys("you")),
        ("userName", "userName", keys()),
        ("#VietNam", "#ViệtNam", keys()),
        ("Ho Chi Minh", "HồChíMinh", keys()),
        # Split brand next to a truncated beam, for a source without Latin letters.
        ("깃허브", "GitHub", keys("Git")),
        ("软银", "SoftBank", keys("Soft")),
        ("ジャバ", "JavaScript", keys("Java")),
        ("微信", "WeChat", keys("We")),
        ("유튜브", "YouTube", keys("You")),
        # One-letter prefixes are spelling, not junk.
        ("ebay", "eBay", keys()),
        ("mrna", "mRNA", keys()),
        ("kwh", "kWh", keys()),
        ("pH", "pH", keys()),
        ("microfarad", "μF", keys()),
        ("the service", "an tSeirbhís", keys()),
        ("in Cork", "i gCorcaigh", keys()),
        ("in France", "i bhFrainc", keys()),
        ("in Dublin", "i mBaile Átha Cliath", keys()),
        ("Government of Ireland", "Rialtas na hÉireann", keys()),
        # Capitalised compounds with accents, no sibling.
        ("ベトナム", "ViệtNam", keys()),
        ("doctors", "ÄrztInnen", keys()),
        ("Mexico City", "MéxicoCity", keys()),
        # A brand after CJK text, other scripts, plain text.
        ("use PowerPoint", "使用PowerPoint", keys()),
        ("엑셀 사용", "使用Excel", keys()),
        ("McDonald's", "McDonald's", keys()),
        ("McDonald's", "МакДональдс", keys()),
        ("IJsselmeer", "IJsselmeer", keys()),
        ("Greece", "Ελλάδα", keys()),
        ("kite", "con diều", keys("con diều")),
        ("communication skills", "kỹ năng giao tiếp", keys()),
        # Out of scope: a spaced piece, and an ASCII head with no clean beam (could be a brand).
        ("communication", "giao tiếp Name", keys("giao tiếp")),
        ("information", "thông tinID", keys()),
        ("ユーチューブ", "xem kênh YouTube", keys()),
        ("communication", "", keys()),
    ],
)
def test_strip_glued_tail_keeps_real_text(source, candidate, siblings):
    assert strip_glued_tail(source, candidate, siblings) == candidate


def test_clean_candidates_merges_glued_junk_into_its_head():
    raw = ["giao tiếpName", "giao tiếp", "Giao tiếpName", "liên lạc", "truyền thông"]
    assert clean_candidates("communication", raw) == ["giao tiếp", "liên lạc", "truyền thông"]


def test_clean_candidates_glued_junk_without_clean_sibling():
    assert clean_candidates("communication", ["giao tiếpName", "liên lạc"]) == ["giao tiếp", "liên lạc"]


def test_clean_candidates_capitalised_source():
    assert clean_candidates("Communication", ["Giao tiếpName", "giao tiếp"]) == ["Giao tiếp"]
    assert clean_candidates("Flower", ["HoaName", "hoa", "Bông hoa"]) == ["Hoa", "Bông hoa"]


def test_clean_candidates_glued_copy_of_a_name():
    raw = ["iPhoneName", "iPhone", "IPhone", "điện thoại iPhone", "iPhoneName"]
    assert clean_candidates("iPhone", raw) == ["iPhone", "điện thoại iPhone"]
    assert clean_candidates("you tube", ["You TubeName", "YouTube"]) == ["You Tube", "YouTube"]


def test_clean_candidates_only_junk_never_empty():
    assert clean_candidates("iPhone", ["iPhoneName"]) == ["iPhone"]
    assert clean_candidates("communication", ["giao tiếpName"]) == ["giao tiếp"]


def test_clean_candidates_unchanged_without_junk():
    assert clean_candidates("kite", ["con diều", "Con diều", "diều", "", "  "]) == ["con diều", "diều"]
    assert clean_candidates("kite.", ["con diều"]) == ["con diều."]
    assert clean_candidates("你好！", ["Xin chào"]) == ["Xin chào!"]
    assert clean_candidates("Kite", ["con diều"]) == ["Con diều"]
    assert clean_candidates("x", []) == []


def test_clean_candidates_logs_each_cut(caplog):
    with caplog.at_level(logging.INFO, logger="madlad400-mt-api"):
        clean_candidates("communication", ["giao tiếpName", "giao tiếp", "giao tiếpName"])
    assert caplog.messages == ["Short-text 'communication': cut glued tails 'giao tiếpName' -> 'giao tiếp'"]


# --- copy handling for camelCase names -------------------------------------------------------


@pytest.mark.parametrize("word", ["iPhone", "YouTube", "PlayStation", "JavaScript", "McDonald's", "eBay"])
def test_camelcase_names_are_not_plain_words(word):
    assert not is_plain_word(word)


@pytest.mark.parametrize(
    "word", ["communication", "Communication", "kite", "first aid kit", "NASA", "Google", "anak-anak", "風箏"]
)
def test_plain_words_unchanged(word):
    assert is_plain_word(word)


@pytest.mark.parametrize("word", ["a@b.com", "2024", "https://x.io", "3.5"])
def test_non_words_are_not_plain(word):
    assert not is_plain_word(word)


def test_english_context_unchanged():
    assert english_context("cap") == "the cap"
    assert english_context("communication") == "the communication"
    assert english_context("Communication") is None
    assert english_context("iPhone") is None
    assert english_context("the cap") is None


# --- existing behaviour the fix must keep ------------------------------------------------------


def test_repetition_and_copy_penalties_unchanged():
    kw = dict(translating_across_languages=True, copy_penalty=1.5, repeat_penalty=2.0)
    assert candidate_penalty("kit", "đi đi", siblings={"đi", "đi đi"}, **kw) == 0.5
    assert candidate_penalty("kit", "đi đi", siblings={"đi"}, backward_scored=False, **kw) == 2.0
    assert candidate_penalty("cat", "con mèo con mèo", **kw) == 2.0
    assert candidate_penalty("x", "a a a", **kw) == 2.0
    assert candidate_penalty("slowly", "từ từ", **kw) == 0.0
    assert candidate_penalty("children", "anak-anak", **kw) == 0.0
    assert candidate_penalty("cape", "cape", **kw) == 1.5
    assert candidate_penalty("pizza", "bánh pizza", **kw) == 0.75
    assert candidate_penalty("kite", "con diều", **kw) == 0.0
    assert repetition_level("đi đi", frozenset({"đi"})) == 1


def test_surface_and_segmentation_unchanged():
    assert match_surface("kite.", "con diều") == "con diều."
    assert match_surface("你好！", "Xin chào") == "Xin chào!"
    assert match_surface("cat", "猫") == "猫"
    assert input_variants("communication") == ["communication", "Communication", "communication."]
    assert is_short_text("猫", 3) and is_short_text("風箏", 3)
    assert not is_short_text("我今天去商店", 3)
    assert contains_source("cape", "áo cape")
    assert normalize("  Giao tiếp. ") == "giao tiếp"


# --- the rerank itself, with a scripted fake model ---------------------------------------------


class _Ids:
    shape = (1, 4)

    def to(self, _device):
        return self


class _Tok:
    """Stands in for the T5 tokenizer: remembers the variant, decodes to scripted beams."""

    def __init__(self, beams):
        self.beams = beams
        self.last = None

    def __call__(self, text, return_tensors=None):
        self.last = text.split(" ", 1)[1]
        return type("Enc", (), {"input_ids": _Ids()})()

    def batch_decode(self, _outputs, skip_special_tokens=True):
        return list(self.beams.get(self.last, []))


def _engine(beams, forward, backward):
    """MTEngine whose generate / scoring are scripted (scores keyed by candidate text)."""
    eng = MTEngine("fake")
    eng.tokenizer = _Tok(beams)
    eng.model = type("Model", (), {"generate": lambda self, **kw: object()})()
    eng._device = lambda: "cpu"

    def logprobs(sources, targets):
        if all(s.startswith("<2vi>") for s in sources):  # forward pass
            return [forward[t] for t in targets]
        return [backward[s.split(" ", 1)[1]] for s in sources]

    eng._sequence_logprobs = logprobs
    return eng


def _translate(eng, text):
    return eng._translate_short(text, target_language="vi", source_language="en", max_new_tokens=32)


def test_rerank_communication_no_longer_returns_glued_name():
    # The junk scores just above the clean form, as length normalisation allows
    # ("Name" + a near-certain EOS lifts the per-token average). HEAD returned it.
    beams = {
        "communication": ["giao tiếpName", "giao tiếp", "liên lạc"],
        "Communication": ["Giao tiếpName", "Giao tiếp"],
        "communication.": ["liên lạc.", "giao tiếp."],
        "the communication": ["liên lạc", "sự giao tiếp"],
    }
    forward = {"giao tiếpName": -0.70, "giao tiếp": -0.77, "liên lạc": -1.0, "sự giao tiếp": -1.4}
    backward = {"giao tiếpName": -0.5, "giao tiếp": -0.5, "liên lạc": -0.9, "sự giao tiếp": -0.8}
    assert _translate(_engine(beams, forward, backward), "communication") == "giao tiếp"


def test_rerank_flower_no_longer_returns_glued_name():
    beams = {"Flower": ["HoaName", "Hoa", "Bông hoa"], "flower": ["hoa", "bông hoa"], "Flower.": ["Hoa."]}
    forward = {"HoaName": -0.5, "Hoa": -0.8, "Bông hoa": -1.2}
    backward = {"HoaName": -0.3, "Hoa": -0.4, "Bông hoa": -0.6}
    assert _translate(_engine(beams, forward, backward), "Flower") == "Hoa"


def test_rerank_iphone_keeps_the_name():
    # HEAD: "iPhone" counted as a plain word, so its copies got the median backward score
    # of the real translations while "iPhoneName" kept its near-copy back-translation.
    beams = {
        "iPhone": ["iPhoneName", "iPhone", "điện thoại iPhone", "điện thoại"],
        "IPhone": ["IPhone", "Điện thoại"],
        "iPhone.": ["iPhone.", "Điện thoại di động."],
    }
    forward = {
        "iPhoneName": -0.6,
        "iPhone": -0.7,
        "điện thoại iPhone": -1.2,
        "điện thoại": -1.5,
        "Điện thoại di động": -1.8,
    }
    backward = {
        "iPhoneName": -0.4,
        "iPhone": -0.1,
        "điện thoại iPhone": -0.3,
        "điện thoại": -4.0,
        "Điện thoại di động": -4.5,
    }
    assert _translate(_engine(beams, forward, backward), "iPhone") == "iPhone"


def test_rerank_brand_copy_keeps_its_back_translation():
    # HEAD gave the copy "iPhone" the backward score of "Điện thoại" (the only real
    # translation), so the generic word won; a camelCase name now keeps its own score.
    beams = {"iPhone": ["iPhone", "Điện thoại"], "IPhone": ["IPhone"], "iPhone.": ["iPhone."]}
    forward = {"iPhone": -1.0, "Điện thoại": -0.6}
    backward = {"iPhone": -0.1, "Điện thoại": -1.5}
    assert _translate(_engine(beams, forward, backward), "iPhone") == "iPhone"


def test_rerank_lowercase_word_still_rejects_copy():
    beams = {"cape": ["cape", "áo choàng", "áo cape"], "Cape": ["Cape", "Mũi"], "cape.": ["cape."]}
    beams["the cape"] = ["áo choàng"]
    forward = {"cape": -0.3, "áo choàng": -0.9, "áo cape": -1.0, "Mũi": -1.6}
    backward = {"cape": -0.05, "áo choàng": -0.8, "áo cape": -0.1, "Mũi": -2.5}
    assert _translate(_engine(beams, forward, backward), "cape") == "áo choàng"
