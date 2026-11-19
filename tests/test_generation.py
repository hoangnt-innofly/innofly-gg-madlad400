from __future__ import annotations

from app.services.generation import (
    GenResult,
    assess,
    find_loop,
    is_loop,
    max_repeats,
    output_budget,
    pack_incomplete,
    plan_batches,
    retry_budget,
    retry_kind,
    row_stats,
    too_short,
    trim_loop,
    wrong_script,
)


def test_budget_covers_measured_ratios():
    # src_len includes <2xx> and </s>: c = src_len - 2.
    assert output_budget(13, 512) >= 27  # en->th sentence, c=11 needs 27
    assert output_budget(149, 512) >= 217  # en->vi whole paragraph, c=147 needs 217
    assert output_budget(210, 512) >= 216  # ja->vi whole paragraph
    assert output_budget(3, 512) == 64  # floor
    assert output_budget(400, 512) == 512  # capped by the request
    assert output_budget(400, 100) == 100
    assert retry_budget(13, 512) == 4 * 11 + 64
    assert retry_budget(400, 512) == 512


def test_plan_batches_sorts_and_caps():
    lengths = [10, 50, 20, 40, 30]
    budgets = [64] * 5
    plan = plan_batches(lengths, budgets, batch_size=2, num_beams=1, max_batch_tokens=10_000)
    assert plan == [[1, 3], [4, 2], [0]]
    # KV-cache cap: 2 rows * 4 beams * (50 + 64) = 912 > 600 -> one row per call.
    plan = plan_batches(lengths, budgets, batch_size=4, num_beams=4, max_batch_tokens=600)
    assert all(len(b) == 1 for b in plan)
    # A single over-size row still runs.
    assert plan_batches([900], [512], batch_size=4, num_beams=5, max_batch_tokens=10) == [[0]]
    assert sorted(i for b in plan for i in b) == [0, 1, 2, 3, 4]


def test_row_stats():
    eos, pad = 1, 0
    assert row_stats([0, 5, 6, 1, 0, 0], eos, pad) == (True, 2)
    assert row_stats([0, 5, 6, 7], eos, pad) == (False, 3)  # ran into max_new_tokens
    assert row_stats([0, 1], eos, pad) == (True, 0)


def test_find_loop():
    assert find_loop("環境問題とは、近近近近近近近近近近近") is not None
    sentence = "環境問題とは、地球の環境に悪影響を及ぼすものです。"
    start, period, copies = find_loop("前置き。" + sentence * 4)
    assert period == len(sentence) and copies == 4
    assert find_loop("con mèo con mèo con mèo con mèo") is not None
    assert find_loop("Giá là 1000000 đồng......") is None  # digits / dots only
    assert find_loop("hahaha, rất vui") is None
    assert find_loop("chúng ta có thể làm, và chúng ta có thể làm tốt") is None
    assert find_loop("") is None


def test_is_loop_ignores_repetition_in_source():
    assert not is_loop("ha ha ha ha ha ha ha ha", "はははははははは")
    # A repeated short source word becomes a repeated longer phrase: faithful, not a loop,
    # although the source span is too short for find_loop's 8-copy threshold.
    assert not is_loop("very very very very good", "とてもとてもとてもとても良い")
    assert not is_loop("Cố lên, cố lên, cố lên, cố lên!", "頑張れ頑張れ頑張れ頑張れ！")
    assert not is_loop("Yes, yes, yes, yes, I understand.", "はいはいはいはい、わかりました。")
    assert not is_loop("Come on, come on, come on, come on!", "加油加油加油加油！")
    assert max_repeats("頑張れ頑張れ頑張れ頑張れ！") == 4 and max_repeats("環境を守る。") == 1
    # More copies than the source has: a decoding loop.
    assert is_loop("very very very very very very good", "とてもとても良い")
    assert is_loop("Bảo vệ " + "môi trường " * 40, "環境を守る。")
    assert is_loop("Đây là câu. " * 4, "はい、これです。")


def test_trim_loop():
    sentence = "Đây là câu. "
    assert trim_loop("Mở đầu. " + sentence * 5) == "Mở đầu. Đây là câu."
    assert trim_loop("không lặp") == "không lặp"


def test_wrong_script():
    assert wrong_script("環境問題とは、地球の環境に悪影響を及ぼすものです。", "vi")
    assert wrong_script("環境問題とは何か", "en")
    assert not wrong_script("Vấn đề môi trường", "vi")
    assert not wrong_script("環境問題とは", "ja")
    assert not wrong_script("环境问题", "zh")
    assert wrong_script("环境问题很严重", "zh_Latn")
    assert not wrong_script("iPhone 東京 is nice and big", "en")  # one name in Han


def test_too_short_and_assess():
    full = GenResult("Một câu dịch đầy đủ.", True, 30, 40, 128)
    dropped = GenResult("Câu cuối.", True, 4, 210, 224)
    assert not too_short(full)
    assert too_short(dropped)
    assert assess("src", full, "vi") == frozenset()
    cut = GenResult("abc", False, 64, 20, 64)
    assert assess("src", cut, "vi") == {"truncated"}


def test_retry_kind():
    cut = GenResult("x", False, 64, 13, 64)
    assert retry_kind(frozenset({"truncated"}), cut, 512) == "longer"
    # No room to grow under the request cap: nothing to retry.
    assert retry_kind(frozenset({"truncated"}), GenResult("x", False, 40, 13, 40), 40) is None
    assert retry_kind(frozenset({"truncated", "loop"}), cut, 512) == "loop"
    assert retry_kind(frozenset({"script"}), cut, 512) == "beam"
    assert retry_kind(frozenset(), cut, 512) is None


def test_pack_fallback_needs_real_evidence():
    ok = frozenset()
    full = GenResult("x", True, 40, 42, 120)  # 40 content tokens out for 40 in
    # One sentence short but full length: two sentences merged in the target, keep it.
    assert pack_incomplete(3, 2, full, ok) is None
    assert pack_incomplete(3, 3, full, ok) is None
    # Two sentences missing.
    assert pack_incomplete(3, 1, full, ok) == "sentences"
    # One short and under 3/4 of the source: a dropped sentence.
    assert pack_incomplete(3, 2, GenResult("x", True, 28, 42, 120), ok) == "dropped"
    assert pack_incomplete(2, 1, GenResult("x", True, 20, 42, 120), ok) == "dropped"
    # Right count but clearly too short.
    assert pack_incomplete(3, 3, GenResult("x", True, 19, 42, 120), ok) == "length"
    assert pack_incomplete(3, 3, GenResult("x", True, 20, 42, 120), ok) is None
    # Any detected issue (loop, wrong script, cut off).
    assert pack_incomplete(3, 3, full, frozenset({"loop"})) == "issues"
    assert pack_incomplete(3, 3, full, frozenset({"script"})) == "issues"
