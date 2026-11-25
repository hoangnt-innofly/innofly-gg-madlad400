"""MTEngine.translate with a fake tokenizer / model: no torch weights, no GPU."""

from __future__ import annotations

import contextlib
import importlib.util
import re
import sys
import types
from types import SimpleNamespace

import pytest

from app.services.mt_engine import MTEngine

PAD, EOS = 0, 1
_PIECE = re.compile(
    r"\s*[぀-ヿ㐀-䶿一-鿿＀-￯　-〿]"  # CJK: one token per character
    r"|\s*[^\W\d_぀-ヿ㐀-䶿一-鿿]+"  # a word
    r"|\s*\d+|\s*\S|\s+"
)


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def to(self, device):
        return self

    def tolist(self) -> list[list[int]]:
        return [list(r) for r in self.rows]


class FakeTokenizer:
    """Lossless toy tokenizer: <2xx> tag, one token per CJK char or per word."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 2
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def pieces(self, text: str) -> list[str]:
        out = []
        m = re.match(r"(<2[\w-]+>) ?", text)
        if m:
            out.append(m.group(1))
            text = text[m.end():]
        return out + _PIECE.findall(text)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._id(p) for p in self.pieces(text)] + ([EOS] if add_special_tokens else [])

    def __call__(self, texts, return_tensors="pt", padding=False):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [self.encode(t) for t in texts]
        width = max(len(r) for r in rows)
        return SimpleNamespace(
            input_ids=FakeTensor([r + [PAD] * (width - len(r)) for r in rows]),
            attention_mask=FakeTensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]),
        )

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(
            self.inv[i] for i in ids if i > EOS and not self.inv[i].startswith("<2")
        )

    def batch_decode(self, rows, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(r) for r in rows]


class OutOfMemoryError(RuntimeError):
    pass


class FakeModel:
    """generate() answers each row with ``fn(source, target, kwargs)``, honouring
    max_new_tokens (no EOS when the answer does not fit) and right-padding rows."""

    config = SimpleNamespace(eos_token_id=EOS, pad_token_id=PAD)

    def __init__(self, tok: FakeTokenizer, fn, max_rows: int | None = None) -> None:
        self.tok = tok
        self.fn = fn
        self.max_rows = max_rows
        self.calls: list[dict] = []

    def parameters(self):
        yield SimpleNamespace(device="cpu")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=20, **kwargs):
        rows = input_ids.tolist()
        assert attention_mask is not None, "batched rows need an attention mask"
        masks = attention_mask.tolist()
        if self.max_rows is not None and len(rows) > self.max_rows:
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        sources = []
        out = []
        for row, mask in zip(rows, masks):
            ids = [i for i, keep in zip(row, mask) if keep]
            target = self.tok.inv[ids[0]][2:-1]
            source = self.tok.decode(ids)
            sources.append(source)
            answer = self.fn(source, target, kwargs)
            body = [self.tok._id(p) for p in self.tok.pieces(answer)]
            out.append([PAD] + (body[:max_new_tokens] if len(body) >= max_new_tokens else body + [EOS]))
        width = max(len(r) for r in out)
        self.calls.append(
            {"sources": sources, "max_new_tokens": max_new_tokens, "lengths": [sum(m) for m in masks], **kwargs}
        )
        return FakeTensor([r + [PAD] * (width - len(r)) for r in out])


@pytest.fixture(autouse=True)
def torch_stub(monkeypatch):
    """translate() imports torch; provide a minimal stand-in when it is not installed."""
    if importlib.util.find_spec("torch") is not None:
        return
    stub = types.ModuleType("torch")
    stub.no_grad = contextlib.nullcontext
    stub.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    stub.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", stub)


def make_engine(fn, *, batch_size=4, max_rows=None, **kwargs) -> tuple[MTEngine, FakeModel]:
    engine = MTEngine("fake", free_vram=False, batch_size=batch_size, **kwargs)
    tok = FakeTokenizer()
    model = FakeModel(tok, fn, max_rows=max_rows)
    engine.tokenizer, engine.model, engine._ready = tok, model, True
    engine.release_vram = lambda: None
    return engine, model


JA_VI = {
    "環境問題と私たちの生活": "Các vấn đề môi trường và cuộc sống của chúng ta",
    "近年、地球温暖化や海洋汚染などの環境問題が世界中で深刻になっています。":
        "Những năm gần đây, các vấn đề môi trường như sự nóng lên toàn cầu và ô nhiễm đại dương "
        "đang trở nên nghiêm trọng trên khắp thế giới.",
    "特に、プラスチックごみによる海の汚染は、多くの生物に悪い影響を与えています。":
        "Đặc biệt, ô nhiễm biển do rác thải nhựa đang gây ảnh hưởng xấu đến nhiều sinh vật.",
    "この問題を解決するためには、国や企業だけでなく、私たち一人一人の努力も必要です。":
        "Để giải quyết vấn đề này, không chỉ nhà nước và doanh nghiệp mà mỗi người chúng ta "
        "cũng cần nỗ lực.",
    "例えば、買い物の際にエコバッグを使ったり、使い捨てのプラスチック製品を減らしたりすることができます。":
        "Ví dụ, chúng ta có thể dùng túi sinh thái khi đi mua sắm hoặc giảm các sản phẩm nhựa "
        "dùng một lần.",
    "また、電気や水を無駄に使わないことも大切です。":
        "Ngoài ra, việc không lãng phí điện và nước cũng rất quan trọng.",
    "小さな行動でも、多くの人が続ければ大きな変化につながります。":
        "Dù là hành động nhỏ, nếu nhiều người cùng duy trì thì sẽ dẫn đến thay đổi lớn.",
    "未来の地球を守るために、今できることから始めましょう。":
        "Để bảo vệ Trái Đất của tương lai, hãy bắt đầu từ những việc có thể làm ngay bây giờ.",
}


def sentence_level_ja_vi(source, target, kwargs):
    """Like MADLAD: fine on one sentence, keeps only the last one of a multi-sentence input."""
    if source in JA_VI:
        return JA_VI[source]
    sentences = re.findall(r"[^。\n]+。?", source)
    return JA_VI.get(sentences[-1].strip(), "?")


GLOSSARY_JA_VI = {"テーマ": "Chủ đề"}


def test_bug_paragraph_ja_vi_keeps_every_sentence(para_ja):
    engine, model = make_engine(sentence_level_ja_vi)
    engine._translate_short = lambda text, **kw: pytest.fail(f"glossary label reranked: {text}")
    out, segments = engine.translate(pivot=False, 
        text=para_ja, target_language="vi", num_beams=1, lookup=GLOSSARY_JA_VI.get
    )
    values = list(JA_VI.values())
    assert out == "Chủ đề: " + values[0] + "\n" + " ".join(values[1:])
    assert segments == 9  # label + title content + 7 sentences
    # The label comes from the glossary; 8 units decoded in 2 padded generate() calls
    # of 4, longest first, no retries.
    assert "テーマ" not in [s for c in model.calls for s in c["sources"]]
    assert [len(c["sources"]) for c in model.calls] == [4, 4]
    lengths = [n for c in model.calls for n in c["lengths"]]
    assert lengths == sorted(lengths, reverse=True)
    assert all("num_beams" not in c for c in model.calls)


def test_bug_paragraph_ja_zh_joins_without_spaces(para_ja):
    zh = {k: f"句{i}。" for i, k in enumerate(JA_VI)}
    zh["テーマ"] = "主题"
    zh["環境問題と私たちの生活"] = "环境问题与我们的生活"
    engine, _ = make_engine(lambda s, t, kw: zh[s], short_text_rerank=False)
    out, _ = engine.translate(pivot=False, text=para_ja, target_language="zh", num_beams=1)
    assert out == "主题：环境问题与我们的生活\n" + "".join(f"句{i}。" for i in range(1, 8))


def test_label_without_glossary_is_reranked_alone():
    engine, model = make_engine(lambda s, t, kw: {"the meeting is moved.": "cuộc họp bị dời."}[s])
    shorts: list[str] = []
    engine._translate_short = lambda text, **kw: shorts.append(text) or "Lưu ý"
    out, segments = engine.translate(pivot=False, text="Note: the meeting is moved.", target_language="vi", num_beams=1)
    assert shorts == ["Note"] and segments == 2
    assert out == "Lưu ý: cuộc họp bị dời."
    assert model.calls[0]["sources"] == ["the meeting is moved."]


# 80 distinct syllables: long output that is not a repetition loop.
WORDS = [c + v for c in "bcdghklmnpqrstvx" for v in ("a", "e", "o", "u", "i")]


def test_truncated_row_is_retried_with_a_larger_budget():
    long_answer = " ".join(WORDS[:70]) + "."

    def fn(source, target, kwargs):
        return long_answer

    engine, model = make_engine(fn)
    text = "Please write a long answer."  # 6 content tokens
    out, _ = engine.translate(pivot=False, text=text, target_language="vi", num_beams=1)
    assert out == long_answer
    assert [c["max_new_tokens"] for c in model.calls] == [64, 4 * 6 + 64]


def test_truncated_row_without_room_is_kept_and_logged(caplog):
    engine, model = make_engine(lambda s, t, kw: " ".join(WORDS[:70]))
    text = "Please write a long answer."
    out, _ = engine.translate(pivot=False, text=text, target_language="vi", num_beams=1, max_new_tokens=40)
    assert len(model.calls) == 1 and out == " ".join(WORDS[:40])
    assert "still ['truncated']" in caplog.text


def test_repetition_loop_is_retried_with_beams_and_ngram_block():
    def fn(source, target, kwargs):
        if kwargs.get("no_repeat_ngram_size") == 4 and kwargs.get("num_beams", 1) >= 4:
            return "Bảo vệ môi trường."
        return "Bảo vệ " + "môi trường " * 40

    engine, model = make_engine(fn)
    out, _ = engine.translate(pivot=False, text="環境を守る。", target_language="vi", num_beams=1)
    assert out == "Bảo vệ môi trường."
    assert model.calls[1]["num_beams"] == 4 and model.calls[1]["no_repeat_ngram_size"] == 4


def test_persistent_loop_is_trimmed():
    engine, _ = make_engine(lambda s, t, kw: "Bảo vệ. " + "môi trường " * 80)
    out, _ = engine.translate(pivot=False, text="環境を守る。", target_language="vi", num_beams=1)
    assert out == "Bảo vệ. môi trường"


def test_wrong_script_output_is_retried_with_beams():
    def fn(source, target, kwargs):
        if kwargs.get("num_beams", 1) >= 4:
            return "Vấn đề môi trường là gì?"
        return "環境問題とは、地球の環境に悪影響を及ぼすものです。"

    engine, model = make_engine(fn)
    out, _ = engine.translate(pivot=False, text="環境問題とは何ですか？", target_language="vi", num_beams=1)
    assert out == "Vấn đề môi trường là gì?"
    assert len(model.calls) == 2


def test_wrong_script_is_not_retried_with_the_same_beams(caplog):
    """At 4+ beams the "beam" retry would repeat the first search exactly."""
    engine, model = make_engine(lambda s, t, kw: "Tôi thích アニメ và マンガ lắm.")
    out, _ = engine.translate(pivot=False, text="アニメとマンガが大好きです。", target_language="vi")
    assert len(model.calls) == 1 and model.calls[0]["num_beams"] == 4
    assert out == "Tôi thích アニメ và マンガ lắm."
    assert "still ['script']" in caplog.text


def test_faithful_repetition_is_not_a_loop():
    def fn(source, target, kwargs):
        if kwargs.get("no_repeat_ngram_size"):
            return "Cố lên!"
        return "Cố lên, cố lên, cố lên, cố lên!"

    engine, model = make_engine(fn)
    out, _ = engine.translate(pivot=False, text="頑張れ頑張れ頑張れ頑張れ！", target_language="vi", num_beams=1)
    assert out == "Cố lên, cố lên, cố lên, cố lên!" and len(model.calls) == 1


def test_retry_that_is_not_better_keeps_the_first_output():
    engine, model = make_engine(lambda s, t, kw: "環境問題です。")
    out, _ = engine.translate(pivot=False, text="環境問題とは何ですか？", target_language="vi", num_beams=1)
    assert out == "環境問題です。" and len(model.calls) == 2


EN_VI = {
    "The sky is blue.": "Bầu trời màu xanh.",
    "The sea is wide.": "Biển rộng.",
    "We like it.": "Chúng tôi thích nó.",
}


def test_pack_with_dropped_sentence_falls_back_to_sentences():
    def fn(source, target, kwargs):
        if source in EN_VI:
            return EN_VI[source]
        return "Bầu trời màu xanh. Chúng tôi thích nó."  # dropped the middle sentence

    engine, model = make_engine(fn)
    out, segments = engine.translate(pivot=False, 
        text="The sky is blue. The sea is wide. We like it.", target_language="vi", num_beams=1
    )
    assert out == "Bầu trời màu xanh. Biển rộng. Chúng tôi thích nó."
    assert segments == 1
    assert [len(c["sources"]) for c in model.calls] == [1, 3]


def test_pack_merged_in_target_is_kept():
    """One sentence short by count but full length: two sentences joined in the target."""
    merged = "Bầu trời màu xanh và biển thì rộng. Chúng tôi thích nó."
    engine, model = make_engine(lambda s, t, kw: merged)
    out, _ = engine.translate(pivot=False, 
        text="The sky is blue. The sea is wide. We like it.", target_language="vi", num_beams=1
    )
    assert out == merged and len(model.calls) == 1


def test_pack_missing_two_sentences_falls_back():
    def fn(source, target, kwargs):
        if source in EN_VI:
            return EN_VI[source]
        return "Bầu trời màu xanh, biển rộng và chúng tôi thích nó lắm lắm."  # 1 sentence for 3

    engine, model = make_engine(fn)
    out, _ = engine.translate(pivot=False, 
        text="The sky is blue. The sea is wide. We like it.", target_language="vi", num_beams=1
    )
    assert out == "Bầu trời màu xanh. Biển rộng. Chúng tôi thích nó."
    assert [len(c["sources"]) for c in model.calls] == [1, 3]


def test_complete_pack_is_one_call():
    engine, model = make_engine(
        lambda s, t, kw: " ".join(EN_VI[x.strip()] for x in re.findall(r"[^.]+\.", s))
    )
    out, _ = engine.translate(pivot=False, 
        text="The sky is blue. The sea is wide. We like it.", target_language="vi", num_beams=1
    )
    assert out == "Bầu trời màu xanh. Biển rộng. Chúng tôi thích nó."
    assert len(model.calls) == 1


def test_thai_target_never_packs():
    engine, model = make_engine(lambda s, t, kw: "ท" + s)
    engine.translate(pivot=False, text="The sky is blue. The sea is wide.", target_language="th", num_beams=1)
    assert sorted(model.calls[0]["sources"]) == ["The sea is wide.", "The sky is blue."]


def _rerank_engine(answers: dict[str, str]):
    engine, model = make_engine(lambda s, t, kw: answers[s])
    shorts: list[str] = []

    def fake_short(text, **kwargs):
        shorts.append(text)
        return f"<{text}>"

    engine._translate_short = fake_short
    return engine, model, shorts


def test_short_text_rerank_only_for_whole_lines():
    engine, _, shorts = _rerank_engine({"次。": "Tiếp.", "それだ。": "Đó."})
    out, segments = engine.translate(pivot=False, text="kite\n\n次。それだ。", target_language="vi", num_beams=1)
    assert shorts == ["kite"]  # a paragraph of its own; "次。" inside a line is not
    assert out == "<kite>\n\nTiếp. Đó."
    assert segments == 3


def test_short_text_rerank_word_list_and_single_line():
    engine, model, shorts = _rerank_engine({})
    out, _ = engine.translate(pivot=False, text="kite\ncape\nkit", target_language="vi", num_beams=1)
    assert shorts == ["kite", "cape", "kit"] and not model.calls
    assert out == "<kite>\n<cape>\n<kit>"
    engine, _, shorts = _rerank_engine({})
    engine.translate(pivot=False, text="kite", target_language="vi", num_beams=1)
    assert shorts == ["kite"]


def test_short_lines_of_a_letter_are_not_reranked():
    answers = {
        "Hi John,": "Chào John,",
        "Thanks for your message. I will send it.": "Cảm ơn tin nhắn của bạn. Tôi sẽ gửi nó.",
        "Best regards,": "Trân trọng,",
        "Anna": "Anna",
    }
    engine, _, shorts = _rerank_engine(answers)
    text = "Hi John,\nThanks for your message. I will send it.\nBest regards,\nAnna"
    out, _ = engine.translate(pivot=False, text=text, target_language="vi", num_beams=1)
    assert shorts == []
    assert out == "\n".join(answers.values())
    # A short line next to a heading line of the same block is not a word list either.
    engine, _, shorts = _rerank_engine({"Hello": "Xin chào", "the meeting is moved.": "dời họp."})
    engine.translate(pivot=False, text="Hello\nNote: the meeting is moved.", target_language="vi", num_beams=1)
    assert shorts == ["Note"]  # the label only


def test_short_heading_content_is_reranked():
    """'Word: kite' was reranked whole before headings were split: its lone content
    word must not fall back to plain decoding ('kite' -> 'đi đi')."""
    engine, model, shorts = _rerank_engine({})
    out, _ = engine.translate(pivot=False, text="Word: kite", target_language="vi", num_beams=1)
    assert shorts == ["Word", "kite"] and not model.calls
    assert out == "<Word>: <kite>"
    engine, model, shorts = _rerank_engine({})
    out, _ = engine.translate(pivot=False, 
        text="テーマ：環境", target_language="vi", num_beams=1, lookup=GLOSSARY_JA_VI.get
    )
    assert shorts == ["環境"] and out == "Chủ đề: <環境>"


def test_dialogue_speaker_names_are_reranked_once():
    answers = {"おはようございます。": "Chào buổi sáng.", "おはよう。": "Chào.", "今日は暑いね。": "Hôm nay nóng nhỉ.",
               "そうですね。": "Đúng vậy."}
    engine, _, shorts = _rerank_engine(answers)
    text = "田中：おはようございます。\n佐藤：おはよう。今日は暑いね。\n田中：そうですね。"
    out, _ = engine.translate(pivot=False, text=text, target_language="vi", num_beams=1)
    assert shorts == ["田中", "佐藤"]
    assert out == "<田中>: Chào buổi sáng.\n<佐藤>: Chào. Hôm nay nóng nhỉ.\n<田中>: Đúng vậy."


def test_bulleted_heading_label_gets_the_glossary():
    engine, model = make_engine(lambda s, t, kw: JA_VI[s])
    engine._translate_short = lambda text, **kw: pytest.fail(f"reranked: {text}")
    out, _ = engine.translate(pivot=False, 
        text="・テーマ：環境問題と私たちの生活", target_language="vi", num_beams=1, lookup=GLOSSARY_JA_VI.get
    )
    assert out == "・Chủ đề: " + JA_VI["環境問題と私たちの生活"]


def test_glossary_matches_a_whole_split_line():
    engine, model = make_engine(lambda s, t, kw: pytest.fail("model must not run"))
    hits = {"おはよう。元気？": "Chào buổi sáng. Khỏe không?"}
    out, _ = engine.translate(pivot=False, 
        text="おはよう。元気？", target_language="vi", num_beams=1, lookup=hits.get
    )
    assert out == "Chào buổi sáng. Khỏe không?"


def test_symbols_are_copied_and_layout_kept():
    engine, _ = make_engine(lambda s, t, kw: "Xin chào.")
    out, _ = engine.translate(pivot=False, text="★ 2024 ★\n\n　こんにちは。", target_language="vi", num_beams=1)
    assert out == "★ 2024 ★\n\n　Xin chào."


def test_oom_batch_is_halved():
    engine, model = make_engine(lambda s, t, kw: JA_VI[s], max_rows=2)
    sentences = list(JA_VI)[1:5]
    out, _ = engine.translate(pivot=False, text="".join(sentences), target_language="vi", num_beams=1)
    assert out == " ".join(JA_VI[s] for s in sentences)
    assert [len(c["sources"]) for c in model.calls] == [2, 2]


def test_batch_size_and_token_cap_are_respected():
    engine, model = make_engine(lambda s, t, kw: JA_VI[s], batch_size=3)
    sentences = list(JA_VI)[1:]
    engine.translate(pivot=False, text="".join(sentences), target_language="vi", num_beams=1)
    assert [len(c["sources"]) for c in model.calls] == [3, 3, 1]

    engine, model = make_engine(lambda s, t, kw: JA_VI[s], batch_size=8, batch_max_tokens=200)
    engine.translate(pivot=False, text="".join(sentences), target_language="vi", num_beams=4)
    assert all(len(c["sources"]) == 1 for c in model.calls)
    assert all(c["num_beams"] == 4 for c in model.calls)


# --------------------------------------------------------------------------- pivot

PIVOT_JA_EN = {
    "日本の大学生活": "University life in Japan",
    "日本の大学生活は、高校までの生活とは大きく違います。":
        "University life in Japan is very different from life up to high school.",
    "私は、新しいことに挑戦することが必要だと思います。":
        "I think it is necessary to try new things. That is my opinion.",
    "２０２４年。": "2024.",
}
PIVOT_EN_VI = {
    "University life in Japan": "Cuộc sống đại học ở Nhật Bản",
    "University life in Japan is very different from life up to high school.":
        "Cuộc sống đại học ở Nhật Bản rất khác với cuộc sống thời trung học.",
    "I think it is necessary to try new things. That is my opinion.":
        "Tôi nghĩ cần phải thử những điều mới. Đó là ý kiến của tôi.",
}


def pivot_model(source, target, kwargs):
    if target == "en":
        return PIVOT_JA_EN.get(source, "")
    if target == "vi":
        return PIVOT_EN_VI.get(source, f"[direct {source}]")
    raise AssertionError(target)


def test_pivot_goes_through_english_and_keeps_layout():
    engine, model = make_engine(pivot_model)
    text = "テーマ：日本の大学生活  日本の大学生活は、高校までの生活とは大きく違います。私は、新しいことに挑戦することが必要だと思います。"
    out, segments = engine.translate(
        text=text, target_language="vi", num_beams=1, lookup=GLOSSARY_JA_VI.get, source_language="ja"
    )
    assert out == (
        "Chủ đề: Cuộc sống đại học ở Nhật Bản  "
        "Cuộc sống đại học ở Nhật Bản rất khác với cuộc sống thời trung học. "
        "Tôi nghĩ cần phải thử những điều mới. Đó là ý kiến của tôi."
    )
    assert segments == 4
    stage1 = [s for c in model.calls[:1] for s in c["sources"]]
    assert all(s in PIVOT_JA_EN for s in stage1) and len(stage1) == 3
    # Stage 2 gets English only; the 2-sentence English output is packed as one unit.
    stage2 = [s for c in model.calls[1:] for s in c["sources"]]
    assert sorted(stage2) == sorted(PIVOT_EN_VI)


def test_pivot_empty_intermediate_falls_back_to_direct(caplog):
    engine, model = make_engine(pivot_model)
    out, _ = engine.translate(
        text="知らない文です。", target_language="vi", num_beams=1, source_language="ja"
    )
    assert out == "[direct 知らない文です。]"
    assert "translated directly instead" in caplog.text


def test_pivot_number_only_intermediate_is_copied():
    engine, model = make_engine(pivot_model)
    out, _ = engine.translate(text="２０２４年。", target_language="vi", num_beams=1, source_language="ja")
    assert out == "2024."
    assert len(model.calls) == 1  # nothing left to translate after stage 1


def test_pivot_false_is_direct():
    engine, model = make_engine(pivot_model)
    out, _ = engine.translate(
        text="日本の大学生活は、高校までの生活とは大きく違います。", target_language="vi", num_beams=1, pivot=False
    )
    assert out.startswith("[direct ")
    assert not any(s in PIVOT_EN_VI for c in model.calls for s in c["sources"])


def test_last_pivot_reports_only_a_pivot_that_ran():
    engine, model = make_engine(pivot_model)
    engine._translate_short = lambda text, **kw: "con mèo"
    engine.translate(text="猫", target_language="vi", num_beams=1, source_language="ja")
    assert engine.last_pivot is None and not model.calls
    engine.translate(text="日本の大学生活", target_language="vi", num_beams=1, source_language="ja")
    assert engine.last_pivot == "en"
    engine.translate(text="知らない文です。", target_language="vi", num_beams=1, source_language="ja")
    assert engine.last_pivot is None  # empty intermediate: translated directly


def test_pivot_is_chosen_per_unit_in_a_mixed_study_sheet():
    """ja->vi sheet with Vietnamese notes and an English line: only Japanese goes via en."""
    seen: list[tuple[str, str]] = []

    def fn(source, target, kwargs):
        seen.append((target, source))
        if target == "en":
            return PIVOT_JA_EN.get(source, "?")
        return PIVOT_EN_VI.get(source, f"[{source}]")

    engine, model = make_engine(fn, short_text_rerank=False)
    text = (
        "日本の大学生活\n"
        "Ghi chú: đây là câu ví dụ của bài học.\n"
        "Please read it twice before class."
    )
    out, _ = engine.translate(text=text, target_language="vi", num_beams=1, source_language="ja")
    assert out.splitlines() == [
        "Cuộc sống đại học ở Nhật Bản",
        "Ghi chú: đây là câu ví dụ của bài học.",
        "[Please read it twice before class.]",
    ]
    assert ("en", "日本の大学生活") in seen
    assert not any("đây là" in src or "Ghi chú" in src for _, src in seen)  # copied, never decoded
    assert ("vi", "Please read it twice before class.") in seen  # English line: direct
    assert ("en", "Please read it twice before class.") not in seen


def test_pivot_requested_on_unknown_source_and_mixed_routes():
    engine, model = make_engine(pivot_model)
    out, _ = engine.translate(
        text="日本の大学生活", target_language="vi", num_beams=1, source_language="zh",
        source_reliable=False, pivot=True,
    )
    assert out == "Cuộc sống đại học ở Nhật Bản" and engine.last_pivot == "en"


def test_pivot_stage1_flagged_unit_is_translated_directly(caplog):
    long_ja = "私は毎日学校へ行って友達と一緒に勉強しています。"  # 25 content tokens

    def fn(source, target, kwargs):
        if target == "en":
            return "I."  # far too short: flagged "short" (not retried at 4 beams)
        return f"[direct {source}]"

    engine, model = make_engine(fn)
    out, _ = engine.translate(text=long_ja, target_language="vi", num_beams=4, source_language="ja")
    assert out == f"[direct {long_ja}]"
    assert engine.last_pivot is None
    assert "still ['short']" in caplog.text
