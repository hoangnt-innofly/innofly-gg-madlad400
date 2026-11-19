from __future__ import annotations

import re

import pytest

from app.services.segmenter import (
    count_sentences,
    hard_split,
    join_units,
    line_sources,
    split_text,
)

CASES = [
    ("zh", "我今天很忙。你呢？我们明天见吧！", ["我今天很忙。", "你呢？", "我们明天见吧！"]),
    ("zh-halfwidth", "我很好.你呢?今天天气不错.", ["我很好.", "你呢?", "今天天气不错."]),
    # A closing quote glued to the next word is followed by its speech tag: no split.
    ("zh-quote", "“我来了。”他说。然后走了。", ["“我来了。”他说。", "然后走了。"]),
    ("ja-quote-cont", "「行こう。」と彼は言った。私も行った。", ["「行こう。」と彼は言った。", "私も行った。"]),
    ("ja-quote-seq", "「おはよう。」「おはよう。」今日は晴れだ。", ["「おはよう。」", "「おはよう。」今日は晴れだ。"]),
    ("ja-quote-spaced", "「おはよう。」「おはよう。」　今日は晴れだ。", ["「おはよう。」", "「おはよう。」", "今日は晴れだ。"]),
    ("ja-reikoron", "例：りんご、みかん。次の文。", ["例", "りんご、みかん。", "次の文。"]),
    ("ja-ellipsis", "そうですね……わかりました。", ["そうですね……わかりました。"]),
    ("ja-fw-decimal", "値は３．５です。次。", ["値は３．５です。", "次。"]),
    ("ja-quotative", "なぜ？と思った。そうだ。", ["なぜ？と思った。", "そうだ。"]),
    ("ko", "안녕하세요. 저는 학생입니다. 만나서 반갑습니다!", ["안녕하세요.", "저는 학생입니다.", "만나서 반갑습니다!"]),
    ("en-abbr", "Mr. Smith met Dr. Jones in the U.S. Army. It was 3.5 km away, e.g. near No. 5 Street. Then they left.",
     ["Mr. Smith met Dr. Jones in the U.S. Army.", "It was 3.5 km away, e.g. near No. 5 Street.", "Then they left."]),
    ("en-initials", "J. K. Rowling wrote it. She lives in the UK.", ["J. K. Rowling wrote it.", "She lives in the UK."]),
    ("en-url", "Visit www.example.com/a.b?x=1 now. Mail me at a.b@c.org. Thanks!",
     ["Visit www.example.com/a.b?x=1 now.", "Mail me at a.b@c.org.", "Thanks!"]),
    ("en-dialogue", '"Stop!" she said. "Why?" He asked. She left.', ['"Stop!" she said.', '"Why?"', "He asked.", "She left."]),
    ("en-ellipsis", "Well... maybe. I don't know... Let's see.", ["Well... maybe.", "I don't know...", "Let's see."]),
    ("en-list", "1. Open the box.\n2. Take the phone. Charge it.\n- Done", ["Open the box.", "Take the phone.", "Charge it.", "Done"]),
    ("en-wrap", "This is a long sentence that was\nhard wrapped by an email client. Next one.",
     ["This is a long sentence that was hard wrapped by an email client.", "Next one."]),
    ("en-title", "Introduction\nThe world is big. It is round.", ["Introduction", "The world is big.", "It is round."]),
    ("word-list", "kite\ncape\nkit", ["kite", "cape", "kit"]),
    ("es", "¡Hola! ¿Cómo estás? Muy bien, gracias. Sra. García llegó.", ["¡Hola!", "¿Cómo estás?", "Muy bien, gracias.", "Sra. García llegó."]),
    ("fr", "M. Dupont est là. Il dit « Bonjour. » Puis il part.", ["M. Dupont est là.", "Il dit « Bonjour. »", "Puis il part."]),
    ("el", "Γεια σου. Τι κάνεις; Είμαι καλά.", ["Γεια σου.", "Τι κάνεις;", "Είμαι καλά."]),
    ("hi", "मेरा नाम राम है। मैं भारत से हूँ। आप कैसे हैं?", ["मेरा नाम राम है।", "मैं भारत से हूँ।", "आप कैसे हैं?"]),
    ("hi-pipe", "मेरा नाम राम है| मैं भारत से हूँ|", ["मेरा नाम राम है|", "मैं भारत से हूँ|"]),
    ("ar", "مرحبا. كيف حالك؟ أنا بخير.", ["مرحبا.", "كيف حالك؟", "أنا بخير."]),
    ("ur", "میرا نام علی ہے۔ آپ کیسے ہیں؟", ["میرا نام علی ہے۔", "آپ کیسے ہیں؟"]),
    ("hy", "Բարև։ Ինչպե՞ս ես։ Ես լավ եմ։", ["Բարև։", "Ինչպե՞ս ես։", "Ես լավ եմ։"]),
    ("am", "ሰላም ነህ። ስምህ ማን ነው፧ እኔ ደህና ነኝ።", ["ሰላም ነህ።", "ስምህ ማን ነው፧", "እኔ ደህና ነኝ።"]),
    ("my", "မင်္ဂလာပါ။ နေကောင်းလား။", ["မင်္ဂလာပါ။", "နေကောင်းလား။"]),
    ("km", "សួស្តី។ អ្នកសុខសប្បាយទេ?", ["សួស្តី។", "អ្នកសុខសប្បាយទេ?"]),
    ("th-be", "ปี พ.ศ. 2500 เป็นปีที่ดี", ["ปี พ.ศ. 2500 เป็นปีที่ดี"]),
    ("mixed-ja-en", "iPhoneを買った。It is great. また来週。", ["iPhoneを買った。", "It is great.", "また来週。"]),
    ("indent-ja", "　近年、温暖化が進む。\n　海も汚れている。", ["近年、温暖化が進む。", "海も汚れている。"]),
    ("blank-paras", "First para. Second sentence.\n\nSecond para.", ["First para.", "Second sentence.", "Second para."]),
    ("unbalanced", "He said “hello. Then left. Bye.", ["He said “hello.", "Then left.", "Bye."]),
    # 1. ASCII "." glued to CJK / Hangul: same abbreviation checks, split only after CJK.
    ("glued-mr-ja", "Mr.田中さんが来た。", ["Mr.田中さんが来た。"]),
    ("glued-dr-ja", "Dr.スミスに会った。", ["Dr.スミスに会った。"]),
    ("glued-us-ja", "彼はU.S.出身です。", ["彼はU.S.出身です。"]),
    ("glued-inc-ja", "Apple Inc.の製品", ["Apple Inc.の製品"]),
    ("glued-mr-ko", "Mr.김이 왔다.", ["Mr.김이 왔다."]),
    ("glued-brand", "Yahoo!ニュースを見た。", ["Yahoo!ニュースを見た。"]),
    ("glued-zh-short", "我好.你好.他好.", ["我好.", "你好.", "他好."]),
    ("glued-ko", "네.좋아요.", ["네.", "좋아요."]),
    # 2. Closing quote glued to any letter: the speech tag follows.
    ("quote-ko-ascii", '"Hi!"라고 말했다.', ['"Hi!"라고 말했다.']),
    ("quote-ko-single", "'좋아.'하고 웃었다.", ["'좋아.'하고 웃었다."]),
    ("quote-ja-excl", "「行こう！」と言った。", ["「行こう！」と言った。"]),
    # 3. Ordinals, month and company abbreviations; 4-digit years still split.
    ("ord-de", "Wir sehen uns am 24. Dezember wieder. Bis dann.", ["Wir sehen uns am 24. Dezember wieder.", "Bis dann."]),
    ("ord-de-century", "Das 21. Jahrhundert ist da.", ["Das 21. Jahrhundert ist da."]),
    ("ord-cs", "Narodil se 5. května 1945.", ["Narodil se 5. května 1945."]),
    ("ord-tr", "Bu 1. Dünya Savaşı idi.", ["Bu 1. Dünya Savaşı idi."]),
    ("year", "I was born in 1990. Then I moved.", ["I was born in 1990.", "Then I moved."]),
    ("month", "We met on Jan. 5 in Paris. It rained.", ["We met on Jan. 5 in Paris.", "It rained."]),
    ("month-fr", "Le 5 déc. 2024 il neigeait.", ["Le 5 déc. 2024 il neigeait."]),
    ("month-word", "It was in Jan. The year was cold.", ["It was in Jan.", "The year was cold."]),
    ("mt-co", "Climb Mt. Fuji with Acme Co. Ltd. Today.", ["Climb Mt. Fuji with Acme Co. Ltd. Today."]),
    # 6. Heading labels: label unit + content units; times / URLs / clauses stay whole.
    ("heading-ja", "テーマ：環境問題と私たちの生活", ["テーマ", "環境問題と私たちの生活"]),
    ("heading-en", "Note: the meeting is moved.", ["Note", "the meeting is moved."]),
    ("heading-time", "Time: 10:30", ["Time", "10:30"]),
    ("heading-fr", "Remarque : texte ici.", ["Remarque", "texte ici."]),
    ("no-heading-time", "会議は10：30からです。", ["会議は10：30からです。"]),
    ("no-heading-ratio", "1:2 is the ratio.", ["1:2 is the ratio."]),
    ("no-heading-url", "http://example.com is down.", ["http://example.com is down."]),
    ("no-heading-clause", "The reasons are simple: cost and time.", ["The reasons are simple: cost and time."]),
    ("no-heading-q", "Q: What is it?", ["Q: What is it?"]),
    ("no-heading-long-ja", "彼はこう言ったのです：行こう。", ["彼はこう言ったのです：行こう。"]),
]


@pytest.mark.parametrize("name,text,expected", CASES, ids=[c[0] for c in CASES])
def test_sentence_units(name, text, expected, token_len):
    units = split_text(text, token_len=token_len, max_tokens=480)
    assert [u.text for u in units] == expected


def _lang(text: str) -> str:
    return "ja" if re.search(r"[぀-ヿ一-鿿]", text) else "en"


@pytest.mark.parametrize("name,text,expected", CASES, ids=[c[0] for c in CASES])
def test_round_trip(name, text, expected, token_len):
    """Joining the source units back (target = source language) rebuilds the source."""
    if name in ("en-wrap", "zh-halfwidth", "glued-zh-short", "glued-ko", "heading-fr", "ja-quote-spaced"):
        pytest.skip(
            "intended: soft wrap merged / ASCII stops before CJK get a space / "
            "' : ' -> ': ' / in-line U+3000 between sentences not copied"
        )
    units = split_text(text, token_len=token_len)
    rebuilt = join_units(units, [u.text for u in units], _lang(text))
    norm = lambda s: re.sub(r"[ \t]+\n", "\n", s).strip()
    assert norm(rebuilt) == norm(text)


def test_bug_paragraph_is_title_plus_seven_sentences(para_ja, token_len):
    units = split_text(para_ja, token_len=token_len, max_tokens=480, pack_tokens=256)
    assert len(units) == 9
    label, content = units[0], units[1]
    assert (label.text, label.sep, label.label, label.whole_line) == ("テーマ", "：", True, False)
    assert (content.text, content.sep, content.label) == ("環境問題と私たちの生活", "\n", False)
    assert label.line == content.line == 0
    assert all(u.text.endswith("。") for u in units[2:])
    assert not any(u.whole_line or u.parts or u.label for u in units[2:])  # never packed
    assert content.whole_line and not content.parts  # the whole content of the heading line
    assert "".join(u.text for u in units[2:]) == para_ja.split("\n", 1)[1]
    assert line_sources(units)[0] == "テーマ：環境問題と私たちの生活"


def test_bug_paragraph_rejoined_per_target(para_ja, token_len):
    units = split_text(para_ja, token_len=token_len)
    vi = ["Chủ đề", "X"] + [f"Câu {i}." for i in range(1, 8)]
    assert join_units(units, vi, "vi") == "Chủ đề: X\n" + " ".join(vi[2:])
    zh = ["主题", "环境"] + [f"句{i}。" for i in range(1, 8)]
    assert join_units(units, zh, "zh") == "主题：环境\n" + "".join(zh[2:])
    ja = ["テーマ", "環境"] + [f"文{i}。" for i in range(1, 8)]
    assert join_units(units, ja, "ja") == "テーマ：環境\n" + "".join(ja[2:])
    assert join_units(units, vi, "zh_Latn") == "Chủ đề: X\n" + " ".join(vi[2:])
    en = ["Theme", "X"] + [f"Sentence {i}." for i in range(1, 8)]
    assert join_units(units, en, "en") == "Theme: X\n" + " ".join(en[2:])


def test_heading_label_join_edge_cases(token_len):
    units = split_text("Note: the meeting is moved.", token_len=token_len)
    assert [(u.text, u.label) for u in units] == [("Note", True), ("the meeting is moved.", False)]
    # The model echoing the colon does not double it.
    assert join_units(units, ["Lưu ý:", "cuộc họp bị dời."], "vi") == "Lưu ý: cuộc họp bị dời."
    # A line-level glossary hit sits in the label's slot, the rest is blank: no colon added.
    assert join_units(units, ["Lưu ý: cuộc họp bị dời.", ""], "vi") == "Lưu ý: cuộc họp bị dời."
    units = split_text("Time: 10:30", token_len=token_len)
    assert [(u.text, u.translate) for u in units] == [("Time", True), ("10:30", False)]
    assert join_units(units, ["Thời gian", "10:30"], "vi") == "Thời gian: 10:30"
    units = split_text("- テーマ：環境", token_len=token_len)
    assert units[0].lead == "- " and units[0].label
    assert join_units(units, ["Chủ đề", "Môi trường"], "vi") == "- Chủ đề: Môi trường"


def test_heading_label_never_packed(token_len):
    units = split_text("Note: The sky is blue. The sea is wide.", token_len=token_len, pack_tokens=256)
    assert [u.text for u in units] == ["Note", "The sky is blue. The sea is wide."]
    assert len(units[1].parts) == 2


def test_blocks_follow_blank_lines(token_len):
    units = split_text("kite\ncape\n\nHi.\n\n\nBye.", token_len=token_len)
    assert [(u.text, u.block) for u in units] == [("kite", 0), ("cape", 0), ("Hi.", 1), ("Bye.", 2)]


def test_spaced_sentences_are_packed_cjk_never(token_len):
    text = "The sky is blue. The sea is wide. We like it.\n空は青い。海は広い。"
    units = split_text(text, token_len=token_len, pack_tokens=256)
    assert [u.text for u in units] == [
        "The sky is blue. The sea is wide. We like it.",
        "空は青い。",
        "海は広い。",
    ]
    assert units[0].whole_line and [p.text for p in units[0].parts] == [
        "The sky is blue.",
        "The sea is wide.",
        "We like it.",
    ]
    assert units[0].sep == "\n"


def test_pack_respects_token_limit_and_mixed_scripts(token_len):
    text = "iPhoneを買った。It is great. Really great. また来週。"
    units = split_text(text, token_len=token_len, pack_tokens=256)
    assert [u.text for u in units] == ["iPhoneを買った。", "It is great. Really great.", "また来週。"]
    long_line = " ".join(f"Sentence number {i} is here." for i in range(60))
    packed = split_text(long_line, token_len=token_len, pack_tokens=64)
    assert len(packed) > 1
    assert all(token_len(u.text) <= 64 for u in packed)
    assert " ".join(u.text for u in packed) == long_line


def test_pack_disabled_gives_one_sentence_per_unit(token_len):
    units = split_text("A cat sat. A dog ran.", token_len=token_len, pack_tokens=0)
    assert [u.text for u in units] == ["A cat sat.", "A dog ran."]
    assert not any(u.whole_line for u in units)


def test_list_markers_and_layout_preserved(token_len):
    text = "1. Open the box.\n2. Take it.\n\n\n　次の段落。"
    units = split_text(text, token_len=token_len)
    out = join_units(units, ["Mở hộp.", "Lấy nó.", "Đoạn tiếp."], "vi")
    assert out == "1. Mở hộp.\n2. Lấy nó.\n\n\n　Đoạn tiếp."


def test_empty_outputs_leave_no_double_spaces(token_len):
    units = split_text("A one. B two. C three.\nD four.", token_len=token_len)
    assert join_units(units, ["X.", "", "Z.", "W."], "en") == "X. Z.\nW."
    assert join_units(units, ["", "", "", "W."], "en") == "W."


def test_symbol_only_units_are_not_translated(token_len):
    units = split_text("★ 2024 ★\nOK.", token_len=token_len)
    assert [(u.text, u.translate) for u in units] == [("★ 2024 ★", False), ("OK.", True)]


def test_line_sources_and_count_sentences(token_len):
    units = split_text("おはよう。元気？\nkite", token_len=token_len)
    assert line_sources(units) == {0: "おはよう。元気？", 1: "kite"}
    assert count_sentences("Xin chào. Tôi là sinh viên. TP. HCM rất đẹp.") == 3
    assert count_sentences("Một câu duy nhất") == 1
    assert count_sentences("") == 0


def test_hard_split_cjk_without_spaces(token_len):
    long_han = "環境問題" * 200
    pieces = hard_split(long_han, token_len, 200)
    assert len(pieces) >= 2 and all(token_len(p) <= 200 for p in pieces)
    assert "".join(pieces) == long_han
    clauses = "、".join(["地球温暖化や海洋汚染などの環境問題が深刻になっています"] * 30) + "。"
    pieces = hard_split(clauses, token_len, 200)
    assert len(pieces) >= 2 and all(token_len(p) <= 200 for p in pieces)
    assert pieces[0].endswith("、") and "".join(pieces) == clauses
    units = split_text(clauses, token_len=token_len, max_tokens=200)
    assert len(units) >= 2
    assert join_units(units, [u.text for u in units], "ja") == clauses


def test_hard_split_of_a_huge_run_needs_few_tokenizer_calls(token_len):
    calls = []

    def counting(s):
        calls.append(1)
        return token_len(s)

    text = "環境問題" * 5000  # 20k characters, no spaces, no punctuation
    pieces = hard_split(text, counting, 480)
    assert "".join(pieces) == text and all(token_len(p) <= 480 for p in pieces)
    assert len(calls) < 40 * len(pieces)


def test_hard_split_keeps_combining_marks(token_len):
    thai = "ที่นี่" * 400  # no spaces, combining vowel/tone marks
    pieces = hard_split(thai, token_len, 64)
    assert "".join(pieces) == thai
    for p in pieces:
        assert not ("ั" <= p[0] <= "ฺ" or "็" <= p[0] <= "๎")


def test_crlf_normalised(token_len):
    units = split_text("Line one\r\nLine two.\r\n\r\nPara two.", token_len=token_len)
    assert [u.text for u in units] == ["Line one", "Line two.", "Para two."]
    assert join_units(units, ["A", "B.", "C."], "vi") == "A\nB.\n\nC."


def test_cjk_title_and_chat_lines_are_not_glued(token_len):
    """Lines of similar length are not a fixed-width wrap: a title line or chat messages
    stay separate units (title + sentence in one input is what drops sentences)."""
    title = "テーマ：地球温暖化と海洋汚染が私たちの日常生活に与える深刻な影響について"
    text = (
        title + "\n近年、地球温暖化や海洋汚染などの環境問題が深刻になっています。私たちも原因です。"
        "\n例えば、プラスチックごみを減らすことが大切です。小さな行動でも大きな変化になる。"
    )
    units = split_text(text, token_len=token_len)
    assert [u.text for u in units[:3]] == [
        "テーマ",
        title.split("：", 1)[1],
        "近年、地球温暖化や海洋汚染などの環境問題が深刻になっています。",
    ]
    assert units[1].sep == "\n"
    chat = (
        "今日は朝から雨が降っていたので傘を持って駅まで歩いて行ったんだけど\n"
        "電車が遅れていて会社に着いたのは十時過ぎだったよ本当に疲れた一日\n"
        "まあ明日は晴れるらしいから少しは気分も良くなるといいな。"
    )
    units = split_text(chat, token_len=token_len)
    assert [u.text for u in units] == chat.split("\n")
    assert join_units(units, ["A", "B", "C."], "vi") == "A\nB\nC."


def test_cjk_fixed_width_wrap_is_merged(token_len):
    sentence = (
        "近年、地球温暖化や海洋汚染などの環境問題が世界中で深刻になっており、"
        "私たち一人一人の努力も必要だと多くの専門家が指摘しています。"
    )
    wrapped = "\n".join(sentence[i : i + 30] for i in range(0, len(sentence), 30))
    assert wrapped.count("\n") == 2
    units = split_text(wrapped + "\n次の段落です。", token_len=token_len)
    assert [u.text for u in units] == [sentence, "次の段落です。"]


@pytest.mark.parametrize("bullet", ["・", "■", "◆", "※", "● "])
def test_bullet_before_a_heading_is_the_lead(bullet, token_len):
    units = split_text(bullet + "テーマ：環境問題", token_len=token_len)
    assert [(u.lead, u.text, u.label) for u in units] == [(bullet, "テーマ", True), ("", "環境問題", False)]
    assert join_units(units, ["Chủ đề", "Môi trường"], "vi") == bullet + "Chủ đề: Môi trường"


def test_markdown_heading_and_glued_ascii_are_not_labels(token_len):
    units = split_text("# Theme: x", token_len=token_len)
    assert [(u.lead, u.text, u.label) for u in units] == [("# ", "Theme", True), ("", "x", False)]
    for text in ("#tag: hello there.", "-5 degrees today.", "★★★"):
        units = split_text(text, token_len=token_len)
        assert [(u.lead, u.text) for u in units] == [("", text)]


@pytest.mark.parametrize("brk", [" ", " ", "\x85", "\x0b", "\x0c"])
def test_unicode_line_breaks_split_lines(brk, token_len):
    text = "テーマ：環境問題と私たちの生活" + brk + "近年、温暖化が進んでいる。私たちにできることは何か。"
    units = split_text(text, token_len=token_len)
    assert [u.text for u in units] == [
        "テーマ", "環境問題と私たちの生活", "近年、温暖化が進んでいる。", "私たちにできることは何か。"
    ]
    assert units[1].sep == "\n"
    assert count_sentences(text) == len(units) - 1  # the label is no sentence of its own


def test_label_colon_follows_the_label_script(token_len):
    assert join_units(split_text("問1：答えなさい。"), ["問1", "答えなさい。"], "ja") == "問1：答えなさい。"
    units = split_text("Step 1: open the box.", token_len=token_len)
    assert join_units(units, ["ステップ1", "箱を開けてください。"], "ja") == "ステップ1：箱を開けてください。"
    assert join_units(units, ["Step 1", "打开盒子。"], "zh") == "Step 1: 打开盒子。"
    assert join_units(units, ["Bước 1", "mở hộp."], "vi") == "Bước 1: mở hộp."


def test_number_before_an_english_sentence_ends_it(token_len):
    units = split_text("I counted to 3. Then we left.", token_len=token_len)
    assert [u.text for u in units] == ["I counted to 3.", "Then we left."]
    units = split_text("Wir sehen uns am 24. Dezember. Das 3. Mal.", token_len=token_len)
    assert [u.text for u in units] == ["Wir sehen uns am 24. Dezember.", "Das 3. Mal."]


def test_hard_split_at_a_spaced_dash(token_len):
    text = " - ".join(["word " * 30] * 4).strip()
    pieces = hard_split(text, token_len, 60)
    assert all(token_len(p) <= 60 for p in pieces)
    assert " ".join(pieces).split() == text.split()
