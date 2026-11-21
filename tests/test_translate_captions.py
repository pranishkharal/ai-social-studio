"""Tests for translate_captions.py (gta6-clipper feature 19).

The Argos model calls are mocked — no 90 MB models are downloaded in
this VM, and the module is reported as implemented-but-untested-live.
The timing, chunking and masking logic is tested for real.
"""

import pytest

import translate_captions as tc
from translate_captions import TranslationUnavailableError


def test_spread_proportional_and_contiguous():
    words = tc.spread(["aa", "bbbbbb", "c"], 1.0, 2.0, True)
    assert words[0]["s"] == 1.0
    assert words[-1]["e"] == 2.0
    assert words[-1]["end"] is True
    assert words[0]["end"] is False
    assert all(a["e"] == b["s"] for a, b in zip(words, words[1:]))
    # The longest word gets the longest slice.
    assert words[1]["e"] - words[1]["s"] > words[0]["e"] - words[0]["s"]


def test_sentences_split_on_end_and_cap():
    words = [dict(w=f"w{i}", s=i * 0.3, e=i * 0.3 + 0.25, end=(i == 5))
             for i in range(50)]
    groups = tc.sentences(words)
    assert [len(g) for g in groups] == [6, 22, 22]  # end flag, then the 22 cap


def test_cjk_chunks():
    assert tc.cjk_chunks("这太疯狂了，我们走吧！") == ["这太疯", "狂了", "我们走吧！"]
    chunks = tc.cjk_chunks("他在维斯普奇海滩抢了银行。")
    assert "".join(chunks) == "他在维斯普奇海滩抢了银行"
    assert all(1 < len(c) <= 4 for c in chunks)


def test_mask():
    assert tc._mask("mierda") == "m*****"
    assert tc._mask("Hola") == "Hola"
    assert tc._mask("كسمك")[0] == "ك"


def test_passthrough_for_english():
    words = [dict(w="hi", s=0.0, e=0.5, end=True)]
    assert tc.translate_words(words, "en") is words
    assert tc.translate_text("hi", "") == "hi"
    assert tc.translate_text("hi", None) == "hi"


def test_translate_words_with_mocked_model(monkeypatch):
    def fake_batch(texts, lang):
        return ["Vamos a la playa." if lang == "es" else " ".join(texts)]

    monkeypatch.setattr(tc, "_translate_batch", fake_batch)
    words = [dict(w=x, s=i * 0.3, e=i * 0.3 + 0.25, end=x.endswith("."))
             for i, x in enumerate("We are going to the beach.".split())]
    out = tc.translate_words(words, "es")
    assert " ".join(w["w"] for w in out) == "Vamos a la playa"
    assert out[0]["s"] == 0
    assert abs(out[-1]["e"] - words[-1]["e"]) < 1e-6
    assert sum(w["end"] for w in out) == 1


def test_translate_text_with_mocked_model(monkeypatch):
    monkeypatch.setattr(tc, "_translate_batch",
                        lambda texts, lang: ["C'est fou"])
    assert tc.translate_text("This is crazy", "fr") == "C'est fou"
    # ALL-CAPS hooks are sentence-cased before translating.
    seen = []
    monkeypatch.setattr(tc, "_translate_batch",
                        lambda texts, lang: seen.append(texts[0]) or ["x"])
    tc.translate_text("THIS IS CRAZY", "fr")
    assert seen == ["This is crazy"]


def test_translate_words_cjk_with_mocked_model(monkeypatch):
    monkeypatch.setattr(tc, "_translate_batch",
                        lambda texts, lang: ["这太疯狂了"])
    words = [dict(w=x, s=i * 0.3, e=i * 0.3 + 0.25, end=(i == 2))
             for i, x in enumerate(["This", "is", "crazy."])]
    out = tc.translate_words(words, "zh")
    assert [w["w"] for w in out] == ["这太疯", "狂了"]


def test_load_without_argos_raises_clear_error(monkeypatch):
    """Without Argos installed, the error says how to install it."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("ctranslate2", "sentencepiece", "argostranslate.package"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    tc._cache.clear()
    with pytest.raises(TranslationUnavailableError, match="pip install argostranslate"):
        tc._load("es")


def test_rtl_and_languages():
    assert tc.is_rtl("ar") and tc.is_rtl("he")
    assert not tc.is_rtl("es")
    assert "hi" in tc.available_languages()
    assert "es" in tc.available_languages()
