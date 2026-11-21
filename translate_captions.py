"""Offline translation of captions + hook (gta6-clipper feature 19).

Argos Translate (OPUS models on ctranslate2, ~90 MB per language, no
torch): the same clip with Hindi or Spanish captions, from the same
render. Handles right-to-left languages, CJK chunking (Chinese and
Japanese have no spaces, so captions need character chunks instead
of words), and re-spreads the word timings over the translated words
so captions stay in sync.

Argos is an OPTIONAL lazy import: nothing is imported — and no model
is downloaded — until a translation is actually requested, and this
module works (raising a clear error) on a machine without Argos
installed. The language models download once into
~/.local/share/argos-translate on first use (~90 MB per language).

Status: implemented and unit-tested with mocked models. NOT tested
live in this VM — no models were downloaded here, and it needs a
machine with Argos installed to run for real.
"""

import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

MODELS_DIR = Path.home() / ".local" / "share" / "argos-translate" / "packages"

# Languages the Argos index offers for English -> X.
LANGS = "es fr pt ar hi de he it nl tr ru pl id ja ko zh".split()

# Right-to-left languages: the hook/caption text direction flips.
RTL = {"ar", "he", "fa", "ur"}

# Chinese/Japanese: written without spaces, so captions need chunks.
CJK = ("zh", "ja")

# Punctuation stripped from translated caption words — only ? and !
# stay (plus ؟ for Arabic), matching the caption cleaner.
_PUNCT = re.compile(r'[,.;:"«»“”،。、¿¡।()]+')

# Short per-language profanity list, masked the same way as the
# English captions (first letter + stars).
_BAD = re.compile(
    r"^(mierda|joder|coño|puta|puto|cabrón|cabron|merde|putain|connard|salope"
    r"|foda-se|porra|caralho|scheiße|scheisse|fick\w*|arschloch|kurwa"
    r"|كس\w*|شرموط\w*)$", re.I)

_cache = {}


class TranslationUnavailableError(RuntimeError):
    """Argos Translate is not installed, or the model could not be
    downloaded. The fix is in the message."""


def _load(lang):
    """(ctranslate2 Translator, SentencePiece processor) for English ->
    ``lang``. Downloads the model once when missing. Raises
    TranslationUnavailableError when Argos is not installed."""
    if lang in _cache:
        return _cache[lang]
    try:
        import ctranslate2
        import sentencepiece
    except ImportError:
        raise TranslationUnavailableError(
            "Argos Translate is not installed. Install it with: "
            "pip install argostranslate ctranslate2 sentencepiece "
            "(then the English -> " + lang + " model downloads once, ~90 MB).")

    def find_model():
        return next((d for d in MODELS_DIR.glob(f"*en_{lang}*")
                     if (d / "model").exists()), None)

    model_dir = find_model()
    if model_dir is None:
        import argostranslate.package as packages
        try:
            packages.update_package_index()
        except Exception as e:
            raise TranslationUnavailableError(
                f"Could not download the translation list (internet?): {str(e)[:120]}")
        package = next((p for p in packages.get_available_packages()
                        if p.from_code == "en" and p.to_code == lang), None)
        if package is None:
            raise TranslationUnavailableError(f"No offline model for English -> {lang}")
        print(f"  downloading English -> {lang} model (~90 MB, once)",
              file=sys.stderr)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=MODELS_DIR))
        try:
            zipfile.ZipFile(package.download()).extractall(tmp)
        except Exception as e:
            shutil.rmtree(tmp, True)
            raise TranslationUnavailableError(
                f"Translation model download failed (internet?): {str(e)[:120]}")
        for item in tmp.iterdir():
            if not (MODELS_DIR / item.name).exists():
                os.replace(item, MODELS_DIR / item.name)
        shutil.rmtree(tmp, True)
        model_dir = find_model()
    _cache[lang] = (
        ctranslate2.Translator(str(model_dir / "model"), device="cpu",
                               inter_threads=1),
        sentencepiece.SentencePieceProcessor(
            model_file=str(model_dir / "sentencepiece.model")))
    return _cache[lang]


def _translate_batch(texts, lang):
    """Translate a list of English strings to ``lang``."""
    translator, sp = _load(lang)
    results = translator.translate_batch(
        [sp.encode(text, out_type=str) for text in texts],
        beam_size=4, length_penalty=0.2, max_batch_size=16)
    return [re.sub(r"\s+", " ", sp.decode(r.hypotheses[0]).replace("▁", " ")).strip()
            for r in results]


def translate_text(text, lang):
    """One line (the hook) -> target language; the same text back
    when ``lang`` is empty or "en". ALL-CAPS hooks are sentence-cased
    first — they translate badly in caps."""
    if not text or lang in ("", "en", None):
        return text
    return _translate_batch(
        [text.capitalize() if text.isupper() else text], lang)[0].translate(
            str.maketrans("", "", "()"))  # Arabic wraps names in ( )


def _mask(word):
    return word[0] + "*" * (len(word) - 1) if _BAD.match(word) else word


def sentences(words, cap=22):
    """Split word timings into sentences on the "end" flags — or after
    ``cap`` words when Whisper gave no punctuation, so one run-on
    never becomes one huge translation."""
    out, current = [], []
    for word in words:
        current.append(word)
        if word.get("end") or len(current) >= cap:
            out.append(current)
            current = []
    return out + ([current] if current else [])


def spread(tokens, start, end, sentence_end):
    """Place translated tokens over [start, end], time share
    proportional to word length. The last token ends the sentence."""
    lengths = [len(token) + 1 for token in tokens]
    total = sum(lengths)
    t0 = start
    out = []
    for i, (token, length) in enumerate(zip(tokens, lengths)):
        t1 = end if i == len(tokens) - 1 else t0 + (end - start) * length / total
        out.append({"w": token, "s": round(t0, 3), "e": round(t1, 3),
                    "end": bool(sentence_end and i == len(tokens) - 1)})
        t0 = t1
    return out


def cjk_chunks(text, n=3):
    """Chinese/Japanese text -> caption "words" of up to ``n``
    characters, cut at punctuation (dropped, ?/! kept on the chunk
    before). Never leaves a 1-character chunk at the end."""
    out = []
    parts = (y for x in re.split(r"[，。、；：,.;:\s“”「」『』（）()]+", text)
             for y in re.split(r"(?<=[？！?!])", x))
    for part in parts:
        part = part.strip()
        while part:
            body = len(part.rstrip("？！?!"))
            k = body if body <= n + 1 else n
            chunk, part = part[:k], part[k:]
            if not part.strip("？！?!"):
                chunk, part = chunk + part, ""
            if chunk.strip("？！?!"):
                out.append(chunk)
            elif out:
                out[-1] += chunk
    return out


def translate_words(words, lang):
    """words = [{"w", "s", "e", "end"}] in English -> the same shape
    in ``lang``, sentence ends kept, timings re-spread over the
    translated words. "en" (or empty) returns the input unchanged."""
    if lang in ("", "en", None):
        return words
    grouped = sentences(words)
    texts = [" ".join(w["w"] for w in group) for group in grouped]
    out = []
    for group, translated in zip(grouped, _translate_batch(texts, lang)):
        if lang in CJK:
            tokens = cjk_chunks(translated)
        else:
            tokens = []
            # French " ?" would otherwise become a caption word of its own.
            for word in re.sub(r"\s+([?!؟])", lambda m: m[1], translated).split():
                word = _PUNCT.sub("", word)
                if word:
                    tokens.append(_mask(word))
        if tokens:
            ends = (group[-1].get("end") or
                    re.search(r"[.?!؟]$", translated) is not None)
            out += spread(tokens, group[0]["s"], group[-1]["e"], ends)
    return out


def is_rtl(lang):
    """True for right-to-left languages (Arabic, Hebrew, ...)."""
    return lang in RTL


def available_languages():
    """Languages this module can translate English into."""
    return list(LANGS)
