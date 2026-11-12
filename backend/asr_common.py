from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

TARGET_SR = 16_000


# ----------------------------------------------------------------------------
# 1. Text normalisation
# ----------------------------------------------------------------------------
_FILLERS = re.compile(r"\b(hmm|mm|mhm|mmm|uh|um)\b")
_CUT_WORD = re.compile(r"\b\w+~")          # AppTek marks cut-off words as "PRE~"

EN_NORMALIZER_NAME = None
_EN_NORM = None


def _simple_en(text: str) -> str:
    """Fallback if the Whisper normaliser cannot be loaded."""
    text = str(text or "").lower()
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    text = _FILLERS.sub(" ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _get_en_normalizer():
    global _EN_NORM, EN_NORMALIZER_NAME
    if _EN_NORM is not None:
        return _EN_NORM
    try:
        from transformers import WhisperTokenizer
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        tok = WhisperTokenizer.from_pretrained("openai/whisper-small")
        _EN_NORM = EnglishTextNormalizer(tok.english_spelling_normalizer)
        EN_NORMALIZER_NAME = "whisper_english_text_normalizer"
    except Exception as exc:  # noqa: BLE001
        print(f"[asr_common] Whisper normaliser unavailable ({exc!r}); using simple fallback.")
        _EN_NORM = _simple_en
        EN_NORMALIZER_NAME = "simple_fallback"
    print(f"[asr_common] English normaliser: {EN_NORMALIZER_NAME}")
    return _EN_NORM


def normalize_en(text: str) -> str:
    """Lower-case, strip punctuation/fillers/bracketed tags, spelled numbers -> digits,
    British -> American spelling. Apply to BOTH reference and hypothesis."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _CUT_WORD.sub(" ", text)
    return " ".join(_get_en_normalizer()(text).split())


def normalize_hi(text: str) -> str:
    """Devanagari-aware normalisation (same logic as the Hindi training cell)."""
    text = unicodedata.normalize("NFC", str(text or ""))
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat == "Cf":                                  # invisible formatting chars
            continue
        if "LATIN" in unicodedata.name(ch, "") and ord(ch) > 127:
            ascii_letters = "".join(
                c for c in unicodedata.normalize("NFKD", ch) if c.isascii() and c.isalpha()
            )
            out.append(ascii_letters.lower() if ascii_letters else " ")
            continue
        if ch.isascii() and ch.isalpha():
            out.append(ch.lower())
            continue
        if cat.startswith("P") or cat.startswith("S"):   # punctuation/symbols -> separator
            out.append(" ")
            continue
        out.append(ch)
    return " ".join("".join(out).split())


NORMALIZERS = {"en": normalize_en, "hi": normalize_hi}


# ----------------------------------------------------------------------------
# 2. Audio
# ----------------------------------------------------------------------------
def to_mono_16k(audio, sr: int) -> np.ndarray:
    """Any (samples,) / (samples, ch) / (ch, samples) array -> mono float32 @ 16 kHz."""
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if audio.shape[0] > audio.shape[1] else audio.mean(axis=0)
    elif audio.ndim > 2:
        raise ValueError(f"Unsupported audio shape {audio.shape}")
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_audio_file(path: str) -> np.ndarray:
    """Read a wav/flac/ogg/... file -> mono 16 kHz float32."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    return to_mono_16k(audio, sr)


# ----------------------------------------------------------------------------
# 3. Long-audio-safe CTC inference
# ----------------------------------------------------------------------------
def load_ctc_pipeline(model_dir: str, device=None, chunk_s: float = 20.0,
                      stride_s=(4, 2), batch_size: int = 8,
                      lm_path: str | None = None, alpha: float = 0.5, beta: float = 1.5):
    """Wav2Vec2-CTC ASR pipeline that splits long audio into overlapping chunks.

    Wav2Vec2 attention memory grows quadratically with length, so feeding a
    7-minute call in one forward pass (what the old AppTek cell did) OOMs a T4
    and degrades quality. Chunking with stride fixes both.
    """
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).eval()
    # "group"-norm models (wav2vec2-base family) have NO attention mask, so zero-padding
    # inside a batch corrupts shorter inputs (huge deletion rates). Batch only
    # "layer"-norm models (large / lv60 / XLSR), which support attention masks.
    if getattr(model.config, "feat_extract_norm", "layer") == "group":
        batch_size = 1
    extra = {}
    if lm_path:   # n-gram (KenLM) beam-search decoding via pyctcdecode
        from pyctcdecode import build_ctcdecoder

        vocab = processor.tokenizer.get_vocab()
        labels = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        extra["decoder"] = build_ctcdecoder(labels=labels, kenlm_model_path=lm_path,
                                            alpha=alpha, beta=beta)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=chunk_s,
        stride_length_s=stride_s,
        batch_size=batch_size,
        **extra,
    )
    pipe.asr_batch_size = batch_size       # remembered for transcribe_many
    return pipe


def transcribe_ctc(pipe, audio_16k: np.ndarray) -> str:
    return pipe({"raw": audio_16k, "sampling_rate": TARGET_SR})["text"]


def transcribe_many(pipe, arrays, batch_size=None):
    batch_size = batch_size or getattr(pipe, "asr_batch_size", 1)
    inputs = [{"raw": a, "sampling_rate": TARGET_SR} for a in arrays]
    return [o["text"] for o in pipe(inputs, batch_size=batch_size)]


# ----------------------------------------------------------------------------
# 4. Scoring
# ----------------------------------------------------------------------------
def score(preds, refs, normalizer=normalize_en) -> dict:
    """Corpus-level WER/CER (percent) plus S/D/I breakdown (percent of reference words).

    High deletions  -> model dropped speech (truncation / chunking problem).
    High insertions -> hallucination / repetition.
    High substitutions -> genuine recognition errors.
    """
    import jiwer

    P, R, kept = [], [], []
    for i, (p, r) in enumerate(zip(preds, refs)):
        r_n = normalizer(r)
        if not r_n:
            continue
        P.append(normalizer(p))
        R.append(r_n)
        kept.append(i)
    if not R:
        return {"n": 0}
    w = jiwer.process_words(R, P)
    n_ref_words = w.hits + w.substitutions + w.deletions
    return {
        "n": len(R),
        "wer": round(w.wer * 100, 2),
        "cer": round(jiwer.cer(R, P) * 100, 2),
        "sub_pct": round(100 * w.substitutions / n_ref_words, 2),
        "del_pct": round(100 * w.deletions / n_ref_words, 2),
        "ins_pct": round(100 * w.insertions / n_ref_words, 2),
        "per_sample_wer": [round(jiwer.wer(r, p) * 100, 1) for r, p in zip(R, P)],
        "kept_indices": kept,
    }


def print_score(title: str, s: dict) -> None:
    if not s.get("n"):
        print(f"{title}: no valid samples")
        return
    print(f"{title}\n  samples={s['n']}  WER={s['wer']}%  CER={s['cer']}%  "
          f"(sub {s['sub_pct']}% / del {s['del_pct']}% / ins {s['ins_pct']}%)")


# ----------------------------------------------------------------------------
# 5. Persist / re-score results
# ----------------------------------------------------------------------------
def save_results(path, *, model: str, dataset: str, ids, preds, refs,
                 audio_seconds: float | None = None, elapsed_seconds: float | None = None,
                 lang: str = "en", extra: dict | None = None) -> None:
    """Store RAW predictions and references (un-normalised) so any run can be
    re-scored later with an improved normaliser without re-running the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "dataset": dataset,
        "lang": lang,
        "n": len(ids),
        "ids": list(ids),
        "preds_raw": list(preds),
        "refs_raw": list(refs),
        "audio_seconds": audio_seconds,
        "elapsed_seconds": elapsed_seconds,
        "rtf": (elapsed_seconds / audio_seconds) if audio_seconds and elapsed_seconds else None,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "extra": extra or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[asr_common] saved {len(ids)} results -> {path}")


def rescore(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    s = score(d["preds_raw"], d["refs_raw"], NORMALIZERS[d.get("lang", "en")])
    s.update(model=d["model"], dataset=d["dataset"], rtf=d.get("rtf"))
    return s


def compare(paths) -> None:
    """Print a one-line-per-run table from saved JSON results."""
    print(f"{'model':45s} {'dataset':32s} {'n':>5s} {'WER%':>7s} {'CER%':>7s} {'del%':>6s} {'ins%':>6s}")
    for p in paths:
        s = rescore(p)
        print(f"{s['model'][:45]:45s} {s['dataset'][:32]:32s} {s['n']:5d} "
              f"{s['wer']:7.2f} {s['cer']:7.2f} {s['del_pct']:6.2f} {s['ins_pct']:6.2f}")
