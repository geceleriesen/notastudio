import math
import inspect
import os
import re
import threading
import time
from pathlib import Path
import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_CACHE_DIR = BASE_DIR / "models_cache" / "ace_step_15"
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RENDER_PROGRESS = {
    "active": False,
    "stage": "Hazır",
    "percent": 0,
    "detail": ""
}

_PIPELINE_LOCK = threading.Lock()
_ACE_PIPELINE = None
_ACE_DEVICE = None
_ACE_MODEL_NAME = None

MODEL_CANDIDATES = [
    "ACE-Step/acestep-v15-turbo",
    "ACE-Step/acestep-v15-xl-turbo-diffusers",
]

INSTRUMENT_PROMPT_TAGS = {
    "violin": "lush symphonic violin section",
    "viola": "warm orchestral viola section",
    "cello": "deep expressive cello section",
    "contrabass": "rich orchestral double bass",
    "harp": "celestial concert harp arpeggios",
    "flute": "airy silver concert flute",
    "oboe": "expressive lyrical oboe",
    "clarinet": "warm woodwind clarinet",
    "trumpet": "brilliant brass trumpet fanfare",
    "horn": "noble French horn ensemble",
    "saxophone": "soulful tenor saxophone",
    "piano": "resonant grand piano",
    "electric_guitar": "emotive electric guitar",
    "acoustic_guitar": "warm steel-string acoustic guitar",
    "bass_guitar": "deep groovy electric bass guitar",
    "oud_baglama": "authentic Turkish oud and anatolian baglama saz",
    "drums": "dynamic studio acoustic drum kit",
    "timpani": "thunderous orchestral timpani drums",
}

ARTIST_PROMPT_TAGS = {
    "diva_pop": "solo female singer singing lead vocals, powerful emotive pop diva voice, clear lyrics pronunciation, front-and-center vocal mix",
    "soprano_opera": "solo female operatic soprano singer singing lead vocals, classical concert voice, soaring vibrato, clear pronunciation",
    "turk_sanat_k": "solo female Turkish singer singing lead vocals, soulful Turkish folk and classical singing voice, clear Turkish pronunciation",
    "soul_mezzo": "solo female mezzo-soprano singer singing lead vocals, warm velvety soulful singing voice, clear articulation",
    "rock_tenor": "solo male tenor singer singing lead vocals, passionate rock and symphonic singing voice, clear lyrics pronunciation",
    "turk_sanat_e": "solo male Turkish singer singing lead vocals, expressive Anatolian and Turkish singing voice, clear Turkish pronunciation",
    "velvet_bariton": "solo male baritone singer singing lead vocals, warm velvet crooner voice, clear pronunciation",
    "deep_bass": "solo male bass-baritone singer singing lead vocals, deep resonant singing voice, clear pronunciation",
}

LEGACY_VOCAL_MAP = {
    "female_soprano": "soprano_opera",
    "male_tenor": "rock_tenor",
}

KEY_TRANSLATIONS = {
    "do majör": "C major",
    "do minör": "C minor",
    "re majör": "D major",
    "re minör": "D minor",
    "mi majör": "E major",
    "mi minör": "E minor",
    "fa majör": "F major",
    "fa minör": "F minor",
    "sol majör": "G major",
    "sol minör": "G minor",
    "la majör": "A major",
    "la minör": "A minor",
    "si majör": "B major",
    "si minör": "B minor",
    "hicaz": "D Phrygian dominant",
    "nihavend": "G minor",
    "hüzzam": "F# Locrian",
}

TR_CHARS = set("ğüşöçıİĞÜŞÖÇ")


def check_studio_ai_status() -> dict:
    has_torch = False
    has_cuda = False
    gpu_name = None
    has_diffusers_acestep = False

    try:
        import torch
        has_torch = True
        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        from diffusers import AceStepPipeline  # noqa: F401
        has_diffusers_acestep = True
    except Exception:
        try:
            from diffusers import ACEStepPipeline  # noqa: F401
            has_diffusers_acestep = True
        except Exception:
            pass

    weights_cached = any(MODELS_CACHE_DIR.rglob("*.safetensors")) or any(MODELS_CACHE_DIR.rglob("*.bin"))

    return {
        "acestep_installed": has_diffusers_acestep,
        "torch_installed": has_torch,
        "cuda_available": has_cuda,
        "gpu_name": gpu_name,
        "weights_cached": weights_cached,
        "model_loaded_in_ram": _ACE_PIPELINE is not None,
        "model_name": _ACE_MODEL_NAME or MODEL_CANDIDATES[0],
    }


def _detect_lang_code(text: str) -> tuple[str, str]:
    """(vocal_language_code, english_label) dondurur."""
    if any(c in TR_CHARS for c in text):
        return "tr", "Turkish"
    tr_words = {
        "ve", "bir", "bu", "ile", "gibi", "yine", "kadar", "için", "icin",
        "ben", "sen", "biz", "bugün", "rüzgar", "aşk", "gece", "gel", "git", "kal"
    }
    tokens = {w.lower().strip(".,!?;:'\"-") for w in text.split()}
    if tokens & tr_words:
        return "tr", "Turkish"
    return "en", "English"


def _normalize_keyscale(raw_key: str) -> str | None:
    if not raw_key:
        return None
    first_part = raw_key.split("/")[0].strip().lower()
    for tr_k, en_k in KEY_TRANSLATIONS.items():
        if tr_k in first_part:
            return en_k
    if re.match(r"^[A-Ga-g][#b]?\s+(major|minor)$", first_part):
        return first_part[0].upper() + first_part[1:]
    return None


def _normalize_timesig(raw_ts: str) -> str | None:
    if not raw_ts:
        return None
    m = re.match(r"^(\d+)\s*/\s*\d+$", raw_ts.strip())
    if m:
        return m.group(1)
    return None


def _clean_hyphenated_lyrics(text: str) -> str:
    """'Üs- kü- dar- a gi- der' gibi hecelenmis metni 'Üsküdara gider' seklinde tam kelimelere birlestirir."""
    # Tire + bosluk veya tire + satir sonu birlestir
    merged = re.sub(r"-\s+", "", text)
    # Kelime icinde kalmis tekil hece tirelerini de temizle (etiketler [verse] haric)
    lines_out = []
    for ln in merged.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            lines_out.append(s)
        else:
            words = [w.replace("-", "") for w in s.split() if w.replace("-", "")]
            lines_out.append(" ".join(words))
    return "\n".join(lines_out)


def _split_long_lines_for_diction(lines: list[str], max_words_per_line: int = 5) -> list[str]:
    """Uzun cumleleri 4-5 kelimelik dogal nefes satirlarina boler; AI'nin kelimeleri yutmasini onler."""
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            out.append(s)
            continue
        words = s.split()
        if len(words) <= max_words_per_line:
            out.append(" ".join(words))
        else:
            for i in range(0, len(words), max_words_per_line):
                out.append(" ".join(words[i:i + max_words_per_line]))
    return out


def format_lyrics_for_studio_ai(raw_lyrics: str, melody_notes: list) -> str:
    text = (raw_lyrics or "").strip()
    if not text:
        words = []
        buf = ""
        for n in melody_notes or []:
            lyr = (n.get("lyric") or "").strip()
            if not lyr or lyr == "-":
                continue
            if lyr.endswith("-"):
                buf += lyr[:-1]
            else:
                words.append(buf + lyr)
                buf = ""
        if buf:
            words.append(buf)
        text = " ".join(words).strip()

    if not text:
        return "[instrumental]"

    cleaned = _clean_hyphenated_lyrics(text)
    if not cleaned:
        return "[instrumental]"

    phrased_lines = _split_long_lines_for_diction(cleaned.splitlines(), max_words_per_line=5)
    phrased = "\n".join(phrased_lines)

    if re.search(r"\[(verse|chorus|bridge|intro|outro|pre-chorus|hook|instrumental)\]", phrased, re.IGNORECASE):
        return phrased

    return "[verse]\n" + phrased


def build_studio_prompt(score: dict, formatted_lyrics: str) -> tuple[str, float, str]:
    title = (score.get("title") or "").strip()
    composer = (score.get("composer") or "").strip()
    bpm = int(score.get("bpm", 116))
    time_sig = (score.get("time_signature") or "4/4").strip()
    key_sig = (score.get("key_signature") or "").strip()
    instruments = score.get("instruments") or {}
    vocals = score.get("vocals") or {}

    inst_desc = []
    for k, cfg in instruments.items():
        cnt = int(cfg.get("count", 0))
        if cnt > 0:
            base_tag = INSTRUMENT_PROMPT_TAGS.get(k, k.replace("_", " "))
            if cnt >= 3:
                inst_desc.append(f"prominent {base_tag}")
            else:
                inst_desc.append(base_tag)
    if not inst_desc:
        inst_desc = ["grand piano", "lush orchestral strings"]

    active_v = []
    for k, cnt in vocals.items():
        nk = LEGACY_VOCAL_MAP.get(k, k)
        if int(cnt or 0) > 0:
            active_v.append((nk, int(cnt)))

    is_instrumental = (formatted_lyrics.strip() == "[instrumental]")
    if not active_v and not is_instrumental:
        active_v = [("diva_pop", 1)]

    lang_code, lang_name = _detect_lang_code(formatted_lyrics)
    vocal_desc = []
    if is_instrumental:
        vocal_desc.append("purely instrumental arrangement, no vocals")
    else:
        vocal_desc.append(
            f"crystal clear {lang_name} diction, every single word and syllable articulated distinctly at an unhurried singing pace, "
            f"no rushed or swallowed words, immediate {lang_name} lead vocals starting on beat 1, vocals prominent in front of the mix"
        )
        if len(active_v) >= 2:
            vocal_desc.append("harmonized vocal duet")
        for vk, cnt in active_v:
            tag = ARTIST_PROMPT_TAGS.get(vk, "solo singer singing lead vocals clearly")
            if cnt >= 2:
                tag += ", layered backing harmonies"
            vocal_desc.append(tag)

    style_hints = ["studio master quality, high fidelity, wide stereo soundstage, clean vocal intelligibility"]
    title_low = title.lower()
    if "türk marşı" in title_low or "turkish march" in title_low or "mozart" in composer.lower():
        style_hints.append("classical symphonic crossover inspired by Mozart Rondo Alla Turca, energetic classical motifs")
    elif "üsküdar" in title_low or "makam" in key_sig.lower() or "hicaz" in key_sig.lower():
        style_hints.append("Turkish classical and Anatolian symphonic fusion, authentic makam phrasing")
    elif "vivaldi" in title_low or "bahar" in title_low:
        style_hints.append("baroque orchestral concerto style, bright expressive strings")

    if key_sig:
        style_hints.append(f"key of {key_sig}")
    style_hints.append(f"{bpm} BPM, {time_sig} meter")

    full_prompt = ", ".join(vocal_desc + inst_desc + style_hints)

    melody = score.get("staves", {}).get("melody", [])
    total_beats = sum(float(n.get("beats", 1.0)) for n in melody)
    score_sec = total_beats * (60.0 / max(45, bpm))
    clean_lyric_body = re.sub(r"\[.*?\]", "", formatted_lyrics).strip()
    words_list = clean_lyric_body.split()
    word_count = len(words_list)
    line_count = len([ln for ln in clean_lyric_body.splitlines() if ln.strip()])
    # Turkce ve cok dilli hece sayisini hesaplayarak sarkinin tam soz suresine oturmasini sagla
    vowel_groups = sum(max(1, len(re.findall(r"[aeıioöuüAEIİOÖUÜyY]+", w))) for w in words_list)
    syllable_count = max(word_count, vowel_groups)
    lyrics_sec = max(10.0, syllable_count * 0.46 + line_count * 1.15 + 3.2)
    duration_sec = float(np.clip(max(score_sec, lyrics_sec), 10.0, 120.0))

    return full_prompt, duration_sec, lang_code


def _get_or_load_acestep():
    global _ACE_PIPELINE, _ACE_DEVICE, _ACE_MODEL_NAME
    with _PIPELINE_LOCK:
        if _ACE_PIPELINE is not None:
            return _ACE_PIPELINE, _ACE_DEVICE, _ACE_MODEL_NAME

        import torch
        try:
            from diffusers import AceStepPipeline as PipelineCls
        except ImportError:
            from diffusers import ACEStepPipeline as PipelineCls

        has_cuda = bool(torch.cuda.is_available())
        dtype = torch.bfloat16 if has_cuda else torch.float32
        device_label = f"CUDA GPU ({torch.cuda.get_device_name(0)})" if has_cuda else "CPU Stüdyo Modu"

        last_err = None
        for repo_id in MODEL_CANDIDATES:
            try:
                RENDER_PROGRESS.update({
                    "active": True,
                    "stage": f"ACE-Step 1.5 Yükleniyor ({repo_id.split('/')[-1]})...",
                    "percent": 15,
                    "detail": f"Donanım: {device_label}. İlk çalıştırmada model ağırlıkları indirilir (~4.8 GB)."
                })
                pipe = PipelineCls.from_pretrained(
                    repo_id,
                    torch_dtype=dtype,
                    cache_dir=str(MODELS_CACHE_DIR),
                )
                if has_cuda:
                    pipe = pipe.to("cuda")
                else:
                    try:
                        pipe.enable_model_cpu_offload()
                    except Exception:
                        pipe = pipe.to("cpu")

                _ACE_PIPELINE = pipe
                _ACE_DEVICE = device_label
                _ACE_MODEL_NAME = repo_id.split("/")[-1]
                return _ACE_PIPELINE, _ACE_DEVICE, _ACE_MODEL_NAME
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"ACE-Step 1.5 modeli yüklenemedi: {last_err}")


def _split_lyrics_into_sections(formatted_lyrics: str, n_parts: int) -> list[str]:
    """Uzun sarki sozlerini bolum sayisina gore anlamli kitalara boler."""
    if n_parts <= 1 or formatted_lyrics.strip() == "[instrumental]":
        return [formatted_lyrics] * max(1, n_parts)

    lines = [ln.strip() for ln in formatted_lyrics.splitlines() if ln.strip() and not (ln.strip().startswith("[") and ln.strip().endswith("]"))]
    if not lines:
        return ["[instrumental]"] * n_parts

    chunk_size = max(1, math.ceil(len(lines) / n_parts))
    sections = []
    tags = ["[verse]", "[chorus]", "[verse]", "[bridge]", "[chorus]", "[outro]"]
    for idx in range(n_parts):
        sub = lines[idx * chunk_size:(idx + 1) * chunk_size]
        if not sub:
            sub = lines[-chunk_size:]
        tag = tags[idx % len(tags)]
        sections.append(f"{tag}\n" + "\n".join(sub))
    return sections


def _crossfade_concat_wavs(chunks: list[np.ndarray], sr: int, fade_sec: float = 0.8) -> np.ndarray:
    """Birden fazla AI ses bolumunu puruzsuz gecisle (crossfade) tek uzun sarkiya birlestirir."""
    if not chunks:
        return np.zeros((sr, 2), dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    fade_samples = int(fade_sec * sr)
    out = chunks[0]
    for nxt in chunks[1:]:
        ov = min(fade_samples, len(out) // 3, len(nxt) // 3)
        if ov > 16:
            fade_out = np.linspace(1.0, 0.0, ov, dtype=np.float32)[:, None]
            fade_in = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None]
            blended = out[-ov:] * fade_out + nxt[:ov] * fade_in
            out = np.vstack([out[:-ov], blended, nxt[ov:]])
        else:
            out = np.vstack([out, nxt])
    return out


def generate_studio_ai_song_sync(score: dict, out_wav: Path, quality_steps: int = 10) -> tuple[bool, str]:
    """Hugging Face Diffusers ACE-Step 1.5 ile tek parca veya cok bolumlu UZUN sarki (6 dakikaya kadar) uretir."""
    status = check_studio_ai_status()
    if not status["acestep_installed"]:
        return False, "Diffusers ACE-Step modülü hazır değil (install_studio_ai.bat çalıştırın)"

    try:
        import torch
        melody = score.get("staves", {}).get("melody", [])
        raw_lyrics = (score.get("lyrics_text") or "").strip()
        formatted_lyrics = format_lyrics_for_studio_ai(raw_lyrics, melody)
        prompt, auto_dur, lang_code = build_studio_prompt(score, formatted_lyrics)

        # Kullanici ozel hedef sure sectiyse (orn. 180 sn = 3 dk) dikkate al
        requested_dur = float(score.get("target_duration_sec") or 0)
        total_target_sec = requested_dur if requested_dur >= 10.0 else auto_dur
        total_target_sec = float(np.clip(total_target_sec, 10.0, 360.0))

        # Tek parca max ~110 sn; daha uzunsa otomatik bolumlere ayirip birlestir
        n_parts = max(1, int(math.ceil(total_target_sec / 105.0)))
        part_dur = float(np.clip(total_target_sec / n_parts, 10.0, 115.0))
        lyric_sections = _split_lyrics_into_sections(formatted_lyrics, n_parts)

        pipe, device_label, model_name = _get_or_load_acestep()

        steps = int(np.clip(quality_steps, 8, 16))
        bpm = int(score.get("bpm", 116))
        keyscale = _normalize_keyscale(score.get("key_signature", ""))
        timesig = _normalize_timesig(score.get("time_signature", ""))

        base_seed = int(time.time() * 1000) % 2147483647
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        generated_chunks = []
        sr_out = int(getattr(pipe, "sample_rate", 48000))

        for p_idx in range(n_parts):
            sec_lyrics = lyric_sections[p_idx]
            part_role_hint = ""
            if n_parts > 1:
                if p_idx == 0:
                    part_role_hint = ", opening section, establish main theme"
                elif p_idx == n_parts - 1:
                    part_role_hint = ", grand finale section, conclusive musical ending"
                else:
                    part_role_hint = f", continuation section {p_idx + 1} of {n_parts}, seamless musical progression"

            pct = int(15 + (p_idx / n_parts) * 75)
            RENDER_PROGRESS.update({
                "active": True,
                "stage": (
                    f"Stüdyo AI Beste Üretiyor (Bölüm {p_idx + 1}/{n_parts})..."
                    if n_parts > 1 else "Stüdyo AI Beste & Gerçekçi Vokal Üretiyor..."
                ),
                "percent": pct,
                "detail": f"Dil: {lang_code.upper()} | Hedef: ~{int(total_target_sec)} sn | Adım: {steps} | {device_label}"
            })

            gen = torch.Generator(device="cpu").manual_seed(base_seed + p_idx)
            raw_kwargs = dict(
                prompt=prompt + part_role_hint,
                lyrics=sec_lyrics,
                vocal_language=lang_code,
                audio_duration=part_dur,
                num_inference_steps=steps,
                shift=3.0,
                vocal_guidance_scale=12.0,
                generator=gen,
            )
            if bpm > 0:
                raw_kwargs["bpm"] = bpm
            if keyscale:
                raw_kwargs["keyscale"] = keyscale
            if timesig:
                raw_kwargs["timesignature"] = timesig

            try:
                sig_params = inspect.signature(pipe.__call__).parameters
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params.values())
                call_kwargs = raw_kwargs if accepts_var_kw else {k: v for k, v in raw_kwargs.items() if k in sig_params}
            except Exception:
                call_kwargs = {
                    "prompt": prompt + part_role_hint,
                    "lyrics": sec_lyrics,
                    "audio_duration": part_dur,
                    "num_inference_steps": steps,
                    "generator": gen,
                }

            with torch.no_grad():
                result = pipe(**call_kwargs)

            audio_tensor = result.audios[0]
            if hasattr(audio_tensor, "detach"):
                wav_np = audio_tensor.detach().cpu().float().numpy()
            else:
                wav_np = np.asarray(audio_tensor, dtype=np.float32)

            if wav_np.ndim == 2 and wav_np.shape[0] <= 2 and wav_np.shape[1] > wav_np.shape[0]:
                wav_np = wav_np.T
            elif wav_np.ndim == 1:
                wav_np = np.column_stack([wav_np, wav_np])

            wav_np = np.nan_to_num(wav_np, nan=0.0).astype(np.float32)
            generated_chunks.append(wav_np)

        RENDER_PROGRESS.update({
            "active": True,
            "stage": "Stüdyo Mastering & Bölüm Birleştirme...",
            "percent": 92,
            "detail": "48 kHz stereo master dosyası kaydediliyor..."
        })

        final_wav = _crossfade_concat_wavs(generated_chunks, sr_out, fade_sec=0.85)
        pk = float(np.max(np.abs(final_wav))) if final_wav.size else 0.0
        if pk > 1e-4:
            final_wav = (final_wav / pk) * 0.94

        sf.write(str(out_wav), final_wav, sr_out, subtype="PCM_16")

        if out_wav.exists() and out_wav.stat().st_size > 1000:
            part_info = f" • {n_parts} Bölüm" if n_parts > 1 else ""
            return True, f"Stüdyo AI ({model_name}{part_info} • {device_label})"

        return False, "Stüdyo AI ses dosyası oluşturamadı"
    except Exception as exc:
        return False, f"Stüdyo AI hatası: {exc}"
    finally:
        RENDER_PROGRESS.update({
            "active": False,
            "stage": "Hazır",
            "percent": 100,
            "detail": ""
        })
