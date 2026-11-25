import asyncio
import hashlib
import os
import re
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample, butter, lfilter

try:
    import pyworld as pw
    HAS_PYWORLD = True
except Exception:
    HAS_PYWORLD = False

CACHE_DIR = Path(__file__).resolve().parent.parent / "exports" / "vocal_cache_v9"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAST_TTS_SOURCE = {"name": ""}

LEGACY_VOCAL_KEYS = {
    "female_soprano": "soprano_opera",
    "male_tenor": "rock_tenor",
}

# (tr_voice, en_voice, gender, formant_ratio, vib_depth, vib_rate, breathiness, pitch_offset_semi, label)
ARTIST_PROFILES = {
    "diva_pop":       ("tr-TR-EmelNeural",  "en-US-JennyNeural",       "female", 1.12, 0.013, 5.4, 0.03, 0,   "Kadın Pop Diva"),
    "soprano_opera":  ("tr-TR-EmelNeural",  "en-GB-SoniaNeural",       "female", 1.20, 0.019, 5.9, 0.02, 12,  "Kadın Dramatik Soprano"),
    "turk_sanat_k":   ("tr-TR-EmelNeural",  "en-US-AriaNeural",        "female", 1.08, 0.016, 5.5, 0.03, 0,   "Kadın Türk Sanat / Makam"),
    "soul_mezzo":     ("tr-TR-EmelNeural",  "en-US-MichelleNeural",    "female", 1.00, 0.011, 4.9, 0.05, 0,   "Kadın Kadife Mezzo"),
    "rock_tenor":     ("tr-TR-AhmetNeural", "en-US-GuyNeural",         "male",   0.96, 0.014, 5.5, 0.04, 0,   "Erkek Rock & Senfonik Tenor"),
    "turk_sanat_e":   ("tr-TR-AhmetNeural", "en-GB-RyanNeural",        "male",   0.92, 0.017, 5.7, 0.03, 0,   "Erkek Türk Sanat / Anadolu"),
    "velvet_bariton": ("tr-TR-AhmetNeural", "en-US-ChristopherNeural", "male",   0.85, 0.010, 4.8, 0.04, -12, "Erkek Kadife Bariton"),
    "deep_bass":      ("tr-TR-AhmetNeural", "en-US-EricNeural",        "male",   0.78, 0.008, 4.4, 0.03, -12, "Erkek Derin Bas-Bariton"),
}

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    "Cb": -1, "B#": 12, "E#": 5, "Fb": 4,
}
_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

# Turkce ve Latin alfabeli dillerde gercek sesli harfler ('y' unsuz olarak kalir!)
PURE_VOWELS = set("aeıioöuüAEIİOÖUÜ")
TR_ONLY_CHARS = set("ğüşöçıİĞÜŞÖÇ")


def note_to_hz(note_str: str) -> float:
    if not note_str or note_str == "REST":
        return 0.0
    m = _PITCH_RE.match(note_str.strip())
    if not m:
        return 0.0
    name = m.group(1).upper() + m.group(2)
    octave = int(m.group(3))
    midi = (octave + 1) * 12 + NOTE_TO_SEMITONE.get(name, 0)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def detect_lyrics_language(text: str) -> str:
    if any(ch in TR_ONLY_CHARS for ch in text):
        return "tr"
    tr_common = {"ve", "bir", "bu", "ile", "gibi", "yine", "kadar", "için", "icin", "ben", "sen", "biz", "ey", "neşe", "nese", "bugün", "rüzgar"}
    words = {w.lower().strip(".,!?;:'\"-") for w in text.split()}
    if words & tr_common:
        return "tr"
    return "en"


def syllabify_word_universal(word: str) -> list[str]:
    """Turkce ve Ingilizce kelimeleri dogru hecelere boler ('yine' -> 'yi-', 'ne')."""
    w = word.strip()
    if not w:
        return []
    if "-" in w:
        parts = [p for p in w.split("-") if p]
        return [p + "-" for p in parts[:-1]] + [parts[-1]] if parts else []

    vowel_positions = [i for i, ch in enumerate(w) if ch in PURE_VOWELS]
    if len(vowel_positions) <= 1:
        return [w]

    # Ingilizce kelime sonu sessiz 'e' (love, shine, time) ayri hece yapilmaz
    if len(w) > 3 and w[-1].lower() == "e" and len(vowel_positions) >= 2:
        prev_v = vowel_positions[-2]
        if vowel_positions[-1] == len(w) - 1 and (len(w) - 1 - prev_v) >= 2:
            vowel_positions = vowel_positions[:-1]
            if len(vowel_positions) <= 1:
                return [w]

    syllables = []
    start = 0
    for idx in range(len(vowel_positions) - 1):
        v1 = vowel_positions[idx]
        v2 = vowel_positions[idx + 1]
        between = v2 - v1 - 1
        if between == 0:
            split_at = v2
        elif between == 1:
            split_at = v1 + 1
        else:
            split_at = v2 - 1
        syllables.append(w[start:split_at] + "-")
        start = split_at
    syllables.append(w[start:])
    return [s for s in syllables if s and s != "-"]


def split_lyrics_to_syllable_tokens(raw_text: str) -> list[str]:
    tokens = []
    for raw_w in (raw_text or "").strip().split():
        clean = re.sub(r"^[^\wçğıöşüÇĞİÖŞÜ]+|[^\wçğıöşüÇĞİÖŞÜ-]+$", "", raw_w)
        if not clean:
            continue
        tokens.extend(syllabify_word_universal(clean))
    return tokens


def _decode_mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        res = subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(mp3_path), "-ar", "44100", "-ac", "1", str(wav_path)],
            capture_output=True,
            timeout=15
        )
        return res.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 200
    except Exception:
        return False


def _turkish_to_phonetic_for_en_voice(text: str) -> str:
    rep = {
        "ş": "sh", "Ş": "Sh", "ç": "ch", "Ç": "Ch",
        "ğ": "",   "Ğ": "",   "ı": "uh", "I": "Uh",
        "ö": "oe", "Ö": "Oe", "ü": "ue", "Ü": "Ue",
        "c": "j",  "C": "J"
    }
    return "".join(rep.get(ch, ch) for ch in text)


def _windows_sapi_wav(text: str, gender: str, lang: str, out_wav: Path) -> tuple[bool, str]:
    try:
        safe_text = text.replace("'", " ").replace('"', " ").strip()
        if not safe_text:
            return False, ""
        target_gender = "Female" if gender == "female" else "Male"
        spoken_en = _turkish_to_phonetic_for_en_voice(safe_text) if lang == "tr" else safe_text

        ps_script = f"""
Add-Type -AssemblyName System.Speech;
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;
$voices = $s.GetInstalledVoices() | Where-Object {{ $_.Enabled }};
$chosen = $voices | Where-Object {{ $_.VoiceInfo.Culture.Name -like '{lang}*' -and $_.VoiceInfo.Gender -eq '{target_gender}' }} | Select-Object -First 1;
$nativeMatch = $true;
if (-not $chosen) {{
    $chosen = $voices | Where-Object {{ $_.VoiceInfo.Gender -eq '{target_gender}' }} | Select-Object -First 1;
    $nativeMatch = $false;
}}
if (-not $chosen) {{
    $chosen = $voices | Select-Object -First 1;
}}
if ($chosen) {{
    $s.SelectVoice($chosen.VoiceInfo.Name);
    Write-Output ($chosen.VoiceInfo.Name + ' [' + $chosen.VoiceInfo.Gender + ']');
}}
$s.Rate = -1;
$s.Volume = 100;
$s.SetOutputToWaveFile('{str(out_wav)}');
if ($nativeMatch) {{
    $s.Speak('{safe_text}');
}} else {{
    $s.Speak('{spoken_en}');
}}
$s.Dispose();
"""
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=15
        )
        lines = (res.stdout or "").strip().splitlines()
        label = lines[-1] if lines else f"SAPI-{target_gender}"
        ok = res.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 200
        return ok, label
    except Exception:
        return False, ""


def _load_wav_mono(path: Path, sr: int) -> np.ndarray:
    y, file_sr = sf.read(str(path), dtype="float64")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if file_sr != sr:
        y = resample(y, int(len(y) * sr / file_sr))
    return np.nan_to_num(y, nan=0.0)


async def _synthesize_word_wav(word_text: str, artist_key: str, sr: int = 44100) -> np.ndarray | None:
    """Her kelimeyi ayri ve net artikulasyonla uretir ki kelimeler birbirine girmeden okunsun."""
    clean_w = word_text.replace("-", "").strip()
    if not clean_w:
        return None

    lang = detect_lyrics_language(clean_w)
    tr_v, en_v, gender, *_ = ARTIST_PROFILES.get(
        artist_key, ("tr-TR-EmelNeural", "en-US-JennyNeural", "female", 1.12, 0.015, 5.5, 0.05, 0, "")
    )
    voice_name = tr_v if lang == "tr" else en_v
    h = hashlib.md5(f"w:{voice_name}:{gender}:{lang}:{clean_w}".encode("utf-8")).hexdigest()[:16]
    edge_wav = CACHE_DIR / f"word_{h}.wav"
    sapi_wav = CACHE_DIR / f"word_{h}_sapi_{gender}_{lang}.wav"

    if edge_wav.exists():
        try:
            y = _load_wav_mono(edge_wav, sr)
            LAST_TTS_SOURCE["name"] = f"Edge-TTS ({voice_name})"
            return y
        except Exception:
            pass

    tmp_mp3 = CACHE_DIR / f"word_{h}.mp3"
    try:
        import edge_tts
        comm = edge_tts.Communicate(clean_w, voice_name, rate="-8%")
        await comm.save(str(tmp_mp3))
        if tmp_mp3.exists() and tmp_mp3.stat().st_size > 250:
            if _decode_mp3_to_wav(tmp_mp3, edge_wav):
                y = _load_wav_mono(edge_wav, sr)
                LAST_TTS_SOURCE["name"] = f"Edge-TTS ({voice_name})"
                return y
    except Exception:
        pass
    finally:
        tmp_mp3.unlink(missing_ok=True)

    if sapi_wav.exists():
        try:
            y = _load_wav_mono(sapi_wav, sr)
            LAST_TTS_SOURCE["name"] = f"SAPI ({'Kadın' if gender == 'female' else 'Erkek'})"
            return y
        except Exception:
            pass

    ok, label = _windows_sapi_wav(clean_w, gender, lang, sapi_wav)
    if ok:
        try:
            y = _load_wav_mono(sapi_wav, sr)
            LAST_TTS_SOURCE["name"] = f"SAPI ({label})"
            return y
        except Exception:
            pass

    return None


def _trim_silence(y: np.ndarray, sr: int) -> np.ndarray:
    if len(y) == 0:
        return y
    pk = np.max(np.abs(y))
    if pk < 1e-5:
        return np.array([], dtype=np.float64)
    nz = np.where(np.abs(y) > pk * 0.025)[0]
    if len(nz) == 0:
        return np.array([], dtype=np.float64)
    s = max(0, nz[0] - int(0.015 * sr))
    e = min(len(y), nz[-1] + int(0.025 * sr))
    return y[s:e]


def _sing_word_across_notes(
    word_wav: np.ndarray,
    note_spans: list[tuple[float, float, float]],  # [(rel_start_sec, dur_sec, target_hz), ...]
    sr: int,
    formant_ratio: float,
    vib_depth: float,
    vib_rate: float,
    breathiness: float,
    is_female: bool,
    pitch_offset_semi: int
) -> np.ndarray:
    """Bir kelimenin TAM sesini (tum unsuz ve unlulerini koruyarak) ilgili notalarin melodik frekansina oturtur."""
    total_dur = sum(d for _, d, _ in note_spans)
    total_len = max(1, int(total_dur * sr))
    w = _trim_silence(word_wav, sr)

    if len(w) < int(sr * 0.04):
        return np.zeros(total_len, dtype=np.float32)

    if not HAS_PYWORLD:
        # Python 3.14 vb. ortamlarda pyworld yoksa saf SciPy/NumPy ile perde kaydirmali vokal sentezi
        try:
            x = np.ascontiguousarray(w, dtype=np.float64)
            pk = np.max(np.abs(x))
            if pk < 1e-4:
                return np.zeros(total_len, dtype=np.float32)
            x = (x / pk) * 0.92
            base_spoken_hz = 205.0 if is_female else 122.0
            out_chunks = []
            n_spans = max(1, len(note_spans))
            for idx_s, (_, d_sec, raw_hz) in enumerate(note_spans):
                seg_len = max(1, int(round(d_sec * sr)))
                s_start = int((idx_s / n_spans) * len(x))
                s_end = max(s_start + 16, int(((idx_s + 1) / n_spans) * len(x)))
                seg_w = x[s_start:min(len(x), s_end)]
                if len(seg_w) < 16:
                    seg_w = x

                eff_hz = raw_hz * (2.0 ** (pitch_offset_semi / 12.0))
                if is_female:
                    while eff_hz < 235.0:
                        eff_hz *= 2.0
                    while eff_hz > 850.0:
                        eff_hz *= 0.5
                else:
                    while eff_hz > 340.0:
                        eff_hz *= 0.5
                    while eff_hz < 95.0:
                        eff_hz *= 2.0

                ratio = float(np.clip(eff_hz / base_spoken_hz, 0.55, 2.2))
                shifted = resample(seg_w, max(16, int(len(seg_w) / ratio)))
                # Hedef nota suresine pürüzsüz uyarla
                tiled = np.resize(shifted, seg_len) if len(shifted) < seg_len else shifted[:seg_len]
                ft = np.arange(seg_len) / float(sr)
                vib = 1.0 + vib_depth * np.sin(2.0 * np.pi * vib_rate * ft)
                carrier = 0.72 + 0.28 * np.sin(2.0 * np.pi * eff_hz * ft * vib)
                chunk = tiled * carrier
                fade_c = min(int(0.012 * sr), len(chunk) // 4)
                if fade_c > 1:
                    chunk[:fade_c] *= np.linspace(0, 1, fade_c)
                    chunk[-fade_c:] *= np.linspace(1, 0, fade_c)
                out_chunks.append(chunk)

            y_fb = np.concatenate(out_chunks) if out_chunks else np.zeros(total_len)
            y_fb = np.nan_to_num(y_fb, nan=0.0).astype(np.float32)
            pk_f = np.max(np.abs(y_fb))
            if pk_f > 1e-4:
                y_fb = (y_fb / pk_f) * 0.90
            return y_fb[:total_len]
        except Exception:
            return np.zeros(total_len, dtype=np.float32)

    try:
        x = np.ascontiguousarray(w, dtype=np.float64)
        pk = np.max(np.abs(x))
        if pk < 1e-4:
            return np.zeros(total_len, dtype=np.float32)
        x = (x / pk) * 0.92

        f0, t = pw.harvest(x, sr, f0_floor=70.0, f0_ceil=780.0, frame_period=5.0)
        sp = pw.cheaptrick(x, f0, t, sr)
        ap = pw.d4c(x, f0, t, sr)

        orig_frames = len(f0)
        if orig_frames < 6:
            return np.zeros(total_len, dtype=np.float32)

        target_frames = max(12, int(total_dur / 0.005))

        # Kelimenin tum akisini (baslangic unsuzu -> unluler -> son unsuz) koruyarak hedef sureye esle
        # Eger kelime cok uzatiliyorsa orta unlu bolgesini daha fazla genislet, bas/son unsuzleri dogal hizda tut
        if target_frames > orig_frames * 1.35 and orig_frames >= 12:
            head_o = max(3, int(orig_frames * 0.22))
            tail_o = max(3, int(orig_frames * 0.22))
            mid_o = max(2, orig_frames - head_o - tail_o)

            head_t = min(head_o + 4, int(target_frames * 0.22))
            tail_t = min(tail_o + 4, int(target_frames * 0.22))
            mid_t = max(4, target_frames - head_t - tail_t)

            idx_h = np.linspace(0, head_o - 1, head_t)
            idx_m = np.linspace(head_o, orig_frames - tail_o - 1, mid_t)
            idx_t = np.linspace(orig_frames - tail_o, orig_frames - 1, tail_t)
            frame_map = np.clip(np.round(np.concatenate([idx_h, idx_m, idx_t])).astype(int), 0, orig_frames - 1)
        else:
            frame_map = np.clip(np.round(np.linspace(0, orig_frames - 1, target_frames)).astype(int), 0, orig_frames - 1)

        sp_new = sp[frame_map].copy()
        ap_new = np.clip(ap[frame_map].copy() * 0.75 + breathiness * 0.15, 0.0, 0.92)

        if abs(formant_ratio - 1.0) > 0.005:
            n_bins = sp_new.shape[1]
            freq_idx = np.clip(np.arange(n_bins) / formant_ratio, 0, n_bins - 1)
            for i in range(len(sp_new)):
                sp_new[i] = np.interp(freq_idx, np.arange(n_bins), sp_new[i])

        # Hedef nota frekans egrisini (melodi konturu) frame bazinda olustur
        f0_target = np.zeros(target_frames, dtype=np.float64)
        cur_f = 0
        for s_idx_n, (_, d_sec, raw_hz) in enumerate(note_spans):
            n_f = max(1, int(round(d_sec / 0.005)))
            end_f = target_frames if s_idx_n == len(note_spans) - 1 else min(target_frames, cur_f + n_f)
            if end_f <= cur_f:
                continue

            eff_hz = raw_hz * (2.0 ** (pitch_offset_semi / 12.0))
            if is_female:
                while eff_hz < 235.0:
                    eff_hz *= 2.0
                while eff_hz > 850.0:
                    eff_hz *= 0.5
            else:
                while eff_hz > 340.0:
                    eff_hz *= 0.5
                while eff_hz < 95.0:
                    eff_hz *= 2.0

            seg_len = end_f - cur_f
            ft = np.arange(seg_len) * 0.005
            glide = 1.0 - 0.025 * np.exp(-ft / 0.035)
            vib_env = np.clip((ft - 0.16) / 0.16, 0.0, 1.0)
            vib = 1.0 + vib_depth * vib_env * np.sin(2.0 * np.pi * vib_rate * ft)
            f0_target[cur_f:end_f] = eff_hz * glide * vib
            cur_f = end_f

        # Orijinal konusmadaki sert unsuzleri (s, ş, ç, t, k, p) koru ki kelimeler net anlasilsin!
        orig_f0_mapped = f0[frame_map]
        f0_new = f0_target.copy()
        f0_new[orig_f0_mapped == 0] = 0.0

        y_out = pw.synthesize(f0_new, sp_new, ap_new, sr, frame_period=5.0)
        y_out = np.nan_to_num(y_out, nan=0.0).astype(np.float32)

        # Konusma netligini artirmak icin hafif varlik (2.5 kHz) vurgusu
        nyq = sr * 0.5
        b_p, a_p = butter(2, [1800.0 / nyq, min(nyq * 0.90, 4200.0) / nyq], btype="bandpass")
        presence = lfilter(b_p, a_p, y_out) * 0.28
        y_out = y_out + presence.astype(np.float32)

        out_pk = np.max(np.abs(y_out))
        if out_pk > 1e-4:
            y_out = (y_out / out_pk) * 0.94

        if len(y_out) > total_len:
            y_out = y_out[:total_len]
        fade = min(int(0.015 * sr), len(y_out) // 4)
        if fade > 1:
            y_out[:fade] *= np.linspace(0, 1, fade)
            y_out[-fade:] *= np.linspace(1, 0, fade)
        return y_out.astype(np.float32)
    except Exception as e:
        print(f"[VocalSynth] Sentez hatasi: {e}")
        return np.zeros(total_len, dtype=np.float32)


def _apply_hall_reverb(stereo: np.ndarray, sr: int = 44100, wet: float = 0.14) -> np.ndarray:
    out = stereo.copy()
    for d_ms, g in zip([27, 43], [0.22, 0.14]):
        d_samp = int(sr * d_ms / 1000.0)
        if d_samp < len(stereo):
            out[d_samp:, 0] += stereo[:-d_samp, 1] * g * wet
            out[d_samp:, 1] += stereo[:-d_samp, 0] * g * wet
    return out


async def synthesize_vocal_track(
    melody_notes: list,
    bpm: int,
    active_vocals: dict,
    sr: int = 44100,
    raw_lyrics_text: str = ""
) -> np.ndarray:
    beat_sec = 60.0 / max(40, bpm)
    total_beats = sum(float(n.get("beats", 1.0)) for n in melody_notes) + 2.8
    total_samples = int(total_beats * beat_sec * sr)
    mix = np.zeros((total_samples, 2), dtype=np.float32)

    normalized = {}
    for k, cnt in (active_vocals or {}).items():
        k = LEGACY_VOCAL_KEYS.get(k, k)
        normalized[k] = normalized.get(k, 0) + int(cnt or 0)
    enabled_artists = [k for k, cnt in normalized.items() if cnt > 0]

    notes_copy = [dict(n) for n in melody_notes]
    if raw_lyrics_text.strip():
        auto_tokens = split_lyrics_to_syllable_tokens(raw_lyrics_text)
        t_i = 0
        for n in notes_copy:
            if n.get("pitch") and n.get("pitch") != "REST":
                n["lyric"] = auto_tokens[t_i] if t_i < len(auto_tokens) else ""
                t_i += 1

    has_any_lyrics = any((n.get("lyric") or "").strip() for n in notes_copy)
    if not enabled_artists and has_any_lyrics:
        enabled_artists = ["diva_pop"]
        normalized["diva_pop"] = 1

    if not enabled_artists:
        return mix

    # Notalari kelimeler halinde grupla: tire ile biten heceler ayni kelimenin notalarina baglanir
    # Ornek: ("bu-", D5), ("gün", C5) -> tek kelime "bugün" iki nota uzerinde melodik olarak soylenir!
    word_groups = []  # [{"word": "bugün", "start_beat": 2.0, "spans": [(0.0, 0.5, 587.3), (0.5, 0.5, 523.2)]}, ...]
    cur_word = ""
    cur_start_beat = None
    cur_spans = []
    rel_sec = 0.0

    cursor_beat = 0.0
    for note in notes_copy:
        beats = float(note.get("beats", 1.0))
        hz = note_to_hz(note.get("pitch", "REST"))
        lyr = (note.get("lyric") or "").strip()
        if hz > 0 and lyr and lyr != "-":
            if cur_start_beat is None:
                cur_start_beat = cursor_beat
                rel_sec = 0.0
            dur_s = beats * beat_sec
            cur_spans.append((rel_sec, dur_s, hz))
            rel_sec += dur_s
            if lyr.endswith("-"):
                cur_word += lyr[:-1]
            else:
                cur_word += lyr
                word_groups.append({
                    "word": cur_word,
                    "start_beat": cur_start_beat,
                    "spans": cur_spans
                })
                cur_word = ""
                cur_start_beat = None
                cur_spans = []
                rel_sec = 0.0
        cursor_beat += beats

    if cur_word and cur_spans:
        word_groups.append({
            "word": cur_word,
            "start_beat": cur_start_beat or 0.0,
            "spans": cur_spans
        })

    if not word_groups:
        return mix

    for a_idx, artist_key in enumerate(enabled_artists):
        count = normalized[artist_key]
        _, _, gender, f_ratio, vib_d, vib_r, breath, p_off, _ = ARTIST_PROFILES.get(
            artist_key, ("", "", "female", 1.12, 0.015, 5.5, 0.04, 0, "")
        )
        pan = -0.10 if a_idx % 2 == 0 else 0.10
        left_g = np.cos((pan + 1.0) * np.pi / 4.0)
        right_g = np.sin((pan + 1.0) * np.pi / 4.0)

        for wg in word_groups:
            w_wav = await _synthesize_word_wav(wg["word"], artist_key, sr=sr)
            if w_wav is None or len(w_wav) == 0:
                continue
            y_sung = _sing_word_across_notes(
                w_wav, wg["spans"], sr, f_ratio, vib_d, vib_r, breath, gender == "female", p_off
            )
            gain = 1.65 * min(1.35, 0.95 + 0.12 * count)
            s_idx = int(wg["start_beat"] * beat_sec * sr)
            e_idx = min(total_samples, s_idx + len(y_sung))
            seg = np.nan_to_num(y_sung[:e_idx - s_idx], nan=0.0) * gain
            mix[s_idx:e_idx, 0] += seg * left_g
            mix[s_idx:e_idx, 1] += seg * right_g

    mix = _apply_hall_reverb(mix, sr=sr, wet=0.14)
    return np.nan_to_num(mix, nan=0.0)
