import os
import re
import subprocess
import shutil
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt
from midiutil import MIDIFile
from vocal_synth import synthesize_vocal_track, note_to_hz, LAST_TTS_SOURCE

GM_PROGRAMS = {
    "violin": 40, "viola": 41, "cello": 42, "contrabass": 43, "harp": 46,
    "flute": 73, "oboe": 68, "clarinet": 71, "trumpet": 56, "horn": 60, "saxophone": 65,
    "piano": 0, "electric_guitar": 29, "acoustic_guitar": 25, "bass_guitar": 33,
    "oud_baglama": 105, "timpani": 47,
}

FAMILY_LAYOUT = {
    "violin":          {"pan": -0.62, "gain": 0.86, "octave_shift": 0},
    "viola":           {"pan": -0.28, "gain": 0.84, "octave_shift": 0},
    "cello":           {"pan":  0.35, "gain": 0.96, "octave_shift": 0},
    "contrabass":      {"pan":  0.65, "gain": 0.92, "octave_shift": -1},
    "harp":            {"pan": -0.72, "gain": 1.05, "octave_shift": 0},
    "flute":           {"pan": -0.38, "gain": 0.94, "octave_shift": 0},
    "oboe":            {"pan": -0.15, "gain": 0.84, "octave_shift": 0},
    "clarinet":        {"pan":  0.18, "gain": 0.90, "octave_shift": 0},
    "trumpet":         {"pan":  0.54, "gain": 0.88, "octave_shift": 0},
    "horn":            {"pan":  0.28, "gain": 0.90, "octave_shift": 0},
    "saxophone":       {"pan":  0.22, "gain": 0.92, "octave_shift": 0},
    "piano":           {"pan":  0.00, "gain": 1.32, "octave_shift": 0},
    "acoustic_guitar": {"pan": -0.42, "gain": 1.15, "octave_shift": 0},
    "electric_guitar": {"pan":  0.55, "gain": 0.95, "octave_shift": 0},
    "bass_guitar":     {"pan":  0.08, "gain": 1.00, "octave_shift": -1},
    "oud_baglama":     {"pan": -0.30, "gain": 1.18, "octave_shift": 0},
    "drums":           {"pan":  0.00, "gain": 0.92, "octave_shift": 0},
    "timpani":         {"pan":  0.68, "gain": 0.94, "octave_shift": -1},
}

NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    "Cb": -1, "B#": 12, "E#": 5, "Fb": 4,
}
_PITCH_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def find_fluidsynth() -> str | None:
    found = shutil.which("fluidsynth")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path("C:/Program Files/FluidSynth"),
        Path("C:/tools/fluidsynth"),
        Path.home() / "scoop" / "apps" / "fluidsynth" / "current",
    ]
    for base in candidates:
        if base.exists():
            for exe in base.rglob("fluidsynth.exe"):
                return str(exe)
    return None


def pitch_to_midi(pitch_str: str) -> int:
    if not pitch_str or pitch_str == "REST":
        return -1
    m = _PITCH_RE.match(pitch_str.strip())
    if not m:
        return -1
    name = m.group(1).upper() + m.group(2)
    return (int(m.group(3)) + 1) * 12 + NOTE_TO_MIDI.get(name, 0)


def _role_notes_for_instrument(inst_key: str, role: str, staves: dict) -> list:
    """Enstrumanin sectigi role gore notalari getirir; ilgili hat bosssa diger hatlardan cok sesli uretime gecer."""
    if role in staves and staves[role]:
        return staves[role]

    melody = staves.get("melody") or []
    harmony = staves.get("harmony") or []
    bass = staves.get("bass") or []
    rhythm = staves.get("rhythm") or []

    if role == "harmony":
        if harmony:
            return harmony
        out = []
        for ev in melody:
            p = ev.get("pitch")
            if not p or p == "REST":
                out.append({"pitches": [], "beats": ev.get("beats", 1.0)})
            else:
                m = pitch_to_midi(p)
                out.append({"pitches": [f"M:{m+4}", f"M:{m+7}"], "beats": ev.get("beats", 1.0)})
        return out
    if role == "bass":
        if bass:
            return bass
        return [{"pitch": ev.get("pitch", "REST"), "beats": ev.get("beats", 1.0)} for ev in melody]
    if role == "rhythm":
        if rhythm:
            return rhythm
        total_b = sum(float(n.get("beats", 1.0)) for n in melody) or 8.0
        steps = int(max(4, round(total_b)))
        return [{"pitch": "C3" if i % 2 == 0 else "G2", "beats": 1.0} for i in range(steps)]
    return melody


def _resolve_midi_num(token: str) -> int:
    if not token or token == "REST":
        return -1
    if token.startswith("M:"):
        try:
            return int(token.split(":", 1)[1])
        except Exception:
            return -1
    return pitch_to_midi(token)


def _build_midi(score: dict, midi_path: Path):
    bpm = int(score.get("bpm", 112))
    instruments = score.get("instruments", {})
    staves = score.get("staves", {})

    active_list = [(k, v) for k, v in instruments.items() if int(v.get("count", 0)) > 0]
    if not active_list:
        active_list = [("piano", {"count": 1, "role": "melody"})]

    midi = MIDIFile(len(active_list))
    channel_cursor = 0

    for track_idx, (inst_key, cfg) in enumerate(active_list):
        midi.addTempo(track_idx, 0, bpm)
        count = max(1, int(cfg.get("count", 1)))
        role = cfg.get("role", "melody")
        notes = _role_notes_for_instrument(inst_key, role, staves)
        layout = FAMILY_LAYOUT.get(inst_key, {"pan": 0.0, "gain": 0.9, "octave_shift": 0})

        if inst_key == "drums":
            ch = 9
        else:
            if channel_cursor == 9:
                channel_cursor += 1
            ch = channel_cursor % 16
            channel_cursor += 1
            prog = GM_PROGRAMS.get(inst_key, 0)
            midi.addProgramChange(track_idx, ch, 0, prog)
            cc_pan = int(np.clip((layout["pan"] + 1.0) * 63.5, 0, 127))
            midi.addControllerEvent(track_idx, ch, 0, 10, cc_pan)

        velocity = min(112, int(68 + 9 * np.sqrt(count)))
        cursor = 0.0
        for step_i, ev in enumerate(notes):
            beats = float(ev.get("beats", 1.0))
            if inst_key == "drums":
                if ev.get("pitch") != "REST":
                    kick = 36 if step_i % 2 == 0 else 38
                    midi.addNote(track_idx, 9, kick, cursor, min(0.35, beats * 0.75), velocity)
                    midi.addNote(track_idx, 9, 42, cursor, 0.18, max(45, velocity - 18))
            else:
                pitches = ev.get("pitches") or ([ev.get("pitch")] if ev.get("pitch") else [])
                for p in pitches:
                    m_num = _resolve_midi_num(p)
                    if m_num >= 0:
                        m_num += layout["octave_shift"] * 12
                        if inst_key in ("bass_guitar", "contrabass"):
                            m_num = max(28, min(55, m_num))
                        elif inst_key == "cello":
                            m_num = max(36, min(69, m_num))
                        elif inst_key == "violin":
                            m_num = max(55, m_num)
                        gate = 0.95 if inst_key in ("violin", "viola", "cello", "contrabass", "flute", "horn") else 0.78
                        vel_dyn = int(np.clip(velocity * (1.04 if step_i % 2 == 0 else 0.92), 35, 120))
                        midi.addNote(track_idx, ch, m_num, cursor, max(0.08, beats * gate), vel_dyn)
            cursor += beats

    with open(midi_path, "wb") as f:
        midi.writeFile(f)


def _bandpass_formant(sig: np.ndarray, sr: int, center_hz: float, q: float = 3.5) -> np.ndarray:
    nyq = sr * 0.5
    lo = max(30.0, center_hz / (1.0 + 0.5 / q))
    hi = min(nyq * 0.92, center_hz * (1.0 + 0.5 / q))
    if hi <= lo + 20.0:
        return sig * 0.0
    sos = butter(2, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return sosfilt(sos, sig)


def _render_instrument_voice(inst_key: str, f: float, t: np.ndarray, sr: int) -> np.ndarray:
    """Her enstruman ailesine ozgu belirgin tını, harmonik yapisi ve filtre karakteri uretir."""
    n = len(t)
    if n == 0 or f <= 10.0:
        return np.zeros(n, dtype=np.float64)

    # 1. Yayli orkestra (Keman, Viyola, Cello, Kontrbas)
    if inst_key in ("violin", "viola", "cello", "contrabass"):
        vib_rate = 5.2 if inst_key == "violin" else (4.8 if inst_key == "viola" else 4.3)
        vib_amp = 0.0045 if inst_key in ("violin", "viola") else 0.0032
        vib_env = np.clip((t - 0.10) / 0.18, 0.0, 1.0)
        inst_f = f * (1.0 + vib_amp * vib_env * np.sin(2.0 * np.pi * vib_rate * t))
        ph = 2.0 * np.pi * np.cumsum(inst_f) / sr
        saw = (
            0.55 * np.sin(ph)
            + 0.34 * np.sin(2 * ph)
            + 0.22 * np.sin(3 * ph)
            + 0.14 * np.sin(4 * ph)
            + 0.08 * np.sin(5 * ph)
            + 0.04 * np.sin(6 * ph)
        )
        bow_noise = np.random.normal(0.0, 0.015, n)
        b_hp, a_hp = butter(1, min(1800.0, sr * 0.4) / (sr * 0.5), btype="high")
        bow_noise = lfilter(b_hp, a_hp, bow_noise)

        if inst_key == "violin":
            body = _bandpass_formant(saw, sr, 460.0, 2.2) * 0.42 + _bandpass_formant(saw, sr, 2400.0, 2.5) * 0.22
            cut = 4200.0
        elif inst_key == "viola":
            body = _bandpass_formant(saw, sr, 320.0, 2.2) * 0.45 + _bandpass_formant(saw, sr, 1750.0, 2.5) * 0.20
            cut = 2800.0
        elif inst_key == "cello":
            body = _bandpass_formant(saw, sr, 210.0, 2.4) * 0.50 + _bandpass_formant(saw, sr, 1150.0, 2.6) * 0.18
            cut = 1650.0
        else:
            body = _bandpass_formant(saw, sr, 125.0, 2.5) * 0.52 + _bandpass_formant(saw, sr, 650.0, 2.8) * 0.15
            cut = 950.0

        b_lp, a_lp = butter(2, min(cut, sr * 0.45) / (sr * 0.5), btype="low")
        return lfilter(b_lp, a_lp, saw * 0.62 + body + bow_noise)

    # 2. Piyano, Arp, Akustik Gitar, Ud/Baglama (dolgun sustain + parlak vurucu harmonikler)
    if inst_key in ("piano", "harp", "acoustic_guitar", "oud_baglama"):
        w = 2.0 * np.pi * f * t
        if inst_key == "piano":
            d1, d2, d3 = 0.85, 2.1, 4.5
            hammer = np.exp(-t * 32.0) * 0.48
        elif inst_key == "harp":
            d1, d2, d3 = 1.6, 3.8, 7.5
            hammer = np.exp(-t * 28.0) * 0.25
        elif inst_key == "oud_baglama":
            d1, d2, d3 = 1.9, 4.8, 9.0
            hammer = np.exp(-t * 45.0) * 0.38
        else:
            d1, d2, d3 = 1.5, 3.6, 7.0
            hammer = np.exp(-t * 34.0) * 0.30

        pluck = (
            0.85 * np.sin(w) * np.exp(-t * d1)
            + 0.48 * np.sin(2 * w + 0.12) * np.exp(-t * d2)
            + 0.28 * np.sin(3 * w + 0.25) * np.exp(-t * d3)
            + 0.16 * np.sin(4 * w) * np.exp(-t * (d3 * 1.3))
            + hammer * np.sin(2.0 * np.pi * min(sr * 0.38, f * 5.0) * t)
        )
        res_hz = 320.0 if inst_key == "piano" else (240.0 if inst_key == "oud_baglama" else 260.0)
        body_res = _bandpass_formant(pluck, sr, res_hz, 2.4) * 0.32
        return pluck + body_res

    # 3. Elektro Gitar
    if inst_key == "electric_guitar":
        w = 2.0 * np.pi * f * t
        raw_g = 0.65 * np.sin(w) + 0.42 * np.sin(2 * w) + 0.24 * np.sin(3 * w)
        sat = np.tanh(2.6 * raw_g) * np.exp(-t * 1.4)
        cab = _bandpass_formant(sat, sr, 950.0, 1.5) * 0.35
        b_lp, a_lp = butter(2, min(3100.0, sr * 0.45) / (sr * 0.5), btype="low")
        return lfilter(b_lp, a_lp, sat * 0.75 + cab)

    # 4. Tahta Uflemeliler (Flut, Klarnet)
    if inst_key in ("flute", "clarinet"):
        w = 2.0 * np.pi * f * t
        air = np.random.normal(0.0, 0.022, n)
        air_bp = _bandpass_formant(air, sr, max(1400.0, f * 3.0), 2.5)
        if inst_key == "flute":
            tone = 0.90 * np.sin(w) + 0.22 * np.sin(2 * w) + 0.05 * np.sin(3 * w)
        else:
            tone = 0.80 * np.sin(w) + 0.06 * np.sin(2 * w) + 0.34 * np.sin(3 * w) + 0.12 * np.sin(5 * w)
        b_lp, a_lp = butter(2, min(4200.0, sr * 0.45) / (sr * 0.5), btype="low")
        return lfilter(b_lp, a_lp, tone + air_bp)

    # 5. Bakir/Kamis Uflemeliler (Trompet, Korno, Saksafon, Obua)
    # Trompet ozellikle parlak, metalik ve yuksek harmonikli olarak ayristirildi ki diger calgilarin icinde duyulsun
    if inst_key in ("horn", "trumpet", "saxophone", "oboe"):
        w = 2.0 * np.pi * f * t
        if inst_key == "trumpet":
            swell = np.minimum(1.0, t / 0.022)
            brass_raw = (
                0.55 * np.sin(w)
                + swell * (
                    0.52 * np.sin(2 * w)
                    + 0.42 * np.sin(3 * w)
                    + 0.30 * np.sin(4 * w)
                    + 0.18 * np.sin(5 * w)
                    + 0.10 * np.sin(6 * w)
                )
            )
            brass_sat = np.tanh(1.8 * brass_raw) * 0.85
            bright_formant = _bandpass_formant(brass_sat, sr, 2100.0, 2.0) * 0.35
            b_lp, a_lp = butter(2, min(5200.0, sr * 0.45) / (sr * 0.5), btype="low")
            return lfilter(b_lp, a_lp, brass_sat + bright_formant)

        swell = np.minimum(1.0, t / (0.075 if inst_key == "horn" else 0.035))
        if inst_key == "horn":
            tone = 0.72 * np.sin(w) + swell * (0.28 * np.sin(2 * w) + 0.10 * np.sin(3 * w))
            b_lp, a_lp = butter(2, min(1950.0, sr * 0.45) / (sr * 0.5), btype="low")
            return lfilter(b_lp, a_lp, tone)
        if inst_key == "saxophone":
            vib = 1.0 + 0.004 * np.clip((t - 0.12) / 0.15, 0.0, 1.0) * np.sin(2 * np.pi * 5.0 * t)
            ph = 2.0 * np.pi * np.cumsum(f * vib) / sr
            tone = 0.62 * np.sin(ph) + swell * (0.38 * np.sin(2 * ph) + 0.22 * np.sin(3 * ph))
            b_lp, a_lp = butter(2, min(3200.0, sr * 0.45) / (sr * 0.5), btype="low")
            return lfilter(b_lp, a_lp, tone)
        tone = 0.52 * np.sin(w) + swell * (0.36 * np.sin(2 * w) + 0.28 * np.sin(3 * w) + 0.16 * np.sin(4 * w))
        b_lp, a_lp = butter(2, min(3400.0, sr * 0.45) / (sr * 0.5), btype="low")
        return lfilter(b_lp, a_lp, tone)

    if inst_key == "bass_guitar":
        w = 2.0 * np.pi * f * t
        s = (0.82 * np.sin(w) + 0.28 * np.sin(2 * w) * np.exp(-t * 5.0)) * np.exp(-t * 1.4)
        b_lp, a_lp = butter(2, min(900.0, sr * 0.45) / (sr * 0.5), btype="low")
        return lfilter(b_lp, a_lp, s)

    w = 2.0 * np.pi * f * t
    return (0.75 * np.sin(w) + 0.22 * np.sin(2 * w)) * np.exp(-t * 1.6)


def _synth_builtin_fallback(score: dict, sr: int = 44100) -> np.ndarray:
    """Her aktif enstrumani bagimsiz kanal ve tınıyla sentezleyen cok enstrumanli motor."""
    bpm = int(score.get("bpm", 112))
    beat_sec = 60.0 / max(40, bpm)
    staves = score.get("staves", {})
    instruments = score.get("instruments", {})

    max_beats = 4.0
    for staff_notes in staves.values():
        b = sum(float(n.get("beats", 1.0)) for n in staff_notes)
        if b > max_beats:
            max_beats = b

    total_samples = int((max_beats + 2.8) * beat_sec * sr)
    mix = np.zeros((total_samples, 2), dtype=np.float32)

    active_items = [(k, v) for k, v in instruments.items() if int(v.get("count", 0)) > 0]
    if not active_items:
        return mix

    # Aktif enstruman sayisina gore her birinin kendi sesini koruyacak sekilde olcekleme
    num_types = len(active_items)
    norm_scale = 0.68 / np.sqrt(max(1, num_types))

    for idx, (inst_key, cfg) in enumerate(active_items):
        count = max(1, int(cfg.get("count", 1)))
        role = cfg.get("role", "melody")
        notes = _role_notes_for_instrument(inst_key, role, staves)
        layout = FAMILY_LAYOUT.get(inst_key, {"pan": np.sin(idx * 1.25) * 0.6, "gain": 0.9, "octave_shift": 0})
        pan = float(layout["pan"])
        left_g = np.cos((pan + 1.0) * np.pi / 4.0)
        right_g = np.sin((pan + 1.0) * np.pi / 4.0)
        cursor = 0.0

        # Ayni rolde birden fazla enstruman varsa aralarinda hafif oktav/aralik acilimi ver
        role_peers = [k for k, v in active_items if v.get("role", "melody") == role and int(v.get("count", 0)) > 0]
        peer_idx = role_peers.index(inst_key) if inst_key in role_peers else 0
        peer_offset_semi = 0
        if len(role_peers) > 1 and role == "harmony":
            peer_offset_semi = 7 if peer_idx % 2 == 1 else 0
        elif len(role_peers) > 1 and role == "melody" and inst_key == "trumpet":
            peer_offset_semi = 0

        for step_i, ev in enumerate(notes):
            beats = float(ev.get("beats", 1.0))
            dur = beats * beat_sec
            n_samp = int(dur * sr)
            if n_samp <= 8:
                cursor += beats
                continue
            t = np.linspace(0, dur, n_samp, endpoint=False)

            if inst_key in ("drums", "timpani"):
                if ev.get("pitch") != "REST":
                    if inst_key == "timpani":
                        hz = max(55.0, note_to_hz(ev.get("pitch", "D3")) * 0.5)
                        pitch_env = hz * (1.0 + 0.14 * np.exp(-t * 18.0))
                        ph = 2.0 * np.pi * np.cumsum(pitch_env) / sr
                        body = np.sin(ph) * np.exp(-t * 3.8)
                        mallet = np.random.normal(0.0, 0.25, n_samp) * np.exp(-t * 45.0)
                        wave = (body * 0.85 + mallet * 0.25) * norm_scale * 1.35
                    else:
                        if step_i % 2 == 0:
                            f_k = 135.0 * np.exp(-t * 24.0) + 46.0
                            kick = np.sin(2.0 * np.pi * np.cumsum(f_k) / sr) * np.exp(-t * 11.0)
                            hat = np.random.normal(0.0, 0.18, n_samp) * np.exp(-t * 38.0)
                            wave = (kick * 0.82 + hat * 0.25) * norm_scale * 1.25
                        else:
                            snare_body = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t * 18.0) * 0.45
                            snare_wire = np.random.normal(0.0, 0.45, n_samp) * np.exp(-t * 15.0)
                            b_hp, a_hp = butter(2, 900.0 / (sr * 0.5), btype="high")
                            snare_wire = lfilter(b_hp, a_hp, snare_wire)
                            wave = (snare_body + snare_wire) * norm_scale * 1.15
                else:
                    wave = np.zeros(n_samp)
            else:
                pitches = ev.get("pitches") or ([ev.get("pitch")] if ev.get("pitch") else [])
                wave = np.zeros(n_samp)
                for p in pitches:
                    m_num = _resolve_midi_num(p)
                    if m_num < 0:
                        continue
                    m_num += layout["octave_shift"] * 12 + peer_offset_semi
                    if inst_key in ("bass_guitar", "contrabass"):
                        m_num = max(28, min(55, m_num))
                    elif inst_key == "cello":
                        m_num = max(36, min(69, m_num))
                    elif inst_key == "violin":
                        m_num = max(55, m_num)
                    hz = 440.0 * (2.0 ** ((m_num - 69) / 12.0))
                    if hz <= 0:
                        continue

                    ens_n = min(count, 5)
                    for c_i in range(ens_n):
                        detune = 1.0 + (c_i - (ens_n - 1) / 2.0) * 0.0032
                        delay_s = int(c_i * 0.006 * sr)
                        v_sig = _render_instrument_voice(inst_key, hz * detune, t, sr)
                        if delay_s > 0 and delay_s < len(v_sig):
                            shifted = np.zeros_like(v_sig)
                            shifted[delay_s:] = v_sig[:-delay_s]
                            v_sig = shifted
                        wave += v_sig / np.sqrt(ens_n)

                att_sec = 0.022 if inst_key == "trumpet" else (0.055 if inst_key in ("violin", "viola", "cello", "contrabass", "horn") else 0.012)
                rel_sec = 0.075 if inst_key in ("violin", "viola", "cello", "contrabass", "piano", "harp") else 0.045
                att = min(int(att_sec * sr), n_samp // 3)
                rel = min(int(rel_sec * sr), n_samp // 3)
                env = np.ones(n_samp)
                if att > 1:
                    env[:att] = np.linspace(0.0, 1.0, att) ** 1.2
                if rel > 1:
                    env[-rel:] = np.linspace(1.0, 0.0, rel) ** 1.2
                dyn = 1.06 if step_i % 2 == 0 else 0.92
                wave = wave * env * norm_scale * float(layout["gain"]) * dyn

            s_idx = int(cursor * beat_sec * sr)
            e_idx = min(total_samples, s_idx + len(wave))
            seg = wave[:e_idx - s_idx].astype(np.float32)
            mix[s_idx:e_idx, 0] += seg * left_g
            mix[s_idx:e_idx, 1] += seg * right_g
            cursor += beats

    # Hafif salon yankisi (cok fazla yanki diger enstrumanlarin netligini bulandirmasin)
    out = mix.copy()
    for d_ms, g in zip([23, 41, 67], [0.14, 0.10, 0.06]):
        d_samp = int(sr * d_ms / 1000.0)
        if d_samp < len(mix):
            out[d_samp:, 0] += mix[:-d_samp, 1] * g
            out[d_samp:, 1] += mix[:-d_samp, 0] * g
    return np.nan_to_num(out, nan=0.0)


def _load_pil_font(size: int, bold: bool = False):
    from PIL import ImageFont
    candidates = (
        [
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for fp in candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _wrap_text_lines(text: str, max_chars: int = 68, max_lines: int = 3) -> list[str]:
    clean = " ".join((text or "").replace("-", "").split())
    if not clean:
        return []
    words = clean.split(" ")
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines[:max_lines]


def render_mp4_video(score: dict, wav_path: Path, mp4_path: Path) -> bool:
    from clip_generator import render_studio_clip_mp4
    return render_studio_clip_mp4(score or {}, wav_path, mp4_path, temp_dir=wav_path.parent)


def convert_wav_to_format(wav_path: Path, out_path: Path, target_fmt: str, score: dict | None = None) -> bool:
    """Mevcut WAV dosyasini yeniden sentezlemeden aninda MP3, FLAC, OGG veya MP4 formatina cevirir."""
    fmt = target_fmt.lower().lstrip(".")
    if fmt == "wav":
        if wav_path != out_path:
            shutil.copy2(wav_path, out_path)
        return out_path.exists()
    if fmt == "mp4":
        return render_mp4_video(score or {}, wav_path, out_path)

    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if fmt == "mp3":
            cmd = [ffmpeg_exe, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "320k", str(out_path)]
        elif fmt == "flac":
            cmd = [ffmpeg_exe, "-y", "-i", str(wav_path), "-codec:a", "flac", str(out_path)]
        elif fmt == "ogg":
            cmd = [ffmpeg_exe, "-y", "-i", str(wav_path), "-codec:a", "libvorbis", "-q:a", "7", str(out_path)]
        else:
            return False
        res = subprocess.run(cmd, capture_output=True, timeout=90)
        return res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 500
    except Exception as exc:
        print(f"[Format Convert] Hata ({fmt}): {exc}")
        return False


async def render_full_mix(score: dict, out_wav: Path, sf_path: Path | None) -> str:
    sr = 44100
    engine_mode = "Senfonik DSP"
    inst_audio = None

    fluidsynth_bin = find_fluidsynth()
    if sf_path and sf_path.exists() and fluidsynth_bin:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mid_file = tmp / "score.mid"
            raw_wav = tmp / "inst.wav"
            _build_midi(score, mid_file)
            cmd = [
                fluidsynth_bin, "-ni", "-F", str(raw_wav),
                "-r", str(sr), "-g", "0.75", str(sf_path), str(mid_file)
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode == 0 and raw_wav.exists():
                inst_audio, _ = sf.read(str(raw_wav), dtype="float32")
                if inst_audio.ndim == 1:
                    inst_audio = np.column_stack([inst_audio, inst_audio])
                engine_mode = f"FluidSynth ({sf_path.name})"

    if inst_audio is None:
        inst_audio = _synth_builtin_fallback(score, sr=sr)

    inst_audio = np.nan_to_num(inst_audio, nan=0.0)

    melody_list = score.get("staves", {}).get("melody", [])
    raw_lyrics = (score.get("lyrics_text") or "").strip()
    has_any_lyrics = bool(raw_lyrics) or any((n.get("lyric") or "").strip() for n in melody_list)
    has_vocals = has_any_lyrics or any(int(cnt or 0) > 0 for cnt in (score.get("vocals") or {}).values())
    vocal_audio = await synthesize_vocal_track(
        melody_list,
        int(score.get("bpm", 112)),
        score.get("vocals", {}),
        sr=sr,
        raw_lyrics_text=raw_lyrics
    )
    vocal_audio = np.nan_to_num(vocal_audio, nan=0.0)

    max_len = max(len(inst_audio), len(vocal_audio))
    final_mix = np.zeros((max_len, 2), dtype=np.float32)
    inst_gain = 0.48 if has_vocals else 0.95
    final_mix[:len(inst_audio)] += inst_audio * inst_gain
    final_mix[:len(vocal_audio)] += vocal_audio * 1.45

    final_mix = np.nan_to_num(final_mix, nan=0.0)
    final_mix = np.tanh(final_mix * 0.90) * 0.92
    sf.write(str(out_wav), final_mix, sr, subtype="PCM_16")
    if has_vocals:
        src = LAST_TTS_SOURCE.get("name") or "?"
        return f"{engine_mode} + Gerçekçi Vokal [{src}]"
    return engine_mode
