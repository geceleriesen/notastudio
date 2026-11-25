import io
import math
import re
import struct
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TREBLE_LINE_NOTES = ["F5", "D5", "B4", "G4", "E4"]  # top to bottom 5 staff lines
TREBLE_DIATONIC_TOP = 5 * 7 + 3  # F5


def midi_to_pitch(midi_num: int) -> str:
    midi_num = max(24, min(108, int(midi_num)))
    octave = (midi_num // 12) - 1
    name = NOTE_NAMES_SHARP[midi_num % 12]
    return f"{name}{octave}"


def _diatonic_to_pitch(dn: int) -> str:
    letters = ["C", "D", "E", "F", "G", "A", "B"]
    dn = max(14, min(62, int(dn)))
    octave = dn // 7
    letter = letters[dn % 7]
    return f"{letter}{octave}"


def _base_score_template(title: str, composer: str, bpm: int = 112, time_sig: str = "4/4", key_sig: str = "Do Majör") -> dict:
    return {
        "id": f"imp_{uuid.uuid4().hex[:8]}",
        "title": title or "İçe Aktarılan Eser",
        "composer": composer or "NotaStüdyo OMR / İçe Aktarım",
        "bpm": int(max(45, min(220, bpm))),
        "time_signature": time_sig or "4/4",
        "key_signature": key_sig or "Do Majör",
        "instruments": {
            "piano": {"count": 1, "role": "harmony"},
            "violin": {"count": 2, "role": "melody"},
            "cello": {"count": 1, "role": "bass"},
            "drums": {"count": 1, "role": "rhythm"},
        },
        "vocals": {
            "diva_pop": 0,
            "rock_tenor": 0,
        },
        "staves": {
            "melody": [],
            "harmony": [],
            "bass": [],
            "rhythm": [],
        },
    }


def _populate_accompaniment_from_melody(score: dict):
    """Armoni, bas veya ritim bos kalmissa melodiden otomatik orkestra esligi uretir."""
    melody = score["staves"].get("melody") or []
    if not melody:
        return
    if not score["staves"].get("harmony"):
        harm = []
        acc_beats = 0.0
        cur_pitches = []
        for ev in melody:
            b = float(ev.get("beats", 1.0))
            p = ev.get("pitch", "REST")
            if p != "REST":
                cur_pitches.append(p)
            acc_beats += b
            if acc_beats >= 2.0:
                root = cur_pitches[0] if cur_pitches else "C4"
                harm.append({"pitches": [root], "beats": round(acc_beats, 2)})
                acc_beats = 0.0
                cur_pitches = []
        if acc_beats > 0:
            harm.append({"pitches": [cur_pitches[0] if cur_pitches else "C4"], "beats": round(acc_beats, 2)})
        score["staves"]["harmony"] = harm

    if not score["staves"].get("bass"):
        bass = []
        for h in score["staves"]["harmony"]:
            p0 = (h.get("pitches") or ["C4"])[0]
            m = re.match(r"^([A-G][#b]?)(\d)$", p0)
            if m:
                oct_b = max(2, int(m.group(2)) - 1)
                bass.append({"pitch": f"{m.group(1)}{oct_b}", "beats": h.get("beats", 2.0)})
            else:
                bass.append({"pitch": "C3", "beats": h.get("beats", 2.0)})
        score["staves"]["bass"] = bass

    if not score["staves"].get("rhythm"):
        total_b = sum(float(n.get("beats", 1.0)) for n in melody)
        rhy = []
        b_cur = 0.0
        idx = 0
        while b_cur < total_b - 0.01:
            step = min(1.0, total_b - b_cur)
            pad = "C3" if idx % 2 == 1 else "G2"
            rhy.append({"pitch": pad, "beats": round(step, 2)})
            b_cur += step
            idx += 1
        score["staves"]["rhythm"] = rhy


# ==============================================================================
# 1. STANDARD MIDI (.mid / .midi) PARSER
# ==============================================================================
def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    val = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return val, pos


def parse_midi_bytes(raw: bytes, filename: str) -> dict:
    if len(raw) < 14 or raw[:4] != b"MThd":
        raise ValueError("Geçerli bir MIDI başlığı bulunamadı")

    hdr_len = struct.unpack(">I", raw[4:8])[0]
    fmt, n_tracks, division = struct.unpack(">HHH", raw[8:14])
    tpq = division if not (division & 0x8000) else 480
    pos = 8 + hdr_len

    bpm = 112
    time_sig = "4/4"
    all_notes = []  # (start_tick, dur_tick, pitch_midi, channel)

    for _ in range(n_tracks):
        if pos + 8 > len(raw) or raw[pos:pos + 4] != b"MTrk":
            break
        trk_len = struct.unpack(">I", raw[pos + 4:pos + 8])[0]
        trk_end = min(len(raw), pos + 8 + trk_len)
        p = pos + 8
        tick = 0
        running_status = 0
        active_notes = {}  # (ch, note) -> start_tick

        while p < trk_end:
            delta, p = _read_vlq(raw, p)
            tick += delta
            if p >= trk_end:
                break
            status = raw[p]
            if status < 0x80:
                status = running_status
            else:
                p += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:  # Meta event
                if p >= trk_end:
                    break
                meta_type = raw[p]
                p += 1
                m_len, p = _read_vlq(raw, p)
                m_data = raw[p:p + m_len]
                p += m_len
                if meta_type == 0x51 and len(m_data) == 3:
                    usec = (m_data[0] << 16) | (m_data[1] << 8) | m_data[2]
                    if usec > 0:
                        bpm = int(round(60_000_000 / usec))
                elif meta_type == 0x58 and len(m_data) >= 2:
                    num = m_data[0]
                    den = 2 ** m_data[1]
                    time_sig = f"{num}/{den}"
            elif status in (0xF0, 0xF7):
                s_len, p = _read_vlq(raw, p)
                p += s_len
            else:
                cmd = status & 0xF0
                ch = status & 0x0F
                if cmd in (0xC0, 0xD0):
                    p += 1
                else:
                    if p + 1 >= trk_end:
                        break
                    d1, d2 = raw[p], raw[p + 1]
                    p += 2
                    if cmd == 0x90 and d2 > 0:
                        active_notes[(ch, d1)] = tick
                    elif cmd == 0x80 or (cmd == 0x90 and d2 == 0):
                        st = active_notes.pop((ch, d1), None)
                        if st is not None and tick > st:
                            all_notes.append((st, tick - st, d1, ch))

        pos = trk_end

    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    score = _base_score_template(title, "MIDI İçe Aktarım", bpm=bpm, time_sig=time_sig)
    if not all_notes:
        return score

    # Kanal 9 (0-indexed) perkusyondur; digerlerini perdesine gore Melodi / Armoni / Bas olarak ayir
    pitched = [n for n in all_notes if n[3] != 9]
    perc = [n for n in all_notes if n[3] == 9]

    # Zaman dilimlerine (0.5 vurusluk izgara) grupla
    pitched.sort(key=lambda x: (x[0], -x[2]))
    melody_events = []
    harmony_events = []
    bass_events = []

    i = 0
    last_tick = 0
    while i < len(pitched):
        st = pitched[i][0]
        if st - last_tick >= tpq * 0.45:
            rest_b = round((st - last_tick) / tpq * 2) / 2.0
            if rest_b >= 0.5:
                melody_events.append({"pitch": "REST", "beats": min(4.0, rest_b), "lyric": ""})

        chord_group = []
        while i < len(pitched) and abs(pitched[i][0] - st) <= tpq * 0.15:
            chord_group.append(pitched[i])
            i += 1

        chord_group.sort(key=lambda x: x[2], reverse=True)
        top_n = chord_group[0]
        beats = max(0.5, min(4.0, round((top_n[1] / tpq) * 2) / 2.0))
        melody_events.append({"pitch": midi_to_pitch(top_n[2]), "beats": beats, "lyric": ""})

        if len(chord_group) >= 2:
            mid_pitches = [midi_to_pitch(x[2]) for x in chord_group[:-1][:4]]
            harmony_events.append({"pitches": mid_pitches, "beats": beats})
            low_n = chord_group[-1]
            bass_events.append({"pitch": midi_to_pitch(low_n[2]), "beats": beats})

        last_tick = st + int(beats * tpq)

    if perc:
        perc.sort(key=lambda x: x[0])
        for st, dur, note_m, _ in perc[:256]:
            b = max(0.5, min(2.0, round((dur / tpq) * 2) / 2.0))
            pad = "C3" if note_m in (38, 40) else ("E3" if note_m in (42, 44, 46) else "G2")
            score["staves"]["rhythm"].append({"pitch": pad, "beats": b})

    score["staves"]["melody"] = melody_events[:320]
    if harmony_events:
        score["staves"]["harmony"] = harmony_events[:200]
    if bass_events:
        score["staves"]["bass"] = bass_events[:200]
    _populate_accompaniment_from_melody(score)
    return score


# ==============================================================================
# 2. MUSICXML (.xml / .musicxml / .mxl) PARSER
# ==============================================================================
def parse_musicxml_bytes(raw: bytes, filename: str) -> dict:
    xml_bytes = raw
    if filename.lower().endswith(".mxl") or raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml") and "meta-inf" not in n.lower()]
            if not xml_names:
                raise ValueError("MXL arşivi içinde MusicXML dosyası bulunamadı")
            xml_bytes = zf.read(xml_names[0])

    root = ET.fromstring(xml_bytes)
    # Namespace temizle
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    title_el = root.find(".//work/work-title") or root.find(".//movement-title")
    title = title_el.text.strip() if (title_el is not None and title_el.text) else Path(filename).stem
    comp_el = root.find(".//identification/creator[@type='composer']") or root.find(".//identification/creator")
    composer = comp_el.text.strip() if (comp_el is not None and comp_el.text) else "MusicXML İçe Aktarım"

    bpm = 112
    sound_el = root.find(".//sound[@tempo]")
    if sound_el is not None:
        try:
            bpm = int(float(sound_el.attrib["tempo"]))
        except Exception:
            pass

    beats_el = root.find(".//attributes/time/beats")
    beat_type_el = root.find(".//attributes/time/beat-type")
    time_sig = f"{beats_el.text}/{beat_type_el.text}" if (beats_el is not None and beat_type_el is not None) else "4/4"

    divisions = 1
    div_el = root.find(".//attributes/divisions")
    if div_el is not None and div_el.text:
        divisions = max(1, int(div_el.text))

    parts = root.findall(".//part")
    score = _base_score_template(title, composer, bpm=bpm, time_sig=time_sig)

    for p_idx, part in enumerate(parts[:3]):
        events = []
        for note in part.findall(".//note"):
            if note.find("chord") is not None:
                continue
            dur_el = note.find("duration")
            dur_val = int(dur_el.text) if (dur_el is not None and dur_el.text) else divisions
            beats = max(0.25, min(4.0, round((dur_val / divisions) * 4) / 4.0))

            lyr_el = note.find(".//lyric/text")
            lyr_txt = lyr_el.text.strip() if (lyr_el is not None and lyr_el.text) else ""

            if note.find("rest") is not None:
                events.append({"pitch": "REST", "beats": beats, "lyric": ""})
            else:
                step_el = note.find("pitch/step")
                oct_el = note.find("pitch/octave")
                alt_el = note.find("pitch/alter")
                if step_el is not None and oct_el is not None:
                    step = step_el.text.strip().upper()
                    octave = oct_el.text.strip()
                    alt = int(float(alt_el.text)) if (alt_el is not None and alt_el.text) else 0
                    acc = "#" if alt > 0 else ("b" if alt < 0 else "")
                    events.append({"pitch": f"{step}{acc}{octave}", "beats": beats, "lyric": lyr_txt})

        if p_idx == 0:
            score["staves"]["melody"] = events[:320]
        elif p_idx == 1:
            score["staves"]["harmony"] = [{"pitches": [e["pitch"]], "beats": e["beats"]} for e in events[:200] if e["pitch"] != "REST"]
        elif p_idx == 2:
            score["staves"]["bass"] = [{"pitch": e["pitch"], "beats": e["beats"]} for e in events[:200] if e["pitch"] != "REST"]

    _populate_accompaniment_from_melody(score)
    return score


# ==============================================================================
# 3. ABC NOTATION (.abc / .txt) PARSER
# ==============================================================================
def parse_abc_bytes(raw: bytes, filename: str) -> dict:
    text = raw.decode("utf-8", errors="replace")
    title = Path(filename).stem
    composer = "ABC Notasyon"
    bpm = 112
    time_sig = "4/4"
    body_lines = []

    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        if s.startswith("T:"):
            title = s[2:].strip() or title
        elif s.startswith("C:"):
            composer = s[2:].strip() or composer
        elif s.startswith("M:"):
            time_sig = s[2:].strip() or time_sig
        elif s.startswith("Q:"):
            m = re.search(r"(\d{2,3})", s)
            if m:
                bpm = int(m.group(1))
        elif len(s) >= 2 and s[1] == ":" and s[0].isalpha():
            continue
        else:
            body_lines.append(s)

    score = _base_score_template(title, composer, bpm=bpm, time_sig=time_sig)
    tokens = re.findall(r"(\^|_)?([A-Ga-gz])([',]*)(\d*(?:/\d*)?)", " ".join(body_lines))
    melody = []
    for acc, letter, oct_mod, dur_str in tokens[:320]:
        beats = 1.0
        if dur_str:
            if "/" in dur_str:
                parts = dur_str.split("/")
                num = float(parts[0]) if parts[0] else 1.0
                den = float(parts[1]) if (len(parts) > 1 and parts[1]) else 2.0
                beats = max(0.25, min(4.0, num / max(1.0, den)))
            else:
                beats = max(0.5, min(4.0, float(dur_str)))

        if letter == "z":
            melody.append({"pitch": "REST", "beats": beats, "lyric": ""})
            continue

        base_oct = 5 if letter.islower() else 4
        base_oct += oct_mod.count("'") - oct_mod.count(",")
        acc_sym = "#" if acc == "^" else ("b" if acc == "_" else "")
        melody.append({"pitch": f"{letter.upper()}{acc_sym}{base_oct}", "beats": beats, "lyric": ""})

    score["staves"]["melody"] = melody
    _populate_accompaniment_from_melody(score)
    return score


# ==============================================================================
# 4. MULTI-SYSTEM SHEET IMAGE OMR (.png / .jpg / .jpeg / .webp)
# ==============================================================================
def analyze_sheet_image_omr(image_bytes: bytes, filename: str) -> dict:
    """Cok satirli nota sayfasi gorselinden dizek sistemlerini ve nota baslarini tespit eder."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    # Analiz icin ideal genislige olcekle
    target_w = 1400
    scale = target_w / max(1, img.width)
    target_h = max(200, int(img.height * scale))
    img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
    img = ImageOps.autocontrast(img)

    arr = np.asarray(img, dtype=np.float32)
    thresh = np.percentile(arr, 32)
    binary = (arr < min(185.0, thresh + 25.0)).astype(np.uint8)

    # Yatay projeksiyon ile dizek cizgilerini bul
    row_sums = np.sum(binary, axis=1)
    line_thresh = target_w * 0.36
    candidate_rows = np.where(row_sums >= line_thresh)[0]

    # Bitisik satirlari tek cizgi merkezine indirge
    staff_lines = []
    if len(candidate_rows) > 0:
        grp = [candidate_rows[0]]
        for r in candidate_rows[1:]:
            if r - grp[-1] <= 3:
                grp.append(r)
            else:
                staff_lines.append(float(np.mean(grp)))
                grp = [r]
        staff_lines.append(float(np.mean(grp)))

    # 5'li dizek gruplarini cikar (ustten alta tum sistemler)
    systems = []
    i = 0
    while i + 4 < len(staff_lines):
        five = staff_lines[i:i + 5]
        spacings = np.diff(five)
        med_sp = float(np.median(spacings))
        if 4.0 <= med_sp <= 28.0 and np.max(np.abs(spacings - med_sp)) <= med_sp * 0.45:
            systems.append((five[0], five[-1], med_sp))
            i += 5
        else:
            i += 1

    # Yatay cizgileri temizleyip nota baslarini ortaya cikar (morfolojik acma)
    h_kernel = np.ones((1, 28), dtype=np.uint8)
    staff_mask = ndimage.binary_opening(binary, structure=h_kernel)
    notes_only = np.clip(binary.astype(int) - staff_mask.astype(int), 0, 1).astype(np.uint8)
    # Dikey saplari inceltip dolu/bos nota baslarini belirginlestir
    blob_clean = ndimage.binary_opening(notes_only, structure=np.ones((3, 4), dtype=np.uint8))

    detected_notes = []
    if systems:
        for sys_top, sys_bot, sp in systems:
            pad = sp * 2.5
            y0 = max(0, int(sys_top - pad))
            y1 = min(target_h, int(sys_bot + pad))
            roi = blob_clean[y0:y1, int(target_w * 0.08):int(target_w * 0.96)]
            labeled, n_obj = ndimage.label(roi)
            objs = ndimage.find_objects(labeled)

            sys_events = []
            for idx_o, slc in enumerate(objs):
                if slc is None:
                    continue
                h_box = slc[0].stop - slc[0].start
                w_box = slc[1].stop - slc[1].start
                if (sp * 0.55) <= h_box <= (sp * 2.1) and (sp * 0.65) <= w_box <= (sp * 2.6):
                    cy = y0 + (slc[0].start + slc[0].stop) / 2.0
                    cx = (slc[1].start + slc[1].stop) / 2.0
                    # Ust cizgiden (F5) kac yarim-aralik asagida oldugunu hesapla
                    half_steps_from_top = int(round((cy - sys_top) / (sp / 2.0)))
                    dn = TREBLE_DIATONIC_TOP - half_steps_from_top
                    pitch = _diatonic_to_pitch(dn)
                    sys_events.append((cx, pitch))

            sys_events.sort(key=lambda x: x[0])
            # Ayni dikey hizadaki akorlari tek melodi/armoni olayinda birlestir
            merged_x = []
            for cx, p in sys_events:
                if merged_x and abs(cx - merged_x[-1][0]) < sp * 0.9:
                    merged_x[-1][1].append(p)
                else:
                    merged_x.append((cx, [p]))

            for _, plist in merged_x:
                # En tiz notayi melodi kabul et
                plist_sorted = sorted(plist)
                detected_notes.append({"pitch": plist_sorted[-1], "beats": 1.0, "lyric": ""})

    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    score = _base_score_template(
        title or "Görselden Okunan Partisyon",
        f"Optik Nota Okuyucu ({len(systems)} Dizek Sistemi)",
        bpm=108,
        time_sig="4/4",
        key_sig="Do Majör",
    )

    if len(detected_notes) >= 4:
        score["staves"]["melody"] = detected_notes[:320]
    else:
        # Gorsel cok dusuk cozunurlukluyse kullanicinin duzenleyebilecegi temel sablonu koy
        score["staves"]["melody"] = [
            {"pitch": "C4", "beats": 1.0, "lyric": ""},
            {"pitch": "D4", "beats": 1.0, "lyric": ""},
            {"pitch": "E4", "beats": 1.0, "lyric": ""},
            {"pitch": "F4", "beats": 1.0, "lyric": ""},
            {"pitch": "G4", "beats": 1.0, "lyric": ""},
            {"pitch": "A4", "beats": 1.0, "lyric": ""},
            {"pitch": "G4", "beats": 2.0, "lyric": ""},
        ]

    _populate_accompaniment_from_melody(score)
    return score


# ==============================================================================
# UNIFIED ENTRYPOINT FOR /api/upload-sheet
# ==============================================================================


def _extract_images_from_pdf_bytes(pdf_bytes: bytes) -> list[bytes]:
    """PDF dosyasindan sayfa goruntulerini cikarir (fitz/pdf2image varsa render eder, yoksa gomulu sayfa gorsellerini ayiklar)."""
    pages_png = []
    # 1. PyMuPDF (fitz) varsa tum sayfalari yuksek cozunurlukte render et
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc[:12]:
            pix = page.get_pixmap(dpi=160)
            pages_png.append(pix.tobytes("png"))
        if pages_png:
            return pages_png
    except Exception:
        pass

    # 2. pdf2image varsa render et
    try:
        from pdf2image import convert_from_bytes
        imgs = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=12)
        for im in imgs:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            pages_png.append(buf.getvalue())
        if pages_png:
            return pages_png
    except Exception:
        pass

    # 3. Saf Python PDF binary JPEG (DCTDecode) akisi ayiklayici (ek paket gerektirmez)
    pos = 0
    while True:
        soi = pdf_bytes.find(b"\xff\xd8\xff", pos)
        if soi < 0:
            break
        eoi = pdf_bytes.find(b"\xff\xd9", soi + 3)
        if eoi < 0:
            break
        jpg_blob = pdf_bytes[soi:eoi + 2]
        if len(jpg_blob) > 4096:
            try:
                im = Image.open(io.BytesIO(jpg_blob))
                if im.width >= 300 and im.height >= 300:
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="PNG")
                    pages_png.append(buf.getvalue())
            except Exception:
                pass
        pos = eoi + 2

    return pages_png


def parse_pdf_sheet_bytes(pdf_bytes: bytes, filename: str) -> dict:
    """Cok sayfali PDF nota dosyasini sayfa sayfa okuyup tek uzun partisyonda birlestirir."""
    page_imgs = _extract_images_from_pdf_bytes(pdf_bytes)
    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    if not page_imgs:
        sc = _base_score_template(title or "PDF Nota", "PDF İçe Aktarım (Boş Şablon)")
        sc["staves"]["melody"] = [
            {"pitch": "C4", "beats": 1.0, "lyric": ""},
            {"pitch": "D4", "beats": 1.0, "lyric": ""},
            {"pitch": "E4", "beats": 1.0, "lyric": ""},
            {"pitch": "G4", "beats": 1.0, "lyric": ""},
        ]
        _populate_accompaniment_from_melody(sc)
        return sc

    combined_melody = []
    for p_img in page_imgs:
        sub_score = analyze_sheet_image_omr(p_img, filename)
        combined_melody.extend(sub_score.get("staves", {}).get("melody", []))
        if len(combined_melody) >= 480:
            break

    score = _base_score_template(
        title or "PDF Nota Partisyonu",
        f"PDF Optik Okuyucu ({len(page_imgs)} Sayfa)",
        bpm=108,
        time_sig="4/4",
        key_sig="Do Majör",
    )
    score["staves"]["melody"] = combined_melody[:480]
    _populate_accompaniment_from_melody(score)
    return score


def export_score_to_musicxml(score: dict) -> str:
    """Mevcut partisyonu MuseScore / Finale / Sibelius uyumlu MusicXML formatina donusturur."""
    title = (score.get("title") or "NotaStudyo Eseri").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    composer = (score.get("composer") or "NotaStudyo Maestro Pro").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    bpm = int(score.get("bpm", 112))
    ts = str(score.get("time_signature", "4/4"))
    ts_parts = ts.split("/")
    beats_num = ts_parts[0] if len(ts_parts) == 2 else "4"
    beat_type = ts_parts[1] if len(ts_parts) == 2 else "4"
    divisions = 4  # 1 quarter note = 4 divisions

    melody = score.get("staves", {}).get("melody", []) or [{"pitch": "C4", "beats": 1.0, "lyric": ""}]

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="3.1">',
        f'  <work><work-title>{title}</work-title></work>',
        f'  <identification><creator type="composer">{composer}</creator></identification>',
        '  <part-list><score-part id="P1"><part-name>Melodi</part-name></score-part></part-list>',
        '  <part id="P1">',
        '    <measure number="1">',
        '      <attributes>',
        f'        <divisions>{divisions}</divisions>',
        f'        <time><beats>{beats_num}</beats><beat-type>{beat_type}</beat-type></time>',
        '        <clef><sign>G</sign><line>2</line></clef>',
        '      </attributes>',
        f'      <sound tempo="{bpm}"/>',
    ]

    measure_beats = 0.0
    measure_num = 1
    max_measure_beats = float(beats_num) * (4.0 / max(1.0, float(beat_type)))

    for ev in melody:
        b = float(ev.get("beats", 1.0))
        dur_div = max(1, int(round(b * divisions)))
        note_type = "whole" if b >= 4.0 else ("half" if b >= 2.0 else ("eighth" if b <= 0.5 else "quarter"))
        pitch = ev.get("pitch", "REST")
        lyric = (ev.get("lyric") or "").strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        if measure_beats >= max_measure_beats - 0.01:
            measure_num += 1
            measure_beats = 0.0
            xml_lines.append('    </measure>')
            xml_lines.append(f'    <measure number="{measure_num}">')

        xml_lines.append('      <note>')
        if pitch == "REST":
            xml_lines.append('        <rest/>')
            xml_lines.append(f'        <duration>{dur_div}</duration>')
            xml_lines.append(f'        <type>{note_type}</type>')
        else:
            m = re.match(r"^([A-G])([#b]?)(-?\d+)$", pitch)
            step = m.group(1) if m else "C"
            acc = m.group(2) if m else ""
            octave = m.group(3) if m else "4"
            alter = "1" if acc == "#" else ("-1" if acc == "b" else "")
            xml_lines.append('        <pitch>')
            xml_lines.append(f'          <step>{step}</step>')
            if alter:
                xml_lines.append(f'          <alter>{alter}</alter>')
            xml_lines.append(f'          <octave>{octave}</octave>')
            xml_lines.append('        </pitch>')
            xml_lines.append(f'        <duration>{dur_div}</duration>')
            xml_lines.append(f'        <type>{note_type}</type>')
            if lyric:
                xml_lines.append(f'        <lyric><text>{lyric}</text></lyric>')
        xml_lines.append('      </note>')
        measure_beats += b

    xml_lines.append('    </measure>')
    xml_lines.append('  </part>')
    xml_lines.append('</score-partwise>')
    return "\n".join(xml_lines)


def analyze_sheet_image(file_bytes: bytes, filename: str) -> dict:
    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf" or file_bytes[:4] == b"%PDF":
        return parse_pdf_sheet_bytes(file_bytes, filename)
    if ext in (".mid", ".midi"):
        return parse_midi_bytes(file_bytes, filename)
    if ext in (".xml", ".musicxml", ".mxl"):
        return parse_musicxml_bytes(file_bytes, filename)
    if ext in (".abc", ".txt"):
        return parse_abc_bytes(file_bytes, filename)
    return analyze_sheet_image_omr(file_bytes, filename)
