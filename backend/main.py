from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import asyncio
import json
import re
import time
import uuid
import uvicorn

from audio_engine import render_full_mix, render_mp4_video, convert_wav_to_format, find_fluidsynth, _build_midi
from studio_ai_engine import (
    check_studio_ai_status,
    generate_studio_ai_song_sync,
    RENDER_PROGRESS,
)
from omr_reader import analyze_sheet_image, export_score_to_musicxml

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SCORES_DIR = BASE_DIR / "scores"
EXPORTS_DIR = BASE_DIR / "exports"
TEMP_RENDER_DIR = BASE_DIR / "temp_renders"
SOUNDFONTS_DIR = BASE_DIR / "soundfonts"

for folder in (FRONTEND_DIR, SCORES_DIR, EXPORTS_DIR, TEMP_RENDER_DIR, SOUNDFONTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NotaStüdyo Maestro Pro")


class ScorePayload(BaseModel):
    id: str
    title: str
    composer: str
    bpm: int
    time_signature: str
    key_signature: str
    instruments: dict
    vocals: dict
    staves: dict
    lyrics_text: str = ""
    engine_choice: str = "studio_ai"  # "studio_ai" | "hybrid" | "dsp_sheet"
    quality_steps: int = 10
    target_duration_sec: float = 0
    export_mp4: bool = False


class ExportFormatRequest(BaseModel):
    wav_filename: str
    format: str  # "wav" | "mp3" | "mp4" | "flac" | "ogg"
    score: dict | None = None


def cleanup_temp_renders(keep_filename: str | None = None, max_age_sec: int = 300):
    now = time.time()
    for f in TEMP_RENDER_DIR.iterdir():
        if not f.is_file():
            continue
        if keep_filename and f.name == keep_filename:
            continue
        try:
            age = now - f.stat().st_mtime
            if keep_filename is None or age > max_age_sec:
                f.unlink(missing_ok=True)
        except Exception:
            pass


@app.on_event("startup")
def on_startup_cleanup():
    cleanup_temp_renders(keep_filename=None, max_age_sec=0)


@app.get("/")
def serve_ui():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
def get_status():
    sf_files = list(SOUNDFONTS_DIR.glob("*.sf2"))
    fs_bin = find_fluidsynth()
    ai_st = check_studio_ai_status()
    return {
        "fluidsynth_installed": fs_bin is not None,
        "soundfont_ready": len(sf_files) > 0,
        "soundfont_name": sf_files[0].name if sf_files else None,
        "studio_ai": ai_st,
    }


@app.get("/api/render-progress")
def get_render_progress():
    return RENDER_PROGRESS


@app.get("/api/scores")
def list_scores():
    scores = []
    for file in sorted(SCORES_DIR.glob("*.json")):
        try:
            scores.append(json.loads(file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return {"scores": scores}


@app.post("/api/scores")
def save_score(payload: ScorePayload):
    target = SCORES_DIR / f"{payload.id}.json"
    data = payload.model_dump(exclude={"export_mp4", "engine_choice", "quality_steps"})
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "saved", "id": payload.id}


@app.delete("/api/scores/{score_id}")
def delete_score(score_id: str):
    target = SCORES_DIR / f"{score_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Partisyon bulunamadı")
    target.unlink()
    return {"status": "deleted", "id": score_id}


@app.post("/api/render")
async def render_audio(payload: ScorePayload):
    uid = uuid.uuid4().hex[:6]
    wav_name = f"{payload.id}_{uid}.wav"
    wav_path = TEMP_RENDER_DIR / wav_name

    cleanup_temp_renders(keep_filename=None, max_age_sec=0)

    sf_files = list(SOUNDFONTS_DIR.glob("*.sf2"))
    sf_path = sf_files[0] if sf_files else None
    score_dict = payload.model_dump(exclude={"export_mp4"})
    choice = (payload.engine_choice or "studio_ai").strip()

    mode = ""
    if choice in ("studio_ai", "hybrid"):
        ok, ai_msg = await asyncio.to_thread(
            generate_studio_ai_song_sync,
            score_dict,
            wav_path,
            int(payload.quality_steps or 10),
        )
        if ok and choice == "studio_ai":
            mode = ai_msg
        elif ok and choice == "hybrid":
            import numpy as np
            import soundfile as sf
            dsp_wav = TEMP_RENDER_DIR / f"{payload.id}_{uid}_dsp.wav"
            await render_full_mix(score_dict, dsp_wav, sf_path)
            a_ai, sr_ai = sf.read(str(wav_path), dtype="float32")
            a_dsp, _ = sf.read(str(dsp_wav), dtype="float32")
            dsp_wav.unlink(missing_ok=True)
            L = max(len(a_ai), len(a_dsp))
            mix = np.zeros((L, 2), dtype="float32")
            mix[:len(a_ai)] += a_ai * 0.82
            mix[:len(a_dsp)] += a_dsp * 0.35
            pk = float(np.max(np.abs(mix)))
            if pk > 1e-4:
                mix = (mix / pk) * 0.94
            sf.write(str(wav_path), mix, sr_ai, subtype="PCM_16")
            mode = f"Hibrit ({ai_msg} + Dizek Orkestra)"
        else:
            fallback_mode = await render_full_mix(score_dict, wav_path, sf_path)
            mode = f"{fallback_mode} [Not: {ai_msg}]"
    else:
        mode = await render_full_mix(score_dict, wav_path, sf_path)

    mp4_url = None
    if payload.export_mp4:
        RENDER_PROGRESS.update({
            "active": True,
            "stage": "MP4 Ses Dalgalı Klip Hazırlanıyor...",
            "percent": 95,
            "detail": ""
        })
        mp4_name = f"{payload.id}_{uid}.mp4"
        mp4_path = TEMP_RENDER_DIR / mp4_name
        if await asyncio.to_thread(render_mp4_video, score_dict, wav_path, mp4_path):
            mp4_url = f"/api/temp-media/{mp4_name}?download=1"
        RENDER_PROGRESS.update({"active": False, "stage": "Hazır", "percent": 100, "detail": ""})

    return {
        "status": "ok",
        "wav_filename": wav_name,
        "audio_url": f"/api/temp-media/{wav_name}",
        "download_wav_url": f"/api/temp-media/{wav_name}?download=1",
        "mp4_url": mp4_url,
        "engine_mode": mode,
    }


@app.post("/api/export-format")
async def export_format(req: ExportFormatRequest):
    safe_wav = Path(req.wav_filename).name
    src_wav = TEMP_RENDER_DIR / safe_wav
    if not src_wav.exists() or not src_wav.is_file():
        raise HTTPException(status_code=404, detail="Önce şarkıyı sentezleyin (geçici WAV bulunamadı)")

    fmt = req.format.lower().lstrip(".")
    if fmt not in ("wav", "mp3", "mp4", "flac", "ogg"):
        raise HTTPException(status_code=400, detail="Desteklenmeyen format")

    out_name = f"{src_wav.stem}.{fmt}"
    out_path = TEMP_RENDER_DIR / out_name

    if not out_path.exists():
        ok = await asyncio.to_thread(convert_wav_to_format, src_wav, out_path, fmt, req.score)
        if not ok or not out_path.exists():
            raise HTTPException(status_code=500, detail=f"{fmt.upper()} dönüştürme başarısız oldu")

    return {
        "status": "ok",
        "filename": out_name,
        "file_url": f"/api/temp-media/{out_name}",
        "download_url": f"/api/temp-media/{out_name}?download=1",
    }


@app.get("/api/temp-media/{filename}")
def serve_temp_media(filename: str, download: int = Query(default=0)):
    safe_name = Path(filename).name
    target = TEMP_RENDER_DIR / safe_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Geçici ses/klip dosyası bulunamadı veya temizlendi")
    ext = target.suffix.lower()
    mime_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".mp4": "video/mp4",
        ".mid": "audio/midi",
        ".musicxml": "application/vnd.recordare.musicxml+xml",
    }
    media_type = mime_map.get(ext, "application/octet-stream")
    if download:
        return FileResponse(path=str(target), media_type=media_type, filename=safe_name)
    return FileResponse(path=str(target), media_type=media_type)


@app.post("/api/cleanup-temp")
def manual_cleanup_temp(keep: str | None = None):
    cleanup_temp_renders(keep_filename=keep, max_age_sec=0)
    return {"status": "cleaned"}



class ScoreFileExportRequest(BaseModel):
    format: str  # "mid" | "musicxml"
    score: dict


@app.post("/api/export-score-file")
def export_score_file(req: ScoreFileExportRequest):
    fmt = req.format.lower().lstrip(".")
    uid = uuid.uuid4().hex[:6]
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(req.score.get("id") or "partisyon"))
    if fmt in ("mid", "midi"):
        out_name = f"{safe_id}_{uid}.mid"
        out_path = TEMP_RENDER_DIR / out_name
        _build_midi(req.score, out_path)
        return {"status": "ok", "filename": out_name, "download_url": f"/api/temp-media/{out_name}?download=1"}
    elif fmt in ("xml", "musicxml"):
        out_name = f"{safe_id}_{uid}.musicxml"
        out_path = TEMP_RENDER_DIR / out_name
        xml_str = export_score_to_musicxml(req.score)
        out_path.write_text(xml_str, encoding="utf-8")
        return {"status": "ok", "filename": out_name, "download_url": f"/api/temp-media/{out_name}?download=1"}
    raise HTTPException(status_code=400, detail="Desteklenmeyen nota formatı (mid veya musicxml seçin)")


@app.post("/api/upload-sheet")
async def upload_sheet(file: UploadFile = File(...)):
    raw = await file.read()
    score_data = analyze_sheet_image(raw, file.filename or "Yuklenen_Nota")
    target = SCORES_DIR / f"{score_data['id']}.json"
    target.write_text(json.dumps(score_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "ok", "score": score_data}


app.mount("/exports", StaticFiles(directory=str(EXPORTS_DIR)), name="exports")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8765, reload=True, app_dir=str(Path(__file__).resolve().parent))
