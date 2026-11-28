import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_CACHE_DIR = BASE_DIR / "models_cache" / "sd_turbo"

INSTRUMENT_TR_NAMES = {
    "violin": "Keman",
    "viola": "Viyola",
    "cello": "Çello",
    "contrabass": "Kontrbas",
    "harp": "Arp",
    "flute": "Flüt",
    "oboe": "Obua",
    "clarinet": "Klarnet",
    "trumpet": "Trompet",
    "horn": "Korno",
    "saxophone": "Saksafon",
    "piano": "Piyano",
    "electric_guitar": "Elektro Gitar",
    "acoustic_guitar": "Akustik Gitar",
    "bass_guitar": "Bas Gitar",
    "oud_baglama": "Ud / Bağlama",
    "drums": "Bateri Seti",
    "timpani": "Timpani",
}

VOCAL_TR_NAMES = {
    "diva_pop": "Pop Diva Vokal",
    "soprano_opera": "Dramatik Soprano",
    "turk_sanat_k": "Türk Sanat (Kadın)",
    "soul_mezzo": "Kadife Mezzo",
    "rock_tenor": "Senfonik Tenor",
    "turk_sanat_e": "Türk Sanat (Erkek)",
    "velvet_bariton": "Kadife Bariton",
    "deep_bass": "Derin Bas-Bariton",
    "female_soprano": "Soprano Vokal",
    "male_tenor": "Tenor Vokal",
}

# Sahne yerlesim koordinatlari (1280x720 tuval uzerinde orkestra oturma duzeni)
STAGE_SLOTS = {
    "drums":           (640, 275, "percussion"),
    "timpani":         (835, 285, "percussion"),
    "horn":            (435, 295, "brass"),
    "trumpet":         (965, 315, "brass"),
    "saxophone":       (1045, 355, "brass"),
    "flute":           (535, 330, "woodwind"),
    "oboe":            (745, 330, "woodwind"),
    "clarinet":        (865, 345, "woodwind"),
    "piano":           (315, 365, "keys"),
    "harp":            (205, 355, "strings"),
    "violin":          (395, 435, "strings"),
    "viola":           (515, 445, "strings"),
    "cello":           (775, 445, "strings"),
    "contrabass":      (945, 415, "strings"),
    "acoustic_guitar": (250, 455, "guitar"),
    "electric_guitar": (1035, 450, "guitar"),
    "bass_guitar":     (1115, 395, "guitar"),
    "oud_baglama":     (470, 385, "guitar"),
}


def _load_pil_font(size: int, bold: bool = False):
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


def build_pure_soundwave_canvas(width: int = 1280, height: int = 720) -> Image.Image:
    """Yazisiz, basliksiz, sadece saf koyu studyo ses dalgasi (soundwave) arka plani."""
    img = Image.new("RGB", (width, height), (5, 8, 17))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)

    # Merkezde yumusak amber ve zumrut ambient isik halesi (sifir yazi)
    for r, alpha in [(420, 14), (300, 20), (190, 26)]:
        gdraw.ellipse(
            [width // 2 - r, height // 2 - r // 2, width // 2 + r, height // 2 + r // 2],
            fill=(245, 158, 11, alpha)
        )
    for r, alpha in [(340, 10), (210, 16)]:
        gdraw.ellipse(
            [width // 2 - r, height // 2 - r // 3, width // 2 + r, height // 2 + r // 3],
            fill=(16, 185, 129, alpha)
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=42))
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")

    draw = ImageDraw.Draw(img)
    # Ince yatay referans merkez cizgisi
    mid_y = height // 2
    draw.line([(70, mid_y), (width - 70, mid_y)], fill=(30, 41, 59), width=1)
    return img


def _draw_instrument_silhouette(draw: ImageDraw.ImageDraw, inst_key: str, cx: int, cy: int, scale: float = 1.0):
    """Sahne uzerinde ilgili enstrumanin siluetini cizer."""
    gold = (251, 191, 36)
    amber = (217, 119, 6)
    wood = (180, 83, 9)
    dark = (15, 23, 42)
    steel = (203, 213, 225)
    s = scale

    if inst_key == "piano":
        # Kuyruklu konser piyanosu
        draw.polygon([
            (cx - int(48*s), cy - int(10*s)),
            (cx + int(42*s), cy - int(22*s)),
            (cx + int(52*s), cy + int(8*s)),
            (cx - int(44*s), cy + int(12*s)),
        ], fill=(17, 24, 39), outline=gold)
        # Acik piyano kapagi
        draw.polygon([
            (cx - int(44*s), cy - int(10*s)),
            (cx + int(36*s), cy - int(44*s)),
            (cx + int(42*s), cy - int(22*s)),
        ], fill=(30, 41, 59), outline=amber)
        # Ayaklar
        for lx in (-38, 12, 42):
            draw.line([(cx + int(lx*s), cy + int(10*s)), (cx + int(lx*s), cy + int(34*s))], fill=gold, width=3)

    elif inst_key == "drums":
        # Kick davul + ziller + tomlar
        draw.ellipse([cx - int(22*s), cy - int(12*s), cx + int(22*s), cy + int(30*s)], fill=(30, 41, 59), outline=gold, width=3)
        draw.ellipse([cx - int(15*s), cy - int(5*s), cx + int(15*s), cy + int(23*s)], fill=(248, 250, 252), outline=amber, width=2)
        draw.rectangle([cx - int(36*s), cy - int(14*s), cx - int(18*s), cy + int(2*s)], fill=wood, outline=gold)
        draw.rectangle([cx + int(18*s), cy - int(14*s), cx + int(36*s), cy + int(2*s)], fill=wood, outline=gold)
        # Ziller
        draw.line([(cx - int(44*s), cy + int(28*s)), (cx - int(44*s), cy - int(26*s))], fill=steel, width=2)
        draw.ellipse([cx - int(58*s), cy - int(30*s), cx - int(30*s), cy - int(23*s)], fill=gold)
        draw.line([(cx + int(44*s), cy + int(28*s)), (cx + int(44*s), cy - int(24*s))], fill=steel, width=2)
        draw.ellipse([cx + int(30*s), cy - int(28*s), cx + int(58*s), cy - int(21*s)], fill=gold)

    elif inst_key == "timpani":
        for dx in (-24, 24):
            draw.pieslice([cx + int((dx-20)*s), cy - int(10*s), cx + int((dx+20)*s), cy + int(28*s)], 0, 180, fill=amber, outline=gold, width=2)
            draw.ellipse([cx + int((dx-20)*s), cy - int(14*s), cx + int((dx+20)*s), cy - int(2*s)], fill=(241, 245, 249), outline=gold)
            draw.line([(cx + int((dx-12)*s), cy + int(18*s)), (cx + int((dx-16)*s), cy + int(32*s))], fill=steel, width=2)
            draw.line([(cx + int((dx+12)*s), cy + int(18*s)), (cx + int((dx+16)*s), cy + int(32*s))], fill=steel, width=2)

    elif inst_key == "harp":
        draw.polygon([
            (cx - int(18*s), cy + int(30*s)),
            (cx - int(18*s), cy - int(36*s)),
            (cx + int(22*s), cy - int(24*s)),
            (cx + int(4*s), cy + int(30*s)),
        ], outline=gold, fill=(120, 53, 15))
        for sx in range(-12, 14, 5):
            draw.line([(cx + int(sx*s), cy - int(26*s)), (cx + int((sx//2)*s), cy + int(26*s))], fill=(253, 230, 138), width=1)

    elif inst_key in ("violin", "viola"):
        # Yayli govde + arşe
        draw.ellipse([cx - int(10*s), cy - int(6*s), cx + int(10*s), cy + int(14*s)], fill=wood, outline=gold, width=2)
        draw.ellipse([cx - int(8*s), cy - int(18*s), cx + int(8*s), cy - int(2*s)], fill=wood, outline=gold, width=2)
        draw.line([(cx, cy - int(18*s)), (cx, cy - int(36*s))], fill=gold, width=3)
        draw.line([(cx - int(24*s), cy - int(4*s)), (cx + int(26*s), cy - int(12*s))], fill=steel, width=2)

    elif inst_key in ("cello", "contrabass"):
        h_mul = 1.25 if inst_key == "contrabass" else 1.05
        draw.ellipse([cx - int(15*s), cy - int(8*s), cx + int(15*s), cy + int(22*s*h_mul)], fill=wood, outline=gold, width=2)
        draw.ellipse([cx - int(12*s), cy - int(24*s*h_mul), cx + int(12*s), cy - int(2*s)], fill=wood, outline=gold, width=2)
        draw.line([(cx, cy - int(24*s*h_mul)), (cx, cy - int(44*s*h_mul))], fill=gold, width=3)
        draw.line([(cx, cy + int(22*s*h_mul)), (cx, cy + int(34*s*h_mul))], fill=steel, width=2)

    elif inst_key in ("acoustic_guitar", "electric_guitar", "bass_guitar", "oud_baglama"):
        body_col = (220, 38, 38) if inst_key == "electric_guitar" else (37, 99, 235) if inst_key == "bass_guitar" else wood
        draw.ellipse([cx - int(15*s), cy - int(4*s), cx + int(15*s), cy + int(22*s)], fill=body_col, outline=gold, width=2)
        if inst_key != "oud_baglama":
            draw.ellipse([cx - int(11*s), cy - int(15*s), cx + int(11*s), cy + int(2*s)], fill=body_col, outline=gold, width=2)
        draw.line([(cx + int(6*s), cy - int(12*s)), (cx + int(28*s), cy - int(34*s))], fill=gold, width=4)

    else:
        # Nefesli / Bakir (Flut, Obua, Klarnet, Trompet, Korno, Saksafon)
        draw.ellipse([cx - int(10*s), cy - int(32*s), cx + int(10*s), cy - int(14*s)], fill=steel)
        draw.rectangle([cx - int(9*s), cy - int(12*s), cx + int(9*s), cy + int(16*s)], fill=dark, outline=gold)
        draw.line([(cx + int(6*s), cy - int(16*s)), (cx + int(32*s), cy - int(22*s))], fill=gold, width=4)
        draw.polygon([
            (cx + int(30*s), cy - int(28*s)),
            (cx + int(42*s), cy - int(32*s)),
            (cx + int(42*s), cy - int(12*s)),
            (cx + int(30*s), cy - int(16*s)),
        ], fill=amber)


def _draw_vocalist_silhouette(draw: ImageDraw.ImageDraw, cx: int, cy: int, label: str, count: int):
    """Sahne onunde mikrofon basinda solist silueti ve spot isigi cizer."""
    gold = (251, 191, 36)
    # Solist silueti
    draw.ellipse([cx - 13, cy - 58, cx + 13, cy - 32], fill=(248, 250, 252), outline=gold, width=2)
    draw.polygon([
        (cx - 20, cy + 16),
        (cx - 15, cy - 28),
        (cx + 15, cy - 28),
        (cx + 20, cy + 16),
    ], fill=(15, 23, 42), outline=gold)
    # Mikrofon standi
    draw.line([(cx + 24, cy + 18), (cx + 24, cy - 34)], fill=(203, 213, 225), width=3)
    draw.ellipse([cx + 19, cy - 41, cx + 29, cy - 31], fill=gold)


def build_concert_hall_stage_image(
    score: dict,
    width: int = 1280,
    height: int = 720,
    cinematic_ai_style: bool = False,
    custom_prompt: str = "",
) -> Image.Image:
    """
    Konser salonu ve kullanicinin havuzda sectigi aktif enstrumanlar + vokalistlerden olusan
    sahne gorseli uretir. Kesinlikle nota sablon basligini (Turk Marsi vb.) yazmaz.
    """
    img = Image.new("RGBA", (width, height), (8, 10, 18, 255))
    draw = ImageDraw.Draw(img)

    # 1. Konser salonu arka plani (sicak akustik ahsap tonlari & derin sahne derinligi)
    for y in range(height):
        t = y / float(height)
        if cinematic_ai_style:
            r = int(14 + 26 * (1 - t))
            g = int(10 + 18 * (1 - t))
            b = int(28 + 34 * (1 - t))
        else:
            r = int(18 + 24 * (1 - t))
            g = int(12 + 14 * (1 - t))
            b = int(10 + 14 * (1 - t))
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # Akustik ahsap duvar panelleri ve kemerli konser salonu tavan cizgileri
    for arc_idx in range(5):
        pad_x = 60 + arc_idx * 45
        pad_y = 20 + arc_idx * 24
        draw.arc(
            [pad_x, pad_y, width - pad_x, height - 80],
            start=185, end=355,
            fill=(120, 72, 32, 95 if not cinematic_ai_style else 75),
            width=3
        )

    # Sahne zemini (perspektif ahsap konser platformu)
    stage_top_y = 250
    stage_bot_y = 565
    draw.polygon([
        (140, stage_top_y),
        (width - 140, stage_top_y),
        (width - 40, stage_bot_y),
        (40, stage_bot_y),
    ], fill=(24, 18, 16, 255), outline=(146, 94, 42, 180))

    # Sahne basamaklari / orkestra yukseltileri
    draw.ellipse([240, 235, width - 240, 360], fill=(32, 23, 20, 220), outline=(180, 120, 52, 120))
    draw.ellipse([150, 315, width - 150, 515], fill=(28, 20, 18, 210), outline=(217, 119, 6, 110))

    # 2. Ustten inen konser sahne spot isiklari (volumetric spotlight beams)
    spot_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(spot_layer)
    spot_xs = [260, 460, 640, 820, 1020]
    for idx_s, sx in enumerate(spot_xs):
        beam_col = (245, 158, 11, 26) if idx_s % 2 == 0 else (56, 189, 248, 20) if cinematic_ai_style else (251, 191, 36, 22)
        sdraw.polygon([
            (sx - 18, 0),
            (sx + 18, 0),
            (sx + 165, 510),
            (sx - 165, 510),
        ], fill=beam_col)
    spot_layer = spot_layer.filter(ImageFilter.GaussianBlur(radius=18))
    img = Image.alpha_composite(img, spot_layer)
    draw = ImageDraw.Draw(img)

    # Yan kadife sahne perdeleri
    draw.polygon([(0, 0), (145, 0), (95, 565), (0, 565)], fill=(69, 10, 22, 245))
    draw.polygon([(width, 0), (width - 145, 0), (width - 95, 565), (width, 565)], fill=(69, 10, 22, 245))

    # 3. Havuzdaki aktif enstrumanlari sahneye yerlestir
    instruments = (score or {}).get("instruments") or {}
    active_insts = [(k, int(v.get("count", 0))) for k, v in instruments.items() if int(v.get("count", 0)) > 0]
    if not active_insts:
        active_insts = [("piano", 1), ("violin", 2)]

    font_tag = _load_pil_font(13, bold=True)

    # Arka siradan on siraya dogru ciz (y koordinatina gore sirala)
    slotted = []
    for k, cnt in active_insts:
        cx, cy, grp = STAGE_SLOTS.get(k, (640, 390, "strings"))
        slotted.append((cy, cx, k, cnt))
    slotted.sort(key=lambda item: item[0])

    for cy, cx, k, cnt in slotted:
        # Enstruman altinda sicak sahne isik halkasi
        draw.ellipse([cx - 52, cy + 16, cx + 52, cy + 38], fill=(245, 158, 11, 38), outline=(245, 158, 11, 110))
        scale = 1.12 if cnt >= 3 else 0.96
        _draw_instrument_silhouette(draw, k, cx, cy, scale=scale)
        lbl = f"{cnt}x {INSTRUMENT_TR_NAMES.get(k, k)}"
        bbox = draw.textbbox((0, 0), lbl, font=font_tag)
        tw = bbox[2] - bbox[0]
        draw.rounded_rectangle(
            [cx - tw // 2 - 7, cy + 34, cx + tw // 2 + 7, cy + 53],
            radius=6,
            fill=(10, 15, 28, 215),
            outline=(245, 158, 11, 140),
        )
        draw.text((cx - tw // 2, cy + 36), lbl, fill=(253, 230, 138, 255), font=font_tag)

    # 4. Aktif vokalist(ler) varsa sahne on orta spotuna yerlestir
    vocals = (score or {}).get("vocals") or {}
    active_vocs = [(k, int(cnt)) for k, cnt in vocals.items() if int(cnt or 0) > 0]
    if active_vocs:
        n_v = min(3, len(active_vocs))
        v_xs = [640] if n_v == 1 else ([565, 715] if n_v == 2 else [510, 640, 770])
        for idx_v, (vk, vcnt) in enumerate(active_vocs[:3]):
            vx = v_xs[idx_v]
            vy = 485
            draw.ellipse([vx - 58, vy + 8, vx + 58, vy + 32], fill=(16, 185, 129, 45), outline=(16, 185, 129, 150))
            _draw_vocalist_silhouette(draw, vx, vy, vk, vcnt)
            vlbl = f"{vcnt}x {VOCAL_TR_NAMES.get(vk, 'Solist')}"
            bbox = draw.textbbox((0, 0), vlbl, font=font_tag)
            tw = bbox[2] - bbox[0]
            draw.rounded_rectangle(
                [vx - tw // 2 - 8, vy + 24, vx + tw // 2 + 8, vy + 43],
                radius=6,
                fill=(6, 24, 20, 225),
                outline=(16, 185, 129, 170),
            )
            draw.text((vx - tw // 2, vy + 26), vlbl, fill=(167, 243, 208, 255), font=font_tag)

    # 5. Sahne on kenari (alt bolgede soundwave alani icin koyu orkestra cukuru)
    draw.rectangle([0, 565, width, height], fill=(5, 8, 16, 250))
    draw.line([(40, 565), (width - 40, 565)], fill=(245, 158, 11, 180), width=2)

    return img.convert("RGB")


def _try_generate_local_ai_image(score: dict, custom_prompt: str, out_png: Path) -> bool:
    """Eger yerel ortama Diffusers SD-Turbo yukluyse gercek AI sahne gorseli uretir."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        if not any(MODELS_CACHE_DIR.rglob("*.safetensors")):
            return False
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=dtype,
            cache_dir=str(MODELS_CACHE_DIR),
        ).to(device)
        instruments = (score or {}).get("instruments") or {}
        active_names = [k.replace("_", " ") for k, v in instruments.items() if int(v.get("count", 0)) > 0]
        inst_str = ", ".join(active_names[:6]) if active_names else "grand piano and orchestra"
        prompt = custom_prompt.strip() or (
            f"Cinematic grand symphony concert hall stage with {inst_str}, dramatic golden volumetric spotlights, "
            f"dark atmospheric theater, ultra detailed, 8k photography, no text, no watermark"
        )
        img = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0, width=768, height=432).images[0]
        img = img.resize((1280, 720), Image.Resampling.LANCZOS)
        img.save(out_png)
        return out_png.exists()
    except Exception:
        return False


def generate_stage_preview_png(score: dict, clip_mode: str, custom_prompt: str, out_png: Path) -> bool:
    mode = (clip_mode or "soundwave").strip()
    if mode == "concert_hall":
        img = build_concert_hall_stage_image(score, cinematic_ai_style=False, custom_prompt=custom_prompt)
        img.save(out_png)
        return True
    if mode == "ai_clip":
        if _try_generate_local_ai_image(score, custom_prompt, out_png):
            return True
        img = build_concert_hall_stage_image(score, cinematic_ai_style=True, custom_prompt=custom_prompt)
        img.save(out_png)
        return True
    img = build_pure_soundwave_canvas()
    img.save(out_png)
    return True


def render_studio_clip_mp4(
    score: dict,
    wav_path: Path,
    mp4_path: Path,
    temp_dir: Path | None = None,
) -> bool:
    """
    Klip moduna gore MP4 video uretir:
    - 'soundwave' (varsayilan): Sadece ses dalgasi videosu, hicbir nota basligi veya yazi koymaz.
    - 'concert_hall': Konser salonu ve aktif enstruman/vokalist sahnesi + ses dalgasi.
    - 'ai_clip': Sinematik AI sahne gorseli + yavas kamera hareketi + ses dalgasi.
    - 'custom_upload': Kullanicinin yukledigi video (.mp4/.mov/.webm) veya gorsel (.png/.jpg).
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        clip_mode = str((score or {}).get("clip_mode") or "soundwave").strip()
        clip_file_id = Path(str((score or {}).get("clip_file_id") or "")).name
        clip_prompt = str((score or {}).get("clip_prompt") or "").strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. Kullanici kendi video veya gorsel klibini yuklediyse:
            if clip_mode == "custom_upload" and clip_file_id and temp_dir:
                uploaded_path = temp_dir / clip_file_id
                if uploaded_path.exists() and uploaded_path.is_file():
                    ext = uploaded_path.suffix.lower()
                    if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
                        cmd = [
                            ffmpeg_exe, "-y",
                            "-stream_loop", "-1", "-i", str(uploaded_path),
                            "-i", str(wav_path),
                            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=#050811,setsar=1",
                            "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "libx264", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                            "-shortest", str(mp4_path)
                        ]
                        res = subprocess.run(cmd, capture_output=True, timeout=240)
                        if res.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 1000:
                            return True
                    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
                        cover_png = tmp / "custom_cover.png"
                        user_img = Image.open(uploaded_path).convert("RGB")
                        user_img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                        canvas = Image.new("RGB", (1280, 720), (5, 8, 17))
                        ox = (1280 - user_img.width) // 2
                        oy = (720 - user_img.height) // 2
                        canvas.paste(user_img, (ox, oy))
                        canvas.save(cover_png)
                        filter_complex = (
                            "[1:a]showwaves=s=1160x130:mode=cline:rate=25:colors=0xf59e0b|0x10b981[waves];"
                            "[0:v][waves]overlay=60:565[outv]"
                        )
                        cmd = [
                            ffmpeg_exe, "-y",
                            "-loop", "1", "-i", str(cover_png),
                            "-i", str(wav_path),
                            "-filter_complex", filter_complex,
                            "-map", "[outv]", "-map", "1:a",
                            "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                            "-shortest", str(mp4_path)
                        ]
                        res = subprocess.run(cmd, capture_output=True, timeout=240)
                        if res.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 1000:
                            return True

            # 2. Konser Salonu veya AI Sahne Klibi
            if clip_mode in ("concert_hall", "ai_clip"):
                stage_png = tmp / "stage.png"
                generate_stage_preview_png(score, clip_mode, clip_prompt, stage_png)
                if clip_mode == "ai_clip":
                    # Sinematik hafif yakinlasma hareketi + sahne alti soundwave
                    filter_complex = (
                        "[0:v]zoompan=z='min(zoom+0.0004,1.08)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720:fps=25[bg];"
                        "[1:a]showwaves=s=1160x130:mode=cline:rate=25:colors=0xf59e0b|0x38bdf8[waves];"
                        "[bg][waves]overlay=60:572[outv]"
                    )
                    cmd = [
                        ffmpeg_exe, "-y",
                        "-loop", "1", "-i", str(stage_png),
                        "-i", str(wav_path),
                        "-filter_complex", filter_complex,
                        "-map", "[outv]", "-map", "1:a",
                        "-c:v", "libx264", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(mp4_path)
                    ]
                else:
                    filter_complex = (
                        "[1:a]showwaves=s=1160x130:mode=cline:rate=25:colors=0xf59e0b|0x10b981[waves];"
                        "[0:v][waves]overlay=60:572[outv]"
                    )
                    cmd = [
                        ffmpeg_exe, "-y",
                        "-loop", "1", "-i", str(stage_png),
                        "-i", str(wav_path),
                        "-filter_complex", filter_complex,
                        "-map", "[outv]", "-map", "1:a",
                        "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", str(mp4_path)
                    ]
                res = subprocess.run(cmd, capture_output=True, timeout=240)
                return res.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 1000

            # 3. Varsayilan: Sadece Saf Soundwave Videosu (Sifir yazi, sifir nota basligi)
            wave_bg = tmp / "soundwave_bg.png"
            build_pure_soundwave_canvas().save(wave_bg)
            filter_complex = (
                "[1:a]showwaves=s=1160x280:mode=cline:rate=25:colors=0xf59e0b|0x10b981[waves];"
                "[0:v][waves]overlay=60:220[outv]"
            )
            cmd = [
                ffmpeg_exe, "-y",
                "-loop", "1", "-i", str(wave_bg),
                "-i", str(wav_path),
                "-filter_complex", filter_complex,
                "-map", "[outv]", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-shortest", str(mp4_path)
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=240)
            return res.returncode == 0 and mp4_path.exists() and mp4_path.stat().st_size > 1000
    except Exception as exc:
        print(f"[MP4 Clip Render] Hata: {exc}")
        return False
