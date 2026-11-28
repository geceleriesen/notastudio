# 𝄞 NotaStüdyo Maestro Pro (DAW Edition)

**Open-Source & Free AI Music Production, Sheet Editor & Vocal Studio**  
Created by **geceleriesen** • Licensed under MIT

NotaStüdyo Maestro Pro; yerel **ACE-Step 1.5** yapay zeka müzik/vokal motorunu ve 4 kanallı interaktif nota kağıdını (`Melodi`, `Armoni`, `Bas`, `Ritim`) tek bir stüdyo arayüzünde birleştiren, tamamen ücretsiz ve aboneliksiz bir müzik prodüksiyon çalışma istasyonudur.

---

## ✨ Özellikler
- **Yerel Stüdyo AI Beste & Vokal (`ACE-Step 1.5`):** GPU veya CPU üzerinde tamamen kendi bilgisayarınızda çok enstrümanlı şarkılar ve çok dilli vokaller üretin.
- **4 Kanallı İnteraktif Nota Kağıdı:** Sol tık ile nota/ritim ekleme, sağ tık ile silme, klavye kısayolları (`C D E F G A B`, `R` es, `Space` oynat/duraklat).
- **Çok Formatlı Nota Okuma (OMR) & Dışa Aktarma:** Çok satırlı nota görselleri (`.png`, `.jpg`), çok sayfalı `.pdf`, `.mid`, `.musicxml`, `.mxl`, `.abc` içe aktarma ve `.MID` / `.XML` indirme.
- **Uzun Şarkı Birleştirici:** 30 saniyeden 4 dakikaya kadar otomatik bölüm ayrımı ve pürüzsüz crossfade birleştirme.
- **Kayıpsız Ses & Video Klip Çıktısı:** `.WAV`, `.MP3`, `.FLAC`, `.OGG` ve ses dalgalı `.MP4` video klip dışa aktarımı.

---

## 🚀 Kurulum & Çalıştırma

> **📦 Yapay Zeka Modeli Hakkında Önemli Not (~4.8 GB):**  
> Depo boyutunu küçük tutmak için **ACE-Step 1.5** yapay zeka model ağırlıkları bu GitHub deposuna **dahil edilmemiştir**. Kurulumdan sonra ilk kez **"Sentezle & Çal"** düğmesine bastığınızda model dosyaları Hugging Face üzerinden **otomatik olarak bir kez** `models_cache/ace_step_15/` klasörüne indirilir ve sonraki tüm kullanımlarda internet olmadan yerel olarak çalışır.  
> *(Modeli indirmeden hemen denemek isterseniz üst menüden **`🎼 Dizek Nota (DSP)`** motorunu seçerek anında sentez yapabilirsiniz.)*

### 1. Depoyu İndirin
```bash
git clone https://github.com/geceleriesen/notastudio.git
cd notastudio
```

### 2. Sanal Ortam ve Kütüphaneleri Kurun (Windows)
Proje klasöründeki **`install_studio_ai.bat`** dosyasına çift tıklayabilir veya terminalden şu komutları çalıştırabilirsiniz:
```bash
python -m venv venv
.env\Scriptsctivate
pip install -r requirements.txt
pip install torch torchaudio
```
*(NVIDIA ekran kartınız varsa CUDA destekli PyTorch kurmanız AI sentezini 10-20 kat hızlandırır.)*

### 3. Stüdyoyu Başlatın
```bash
.env\Scripts\python.exe backend\main.py
```
Ardından tarayıcınızda **`http://127.0.0.1:8765`** adresini açın.

---

## 📁 Proje Klasör Yapısı
```text
notastudio/
├── backend/              # FastAPI sunucusu, OMR nota okuyucu, DSP & ACE-Step AI motorları
├── frontend/             # DAW Stüdyo arayüzü (index.html)
├── scores/               # Hazır başlangıç partisyonları (JSON)
├── svs/                  # Vokal sentezleyici notları
├── install_studio_ai.bat # Tek tıkla Windows kurulum betiği
├── requirements.txt      # Python bağımlılıkları
├── LICENSE               # MIT Lisansı
└── README.md             # Proje dokümantasyonu
```
