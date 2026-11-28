# Opsiyonel Harici SVS / DiffSinger Kancası (NotaStüdyo)

Eğer sistemine **DiffSinger** veya **OpenUtau** tabanlı bir şarkı söyleme modeli kurarsan,
bu klasöre `run_svs.py` adında bir betik koyduğunda `vocal_synth.py` otomatik olarak
dahili vokal motoru yerine senin harici SVS motorunu çağırır.

Girdi JSON yapısı (`sys.argv[1]`):
- `bpm`: tempo
- `artists`: seçilen vokal profilleri
- `notes`: `[{"beat": 0.0, "beats": 1.0, "hz": 261.63, "lyric": "Üs-"}, ...]`

Çıktı (`sys.argv[2]`):
- 44100 Hz mono/stereo WAV dosyası.
