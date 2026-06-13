# Video narrado de la defensa

Genera un video (español) que recorre las 25 diapositivas con narración por voz.

```bash
python video/make_video.py                       # Kokoro neural, voz em_santa (es)
VOICE=ef_dora python video/make_video.py         # otra voz Kokoro (femenina)
TTS=say VOICE=Paulina python video/make_video.py # fallback macOS `say`
```

Voz por defecto: **Kokoro** (TTS neural local del toolkit `lnm_animations`), voz
`em_santa` en español. `synth_kokoro.py` carga el modelo una vez y sintetiza las 30
líneas. Si Kokoro no está disponible, usa `TTS=say` (voces de macOS).

**Cadena:** `slides/latex/defensa_winoground.pdf` → PNG por página (`pdftoppm`) ·
narración de `video/narration.py` → audio con `say` de macOS → segmentos con
`ffmpeg` → `video/defensa_winoground_video.mp4`.

- `narration.py`: el texto hablado, una entrada por página del PDF (30).
- El `.mp4` y `video/build/` están en `.gitignore` (archivos pesados); se regeneran
  con el comando de arriba. Requiere macOS (`say`), `ffmpeg` y `pdftoppm`.
