# Video narrado de la defensa

Genera un video (español) que recorre las 25 diapositivas con narración por voz.

```bash
python video/make_video.py            # voz Paulina (es_MX) por defecto
VOICE=Monica python video/make_video.py   # voz Mónica (es_ES)
```

**Cadena:** `slides/latex/defensa_winoground.pdf` → PNG por página (`pdftoppm`) ·
narración de `video/narration.py` → audio con `say` de macOS → segmentos con
`ffmpeg` → `video/defensa_winoground_video.mp4`.

- `narration.py`: el texto hablado, una entrada por página del PDF (30).
- El `.mp4` y `video/build/` están en `.gitignore` (archivos pesados); se regeneran
  con el comando de arriba. Requiere macOS (`say`), `ffmpeg` y `pdftoppm`.
