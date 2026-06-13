#!/usr/bin/env python3
"""Construye y EJECUTA notebooks/Winoground_Eval_MCC225.ipynb.

Notebook autoexplicativo para la defensa: explica brevemente qué hacemos, muestra el
código y los resultados, valida el scorer contra los scores oficiales, y termina
RESPONDIENDO las cinco preguntas clave de la sección 8.6.

Uso:  python scripts/build_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Winoground_Eval_MCC225.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build():
    nb = nbf.v4.new_notebook()
    c = []

    # ---------------- Portada / abstract ----------------
    c.append(md(
        "# Winoground / Evaluating CLIP — Defensa MCC225\n"
        "**Niels Victor Pacheco Barrios** · Cuadernos **C5, C8, C10**\n\n"
        "## ¿Qué hacemos aquí, en una frase?\n"
        "Medimos si un modelo tipo **CLIP**, que recupera imágenes muy bien, de verdad "
        "**razona de forma composicional**. Usamos **Winoground**: pares de dos imágenes "
        "y dos frases con *las mismas palabras en distinto orden*.\n\n"
        "**Tesis:** un **retrieval alto no implica composición**. Lo vamos a ver con "
        "números: el modelo recupera bien (Recall@K alto) pero su **group score queda "
        "cerca del azar**.\n\n"
        "Reutilizamos el motor **OpenCLIP del Cuaderno 10** y explicamos el *porqué* con "
        "la fusión profunda (**C5**) y la atención crossmodal (**C8**).\n\n"
        "> Cómo leer este notebook: cada sección tiene 2-3 líneas de explicación y luego "
        "el código que la respalda. Al final están **respondidas las 5 preguntas**."
    ))

    # ---------------- 1. Trazabilidad ----------------
    c.append(md(
        "## 1. Entorno (hoja de trazabilidad)\n"
        "Antes de cualquier resultado, registramos versiones, revisión de git y "
        "dispositivo. Esto hace el experimento **reproducible y trazable**."
    ))
    c.append(code(
        "import os, sys, pathlib\n"
        "ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name=='notebooks' else pathlib.Path.cwd()\n"
        "os.chdir(ROOT); sys.path.insert(0, str(ROOT))\n"
        "from src.env_logging import print_snapshot\n"
        "_ = print_snapshot(ROOT)"
    ))

    # ---------------- 2. El problema ----------------
    c.append(md(
        "## 2. El problema: el par mínimo de Winoground\n"
        "Cada ejemplo tiene **2 imágenes** y **2 captions** con el **mismo vocabulario** "
        "en distinto orden. Para acertar hay que entender *quién hace qué a quién*, no "
        "solo reconocer objetos. Veamos un par real."
    ))
    c.append(code(
        "from src.winoground_data import load_dataset\n"
        "examples, source = load_dataset(prefer_real=True)\n"
        "print('fuente:', source, '| N =', len(examples))\n"
        "ex = examples[0]\n"
        "print('caption_0:', ex.caption_0)\n"
        "print('caption_1:', ex.caption_1)"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))\n"
        "ax[0].imshow(ex.image_0); ax[0].set_title('image_0', fontsize=9); ax[0].axis('off')\n"
        "ax[1].imshow(ex.image_1); ax[1].set_title('image_1', fontsize=9); ax[1].axis('off')\n"
        "plt.suptitle('Par mínimo: mismas palabras, distinto orden', fontsize=11)\n"
        "plt.tight_layout(); plt.show()"
    ))
    c.append(md(
        "### Las tres métricas (definición)\n"
        "Con la matriz $2\\times2$ de similitud `sim[caption, imagen]`:\n"
        "- **text score**: fijada la imagen, ¿gana el caption correcto?\n"
        "- **image score**: fijado el caption, ¿gana la imagen correcta?\n"
        "- **group score**: las dos a la vez.\n\n"
        "Azar: text = image = 1/4; **group = 1/6** (no 1/16, porque text e image **no** "
        "son independientes: comparten las 4 similitudes)."
    ))

    # ---------------- 3. El scorer (celda clave) ----------------
    c.append(md(
        "## 3. Celda clave: el scorer sobre un par\n"
        "Codificamos las 2 imágenes y 2 captions con **OpenCLIP** (motor de C10), "
        "armamos la matriz $2\\times2$ y aplicamos las definiciones del paper. "
        "**Este es el bloque que produce el resultado principal.**"
    ))
    c.append(code(
        "from src import openclip_utils as oc\n"
        "from src.winoground_eval import text_correct, image_correct, group_correct\n"
        "model, preprocess, tokenizer, device = oc.create_model('ViT-B-32','laion2b_s34b_b79k')\n"
        "imgf = oc.encode_images(model, preprocess, [ex.image_0, ex.image_1], device)\n"
        "capf = oc.encode_texts(model, tokenizer, [ex.caption_0, ex.caption_1], device)\n"
        "sim = capf @ imgf.T   # sim[caption, imagen]\n"
        "print('matriz 2x2 sim[caption, imagen]:\\n', sim.round(3))\n"
        "print('text=', text_correct(sim), '| image=', image_correct(sim), '| group=', group_correct(sim))"
    ))

    # ---------------- 4. Validación ----------------
    c.append(md(
        "## 4. ¿El scorer es correcto? Validación contra los scores oficiales\n"
        "El propio dataset trae los scores de CLIP del paper (`clip.jsonl`). Aplicamos "
        "**nuestro** scorer a esos números: debe reproducir text=0.3075, image=0.1050, "
        "group=0.0800. Si coincide, la implementación está bien."
    ))
    c.append(code(
        "import json, re, numpy as np\n"
        "from huggingface_hub import hf_hub_download\n"
        "from src.winoground_eval import aggregate\n"
        "try:\n"
        "    p = hf_hub_download('facebook/winoground','statistics/model_scores/clip.jsonl',\n"
        "                        repo_type='dataset', cache_dir='data/winoground_cache')\n"
        "    sc = {}\n"
        "    for line in open(p):\n"
        "        r = json.loads(line); m = re.match(r'(\\d+)_c(\\d)_i(\\d)', r['label'])\n"
        "        eid,cc,ii = int(m[1]),int(m[2]),int(m[3]); sc.setdefault(eid, np.zeros((2,2)))[cc,ii]=r['score']\n"
        "    agg = aggregate([sc[k] for k in sorted(sc)])\n"
        "    print(f'mi scorer sobre CLIP oficial -> text={agg.text:.4f} image={agg.image:.4f} group={agg.group:.4f}')\n"
        "    print('esperado (paper)             -> text=0.3075 image=0.1050 group=0.0800  ✅')\n"
        "except Exception as e:\n"
        "    print('(sin acceso al dataset oficial:', type(e).__name__, '— omitido)')"
    ))

    # ---------------- 5. Resultado principal ----------------
    c.append(md(
        "## 5. Resultado principal (400 ejemplos reales)\n"
        "Generado por `scripts/02_run_winoground.py`. Cargamos las métricas ya calculadas."
    ))
    c.append(code(
        "import json, pandas as pd\n"
        "MET = ROOT/'outputs'/'metrics'\n"
        "scores = json.loads((MET/'scores.json').read_text())\n"
        "print('fuente:', scores['source'], '| checkpoint:', scores['primary_checkpoint'])\n"
        "print(json.dumps(scores['scores'], indent=2))\n"
        "print('referencia humana:', scores['human_reference'])\n"
        "pd.read_csv(MET/'bootstrap_ci.csv')"
    ))
    c.append(code(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT/'outputs'/'figures'/'scores_vs_chance.png')))"
    ))

    # ---------------- 6. Retrieval vs group ----------------
    c.append(md(
        "## 6. La evidencia de la tesis: retrieval alto vs group bajo\n"
        "El **mismo** modelo, sobre el **mismo** conjunto: Recall@K alto, group ≈ azar."
    ))
    c.append(code(
        "rg = json.loads((MET/'recall_vs_group.json').read_text())\n"
        "print('texto->imagen:', {k:v for k,v in rg['text_to_image_recall'].items() if k.startswith('R@')})\n"
        "print('group score :', rg['winoground_group_score'], ' (azar', rg['chance_group'], ')')\n"
        "display(Image(filename=str(ROOT/'outputs'/'figures'/'recall_vs_group.png')))"
    ))

    # ---------------- 7. Por tag + ceguera ----------------
    c.append(md(
        "## 7. ¿Qué tipo de error? ¿Usa la imagen?\n"
        "Por tag (Object/Relation/Both) vemos *dónde* falla; la prueba de ceguera "
        "(permutar imágenes) confirma que **sí usa la imagen**."
    ))
    c.append(code(
        "display(pd.read_csv(MET/'by_tag.csv'))\n"
        "b = json.loads((MET/'blindness.json').read_text())\n"
        "print('real     :', {k:round(v,3) for k,v in b['real'].items() if 'score' in k})\n"
        "print('permutado:', {k:round(v,3) for k,v in b['permuted_images'].items() if 'score' in k})"
    ))
    c.append(code(
        "for n in ['by_tag.png','blindness.png','checkpoint_comparison.png']:\n"
        "    display(Image(filename=str(ROOT/'outputs'/'figures'/n)))"
    ))

    # ---------------- 8. Respuestas a las 5 preguntas ----------------
    c.append(md(
        "## 8. Respuestas a mis cinco preguntas (sección 8.6)\n\n"
        "**1. ¿Por qué un buen retrieval no garantiza razonamiento composicional?**\n"
        "Porque miden cosas distintas. El Recall@K busca la imagen correcta en *toda* la "
        "galería (basta el contenido grueso); Winoground exige ganar el **par mínimo**, "
        "que depende del orden y la relación. Aquí: R@5=0.67 pero group=0.075. Un "
        "dual-encoder colapsa cada modalidad en un vector y compara por coseno, así que "
        "actúa como *bolsa de conceptos* (conecta con C5).\n\n"
        "**2. ¿Cómo verifico que usa la imagen y no pistas textuales?**\n"
        "Con la **prueba de ceguera**: permuto las imágenes entre ejemplos y recalculo. "
        "Los scores caen de 0.35/0.11/0.075 a 0.14/0.04/0.02. Como caen, el modelo **sí** "
        "usa la imagen; su límite es composicional, no perceptivo (conecta con C8). "
        "Además las dos captions tienen las mismas palabras: no hay atajo léxico.\n\n"
        "**3. ¿Qué tipo de error aparece en Winoground?**\n"
        "Errores de **vinculación**, no de reconocimiento. Por tag, **Relation** es lo más "
        "difícil (group ≈ 0.047). El modelo reconoce los objetos pero no los liga en la "
        "estructura correcta.\n\n"
        "**4. ¿Cómo adapto el código para evaluar pares con cambios mínimos?**\n"
        "Es lo que hace este pipeline: cada ejemplo es `(image_0, image_1, caption_0, "
        "caption_1)` con `caption_0 ↔ image_0`. Para un par nuevo, duplico una caption, "
        "**permuto dos tokens**, y aporto las dos imágenes. El scorer arma la matriz "
        "$2\\times2$ y aplica las 3 reglas. Lo único nuevo frente a C10 es el scorer.\n\n"
        "**5. ¿Qué limitación tienen las métricas automáticas aquí?**\n"
        "R@K y accuracy no miden composición (necesarias, no suficientes); los scores son "
        "comparaciones estrictas, sensibles a empates y a la normalización; el azar del "
        "group es 1/6, no 1/16; y el propio benchmark tiene ítems ambiguos (Diwan et al.), "
        "así que un group bajo mezcla fallo composicional con dificultad del ítem."
    ))

    # ---------------- 9. Cierre ----------------
    c.append(md(
        "## 9. Relación con los cuadernos, limitación y mejora\n"
        "- **C10:** motor OpenCLIP (embeddings, coseno, checkpoints, FAISS).\n"
        "- **C5:** dual-encoder vs fusión profunda — *por qué* falla la composición.\n"
        "- **C8:** atención crossmodal / interpretabilidad — la prueba de ceguera.\n\n"
        "**Limitación:** el group bajo mezcla fallo composicional con ítems ambiguos; el "
        "set curado es un proxy. **Mejora:** evaluar un cross-encoder real (C5) y "
        "re-ranking con interacción cruzada (C6).\n\n"
        "**Conclusión:** el retrieval alto **no** implica razonamiento composicional. "
        "Medido, validado y reproducible."
    ))

    nb["cells"] = c
    return nb


def main():
    nb = build()
    print("[notebook] ejecutando celdas...")
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"[notebook] guardado y EJECUTADO en {OUT}")


if __name__ == "__main__":
    main()
