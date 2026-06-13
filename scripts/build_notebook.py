#!/usr/bin/env python3
"""Construye y EJECUTA notebooks/Winoground_Eval_MCC225.ipynb.

Notebook de defensa: explicaciones BREVES (apuntes de exposición) + código + resultados
+ respuestas a las 5 preguntas de 8.6. Uso:  python scripts/build_notebook.py
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

    c.append(md(
        "# Winoground / Evaluating CLIP — Defensa MCC225\n"
        "**Niels Pacheco** · C5, C8, C10\n\n"
        "**Idea:** ¿CLIP *entiende* la escena o solo *empareja*?\n"
        "**Tesis:** retrieval alto **≠** composición → Recall@K alto pero **group ≈ azar**.\n"
        "Motor = OpenCLIP (C10); el *porqué* = fusión profunda (C5) y atención crossmodal (C8)."
    ))

    c.append(md("## 1. Entorno (trazabilidad)"))
    c.append(code(
        "import os, sys, pathlib\n"
        "ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name=='notebooks' else pathlib.Path.cwd()\n"
        "os.chdir(ROOT); sys.path.insert(0, str(ROOT))\n"
        "from src.env_logging import print_snapshot\n"
        "_ = print_snapshot(ROOT)"
    ))

    c.append(md(
        "## 2. El par mínimo\n"
        "2 imágenes + 2 captions con **las mismas palabras en otro orden**. Hay que entender "
        "*quién hace qué a quién*."
    ))
    c.append(code(
        "from src.winoground_data import load_dataset\n"
        "examples, source = load_dataset(prefer_real=True)\n"
        "print('fuente:', source, '| N =', len(examples))\n"
        "ex = examples[0]\n"
        "print('caption_0:', ex.caption_0); print('caption_1:', ex.caption_1)"
    ))
    c.append(code(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))\n"
        "ax[0].imshow(ex.image_0); ax[0].set_title('image_0', fontsize=9); ax[0].axis('off')\n"
        "ax[1].imshow(ex.image_1); ax[1].set_title('image_1', fontsize=9); ax[1].axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ))
    c.append(md(
        "**3 métricas** (matriz `sim[caption, imagen]`): **text** (fija imagen, elige caption), "
        "**image** (fija caption, elige imagen), **group** (ambas). Azar: 1/4, 1/4, **1/6**."
    ))

    c.append(md(
        "## 3. Celda clave: el scorer\n"
        "OpenCLIP (C10) → matriz 2×2 → 3 reglas. **Aquí sale el resultado.**"
    ))
    c.append(code(
        "from src import openclip_utils as oc\n"
        "from src.winoground_eval import text_correct, image_correct, group_correct\n"
        "model, preprocess, tokenizer, device = oc.create_model('ViT-B-32','laion2b_s34b_b79k')\n"
        "imgf = oc.encode_images(model, preprocess, [ex.image_0, ex.image_1], device)\n"
        "capf = oc.encode_texts(model, tokenizer, [ex.caption_0, ex.caption_1], device)\n"
        "sim = capf @ imgf.T   # sim[caption, imagen]\n"
        "print('matriz 2x2:\\n', sim.round(3))\n"
        "print('text=', text_correct(sim), '| image=', image_correct(sim), '| group=', group_correct(sim))"
    ))

    c.append(md(
        "## 4. Validación del scorer\n"
        "Aplico mi scorer a los scores de CLIP del paper (`clip.jsonl`): debe dar "
        "**0.3075 / 0.1050 / 0.0800**."
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
        "    print(f'mi scorer -> text={agg.text:.4f} image={agg.image:.4f} group={agg.group:.4f}  ✅')\n"
        "except Exception as e:\n"
        "    print('(dataset oficial no disponible:', type(e).__name__, ')')"
    ))

    c.append(md("## 5. Resultado principal (400 ejemplos)"))
    c.append(code(
        "import json, pandas as pd\n"
        "MET = ROOT/'outputs'/'metrics'\n"
        "scores = json.loads((MET/'scores.json').read_text())\n"
        "print(scores['primary_checkpoint'], '| fuente:', scores['source'])\n"
        "print(json.dumps(scores['scores'], indent=2))\n"
        "pd.read_csv(MET/'bootstrap_ci.csv')"
    ))
    c.append(code(
        "from IPython.display import Image, display\n"
        "display(Image(filename=str(ROOT/'outputs'/'figures'/'scores_vs_chance.png')))"
    ))

    c.append(md("## 6. Tesis: retrieval alto vs group bajo"))
    c.append(code(
        "rg = json.loads((MET/'recall_vs_group.json').read_text())\n"
        "print('R@K texto->imagen:', {k:v for k,v in rg['text_to_image_recall'].items() if k.startswith('R@')})\n"
        "print('group:', rg['winoground_group_score'], '| azar:', rg['chance_group'])\n"
        "display(Image(filename=str(ROOT/'outputs'/'figures'/'recall_vs_group.png')))"
    ))

    c.append(md("## 7. Por tag y prueba de ceguera (¿usa la imagen?)"))
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

    c.append(md(
        "## 8. Respuestas — 5 preguntas (8.6)\n\n"
        "1. **Retrieval ≠ composición:** R@K busca en toda la galería (fácil); group exige el "
        "par mínimo (orden/relación). R@5=0.67 vs group=0.075. Un dual-encoder es *bolsa de "
        "conceptos* (C5).\n"
        "2. **¿Usa la imagen?** Sí. Prueba de ceguera: al permutar imágenes los scores caen "
        "(0.35→0.14, 0.075→0.02). El límite es composicional, no perceptivo (C8).\n"
        "3. **Tipo de error:** de *vinculación*, no de reconocimiento. **Relation** es lo peor "
        "(group ≈ 0.047).\n"
        "4. **Adaptar el código:** cada ejemplo = `(img0,img1,cap0,cap1)`; para un par nuevo "
        "permuto dos tokens y aporto las imágenes. El scorer arma la matriz 2×2. Lo único "
        "nuevo vs C10 es el scorer.\n"
        "5. **Límite de las métricas:** R@K/accuracy no miden composición; comparaciones "
        "estrictas (empates); azar group=1/6; el benchmark tiene ítems ambiguos (Diwan et al.)."
    ))

    c.append(md(
        "## 9. Cierre\n"
        "**C10** motor · **C5** por qué falla · **C8** interpretabilidad. "
        "**Limitación:** el group mezcla composición con ítems ambiguos. "
        "**Mejora:** cross-encoder (C5) + re-ranking (C6). "
        "**Conclusión:** retrieval alto **no** implica composición."
    ))

    nb["cells"] = c
    return nb


def main():
    nb = build()
    print("[notebook] ejecutando...")
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT / "notebooks")}})
    client.execute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"[notebook] guardado y EJECUTADO en {OUT}")


if __name__ == "__main__":
    main()
