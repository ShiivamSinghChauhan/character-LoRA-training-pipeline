"""usage: python bias_eval.py OUT_DIR   (folder with {model}_{prompt}_{seed}_00001_.png from bias_job.py)
Writes contact sheets + bias_report.json: per model/prompt -> mean ArcFace sim to the character centroid, mean pairwise sim between seeds (low = diverse faces)."""
import sys, os, glob, json, itertools, numpy as np, cv2
from PIL import Image, ImageDraw
from insightface.app import FaceAnalysis
OUT = sys.argv[1]
REF = [p for p in glob.glob("M:/lora-training/grey_b_final/*_klein_b.png")] + glob.glob("M:/lora-training/batch1_out/d_*_klein_00001_.png")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); app.prepare(ctx_id=-1, det_size=(640, 640))
def emb(p):
    im = cv2.imdecode(np.fromfile(p, np.uint8), 1)
    fs = app.get(im) if im is not None else []
    return max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding if fs else None
refs = [e for e in (emb(p) for p in REF) if e is not None]
cen = np.mean(refs, 0); cen /= np.linalg.norm(cen)
print("reference faces:", len(refs))
files = sorted(glob.glob(OUT + "/*_*_*_00001_.png"))
rows = {}
for f in files:
    m, k, s = os.path.basename(f).split("_")[:3]
    rows.setdefault((m, k), []).append((f, emb(f)))
rep = {}
for (m, k), v in rows.items():
    es = [e for _, e in v if e is not None]
    sim = [float(e @ cen) for e in es]
    pair = [float(a @ b) for a, b in itertools.combinations(es, 2)]
    rep.setdefault(m, {})[k] = {"n": len(v), "faces": len(es), "to_character": round(float(np.mean(sim)), 3) if sim else None,
                                "between_seeds": round(float(np.mean(pair)), 3) if pair else None}
json.dump(rep, open(OUT + "/bias_report.json", "w"), indent=1)
for m, d in rep.items():
    print(m); [print("  ", k, v) for k, v in d.items()]
models = sorted({m for m, _ in rows}); prompts = ["default", "medit", "arab", "persian", "nordic"]
T = 256; Hh = 340
for m in models:
    sh = Image.new("RGB", (T * 4, (Hh + 18) * len(prompts)), "white"); dr = ImageDraw.Draw(sh)
    for r, k in enumerate(prompts):
        for c, (f, e) in enumerate(rows.get((m, k), [])[:4]):
            sh.paste(Image.open(f).convert("RGB").resize((T, Hh)), (c * T, r * (Hh + 18) + 18))
        dr.text((4, r * (Hh + 18) + 3), f"{m} / {k}", fill="black")
    sh.save(f"{OUT}/sheet_{m}.jpg", quality=88)
