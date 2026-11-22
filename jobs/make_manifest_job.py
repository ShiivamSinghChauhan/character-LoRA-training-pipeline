"""Builds E:/train-lora/_tools/klein_manifest_job.py from klein_base.py: processes every image listed in manifest.json (from the Kaggle dataset)."""
import ast
s = open("E:/train-lora/_tools/klein_base.py", encoding="utf-8").read()
a = s.index("found = {}")
b = s.index("os.makedirs(COMFY + \"/input\"")
s = s[:a] + '''MAN = json.load(open(glob.glob("/kaggle/input/**/manifest.json", recursive=True)[0]))
found = {}
for p in glob.glob("/kaggle/input/**/*.png", recursive=True):
    if os.path.basename(p) in MAN: found[os.path.basename(p)] = p
say("frames found:", len(found), "of", len(MAN))
todo = [k for k in sorted(MAN, key=lambda x: (x[0] != "n", x[0] != "b", x[0] != "g", int(re.search(r"(\\d+)", x).group(1)))) if k in found]
LIMIT = 11.3 * 3600
''' + s[b:]
# zip: names are <code>_<stem>_klein_*.png
c = s.index("def zip_new():")
d = s.index("@@VARIANTS@@")
s = s[:c] + '''def zip_new():
    for f in glob.glob(ZIPS + "/*.zip"): os.remove(f)
    pngs = sorted(glob.glob(OUT + "/*_klein_*.png"))
    for k in range(0, len(pngs), 20):
        with zipfile.ZipFile(f"{ZIPS}/set_part{k // 20 + 1:02d}.zip", "w", zipfile.ZIP_STORED) as z:
            for f in pngs[k:k + 20]: z.write(f, os.path.basename(f))
''' + s[d + len("@@VARIANTS@@"):]
e = s.index("for n, i in enumerate(todo, 1):")
s = s[:e] + '''failed = []
for n, nm in enumerate(todo, 1):
    if time.time() - T0 > LIMIT: say("time limit reached"); break
    stem = nm[:-4]; shutil.copy(found[nm], COMFY + "/input/" + nm)
    PROMPT = MAN[nm]["prompt"]; STRENGTH = 0.8; t = time.time(); ok = False
    for attempt in range(3):
        try:
            if proc.poll() is not None: proc = start(); wait_up()
            ok = run(graph(nm, stem + "_klein", mp=1.0 if attempt < 2 else 0.7))
            if ok: break
        except Exception as ex:
            say("err", nm, repr(ex)[:150]); time.sleep(5)
            if proc.poll() is not None: proc = start(); wait_up()
    if not ok: failed.append(nm)
    say(f"[{n}/{len(todo)}] {nm} {MAN[nm]['tag']} {'ok' if ok else 'FAILED'} {time.time()-t:.0f}s")
    if n % 10 == 0: zip_new()
zip_new()
json.dump({"failed": failed}, open("/kaggle/working/summary.json", "w"))
say("ALL DONE", len(todo) - len(failed), "ok,", len(failed), "failed,", int(time.time() - T0), "s")
proc.terminate()
'''
s = s.replace("@@ORDER@@", "[]")
ast.parse(s)
open("E:/train-lora/_tools/klein_manifest_job.py", "w", encoding="utf-8", newline="\n").write(s)
print("ok, longest line", max(len(l) for l in s.split("\n")))
