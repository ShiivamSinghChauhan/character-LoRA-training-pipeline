"""Playground web UI + job queue in front of one or more headless ComfyUI instances (FLUX.2 Klein 9B multi-reference edit).

Env: PG_PASS (required password), PG_MODE=split|dual, COMFY_DIR, PG_IN, PG_OUT, PG_PORT.
  split: one ComfyUI, UNET on cuda:0, text encoder on cuda:1 (needs ComfyUI-MultiGPU)  -> no model swapping, 1 job at a time
  dual : one ComfyUI per GPU (CUDA_VISIBLE_DEVICES), each pinned to its GPU          -> 2 jobs in parallel, models swap on a 15 GB T4
"""
import os, sys, json, time, uuid, queue, threading, socket, subprocess, urllib.request
from flask import Flask, request, jsonify, send_from_directory, make_response, abort

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graph  # noqa: E402
from PIL import Image  # noqa: E402

PASS = os.environ["PG_PASS"]
MODE = os.environ.get("PG_MODE", "split")
COMFY = os.environ.get("COMFY_DIR", "/kaggle/temp/klein/ComfyUI")
IN = os.environ.get("PG_IN", "/kaggle/temp/pg_in")
OUT = os.environ.get("PG_OUT", "/kaggle/working/pg_out")
PORT = int(os.environ.get("PG_PORT", "7860"))
for d in (IN, OUT):
    os.makedirs(d, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ ComfyUI instances
class Inst:
    def __init__(self, idx, gpu):
        self.idx, self.gpu, self.port = idx, gpu, 8190 + idx
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.loaders = None  # None = stock loaders
        self.busy = None

    def start(self):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        self.proc = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(self.port), "--input-directory", IN, "--output-directory", OUT],
            cwd=COMFY, env=env, stdout=open(f"/kaggle/temp/comfy_{self.idx}.log", "ab"), stderr=subprocess.STDOUT)

    def up(self, tries=240):
        for _ in range(tries):
            try:
                socket.create_connection(("127.0.0.1", self.port), timeout=2).close()
                return True
            except Exception:
                time.sleep(2)
        return False

    def j(self, path, data=None):
        req = urllib.request.Request(self.url + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))

    def detect_split(self):
        """Use ComfyUI-MultiGPU loaders (UNET on cuda:0, text encoder on cuda:1) when available."""
        try:
            oi = self.j("/object_info")
            res = {}
            for kind, cls, dev in (("unet", "UNETLoaderMultiGPU", "cuda:0"), ("clip", "CLIPLoaderMultiGPU", "cuda:1")):
                if cls not in oi:
                    return None
                req = oi[cls]["input"]["required"]
                key = next((k for k in ("device", "compute_device") if k in req), None)
                if key is None or dev not in req[key][0]:
                    return None
                res[kind] = (cls, {key: dev})
            return res
        except Exception as e:
            log("split detect failed", repr(e)[:120])
            return None

    def run(self, g, timeout=1800):
        pid = self.j("/prompt", {"prompt": g})["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < timeout:
            h = self.j("/history/" + pid)
            if pid in h and h[pid].get("status", {}).get("completed") is not None:
                st = h[pid]["status"]
                if st.get("status_str") != "success":
                    raise RuntimeError(json.dumps(st.get("messages", ""))[-400:])
                outs = [im for o in h[pid]["outputs"].values() for im in o.get("images", [])]
                return outs[0]
            time.sleep(1.5)
        raise TimeoutError("comfy job timeout")


INSTS = []
JOBS = {}       # id -> dict
ORDER = []      # newest first
Q = queue.Queue()
LOCK = threading.Lock()
HIST = os.path.join(OUT, "jobs.json")


def save_hist():
    with LOCK:
        json.dump({"jobs": JOBS, "order": ORDER}, open(HIST, "w"))


def load_hist():
    if os.path.exists(HIST):
        try:
            d = json.load(open(HIST))
            JOBS.update(d["jobs"])
            ORDER.extend(d["order"])
            for j in JOBS.values():
                if j["status"] in ("queued", "running"):
                    j["status"] = "lost"
        except Exception:
            pass


def worker(inst):
    while True:
        jid = Q.get()
        j = JOBS.get(jid)
        if not j or j["status"] != "queued":
            continue
        j["status"], j["gpu"], j["t_start"] = "running", inst.idx, time.time()
        inst.busy = jid
        try:
            try:
                im = inst.run(graph.build(j["params"], inst.loaders))
            except Exception as e:
                if inst.loaders:  # split loaders failed -> permanently fall back to stock loaders
                    log("split loaders failed, falling back:", repr(e)[:200])
                    inst.loaders = None
                    im = inst.run(graph.build(j["params"], None))
                else:
                    raise
            j["file"] = im["filename"]
            j["status"] = "done"
            j["secs"] = round(time.time() - j["t_start"], 1)
        except Exception as e:
            j["status"], j["error"] = "error", repr(e)[:500]
            log("job error", jid, j["error"])
        inst.busy = None
        save_hist()


# ------------------------------------------------------------------ web app
app = Flask(__name__)

LOGIN = ("<form method=post action=/login style='font:16px sans-serif;margin:20vh auto;width:260px'>"
         "<p>Password</p><input name=p type=password autofocus style='width:100%;padding:8px'>"
         "<button style='margin-top:8px;padding:8px'>Enter</button></form>")


@app.before_request
def auth():
    if request.path == "/login":
        return None
    if request.cookies.get("pg") != PASS and request.headers.get("X-PG") != PASS:
        if request.path.startswith("/api") or request.path.startswith("/files"):
            abort(401)
        return LOGIN
    return None


@app.post("/login")
def login():
    if request.form.get("p") == PASS:
        r = make_response("<script>location='/'</script>")
        r.set_cookie("pg", PASS, max_age=86400 * 3, httponly=True)
        return r
    return LOGIN, 403


@app.get("/")
def index():
    return send_from_directory(HERE, "ui.html")


@app.get("/files/<path:name>")
def files(name):
    return send_from_directory(OUT, name)


@app.get("/in/<path:name>")
def inputs(name):
    return send_from_directory(IN, name)


@app.get("/api/prompts")
def prompts():
    d = os.path.join(HERE, "..", "prompts")
    out = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".txt"):
            out[f[:-4]] = open(os.path.join(d, f), encoding="utf-8").read().strip()
    return jsonify(out)


@app.get("/api/debug")
def debug():
    out = {"insts": [{"idx": i.idx, "gpu": i.gpu, "loaders": i.loaders} for i in INSTS]}
    for i in INSTS:
        try:
            oi = i.j("/object_info")
            out[f"multigpu_nodes_{i.idx}"] = [k for k in oi if "MultiGPU" in k][:12]
            out[f"unet_multigpu_inputs_{i.idx}"] = {k: str(v)[:160] for k, v in oi.get("UNETLoaderMultiGPU", {}).get("input", {}).get("required", {}).items()}
            out[f"clip_multigpu_inputs_{i.idx}"] = {k: str(v)[:160] for k, v in oi.get("CLIPLoaderMultiGPU", {}).get("input", {}).get("required", {}).items()}
        except Exception as e:
            out[f"err_{i.idx}"] = repr(e)[:200]
        try:
            lines = open(f"/kaggle/temp/comfy_{i.idx}.log", errors="ignore").read().splitlines()
            out[f"log_{i.idx}"] = [ln for ln in lines if "MultiGPU" in ln or "rror" in ln or "IMPORT" in ln][-15:]
        except Exception as e:
            out[f"log_err_{i.idx}"] = repr(e)[:100]
    return jsonify(out)


@app.get("/api/state")
def state():
    jobs = [JOBS[i] for i in ORDER[:200] if i in JOBS]
    return jsonify({"mode": MODE, "workers": [{"idx": i.idx, "busy": i.busy, "split": bool(i.loaders)} for i in INSTS],
                    "queued": Q.qsize(), "jobs": jobs})


def prep_image(src, jid, n):
    im = Image.open(src).convert("RGB")
    name = f"{jid}_{n}.png"
    im.save(os.path.join(IN, name))
    return name, im.size


@app.post("/api/submit")
def submit():
    p = json.loads(request.form["params"])
    names, sizes = [], []
    base = uuid.uuid4().hex[:8]
    for n in range(int(p.get("n_images", 0))):
        f = request.files.get(f"img{n}")
        ref = p.get("reuse", {}).get(str(n))
        if f:
            nm, sz = prep_image(f.stream, base, n)
        elif ref:
            nm, sz = prep_image(os.path.join(OUT, os.path.basename(ref)), base, n)
        else:
            continue
        names.append(nm)
        sizes.append(sz)
    if not names:
        return jsonify({"error": "no images"}), 400
    k = int(p.get("size_from", 0))
    if p.get("custom_size"):
        w, h = int(p["width"]), int(p["height"])
    else:
        w, h = graph.out_size(*sizes[min(k, len(sizes) - 1)], float(p.get("out_mp", 1.0)))
    ids = []
    for i in range(int(p.get("count", 1))):
        jid = uuid.uuid4().hex[:10]
        params = {"images": names, "prompt": p["prompt"], "seed": int(p.get("seed", 0)) + i, "steps": int(p.get("steps", 8)),
                  "ref_mp": float(p.get("ref_mp", 1.0)), "width": w, "height": h, "lora": bool(p.get("lora", True)),
                  "lora_strength": float(p.get("lora_strength", 0.8)), "prefix": f"pg_{jid}"}
        JOBS[jid] = {"id": jid, "status": "queued", "params": params, "t_submit": time.time(), "inputs": names}
        ORDER.insert(0, jid)
        Q.put(jid)
        ids.append(jid)
    save_hist()
    return jsonify({"ids": ids})


@app.post("/api/cancel/<jid>")
def cancel(jid):
    j = JOBS.get(jid)
    if j and j["status"] == "queued":
        j["status"] = "cancelled"
    return jsonify({"ok": True})


@app.post("/api/delete/<jid>")
def delete(jid):
    j = JOBS.pop(jid, None)
    if j is not None:
        if jid in ORDER:
            ORDER.remove(jid)
        if j.get("file"):
            try:
                os.remove(os.path.join(OUT, j["file"]))
            except OSError:
                pass
        save_hist()
    return jsonify({"ok": True})


def boot():
    load_hist()
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    ngpu = len([ln for ln in smi if ln.startswith("GPU")]) or 1
    plan = [Inst(0, None)] if MODE == "split" or ngpu < 2 else [Inst(0, 0), Inst(1, 1)]
    for inst in plan:
        inst.start()
        if not inst.up():
            raise SystemExit("ComfyUI failed to start: " + str(inst.idx))
        log("ComfyUI up", inst.idx, "gpu", inst.gpu)
        time.sleep(5)
    if MODE == "split" and ngpu >= 2:
        plan[0].loaders = plan[0].detect_split()
        log("split loaders:", plan[0].loaders)
    INSTS.extend(plan)
    for inst in plan:
        threading.Thread(target=worker, args=(inst,), daemon=True).start()


if __name__ == "__main__":
    boot()
    log("playground listening on", PORT, "mode", MODE)
    app.run(host="127.0.0.1", port=PORT, threaded=True)
