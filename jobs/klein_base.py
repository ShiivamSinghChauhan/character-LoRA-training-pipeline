import os, sys, re, json, time, glob, shutil, subprocess, socket, urllib.request, zipfile
T0 = time.time()
ORDER = @@ORDER@@
def say(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def sh(c, show=False):
    r = subprocess.run(["bash", "-c", c], capture_output=True, text=True)
    if show or r.returncode: print((r.stdout + r.stderr)[-1500:], flush=True)
    return r.returncode
try:
    from kaggle_secrets import UserSecretsClient
    HF = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF = os.environ.get("HF_TOKEN", "")
BASE = "/kaggle/temp/klein"; COMFY = BASE + "/ComfyUI"; OUT = "/kaggle/working/out"; ZIPS = "/kaggle/working/zips"
for d in (BASE, OUT, ZIPS): os.makedirs(d, exist_ok=True)
if not os.path.exists(COMFY + "/main.py"): sh(f"git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY}", True)
lines = [l.strip() for l in open(COMFY + "/requirements.txt") if l.strip() and not re.match(r"^(torch|torchvision|torchaudio)([=<>~ ]|$)", l.strip())]
open(BASE + "/req.txt", "w").write("\n".join(lines) + "\n")
say("pip install ...")
if sh(f"pip install -q -r {BASE}/req.txt > {BASE}/pip.log 2>&1") != 0:
    for l in lines:
        if not l.startswith("#"): sh(f"pip install -q '{l}' > /dev/null 2>&1")
sh("pip install -q -U 'huggingface_hub[hf_xet]' > /dev/null 2>&1")
from huggingface_hub import hf_hub_download
M = COMFY + "/models"
def get(repo, path, d, tok=None):
    dst = f"{M}/{d}/{os.path.basename(path)}"
    if os.path.exists(dst): return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download(repo, path, local_dir=BASE + "/stage", token=tok or None); shutil.move(p, dst); say("got", d, os.path.basename(dst))
get("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors", "diffusion_models", HF)
get("Comfy-Org/Flux-2-Klein-9B", "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", "text_encoders")
get("Comfy-Org/Flux-2-Klein-9B", "split_files/vae/flux2-vae.safetensors", "vae")
get("dx8152/Flux2-Klein-9B-Consistency", "Flux2-Klein-9B-consistency-V2.safetensors", "loras")
found = {}
for p in glob.glob("/kaggle/input/**/*.png", recursive=True):
    m = re.match(r"^p(\d+)\.png$", os.path.basename(p))
    if m: found[int(m.group(1))] = p
say("frames found:", len(found), "of", len(ORDER))
todo = [i for i in ORDER if i in found]
os.makedirs(COMFY + "/input", exist_ok=True)
def start():
    return subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8190", "--output-directory", OUT], cwd=COMFY,
        stdout=open(BASE + "/comfy.log", "ab"), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
def wait_up(n=240):
    for _ in range(n):
        try: socket.create_connection(("127.0.0.1", 8190), timeout=2).close(); return True
        except Exception: time.sleep(2)
    return False
proc = start(); assert wait_up(), "comfy failed to start"; say("ComfyUI up")
U = "http://127.0.0.1:8190"
def post(p, d): return json.load(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=60))
def getj(p): return json.load(urllib.request.urlopen(U + p, timeout=60))
PROMPT = ""
STRENGTH = 0.8
def graph(img, tag, mp=1.0, seed=123, steps=8):
    return {
     "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-9b.safetensors", "weight_dtype": "fp8_e4m3fn"}},
     "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "Flux2-Klein-9B-consistency-V2.safetensors", "strength_model": STRENGTH}},
     "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_8b_fp8mixed.safetensors", "type": "flux2", "device": "default"}},
     "4": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
     "5": {"class_type": "LoadImage", "inputs": {"image": img}},
     "6": {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "megapixels": mp, "resolution_steps": 16}},
     "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["4", 0]}},
     "8": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["3", 0]}},
     "9": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["8", 0], "latent": ["7", 0]}},
     "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}},
     "11": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["10", 0], "latent": ["7", 0]}},
     "12": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["9", 0], "negative": ["11", 0], "cfg": 1.0}},
     "13": {"class_type": "GetImageSize", "inputs": {"image": ["6", 0]}},
     "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": steps, "width": ["13", 0], "height": ["13", 1]}},
     "15": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["13", 0], "height": ["13", 1], "batch_size": 1}},
     "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
     "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
     "18": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["16", 0], "guider": ["12", 0], "sampler": ["17", 0], "sigmas": ["14", 0], "latent_image": ["15", 0]}},
     "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["4", 0]}},
     "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": tag}},
    }
def run(g):
    r = post("/prompt", {"prompt": g}); pid = r["prompt_id"]
    while True:
        h = getj("/history/" + pid)
        if pid in h and h[pid].get("status", {}).get("completed") is not None: return h[pid]["status"].get("status_str") == "success"
        time.sleep(4)
def zip_new():
    for f in glob.glob(ZIPS + "/*.zip"): os.remove(f)
    pngs = sorted(glob.glob(OUT + "/f*_@@TAGRE@@_*.png"), key=lambda s: int(re.search(r"/f(\d+)_", s).group(1)))
    for k in range(0, len(pngs), 25):
        with zipfile.ZipFile(f"{ZIPS}/@@ZIPNAME@@{k // 25 + 1:02d}.zip", "w", zipfile.ZIP_STORED) as z:
            for f in pngs[k:k + 25]: z.write(f, os.path.basename(f))
@@VARIANTS@@
for n, i in enumerate(todo, 1):
    nm = f"f{i}_orig.png"; shutil.copy(found[i], COMFY + "/input/" + nm)
    for tag, pr in VARIANTS:
        PROMPT = pr; STRENGTH = 0.8; t = time.time(); ok = False
        for attempt in range(2):
            try:
                if proc.poll() is not None: proc = start(); wait_up()
                ok = run(graph(nm, f"f{i}_{tag}"))
                if ok: break
            except Exception as e:
                say("err", i, repr(e)[:150]); time.sleep(5)
        say(f"[{n}/{len(todo)}] f{i} {tag} {'ok' if ok else 'FAILED'} {time.time()-t:.0f}s")
zip_new()
say("ALL DONE", int(time.time() - T0), "s")
proc.terminate()
