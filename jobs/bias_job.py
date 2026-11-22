import os, sys, re, json, time, glob, shutil, subprocess, socket, urllib.request, zipfile
T0 = time.time()
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
BASE = "/kaggle/temp/bias"; COMFY = BASE + "/ComfyUI"; OUT = "/kaggle/working/out"; ZIPS = "/kaggle/working/zips"
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
def get(repo, path, d):
    dst = f"{M}/{d}/{os.path.basename(path)}"
    if os.path.exists(dst): return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download(repo, path, local_dir=BASE + "/stage", token=HF or None); shutil.move(p, dst); say("got", d, os.path.basename(path))
def wipe():
    for d in ("diffusion_models", "text_encoders", "vae"): shutil.rmtree(f"{M}/{d}", ignore_errors=True)
    shutil.rmtree(BASE + "/stage", ignore_errors=True)
U = "http://127.0.0.1:8190"
def post(p, d): return json.load(urllib.request.urlopen(urllib.request.Request(U + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), timeout=60))
def getj(p): return json.load(urllib.request.urlopen(U + p, timeout=60))
def start():
    return subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8190", "--output-directory", OUT], cwd=COMFY,
        stdout=open(BASE + "/comfy.log", "ab"), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
def wait_up(n=240):
    for _ in range(n):
        try: socket.create_connection(("127.0.0.1", 8190), timeout=2).close(); return True
        except Exception: time.sleep(2)
    return False
def run(g, limit=1500):
    pid = post("/prompt", {"prompt": g})["prompt_id"]; t = time.time()
    while time.time() - t < limit:
        h = getj("/history/" + pid)
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            if h[pid]["status"].get("status_str") != "success": say("ERR", json.dumps(h[pid]["status"])[-600:])
            return h[pid]["status"].get("status_str") == "success"
        time.sleep(4)
    return False

W, H = 768, 1024
DESC = {
  "default": "a woman in her late twenties",
  "medit": "a Mediterranean woman (southern Italian) in her late twenties",
  "arab": "a Levantine Arab woman (Lebanese) in her late twenties",
  "persian": "a Persian woman (Iranian) in her late twenties",
  "nordic": "a Northern European woman (Scandinavian) in her late twenties",
}
def prompt(k): return f"A candid portrait photograph of {DESC[k]}, head and shoulders, wearing a plain grey t-shirt, looking at the camera with a relaxed neutral expression, soft natural window light, realistic skin texture, shot on a 85mm lens."
JOBS = [("default", s) for s in (11, 12, 13, 14)] + [(k, s) for k in ("medit", "arab", "persian", "nordic") for s in (11, 12)]
NEG = "low quality, blurry, deformed, cartoon, illustration, painting, 3d render, oversmoothed skin, watermark, text"

def common(m, unet, clipn, cliptype, vae, wdt="fp8_e4m3fn"):
    return {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": wdt}},
            "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": clipn, "type": cliptype, "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": vae}}}

def g_klein(p, seed, tag):
    g = common("klein", "flux-2-klein-base-4b.safetensors", "qwen_3_4b.safetensors", "flux2", "flux2-vae.safetensors")
    g.update({"8": {"class_type": "CLIPTextEncode", "inputs": {"text": p, "clip": ["3", 0]}},
     "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}},
     "12": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["8", 0], "negative": ["10", 0], "cfg": 4.0}},
     "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": 30, "width": W, "height": H}},
     "15": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
     "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
     "18": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["16", 0], "guider": ["12", 0], "sampler": ["17", 0], "sigmas": ["14", 0], "latent_image": ["15", 0]}},
     "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["4", 0]}},
     "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": tag}}})
    return g

def g_ksampler(unet, clipn, cliptype, vae, steps, cfg, sampler, sched, latent, shift=None, neg_text=None, wdt="fp8_e4m3fn"):
    def f(p, seed, tag):
        g = common("x", unet, clipn, cliptype, vae, wdt)
        mdl = ["1", 0]
        if shift is not None:
            g["2"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": shift}}; mdl = ["2", 0]
        g["8"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p, "clip": ["3", 0]}}
        if neg_text is None: g["10"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}}
        else: g["10"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg_text, "clip": ["3", 0]}}
        g["15"] = {"class_type": latent, "inputs": {"width": W, "height": H, "batch_size": 1}}
        g["18"] = {"class_type": "KSampler", "inputs": {"model": mdl, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": sched,
                   "positive": ["8", 0], "negative": ["10", 0], "latent_image": ["15", 0], "denoise": 1.0}}
        g["19"] = {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["4", 0]}}
        g["20"] = {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": tag}}
        return g
    return f

def g_chroma(p, seed, tag):
    g = common("chroma", "Chroma1-HD-fp8mixed.safetensors", "t5xxl_fp8_e4m3fn_scaled.safetensors", "chroma", "ae.safetensors", "default")
    g.update({"2": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 1.0}},
     "7": {"class_type": "T5TokenizerOptions", "inputs": {"clip": ["3", 0], "min_padding": 0, "min_length": 0}},
     "8": {"class_type": "CLIPTextEncode", "inputs": {"text": p, "clip": ["7", 0]}},
     "10": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["7", 0]}},
     "12": {"class_type": "CFGGuider", "inputs": {"model": ["2", 0], "positive": ["8", 0], "negative": ["10", 0], "cfg": 3.5}},
     "14": {"class_type": "BasicScheduler", "inputs": {"model": ["2", 0], "scheduler": "beta", "steps": 26, "denoise": 1.0}},
     "15": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
     "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
     "18": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["16", 0], "guider": ["12", 0], "sampler": ["17", 0], "sigmas": ["14", 0], "latent_image": ["15", 0]}},
     "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["4", 0]}},
     "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": tag}}})
    return g

MODELS = [
 ("klein4b", [("Comfy-Org/Flux-2-Klein-4B", "split_files/diffusion_models/flux-2-klein-base-4b.safetensors", "diffusion_models"),
              ("Comfy-Org/Flux-2-Klein-4B", "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
              ("Comfy-Org/Flux-2-Klein-4B", "split_files/vae/flux2-vae.safetensors", "vae")], g_klein),
 ("krea2raw", [("Comfy-Org/Krea-2", "diffusion_models/krea2_raw_fp8_scaled.safetensors", "diffusion_models"),
               ("Comfy-Org/Krea-2", "text_encoders/qwen3vl_4b_fp8_scaled.safetensors", "text_encoders"),
               ("Comfy-Org/Krea-2", "vae/qwen_image_vae.safetensors", "vae")],
  g_ksampler("krea2_raw_fp8_scaled.safetensors", "qwen3vl_4b_fp8_scaled.safetensors", "krea2", "qwen_image_vae.safetensors", 30, 3.5, "euler", "simple", "EmptyLatentImage", wdt="default")),
 ("zimagebase", [("Comfy-Org/Z-Image", "split_files/diffusion_models/z_image_bf16.safetensors", "diffusion_models"),
                 ("Comfy-Org/Z-Image", "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
                 ("Comfy-Org/Z-Image", "split_files/vae/ae.safetensors", "vae")],
  g_ksampler("z_image_bf16.safetensors", "qwen_3_4b.safetensors", "lumina2", "ae.safetensors", 28, 4.0, "res_multistep", "simple", "EmptySD3LatentImage", shift=3.0, neg_text=NEG)),
 ("chroma", [("Comfy-Org/Chroma1-HD", "split_files/diffusion_models/Chroma1-HD-fp8mixed.safetensors", "diffusion_models"),
             ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
             ("Comfy-Org/Z-Image", "split_files/vae/ae.safetensors", "vae")], g_chroma),
]
CAP = 55 * 60
done = {}
for name, files, gf in MODELS:
    t0 = time.time(); proc = None; ok_n = 0
    try:
        wipe()
        for repo, path, d in files: get(repo, path, d)
        say(name, "downloaded in", int(time.time() - t0), "s")
        proc = start(); assert wait_up(), "comfy failed to start"
        for k, s in JOBS:
            if time.time() - t0 > CAP: say(name, "time cap reached"); break
            t = time.time(); tag = f"{name}_{k}_{s}"
            try: ok = run(gf(prompt(k), s, tag))
            except Exception as ex: say("err", tag, repr(ex)[:200]); ok = False
            ok_n += ok; say(f"[{name}] {k} seed {s} {'ok' if ok else 'FAILED'} {time.time()-t:.0f}s")
            if not ok and ok_n == 0: say(name, "first image failed, abandoning model"); break
    except Exception as ex:
        say(name, "MODEL ERROR", repr(ex)[:300])
    finally:
        if proc and proc.poll() is None: proc.terminate(); time.sleep(5)
        done[name] = ok_n
    with zipfile.ZipFile(f"{ZIPS}/{name}.zip", "w", zipfile.ZIP_STORED) as z:
        for f in sorted(glob.glob(f"{OUT}/{name}_*.png")): z.write(f, os.path.basename(f))
    say(name, "finished", ok_n, "images in", int(time.time() - t0), "s")
json.dump({"done": done, "prompts": {k: prompt(k) for k in DESC}}, open("/kaggle/working/summary.json", "w"), indent=1)
say("ALL DONE", done, int(time.time() - T0), "s")
