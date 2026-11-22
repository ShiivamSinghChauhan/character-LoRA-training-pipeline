"""Kaggle bootstrap for the Klein playground. Run as a notebook cell (interactive GPU session, internet on, HF_TOKEN secret attached):

    import os, subprocess
    os.environ["PG_PASS"] = "choose-a-password"          # optional, random if unset
    os.environ["PG_MODE"] = "split"                      # or "dual" (2 parallel workers, one per T4)
    subprocess.run("rm -rf /kaggle/working/pipe; git clone -q --depth 1 https://github.com/ShiivamSinghChauhan/character-LoRA-training-pipeline /kaggle/working/pipe", shell=True)
    exec(open("/kaggle/working/pipe/playground/launch.py").read())

It installs ComfyUI, downloads FLUX.2 Klein 9B + text encoder + VAE + consistency LoRA (needs the gated-repo HF token),
starts the playground app and a Cloudflare quick tunnel, then prints `PLAYGROUND_URL <url>` every minute (keeps the cell alive).
"""
import os, re, sys, json, time, glob, shutil, secrets, subprocess, threading, urllib.request

T0 = time.time()
PIPE = "/kaggle/working/pipe"
BASE = "/kaggle/temp/klein"
COMFY = BASE + "/ComfyUI"
PASSWORD = os.environ.get("PG_PASS") or secrets.token_urlsafe(6)
MODE = os.environ.get("PG_MODE", "split")
os.makedirs(BASE, exist_ok=True)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd, show=False):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if show or r.returncode:
        print((r.stdout + r.stderr)[-1500:], flush=True)
    return r.returncode


HF = os.environ.get("HF_TOKEN", "")
try:
    from kaggle_secrets import UserSecretsClient
    for _label in ("HF_WRITE", "HF_TOKEN"):   # HF_WRITE (write token) preferred, HF_TOKEN fallback
        try:
            HF = UserSecretsClient().get_secret(_label)
            break
        except Exception:
            pass
except Exception:
    pass
if not HF:
    say("WARNING: no HF_TOKEN secret - the gated Klein 9B download will fail")

# ---- ComfyUI + requirements
if not os.path.exists(COMFY + "/main.py"):
    sh(f"git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY}", True)
lines = [l.strip() for l in open(COMFY + "/requirements.txt") if l.strip() and not re.match(r"^(torch|torchvision|torchaudio)([=<>~ ]|$)", l.strip())]
open(BASE + "/req.txt", "w").write("\n".join(lines) + "\n")
say("pip install ...")
if sh(f"pip install -q -r {BASE}/req.txt > {BASE}/pip.log 2>&1") != 0:
    for l in lines:
        if not l.startswith("#"):
            sh(f"pip install -q '{l}' > /dev/null 2>&1")
sh("pip install -q -U 'huggingface_hub[hf_xet]' flask pillow > /dev/null 2>&1")
if MODE == "split" and not os.path.exists(COMFY + "/custom_nodes/ComfyUI-MultiGPU"):
    sh(f"git clone -q --depth 1 https://github.com/pollockjj/ComfyUI-MultiGPU {COMFY}/custom_nodes/ComfyUI-MultiGPU", True)

# ---- models (parallel downloads)
from huggingface_hub import hf_hub_download

M = COMFY + "/models"
FILES = [
    ("black-forest-labs/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors", "diffusion_models", True),
    ("Comfy-Org/Flux-2-Klein-9B", "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", "text_encoders", False),
    ("Comfy-Org/Flux-2-Klein-9B", "split_files/vae/flux2-vae.safetensors", "vae", False),
    ("dx8152/Flux2-Klein-9B-Consistency", "Flux2-Klein-9B-consistency-V2.safetensors", "loras", False),
]


def fetch(repo, path, sub, gated):
    dst = f"{M}/{sub}/{os.path.basename(path)}"
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    p = hf_hub_download(repo, path, local_dir=BASE + "/stage", token=(HF or None))
    shutil.move(p, dst)
    say("got", sub, os.path.basename(dst))


ths = [threading.Thread(target=fetch, args=f) for f in FILES]
[t.start() for t in ths]
[t.join() for t in ths]
for _, path, sub, _ in FILES:
    assert os.path.exists(f"{M}/{sub}/{os.path.basename(path)}"), "download failed: " + path

# ---- app
env = dict(os.environ, PG_PASS=PASSWORD, PG_MODE=MODE, COMFY_DIR=COMFY, PYTHONUNBUFFERED="1")
app = subprocess.Popen([sys.executable, PIPE + "/playground/app.py"], env=env, stdout=open("/kaggle/temp/pg_app.log", "ab"), stderr=subprocess.STDOUT)
say("app starting (ComfyUI boot takes ~1 min) ...")

# ---- tunnel: cloudflared quick tunnel, ssh (localhost.run) as fallback
CF = BASE + "/cloudflared"
if not os.path.exists(CF):
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", CF)
    os.chmod(CF, 0o755)
URL = None


def start_tunnel():
    lf = "/kaggle/temp/tunnel.log"
    open(lf, "w").close()
    p = subprocess.Popen([CF, "tunnel", "--url", "http://127.0.0.1:7860", "--no-autoupdate"], stdout=open(lf, "ab"), stderr=subprocess.STDOUT)
    for _ in range(45):
        time.sleep(2)
        m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", open(lf, errors="ignore").read())
        if m:
            return p, m.group(0)
    p.kill()
    p = subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30", "-R", "80:127.0.0.1:7860", "nokey@localhost.run"],
                         stdout=open(lf, "ab"), stderr=subprocess.STDOUT)
    for _ in range(45):
        time.sleep(2)
        m = re.search(r"https://[-a-z0-9]+\.lhr\.life", open(lf, errors="ignore").read())
        if m:
            return p, m.group(0)
    return p, None


tun, URL = start_tunnel()
say("tunnel:", URL)


def app_state():
    try:
        r = urllib.request.Request("http://127.0.0.1:7860/api/state", headers={"X-PG": PASSWORD})
        s = json.load(urllib.request.urlopen(r, timeout=5))
        return f"queued={s['queued']} workers=" + ",".join(("busy" if w["busy"] else "idle") for w in s["workers"])
    except Exception as e:
        return "app not ready: " + type(e).__name__


while True:
    if tun.poll() is not None:
        tun, URL = start_tunnel()
    if app.poll() is not None:
        say("APP DIED, log tail:\n" + open("/kaggle/temp/pg_app.log", errors="ignore").read()[-1500:])
        break
    say(f"PLAYGROUND_URL {URL} PASS {PASSWORD} | {app_state()} | up {int(time.time() - T0)}s")
    time.sleep(60)
