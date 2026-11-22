"""CLI client for a running playground.  usage:
python pg_client.py URL PASSWORD "prompt text" img1.png [img2.png ...] [--seed 1] [--steps 8] [--out out.png] [--lora 0.8|off] [--ref-mp 1.0] [--out-mp 1.0]
Submits one job, waits, downloads the result and prints timing."""
import sys, json, time, uuid, urllib.request, argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("password")
ap.add_argument("prompt")
ap.add_argument("images", nargs="+")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--out", default="pg_result.png")
ap.add_argument("--lora", default="0.8")
ap.add_argument("--ref-mp", type=float, default=1.0)
ap.add_argument("--out-mp", type=float, default=1.0)
a = ap.parse_args()
H = {"X-PG": a.password, "User-Agent": "Mozilla/5.0"}


def multipart(fields, files):
    b = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, path in files.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{os.path.basename(path)}"\r\nContent-Type: image/png\r\n\r\n'.encode()
        body += open(path, "rb").read() + b"\r\n"
    body += f"--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


p = {"n_images": len(a.images), "prompt": a.prompt, "seed": a.seed, "count": 1, "steps": a.steps, "ref_mp": a.ref_mp, "out_mp": a.out_mp,
     "size_from": 0, "lora": a.lora != "off", "lora_strength": 0.8 if a.lora == "off" else float(a.lora)}
body, ct = multipart({"params": json.dumps(p)}, {f"img{i}": f for i, f in enumerate(a.images)})
r = urllib.request.Request(a.url + "/api/submit", data=body, headers=dict(H, **{"Content-Type": ct}))
jid = json.load(urllib.request.urlopen(r, timeout=120))["ids"][0]
print("submitted", jid, flush=True)
t0 = time.time()
while True:
    s = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/api/state", headers=H), timeout=60))
    j = next(x for x in s["jobs"] if x["id"] == jid)
    if j["status"] in ("done", "error", "lost", "cancelled"):
        break
    time.sleep(5)
print(j["status"], j.get("secs"), "s", j.get("error", ""), f"(wall {time.time() - t0:.0f}s)")
if j.get("file"):
    open(a.out, "wb").write(urllib.request.urlopen(urllib.request.Request(a.url + "/files/" + j["file"], headers=H), timeout=120).read())
    print("saved", a.out)
