"""ComfyUI API graph for FLUX.2 Klein 9B multi-reference editing (+ optional consistency LoRA).

Pure function, no I/O: build(params, loaders) -> graph dict. `loaders` says which loader node classes exist on the
running ComfyUI (stock or ComfyUI-MultiGPU) so the same code works in both modes.
"""

UNET = "flux-2-klein-9b.safetensors"
LORA = "Flux2-Klein-9B-consistency-V2.safetensors"
CLIP = "qwen_3_8b_fp8mixed.safetensors"
VAE = "flux2-vae.safetensors"


def round16(v):
    return max(256, int(round(v / 16.0)) * 16)


def out_size(w, h, mp):
    """Scale (w, h) to ~mp megapixels keeping the aspect ratio; both sides multiples of 16."""
    s = (mp * 1_000_000 / float(w * h)) ** 0.5
    return round16(w * s), round16(h * s)


def build(p, loaders=None):
    """p keys: images [filenames in ComfyUI input dir], prompt, seed, steps, ref_mp, width, height,
    lora (bool), lora_strength, prefix. loaders: {"unet": (class, extra_inputs), "clip": (class, extra_inputs)}."""
    loaders = loaders or {}
    uc, ux = loaders.get("unet", ("UNETLoader", {}))
    cc, cx = loaders.get("clip", ("CLIPLoader", {}))
    g = {
        "1": {"class_type": uc, "inputs": dict({"unet_name": p.get("unet", UNET), "weight_dtype": "fp8_e4m3fn"}, **ux)},
        "3": {"class_type": cc, "inputs": dict({"clip_name": CLIP, "type": "flux2", "device": "default"}, **cx)},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["3", 0]}},
        "10": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["8", 0]}},
    }
    model = ["1", 0]
    if p.get("lora", True):
        g["2"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": LORA, "strength_model": float(p.get("lora_strength", 0.8))}}
        model = ["2", 0]
    pos, neg = ["8", 0], ["10", 0]
    for i, img in enumerate(p.get("images", [])):
        k = str(100 + i * 10)
        g[k] = {"class_type": "LoadImage", "inputs": {"image": img}}
        g[k + "1"] = {"class_type": "ImageScaleToTotalPixels",
                      "inputs": {"image": [k, 0], "upscale_method": "lanczos", "megapixels": float(p.get("ref_mp", 1.0)), "resolution_steps": 16}}
        g[k + "2"] = {"class_type": "VAEEncode", "inputs": {"pixels": [k + "1", 0], "vae": ["4", 0]}}
        g[k + "3"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [k + "2", 0]}}
        g[k + "4"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": neg, "latent": [k + "2", 0]}}
        pos, neg = [k + "3", 0], [k + "4", 0]
    w, h = int(p["width"]), int(p["height"])
    g.update({
        "12": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": pos, "negative": neg, "cfg": 1.0}},
        "14": {"class_type": "Flux2Scheduler", "inputs": {"steps": int(p.get("steps", 8)), "width": w, "height": h}},
        "15": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "16": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(p["seed"])}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "18": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["16", 0], "guider": ["12", 0], "sampler": ["17", 0],
                                                              "sigmas": ["14", 0], "latent_image": ["15", 0]}},
        "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["4", 0]}},
        "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": p["prefix"]}},
    })
    return g
