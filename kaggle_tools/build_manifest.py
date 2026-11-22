"""Builds E:/train-lora/kaggle_upload/ (renamed copies of raw_data images + manifest.json with one prompt per image).
Re-run any time the raw_data folders change. Deterministic (seeded)."""
import os, re, json, random, shutil

RAW = "E:/train-lora/dataset/raw_data"
OUT = "E:/train-lora/kaggle_upload"
FOLDERS = {"change-bg-n-outfit": "n", "change-bg-but-not-outfit": "b", "dont-change-bg": "d", "grey-or-light-grey-bg": "g"}

CORE = ("Restore and enhance this photo of the same person. Sharpen details and improve clarity. Keep the exact same face, "
        "identity, expression, pose, hair and hair length. Do not change her body shape, proportions, height or muscle definition. "
        "Preserve the natural fine skin texture and subtle details of the original, without adding marks, dark spots, blotches, "
        "freckles or discoloration, and without airbrushing. ")
REAL_BG = (" The background must look like a real, naturally photographed place, slightly out of focus with natural depth of field, "
           "not a cutout and not a composite.")
NOT_AI = " Photorealistic, candid, not stylized, not AI-looking."
CLOTH_REAL = " The clothing must look like real, well-fitting fabric with natural folds and seams that follow her body."
SOFT = " Use soft, diffused light, and adjust the light and skin tone on the person subtly so they match the place realistically, with natural shadows."

rng = random.Random(20261119)


def pick(lst):
    return rng.choice(lst)


# ---------- folder d: upscale only
def prompt_d():
    return CORE + "Keep the original background and clothing exactly as they are."


# ---------- folder g: grey / light grey
def prompt_g(i):
    shade = "light grey" if i % 2 == 0 else "medium grey"
    return (CORE + "Keep the exact same clothing. Remove the background and replace it with a plain, seamless, solid " + shade +
            " studio background, with soft even lighting on the person and clean natural hair edges.")


# ---------- folder b: background only
PLACES = [
    "empty sandy beach with the sea behind her", "quiet green garden with trees and a stone path", "minimal modern living room with a sofa and a plant",
    "bright bedroom with a bed and a window", "hotel balcony overlooking a calm city skyline", "poolside terrace with stone tiles and plants",
    "rooftop terrace with a low wall and the sky behind her", "plain textured wall in a minimal room", "forest path with tall trees",
    "empty quiet street in the early morning", "calm lakeside with a wooden deck", "sunlit hallway with large windows",
    "empty cafe corner with a wooden table", "minimal studio room with a beige wall and a large window", "field of tall grass at the edge of a village",
    "stone courtyard with arches and plants",
]
LIGHTS = [
    "Soft gentle sunlight.", "Normal natural daylight.", "Direct sunlight with defined natural shadows.",
    "No direct sunlight, soft overcast light.", "Light reflecting softly off water and nearby surfaces.", "Warm low sunlight from the side.",
]


def prompt_b():
    s = CORE + "Keep her outfit exactly as it is. Replace the background with a real " + pick(PLACES) + ", with no other people around." + REAL_BG
    if rng.random() < 0.30:
        s += " " + pick(LIGHTS)
    return s + NOT_AI


# ---------- folder n: outfit + background
SAREE_COL = ["emerald green", "wine red", "royal blue", "black", "pastel pink", "soft gold", "ivory", "teal", "maroon", "mustard yellow", "deep purple", "peach"]
SAREE_BLOUSE = ["sleeveless", "short-sleeved", "halter-neck", "backless-style"]
BLOUSE_COL = ["red", "royal blue", "emerald green", "pink", "black", "orange", "golden yellow", "white"]
KURTA_COL = ["white", "pastel blue", "mint green", "peach", "lavender", "yellow", "light pink", "black"]
KURTA_NECK = [
    "a normal neckline",
    "a slightly low neckline showing a little cleavage, no nudity",
    "a deep neckline showing some cleavage, no nudity",
]
SET_COL = ["black", "white", "red", "nude-beige", "navy", "burgundy"]
ONEPIECE = ["red", "orange", "yellow", "pink", "light pink"]
HOME_BG = [
    "Indian home living room with a sofa and warm lamps", "Indian home balcony with plants and soft daylight", "cozy bedroom with a bed and a window",
    "courtyard of an Indian house with plants and stone floor", "hotel lobby with soft warm lighting", "quiet garden terrace with plants",
    "modern apartment living room with large windows", "cafe interior with wooden furniture and warm light",
]


def outfit(kind):
    if kind == "saree":
        return ("Change her outfit to a modern " + pick(SAREE_COL) + " silk saree with a " + pick(SAREE_BLOUSE) +
                " blouse with a deep neckline showing some cleavage, no nudity, draped elegantly with simple jewelry")
    if kind == "blouse":
        return ("Change her outfit to a " + pick(BLOUSE_COL) + " Indian blouse with a low neckline showing some cleavage, no nudity, "
                "and a matching petticoat skirt, an elegant modern look")
    if kind == "set":
        return "Change her outfit to a " + pick(SET_COL) + " bra and panty set styled like a bikini, no nudity"
    if kind == "kurta":
        return "Change her outfit to a " + pick(KURTA_COL) + " salwar kurta with " + pick(KURTA_NECK)
    return "Change her outfit to a " + pick(ONEPIECE) + " one-piece swimsuit, no nudity"


def kinds_for(n):
    w = [("saree", 0.40), ("blouse", 0.20), ("kurta", 0.20), ("set", 0.10), ("one", 0.10)]
    out = []
    for k, p in w:
        out += [k] * round(n * p)
    while len(out) < n:
        out.append("saree")
    out = out[:n]
    rng.shuffle(out)
    return out


def prompt_n(kind):
    return CORE + outfit(kind) + ". Replace the background with a real " + pick(HOME_BG) + "." + REAL_BG + CLOTH_REAL + SOFT + NOT_AI


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    man = {}
    for folder, code in FOLDERS.items():
        files = sorted(os.listdir(os.path.join(RAW, folder)), key=lambda s: int(re.search(r"(\d+)", s).group(1)))
        kinds = kinds_for(len(files)) if code == "n" else None
        for k, f in enumerate(files):
            stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(f)[0]).strip("_")
            new = f"{code}_{stem}.png"
            shutil.copy(os.path.join(RAW, folder, f), os.path.join(OUT, new))
            if code == "d":
                pr, tag = prompt_d(), "upscale"
            elif code == "g":
                pr, tag = prompt_g(k), "grey"
            elif code == "b":
                pr, tag = prompt_b(), "bgonly"
            else:
                pr, tag = prompt_n(kinds[k]), "outfit_" + kinds[k]
            man[new] = {"orig": f, "folder": folder, "tag": tag, "prompt": pr}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"), indent=1)
    from collections import Counter
    print(len(man), "images;", dict(Counter(v["tag"] for v in man.values())))
    mb = sum(os.path.getsize(os.path.join(OUT, x)) for x in os.listdir(OUT)) / 1e6
    print(f"{mb:.0f} MB in {OUT}")


if __name__ == "__main__":
    main()
