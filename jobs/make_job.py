"""usage: python make_job.py OUT.py   -- builds the outfit+background test job (frame 99) from klein_base.py"""
import sys, textwrap, ast
base = ("Restore and enhance this photo of the same person. Sharpen details and improve clarity. Keep the exact same face, identity, "
        "expression, pose, hair and hair length. Do not change her body shape, proportions, height or muscle definition. Preserve the "
        "natural fine skin texture and subtle details of the original, without adding marks, dark spots, blotches, freckles or "
        "discoloration, and without airbrushing. ")
tail = (" The clothing must look like real, well-fitting fabric with natural folds and seams that follow her body. The background must look "
        "like a real, naturally photographed place, slightly out of focus with natural depth of field, not a cutout and not a composite. "
        "Use soft, diffused light, no harsh direct sun, and adjust the light and skin tone on the person subtly so they match the place "
        "realistically, with natural shadows. Photorealistic, candid, not stylized, not AI-looking.")
V = {
    "C1": "Change her outfit to a white linen shirt with the sleeves rolled up and blue jeans. Replace the background with a real living room with a sofa and a plant, lit by warm late-afternoon window light.",
    "C2": "Change her outfit to a red summer dress. Replace the background with the interior of a real cafe with wooden tables and chairs and warm ambient indoor light.",
    "C3": "Change her outfit to a black blazer over a white shirt with tailored trousers. Replace the background with a real bright modern office with a desk and large windows, in soft daylight.",
    "C4": "Change her outfit to a beige trench coat over a knit sweater and jeans. Replace the background with a real quiet city street in autumn, under soft overcast daylight.",
    "C5": "Change her outfit to an oversized grey hoodie and black leggings. Replace the background with a real cozy bedroom with a bed and soft daylight from a window.",
}
out = ["VARIANTS = ["]
for k, v in V.items():
    out.append(f"    ('{k}', (")
    for p in textwrap.wrap(base + v + tail, 100):
        out.append("        " + repr(p + " "))
    out.append("    )),")
out.append("]")
s = open("E:/train-lora/_tools/klein_base.py", encoding="utf-8").read()
s = s.replace("@@ORDER@@", "[99]").replace("@@TAGRE@@", "C[0-9]").replace("@@ZIPNAME@@", "klein10_part").replace("@@VARIANTS@@", "\n".join(out))
ast.parse(s)
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write(s)
print("ok, longest line", max(len(l) for l in s.split("\n")))
