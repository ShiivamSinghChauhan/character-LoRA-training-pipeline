"""usage: python addsecret.py PORT NOTEBOOK_URL [LABEL] -- adds the secret LABEL (default HF_TOKEN) from _secrets/hf_token.txt to the notebook and prints the panel state (never the value)"""
import sys, os
from playwright.sync_api import sync_playwright
port, url = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else "HF_TOKEN"
val = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_secrets", "hf_token.txt")).read().strip()
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    pg = [x for c in b.contexts for x in c.pages if "kaggle.com" in x.url][0]
    pg.goto(url, wait_until="domcontentloaded", timeout=60000); pg.wait_for_timeout(15000)
    pg.get_by_text("Add-ons", exact=True).first.click(timeout=10000); pg.wait_for_timeout(1000)
    pg.locator("li", has_text="Secrets").first.click(); pg.wait_for_timeout(3000)
    if label in pg.evaluate("()=>document.body.innerText").split("Secrets", 2)[-1]:
        print("already has", label)
    else:
        pg.get_by_role("button", name="Add secret").first.click(); pg.wait_for_timeout(2000)
        inputs = pg.locator("input:visible")
        n = inputs.count()
        print("visible inputs", n)
        inputs.nth(n - 2).fill(label); inputs.nth(n - 1).fill(val); pg.wait_for_timeout(500)
        for name in ("Save", "Add", "Done"):
            btn = pg.get_by_role("button", name=name, exact=True)
            if btn.count():
                btn.last.click(); print("clicked", name); break
        pg.wait_for_timeout(3000)
    t = pg.evaluate("()=>document.body.innerText")
    print(repr(t[t.find("Add-Ons"):][:300]).replace(val, "<token>"))
