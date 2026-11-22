"""usage: python push_job.py JOB.py NOTEBOOK_SLUG  -- types JOB.py into the notebook cell on account A and commits with GPU"""
import sys, re
from playwright.sync_api import sync_playwright
code = open(sys.argv[1], encoding="utf-8").read().rstrip("\n")
NB = sys.argv[2]
USER = sys.argv[3] if len(sys.argv) > 3 else "sh1vamsinghchauhan"
PORT = sys.argv[4] if len(sys.argv) > 4 else "9222"
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
    pg = [x for c in b.contexts for x in c.pages if "kaggle.com" in x.url][0]
    pg.bring_to_front()
    pg.goto(f"https://www.kaggle.com/code/{USER}/{NB}/edit", wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_timeout(25000)
    fr = [f for f in pg.frames if f.locator(".cm-content").count() > 0][0]
    c = fr.locator(".cm-content").first
    c.click(); pg.wait_for_timeout(400); pg.keyboard.press("Control+a"); pg.keyboard.press("Delete"); pg.wait_for_timeout(500)
    chunk = ""
    for ln in code.split("\n"):
        while len(ln) > 900:
            pg.keyboard.insert_text(ln[:900]); pg.wait_for_timeout(250); ln = ln[900:]
        if len(chunk) + len(ln) > 900:
            pg.keyboard.insert_text(chunk); pg.wait_for_timeout(250); chunk = ""
        chunk += ln + "\n"
    pg.keyboard.insert_text(chunk.rstrip("\n")); pg.wait_for_timeout(2000)
    seen = set(); pg.keyboard.press("Control+Home"); pg.wait_for_timeout(500)
    for i in range(30):
        for ln in c.inner_text().split("\n"): seen.add(ln.strip())
        pg.keyboard.press("PageDown"); pg.wait_for_timeout(300)
    miss = [l for l in (x.strip() for x in code.split("\n") if x.strip()) if l not in seen and len(l) < 300]
    print("missing", len(miss), miss[:2])
    if not miss:
        pg.get_by_role("button", name="Save Version").first.click(); pg.wait_for_timeout(3000)
        pg.get_by_text("Advanced Settings", exact=True).first.click(); pg.wait_for_timeout(1500)
        print(re.findall(r"Run with [^\n]*", pg.evaluate("()=>document.body.innerText")))
        pg.get_by_role("button", name="Save", exact=True).last.click(); pg.wait_for_timeout(15000); print("committed")
