"""usage: python run_cell.py PORT NOTEBOOK_EDIT_URL CELL.py [WAIT_PATTERN] [WAIT_SECONDS]
Types CELL.py into the notebook's code cell, runs it in the interactive session, then polls the page for WAIT_PATTERN
and prints the matching line (or the latest timestamped log line while waiting)."""
import sys, re, time
from playwright.sync_api import sync_playwright

port, url, cellfile = sys.argv[1], sys.argv[2], sys.argv[3]
pat = sys.argv[4] if len(sys.argv) > 4 else "PLAYGROUND_URL"
wait = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
code = open(cellfile, encoding="utf-8").read().rstrip("\n")
NL = "\n"
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    pg = [x for c in b.contexts for x in c.pages if "kaggle.com" in x.url][0]
    pg.goto(url, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_timeout(25000)
    try:  # stop a still-running previous cell so this one can start
        pg.get_by_role("button", name="More settings").first.click(timeout=5000)
        pg.wait_for_timeout(800)
        it = pg.get_by_text("Interrupt execution")
        if it.count():
            it.first.click()
            pg.wait_for_timeout(6000)
            print("interrupted previous cell")
        else:
            pg.keyboard.press("Escape")
    except Exception as e:
        print("interrupt skipped", repr(e)[:80])
    fr = [f for f in pg.frames if f.locator(".cm-content").count() > 0][0]
    c = fr.locator(".cm-content").first
    c.click()
    pg.wait_for_timeout(400)
    pg.keyboard.press("Control+a")
    pg.keyboard.press("Delete")
    pg.wait_for_timeout(500)
    chunk = ""
    for ln in code.split(NL):
        if len(chunk) + len(ln) > 900:
            pg.keyboard.insert_text(chunk)
            pg.wait_for_timeout(250)
            chunk = ""
        chunk += ln + NL
    pg.keyboard.insert_text(chunk.rstrip(NL))
    pg.wait_for_timeout(1500)
    pg.keyboard.press("Shift+Enter")
    print("cell started", time.strftime("%H:%M:%S"), flush=True)
    t0 = time.time()
    last = ""
    while time.time() - t0 < wait:
        pg.wait_for_timeout(20000)
        txt = ""
        for f in pg.frames:
            try:
                txt += f.evaluate("()=>document.body?document.body.innerText:''") + NL
            except Exception:
                pass
        hits = [ln for ln in txt.split(NL) if re.search(pat, ln)]
        tail = [ln for ln in txt.split(NL) if re.match(r"\d\d:\d\d:\d\d ", ln)][-1:]
        cur = hits[-1] if hits else (tail[0] if tail else "")
        if cur != last:
            print(time.strftime("%H:%M:%S"), cur.encode("ascii", "ignore").decode()[:300], flush=True)
            last = cur
        if hits and "None" not in cur:
            break
