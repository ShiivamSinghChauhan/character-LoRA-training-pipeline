"""Polls the commit run on account C (CDP 9223); when it finishes, downloads every set_partNN.zip to DEST and extracts the images.
usage: python overnight_watch.py NOTEBOOK USER PORT DEST_DIR"""
import sys, re, os, time, shutil, tempfile, zipfile
from playwright.sync_api import sync_playwright

NB, USER, PORT, DEST = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(DEST, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def status(pg):
    try: pg.goto(f"https://www.kaggle.com/code/{USER}/{NB}/edit", wait_until="domcontentloaded", timeout=60000)
    except Exception: pass
    pg.wait_for_timeout(18000)
    try: pg.get_by_text("View Active Events", exact=False).first.click(); pg.wait_for_timeout(2500)
    except Exception: pass
    t = pg.evaluate("()=>document.body.innerText")
    m = re.findall(r"Version #(\d+)(?: with GPU T4 x2)?\s*([A-Za-z]+:? ?[^\n]*)", t)
    return m[:2]


def download_all(b, pg):
    cdp = b.new_browser_cdp_session()
    DL = tempfile.mkdtemp(prefix="kdl_")
    cdp.send("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": DL, "eventsEnabled": True})
    try: pg.goto(f"https://www.kaggle.com/code/{USER}/{NB}/output", wait_until="commit", timeout=60000)
    except Exception: pass
    pg.wait_for_timeout(15000)
    pg.get_by_text("zips", exact=True).first.click(); pg.wait_for_timeout(3000)
    names = sorted(set(re.findall(r"set_part\d+\.zip", pg.evaluate("()=>document.body.innerText"))))
    log("zips on output:", names)
    got = []
    for n in names:
        for f in os.listdir(DL): os.remove(os.path.join(DL, f))
        pg.get_by_text(n, exact=True).first.click(); pg.wait_for_timeout(3000)
        pg.locator('span[aria-label="Download"][role="button"]').first.click()
        t0 = time.time(); new = []
        while time.time() - t0 < 400 and not new:
            time.sleep(4); new = [f for f in os.listdir(DL) if not f.endswith(".crdownload")]
        if new:
            time.sleep(2); dst = os.path.join(DEST, n); shutil.move(os.path.join(DL, new[0]), dst)
            with zipfile.ZipFile(dst) as z: z.extractall(DEST)
            got.append(n); log("saved", n)
        else:
            log("FAILED to download", n)
        back = pg.locator('[aria-label^="Back to"]')
        if back.count(): back.first.click(); pg.wait_for_timeout(2000)
    return got


with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
    pg = [x for c in b.contexts for x in c.pages if "kaggle.com" in x.url][0]
    pg.bring_to_front()
    log("status", status(pg))
    log("DOWNLOADED", download_all(b, pg))
