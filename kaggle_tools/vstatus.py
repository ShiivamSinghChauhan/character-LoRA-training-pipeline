import sys, re
from playwright.sync_api import sync_playwright
NB = sys.argv[1]
USER = sys.argv[2] if len(sys.argv) > 2 else "sh1vamsinghchauhan"
PORT = sys.argv[3] if len(sys.argv) > 3 else "9222"
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
    pg = [x for c in b.contexts for x in c.pages if "kaggle.com" in x.url][0]
    pg.bring_to_front()
    try: pg.goto(f"https://www.kaggle.com/code/{USER}/{NB}/edit", wait_until="domcontentloaded", timeout=60000)
    except Exception: pass
    pg.wait_for_timeout(18000)
    try: pg.get_by_text("View Active Events", exact=False).first.click(); pg.wait_for_timeout(2500)
    except Exception: pass
    t = pg.evaluate("()=>document.body.innerText")
    m = re.findall(r"Version #(\d+) with GPU T4 x2\s*([A-Za-z]+:? ?[^\n]*)", t)
    print("STATUS", m[:3])
