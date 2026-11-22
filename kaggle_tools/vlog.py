"""usage: python vlog.py NOTEBOOK USER PORT N PATTERN -- prints the last N log lines matching PATTERN from the notebook's latest *successful* version log
(while a version is still running Kaggle shows only the previous one; use the Active Events page for state)."""
import sys, re
from playwright.sync_api import sync_playwright
NB, USER, PORT, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
PAT = sys.argv[5] if len(sys.argv) > 5 else "."
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
    pg = [x for c in b.contexts for x in c.pages if "kaggle.com" in x.url][0]
    pg.bring_to_front()
    try: pg.goto(f"https://www.kaggle.com/code/{USER}/{NB}/log", wait_until="commit", timeout=60000)
    except Exception: pass
    pg.wait_for_timeout(20000)
    try: pg.get_by_role("tab", name="Logs").first.click(); pg.wait_for_timeout(8000)
    except Exception as e: print("tab?", e)
    t = pg.evaluate("()=>document.body.innerText")
    j = t.find("Log Message"); body = t[j:] if j >= 0 else t
    lines = [re.sub(r"^[0-9.]+s\t\d+\t", "", l) for l in body.split("\n") if l.strip()]
    keep = [l for l in lines if re.search(PAT, l)]
    print("\n".join(x[:230].encode("ascii", "ignore").decode() for x in keep[-N:]))
