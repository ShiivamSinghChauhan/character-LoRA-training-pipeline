"""usage: python who.py PORT [PORT...] -- prints the Kaggle username logged in on each Chrome debug port (via the profile link in /settings)"""
import sys, re
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    for port in sys.argv[1:]:
        try:
            b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            pg = [x for c in b.contexts for x in c.pages][0]
            pg.goto("https://www.kaggle.com/settings", wait_until="domcontentloaded", timeout=45000)
            pg.wait_for_timeout(6000)
            t = pg.evaluate("()=>document.body.innerText")
            hrefs = pg.evaluate("()=>[...document.querySelectorAll('a')].map(a=>a.getAttribute('href')).filter(h=>h&&/^\/[A-Za-z0-9_-]+$/.test(h))")
            emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", t)
            print(port, "login" if "login" in pg.url else "ok", sorted(set(hrefs))[:12], emails[:2])
        except Exception as e:
            print(port, "ERR", repr(e)[:100])
