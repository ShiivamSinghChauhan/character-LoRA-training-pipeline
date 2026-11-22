"""usage: python setup_session.py PORT NOTEBOOK_EDIT_URL -- turns internet ON and picks the T4 x2 accelerator for the notebook's interactive session"""
import sys
from playwright.sync_api import sync_playwright
port, url = sys.argv[1], sys.argv[2]
JS = "()=>[...document.querySelectorAll('li,[role=menuitem],[role=option],[role=radio]')].map(e=>e.innerText.split(String.fromCharCode(10)).join(' ')).filter(Boolean).slice(-14)"
with sync_playwright() as p:
    b = p.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
    pg = [x for c in b.contexts for x in c.pages if "kaggle.com" in x.url][0]
    pg.goto(url, wait_until="domcontentloaded", timeout=60000); pg.wait_for_timeout(15000)
    pg.get_by_text("Settings", exact=True).first.click(); pg.wait_for_timeout(1000)
    if pg.get_by_text("Turn on internet").count():
        pg.get_by_text("Turn on internet").first.click(); pg.wait_for_timeout(2500)
        print("internet turned on")
        pg.get_by_text("Settings", exact=True).first.click(); pg.wait_for_timeout(1000)
    pg.get_by_text("Accelerator").first.click(); pg.wait_for_timeout(1500)
    print(pg.evaluate(JS))
    t4 = pg.get_by_text("GPU T4 x2")
    if t4.count():
        t4.first.click(); pg.wait_for_timeout(2500); print("clicked T4 x2")
    bt = pg.get_by_role("button", name="Turn on GPU T4 x2")
    if bt.count():
        bt.first.click(); pg.wait_for_timeout(3000); print("confirmed GPU dialog")
    print(pg.evaluate("()=>document.body.innerText").split("Draft Session")[0][-150:].replace(chr(10), " | "))
