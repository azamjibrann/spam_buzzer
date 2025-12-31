from playwright.sync_api import sync_playwright
import time
import pandas as pd

URL = "https://www.tiktok.com/@detikcom/video/7543556637768846597"

comments = []

def handle_response(response):
    try:
        if "api/comment/list" in response.url:
            data = response.json()
            for c in data.get("comments", []):
                comments.append({
                    "author": c["user"]["nickname"],
                    "author_id": c["user"]["unique_id"],
                    "comment": c["text"]
                })
    except:
        pass

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        locale="en-US"
    )
    page = context.new_page()

    page.on("response", handle_response)

    print("Membuka halaman...")
    page.goto(URL, timeout=60000)

    # ⏳ tunggu awal load komentar
    time.sleep(20)

    # ===== SCROLL BERDASARKAN DURASI =====
    DURATION = 180  # detik → 3 menit
    start_time = time.time()

    print("Mulai scroll komentar...")
    while time.time() - start_time < DURATION:
        page.mouse.wheel(0, 3000)
        time.sleep(2)

    print("Selesai scroll")

    browser.close()

# ===== SIMPAN HASIL =====
df = pd.DataFrame(comments).drop_duplicates()
print(df.head())
print("Total komentar:", len(df))

df.to_csv("tiktok_comments.csv", index=False, encoding="utf-8-sig")
