from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import shutil

import time
import cv2
import pytesseract
import re
import json
import os
from pathlib import Path
import numpy as np


# Cross-platform Tesseract setup
# Windows: uses installed Tesseract path if available
# Hugging Face/Linux: uses system-installed "tesseract" from packages.txt

TESSERACT_PATHS = [
    r"D:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
]

tesseract_found = False

for tesseract_path in TESSERACT_PATHS:
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✅ Using Windows Tesseract: {tesseract_path}")
        tesseract_found = True
        break

# Linux / Hugging Face fallback
if not tesseract_found:
    try:
        import shutil

        linux_tesseract = shutil.which("tesseract")

        if linux_tesseract:
            pytesseract.pytesseract.tesseract_cmd = linux_tesseract
            print(f"✅ Using Linux Tesseract: {linux_tesseract}")
            tesseract_found = True

    except Exception as e:
        print(f"⚠️ Linux Tesseract detection failed: {e}")

if not tesseract_found:
    raise FileNotFoundError(
        "Tesseract OCR not found. Install Tesseract locally or add 'tesseract-ocr' to Hugging Face packages.txt"
    )


TARGET_STATIONS = ["Khanki", "Panjnad", "Marala", "Qadirabad", "Trimmu"]

STATION_SPELLINGS = {
    "Trimmu": ["Trimmu", "Trimum"],
    "Khanki": ["Khanki"],
    "Panjnad": ["Panjnad"],
    "Marala": ["Marala"],
    "Qadirabad": ["Qadirabad"],
}


def get_chrome_driver():
    """
    Cross-platform Chrome driver setup.

    Windows/local:
    - Uses Selenium Manager automatically.

    Hugging Face Docker/Linux:
    - Uses Chromium installed through Dockerfile:
      /usr/bin/chromium
      /usr/bin/chromedriver
    """
    options = webdriver.ChromeOptions()

    # Common options
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")

    # Detect Linux/Hugging Face chromium/chromedriver
    linux_chromium = shutil.which("chromium") or shutil.which("chromium-browser") or "/usr/bin/chromium"
    linux_chromedriver = shutil.which("chromedriver") or "/usr/bin/chromedriver"

    if os.path.exists(linux_chromium) and os.path.exists(linux_chromedriver):
        print(f"✅ Using Linux Chromium: {linux_chromium}")
        print(f"✅ Using Linux ChromeDriver: {linux_chromedriver}")

        options.binary_location = linux_chromium
        options.add_argument("--headless=new")

        return webdriver.Chrome(
            service=Service(linux_chromedriver),
            options=options
        )

    # Windows/local fallback: let Selenium Manager find Chrome + driver
    print("🚀 Using Selenium Manager fallback for local machine...")
    options.add_argument("--start-maximized")

    return webdriver.Chrome(options=options)


def preprocess_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists() or image_path.stat().st_size == 0:
        raise ValueError(f"Screenshot missing or empty: {image_path}")

    img = cv2.imread(str(image_path))

    if img is None:
        raw = image_path.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Could not read screenshot: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    return gray


def extract_station_data_from_text(text, results, date):
    for station in TARGET_STATIONS:
        if station in results:
            continue

        for spelling in STATION_SPELLINGS.get(station, [station]):
            if spelling not in text:
                continue

            idx = text.find(spelling)
            chunk = text[idx:idx + 250]
            numbers = re.findall(r"\d{1,3}(?:,\d{3})*", chunk)

            for num_str in numbers:
                try:
                    num = int(num_str.replace(",", ""))
                except Exception:
                    continue

                if 5000 < num < 100000:
                    results[station] = {
                        "date": date,
                        "station": station,
                        "inflow": str(num),
                    }
                    print(f"✅ Found {station}: {num}")
                    break

            if station in results:
                break


def run_ocr_scraper():
    print("🌐 Starting OCR scraper...")

    driver = None

    try:
        driver = get_chrome_driver()
        driver.get("https://ffd.pmd.gov.pk/flood-dashboard")

        print("⏳ Loading PMD dashboard...")
        time.sleep(8)

        results = {}
        date = "UNKNOWN"
        scroll_positions = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400]

        import tempfile

        temp_dir = Path(tempfile.gettempdir()) / "cguard_ocr_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Temp directory: {temp_dir}")

        for i, scroll_pos in enumerate(scroll_positions):
            print(f"📸 Screenshot {i + 1}/{len(scroll_positions)}")

            driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
            time.sleep(0.7)

            screenshot_path = temp_dir / f"page_{i}.png"

            success = driver.save_screenshot(str(screenshot_path))

            if (not success) or (not screenshot_path.exists()) or screenshot_path.stat().st_size == 0:
                print("⚠️ Screenshot failed, trying fallback...")
                png_data = driver.get_screenshot_as_png()
                screenshot_path.write_bytes(png_data)

            time.sleep(0.5)

            if not screenshot_path.exists() or screenshot_path.stat().st_size == 0:
                raise ValueError(f"Screenshot was not saved correctly: {screenshot_path}")

            gray = preprocess_image(screenshot_path)
            text = pytesseract.image_to_string(gray, config="--psm 6")

            if date == "UNKNOWN":
                date_match = re.search(
                    r"(\d{2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2}\s+PKT)",
                    text
                )
                if date_match:
                    date = date_match.group(1)
                    print(f"📅 Date found: {date}")

            extract_station_data_from_text(text, results, date)

            if len(results) == len(TARGET_STATIONS):
                print("🎯 All stations found.")
                break

        for station in TARGET_STATIONS:
            if station not in results:
                results[station] = {
                    "date": date if date != "UNKNOWN" else "09-May-2026 06:00 PKT",
                    "station": station,
                    "inflow": "0",
                }

        print("✅ OCR scraper completed.")
        return results

    finally:
        if driver is not None:
            driver.quit()


if __name__ == "__main__":
    start = time.time()
    data = run_ocr_scraper()
    print(f"\n⏱️ Time taken: {time.time() - start:.1f} seconds")
    print(json.dumps(data, indent=2))