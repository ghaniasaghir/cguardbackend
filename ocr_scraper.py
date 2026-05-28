from curl_cffi import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

TARGET_STATIONS = ["Khanki", "Panjnad", "Marala", "Qadirabad", "Trimmu"]

def clean_number(value):
    if value is None:
        return None
    value = str(value).replace(",", "").strip()
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return None
    return float(match.group())

def extract_date(text):
    patterns = [
        r"\d{1,2}-[A-Za-z]{3,9}-\d{4}\s+\d{1,2}:\d{2}\s*PKT",
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\s+\d{1,2}:\d{2}",
        r"\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_ocr_scraper():
    print("🌐 Connecting to PMD FFD Dashboard via curl_cffi...")

    url = "https://ffd.pmd.gov.pk/staff/discharge-report-carousel"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://ffd.pmd.gov.pk/",
    }

    response = requests.get(
        url,
        headers=headers,
        impersonate="chrome120",
        timeout=20
    )

    if response.status_code != 200:
        raise Exception(f"PMD request failed with status code: {response.status_code}")

    html = response.text
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    print("✅ Successfully bypassed Cloudflare/WAF layer!")

    report_date = extract_date(text)
    print(f"📅 Extracted Date: {report_date}")

    results = {}

    # Table/row based parsing first
    rows = soup.find_all(["tr", "div", "li"])

    for row in rows:
        row_text = row.get_text(" ", strip=True)

        for station in TARGET_STATIONS:
            if station.lower() in row_text.lower():
                numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", row_text)
                numbers = [clean_number(n) for n in numbers]
                numbers = [n for n in numbers if n is not None]

                # PMD discharge values are usually larger than 1000.
                discharge_candidates = [n for n in numbers if n >= 1000]

                if discharge_candidates:
                    discharge = discharge_candidates[0]

                    results[station] = {
                        "date": report_date,
                        "station": station,
                        "inflow": discharge,
                        "discharge": discharge,
                    }

    # Regex fallback only if table parsing missed stations
    for station in TARGET_STATIONS:
        if station in results:
            continue

        pattern = rf"{station}\D{{0,120}}(\d[\d,]*(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            discharge = clean_number(match.group(1))

            if discharge and discharge >= 1000:
                results[station] = {
                    "date": report_date,
                    "station": station,
                    "inflow": discharge,
                    "discharge": discharge,
                }

    for station in TARGET_STATIONS:
        if station in results:
            print(f"✅ Found {station}: {results[station]['discharge']}")
        else:
            print(f"⚠️ Missing {station}")

    if not results:
        raise Exception("No station discharge values found from PMD page.")

    return results

if __name__ == "__main__":
    import json
    print(json.dumps(run_ocr_scraper(), indent=2))
