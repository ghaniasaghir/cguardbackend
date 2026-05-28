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
    print("🌐 Connecting to PMD FFD Dashboard via curl_cffi + Dynamic Route Tunnel...")

    url = "https://ffd.pmd.gov.pk/staff/discharge-report-carousel"

    # REINFORCED BROWSER HEADERS LAYER (Bypasses advanced Cloudflare structure checks)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://ffd.pmd.gov.pk/",
        "Origin": "https://ffd.pmd.gov.pk",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"'
    }

    # fallback order of proxies if the primary connection gets rejected
    proxy_attempts = [
        # Attempt 1: Your Premium Webshare Connection
        "http://hnearlfg:wozfs4njseds@38.154.203.95:5863",
        # Attempt 2: Verified Open Pakistan Residential Proxy 
        "http://110.39.183.34:8080",
        # Attempt 3: Secondary Open Pakistan Proxy
        "http://202.163.111.130:8080"
    ]

    html = None
    response_status = None

    for idx, proxy_str in enumerate(proxy_attempts):
        try:
            print(f"📡 Trying tunnel connection layer {idx + 1}...")
            proxies = {"http": proxy_str, "https": proxy_str}
            
            response = requests.get(
                url,
                headers=headers,
                proxies=proxies,
                impersonate="chrome120",
                timeout=15
            )
            
            response_status = response.status_code
            if response.status_code == 200:
                html = response.text
                print(f"✅ Route layer {idx + 1} cleared the firewall seamlessly!")
                break
            else:
                print(f"⚠️ Layer {idx + 1} returned status: {response.status_code}. Retrying alternate path...")
        except Exception as e:
            print(f"❌ Layer {idx + 1} network connection timed out: {e}")

    # If all tunnels fail, attempt a direct residential pass as a final recovery option
    if not html:
        print("🔄 All proxies exhausted. Attempting final direct request layer...")
        try:
            response = requests.get(url, headers=headers, impersonate="chrome120", timeout=15)
            response_status = response.status_code
            if response.status_code == 200:
                html = response.text
                print("✅ Direct fallback successfully cleared firewall!")
        except Exception as e:
            print(f"❌ Direct pass failed: {e}")

    if not html:
        raise Exception(f"PMD request failed completely. End status code: {response_status}")

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    report_date = extract_date(text)
    print(f"📅 Extracted Date: {report_date}")

    results = {}
    rows = soup.find_all(["tr", "div", "li"])

    for row in rows:
        row_text = row.get_text(" ", strip=True)
        for station in TARGET_STATIONS:
            if station.lower() in row_text.lower():
                numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", row_text)
                numbers = [clean_number(n) for n in numbers]
                numbers = [n for n in numbers if n is not None]
                discharge_candidates = [n for n in numbers if n >= 1000]

                if discharge_candidates:
                    discharge = discharge_candidates[0]
                    results[station] = {
                        "date": report_date,
                        "station": station,
                        "inflow": discharge,
                        "discharge": discharge,
                    }

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
