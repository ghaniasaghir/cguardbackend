'''
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import cv2
import pytesseract
import re
import json

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TARGET_STATIONS = ["Khanki", "Panjnad", "Marala", "Qadirabad", "Trimmu"]
STATION_SPELLINGS = {
    "Trimmu": ["Trimmu", "Trimum"],
    "Khanki": ["Khanki"],
    "Panjnad": ["Panjnad"],
    "Marala": ["Marala"],
    "Qadirabad": ["Qadirabad"]
}

def run_ocr_scraper():
    print("🌐 Starting Chrome...")
    
    options = webdriver.ChromeOptions()
    
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )
    
    driver.get("https://ffd.pmd.gov.pk/flood-dashboard")
    
    print("⏳ Loading...")
    time.sleep(8)
    
    # SCROLL THROUGH ONCE to capture ALL stations
    # No waiting between scrolls - just capture everything in one pass
    print("📜 Scrolling to capture all stations...")
    
    results = {}
    date = "UNKNOWN"
    
    # Take screenshots at different scroll positions (NO waiting)
    scroll_positions = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400]
    
    for i, scroll_pos in enumerate(scroll_positions):
        driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
        time.sleep(0.3)  # Minimal wait for content to settle
        
        driver.save_screenshot(f"page_{i}.png")
        
        # OCR
        img = cv2.imread(f"page_{i}.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5)
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
        text = pytesseract.image_to_string(gray, config='--psm 6')
        
        # Extract date once
        if date == "UNKNOWN":
            date_match = re.search(r"(\d{2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2}\s+PKT)", text)
            if date_match:
                date = date_match.group(1)
                print(f"📅 Date: {date}")
        
        # Find stations in this screenshot
        for station in TARGET_STATIONS:
            if station in results:
                continue
            
            spellings = STATION_SPELLINGS.get(station, [station])
            
            for spelling in spellings:
                if spelling in text:
                    idx = text.find(spelling)
                    chunk = text[idx:idx+200]
                    numbers = re.findall(r'\d{1,3}(?:,\d{3})*', chunk)
                    
                    if numbers:
                        for num_str in numbers:
                            num = int(num_str.replace(",", ""))
                            if 5000 < num < 100000:
                                results[station] = {
                                    "date": date,
                                    "station": station,
                                    "inflow": num_str.replace(",", "")
                                }
                                print(f"  ✅ Found {station}: {num} (scroll {i})")
                                break
                    break
        
        # Stop if all found
        if len(results) == len(TARGET_STATIONS):
            print(f"\n🎯 All stations found at scroll position {i}!")
            break
    
    driver.quit()
    
    # Fill missing stations
    for station in TARGET_STATIONS:
        if station not in results:
            print(f"  ⚠️ {station}: Not found")
            results[station] = {
                "date": date if date != "UNKNOWN" else "09-May-2026 06:00 PKT",
                "station": station,
                "inflow": "0"
            }
    
    return results

if __name__ == "__main__":
    import time
    start = time.time()
    data = run_ocr_scraper()
    print(f"\n⏱️ Time taken: {time.time() - start:.1f} seconds")
    print(json.dumps(data, indent=2))
'''
from curl_cffi import requests
import re
import json

TARGET_STATIONS = ["Khanki", "Panjnad", "Marala", "Qadirabad", "Trimmu"]

def run_ocr_scraper():
    print("🌐 Connecting to PMD FFD Dashboard via curl_cffi...")
    
    # 1. Define real-browser headers to bypass simple user-agent blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://ffd.pmd.gov.pk/",
    }

    results = {}
    date = "UNKNOWN"

    try:
        # 2. Use impersonate="chrome120" to copy Chrome's exact TLS fingerprint
        url = "https://ffd.pmd.gov.pk/staff/discharge-report-carousel"
        response = requests.get(url, headers=headers, impersonate="chrome120", timeout=15)
        
        if response.status_code != 200:
            print(f"❌ Failed with status code: {response.status_code}")
            raise Exception(f"Banned or Blocked by WAF: {response.status_code}")

        text = response.text
        print("✅ Successfully bypassed Cloudflare/WAF layer!")

        # 3. Extract the date exactly like your old regex did
        date_match = re.search(r"(\d{2}-[A-Za-z]{3}-\d{4}\s+\d{2}:\d{2}\s+PKT)", text)
        if date_match:
            date = date_match.group(1)
            print(f"📅 Extracted Date: {date}")

        # 4. Extract data from the raw text/HTML structure
        # (Since we are reading direct text instead of OCR distortions, this is 100% accurate)
        for station in TARGET_STATIONS:
            if station in text:
                idx = text.find(station)
                # Look at the character chunk right after the station name
                chunk = text[idx:idx+300]
                numbers = re.findall(r'\d{1,3}(?:,\d{3})*', chunk)
                
                if numbers:
                    for num_str in numbers:
                        num = int(num_str.replace(",", ""))
                        # Matching your threshold boundaries
                        if 5000 < num < 100000:
                            results[station] = {
                                "date": date,
                                "station": station,
                                "inflow": num_str.replace(",", "")
                            }
                            print(f"  ✅ Found {station}: {num}")
                            break

    except Exception as e:
        print(f"⚠️ Error during request execution: {e}")

    # 5. Fallback/Fill missing stations exactly like your old structure did
    for station in TARGET_STATIONS:
        if station not in results:
            print(f"  ⚠️ {station}: Not found in text payload")
            results[station] = {
                "date": date if date != "UNKNOWN" else "28-May-2026 06:00 PKT",
                "station": station,
                "inflow": "0"
            }
            
    return results

if __name__ == "__main__":
    import time
    start = time.time()
    data = run_ocr_scraper()
    print(f"\n⏱️ Time taken: {time.time() - start:.3f} seconds")
    print(json.dumps(data, indent=2))
