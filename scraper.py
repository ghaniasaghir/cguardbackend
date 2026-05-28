from ocr_scraper import run_ocr_scraper

def get_flood_data():
    raw = run_ocr_scraper()

    readings = []

    for station, item in raw.items():
        discharge = item.get("discharge") or item.get("inflow")

        if discharge is None:
            continue

        readings.append({
            "station": station,
            "date": item.get("date"),
            "discharge": float(discharge),
            "rainfall_mm": 0,
            "temperature_c": 25,
            "soil_moisture_mm": 25,
            "source": "curl_cffi_scraper"
        })

    if not readings:
        raise Exception("curl_cffi scraper returned no usable readings.")

    return readings
