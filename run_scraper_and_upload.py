import requests
from scraper import get_flood_data

RAILWAY_BACKEND_URL = "https://cguardbackend-production.up.railway.app"

data = get_flood_data()

print("SCRAPER OUTPUT BEFORE UPLOAD:")
for row in data:
    print(row)

for row in data:
    payload = {
        "station": row["station"],
        "discharge": row["discharge"],
        "rainfall_mm": row.get("rainfall_mm", 0),
        "temperature_c": row.get("temperature_c", 25),
        "soil_moisture_mm": row.get("soil_moisture_mm", 25),
        "source": "local_scheduled_scraper",
        "reading_time": None
    }

    response = requests.post(
        f"{RAILWAY_BACKEND_URL}/station-readings/manual",
        json=payload,
        timeout=30
    )

    print(row["station"], response.status_code, response.text)
