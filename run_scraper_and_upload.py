import os
import requests
from weather import get_weather_for_station
from scraper import get_flood_data



RAILWAY_BACKEND_URL = os.getenv("RAILWAY_BACKEND_URL", "https://cguardbackend-production.up.railway.app")

data = get_flood_data()

print("SCRAPER OUTPUT BEFORE UPLOAD:")
for row in data:
    print(row)

for row in data:
    weather = get_weather_for_station(row["station"])

    payload = {
        "station": row["station"],
        "discharge": row["discharge"],
        "rainfall_mm": weather.get("rainfall", 0),
        "temperature_c": weather.get("temperature", 25),
        "soil_moisture_mm": weather.get("soil_moisture", 25),
        "source": "local_scheduled_scraper",
        "reading_time": None
    }

    response = requests.post(
        f"{RAILWAY_BACKEND_URL}/station-readings/manual",
        json=payload,
        timeout=30
    )

    print(row["station"], response.status_code, response.text)
