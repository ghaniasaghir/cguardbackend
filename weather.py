
import requests

# Exact coordinates for each station
STATION_COORDS = {
    "Marala": {"lat": 32.6667, "lon": 74.4667},
    "Khanki": {"lat": 32.4167, "lon": 73.9667},
    "Qadirabad": {"lat": 32.3167, "lon": 73.6667},
    "Trimmu": {"lat": 31.1667, "lon": 72.1333},
    "Panjnad": {"lat": 29.3833, "lon": 71.6833}
}

# Soil moisture ranges for each station (mm)
# From your historical data analysis
SOIL_MOISTURE_RANGES = {
    "Marala": {"dry": 10, "normal": 25, "wet": 45},
    "Khanki": {"dry": 8, "normal": 20, "wet": 40},
    "Qadirabad": {"dry": 8, "normal": 22, "wet": 42},
    "Trimmu": {"dry": 12, "normal": 28, "wet": 48},
    "Panjnad": {"dry": 10, "normal": 25, "wet": 45}
}

def get_weather_for_station(station_name):
    """Get weather data for a specific station"""
    
    coords = STATION_COORDS.get(station_name)
    if not coords:
        # Fallback to Trimmu coordinates
        coords = {"lat": 31.1667, "lon": 72.1333}
    
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,rain,soil_moisture_0_to_1cm"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        current = data.get("current", {})
        
        # Soil moisture from API (0-1 scale) convert to mm (0-100)
        soil_fraction = current.get("soil_moisture_0_to_1cm", 0.13)
        soil_moisture_mm = round(soil_fraction * 100, 1)
        
        # Validate soil moisture against expected range
        ranges = SOIL_MOISTURE_RANGES.get(station_name, {"normal": 25})
        if soil_moisture_mm < 5:
            soil_moisture_mm = ranges["normal"]
        
        return {
            "station": station_name,
            "temperature": current.get("temperature_2m", 25.0),
            "rainfall": current.get("rain", 0.0),
            "soil_moisture": soil_moisture_mm,
            "timestamp": data.get("current", {}).get("time", "UNKNOWN")
        }
        
    except Exception as e:
        print(f"⚠️ Weather error for {station_name}: {e}")
        # Return fallback values
        return {
            "station": station_name,
            "temperature": 25.0,
            "rainfall": 0.0,
            "soil_moisture": SOIL_MOISTURE_RANGES.get(station_name, {"normal": 25})["normal"],
            "timestamp": "UNKNOWN"
        }

def get_all_weather_data():
    """Get weather for ALL stations"""
    results = {}
    for station in STATION_COORDS.keys():
        results[station] = get_weather_for_station(station)
        print(f"  🌡️ {station}: {results[station]['temperature']}°C, {results[station]['rainfall']}mm rain, {results[station]['soil_moisture']}mm soil")
    return results

def get_weather_data():
    """Legacy function for backward compatibility - returns Trimmu weather"""
    return get_weather_for_station("Trimmu")

if __name__ == "__main__":
    print("\n📊 Weather for all stations:")
    print("="*50)
    all_weather = get_all_weather_data()
    print("\n" + "="*50)
    for station, data in all_weather.items():
        print(f"{station}: {data['temperature']}°C, {data['rainfall']}mm, soil: {data['soil_moisture']}mm")
