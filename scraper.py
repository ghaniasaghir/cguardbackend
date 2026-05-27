from ocr_scraper import run_ocr_scraper
from datetime import datetime
import re
import json
import requests

try:
    import pandas as pd
except Exception:
    pd = None


PMD_DISCHARGE_URL = "https://ffd.pmd.gov.pk/staff/discharge-report-carousel"

EXPECTED_STATIONS = ["Khanki", "Marala", "Panjnad", "Qadirabad", "Trimmu"]

STATION_ALIASES = {
    "khanki": "Khanki",
    "khankl": "Khanki",
    "khank": "Khanki",

    "marala": "Marala",
    "maria": "Marala",
    "mirala": "Marala",

    "panjnad": "Panjnad",
    "punjnad": "Panjnad",
    "panjned": "Panjnad",
    "panj nad": "Panjnad",

    "qadirabad": "Qadirabad",
    "qadir abad": "Qadirabad",
    "qadirbad": "Qadirabad",
    "qadirabed": "Qadirabad",

    "trimmu": "Trimmu",
    "trimu": "Trimmu",
    "trimum": "Trimmu",
}


def _normalize_station(value):
    if value is None:
        return None

    cleaned = str(value).strip()
    lowered = re.sub(r"\s+", " ", cleaned.lower())

    if lowered in STATION_ALIASES:
        return STATION_ALIASES[lowered]

    for key, station in STATION_ALIASES.items():
        if key in lowered:
            return station

    for station in EXPECTED_STATIONS:
        if station.lower() in lowered:
            return station

    return None


def _to_float(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


def _extract_discharge_from_record(record):
    """
    Handles common scraper/OCR output shapes:
    - {"station": "Khanki", "discharge": 12345}
    - {"Station": "Khanki", "Inflow": "12,345"}
    - {"Khanki": 12345}
    - string rows from OCR
    """
    if isinstance(record, dict):
        # Case: {"Khanki": 12345}
        for key, value in record.items():
            station = _normalize_station(key)
            if station:
                discharge = _to_float(value)
                if discharge is not None:
                    return station, discharge

        station = (
            record.get("station")
            or record.get("Station")
            or record.get("name")
            or record.get("Name")
            or record.get("site")
            or record.get("Site")
        )
        station = _normalize_station(station)

        discharge = None
        for key in [
            "discharge", "Discharge",
            "inflow", "Inflow",
            "flow", "Flow",
            "current_discharge", "Current_Discharge",
            "value", "Value",
            "reading", "Reading",
        ]:
            if key in record:
                discharge = _to_float(record.get(key))
                if discharge is not None:
                    break

        if station and discharge is not None:
            return station, discharge

    if isinstance(record, str):
        station = _normalize_station(record)
        if not station:
            return None, None

        numbers = [_to_float(x) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", record)]
        numbers = [x for x in numbers if x is not None]

        if numbers:
            # Usually discharge is the largest numeric value in the station row.
            return station, max(numbers)

    return None, None


def _normalize_scraper_output(data):
    """
    Converts OCR/direct scraper output into:
    {
        "Khanki": {"station": "Khanki", "discharge": 123, "source": "..."},
        ...
    }
    """
    readings = {}

    if data is None:
        return readings

    # If OCR returns JSON string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = data.splitlines()

    # If OCR returns dict of station -> values OR {"data": [...]}.
    if isinstance(data, dict):
        possible_lists = [
            data.get("data"),
            data.get("readings"),
            data.get("stations"),
            data.get("results"),
        ]

        nested = next((x for x in possible_lists if isinstance(x, list)), None)

        if nested is not None:
            data = nested
        else:
            data = [data]

    if not isinstance(data, list):
        data = [data]

    for record in data:
        station, discharge = _extract_discharge_from_record(record)
        if station and discharge is not None:
            readings[station] = {
                "station": station,
                "discharge": float(discharge),
                "source": "ocr_scraper",
                "reading_time": datetime.utcnow().isoformat(),
            }

    return readings


def _read_pmd_tables():
    """
    Reads PMD discharge-report-carousel directly.
    This is used as a correction layer for OCR issues like:
    - Khanki accidentally copying Marala
    - Panjnad missing from OCR
    """
    readings = {}

    try:
        response = requests.get(
            PMD_DISCHARGE_URL,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 C-Guard backend scraper"
            }
        )
        response.raise_for_status()
        html = response.text
    except Exception as e:
        print(f"⚠️ Could not fetch PMD discharge carousel directly: {e}")
        return readings

    # Best path: parse HTML tables with pandas.
    if pd is not None:
        try:
            tables = pd.read_html(html)

            for table in tables:
                table = table.fillna("")
                columns = [str(c).strip().lower() for c in table.columns]

                for _, row in table.iterrows():
                    row_values = [str(v).strip() for v in row.tolist()]
                    row_text = " ".join(row_values)
                    station = _normalize_station(row_text)

                    if not station:
                        continue

                    discharge = None

                    # Prefer inflow/discharge/current columns if table headers exist.
                    preferred_keywords = [
                        "inflow",
                        "discharge",
                        "current",
                        "flow",
                        "value",
                    ]

                    for i, col in enumerate(columns):
                        if any(key in col for key in preferred_keywords):
                            discharge = _to_float(row_values[i])
                            if discharge is not None:
                                break

                    # Fallback: choose largest realistic number from row.
                    if discharge is None:
                        nums = [_to_float(v) for v in row_values]
                        nums = [n for n in nums if n is not None]
                        if nums:
                            discharge = max(nums)

                    if discharge is not None:
                        readings[station] = {
                            "station": station,
                            "discharge": float(discharge),
                            "source": "pmd_discharge_report_carousel",
                            "reading_time": datetime.utcnow().isoformat(),
                        }

        except Exception as e:
            print(f"⚠️ pandas table parse failed: {e}")

    # Regex fallback if pandas cannot parse.
    if not readings:
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)

        for station in EXPECTED_STATIONS:
            pattern = rf"({station}).{{0,500}}?(\d[\d,]{{3,}}(?:\.\d+)?)"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                discharge = _to_float(match.group(2))
                if discharge is not None:
                    readings[station] = {
                        "station": station,
                        "discharge": float(discharge),
                        "source": "pmd_discharge_report_carousel_regex",
                        "reading_time": datetime.utcnow().isoformat(),
                    }

    print("Direct PMD parsed readings:", readings)
    return readings


def _fix_known_ocr_station_issues(ocr_readings, direct_readings):
    """
    Merge direct PMD readings with OCR readings.

    Direct PMD page is preferred because the ML partner confirmed:
    https://ffd.pmd.gov.pk/staff/discharge-report-carousel
    contains the current values.

    This specifically prevents:
    - Khanki wrongly copying Marala OCR value
    - Panjnad missing when direct page has it
    """
    final = {}

    for station in EXPECTED_STATIONS:
        if station in direct_readings:
            final[station] = direct_readings[station]
        elif station in ocr_readings:
            final[station] = ocr_readings[station]

    # Safety: if OCR gave Khanki exactly same as Marala and direct did not confirm Khanki,
    # do not return a wrong copied Khanki value.
    if (
        "Khanki" in final
        and "Marala" in final
        and "Khanki" not in direct_readings
        and final["Khanki"]["discharge"] == final["Marala"]["discharge"]
    ):
        print("⚠️ Removing suspicious Khanki value because it equals Marala and was not confirmed by direct PMD page.")
        final.pop("Khanki", None)

    return final


def get_flood_data():
    print("Calling OCR scraper...")

    ocr_data = run_ocr_scraper()
    print("OCR returned:", ocr_data)

    ocr_readings = _normalize_scraper_output(ocr_data)
    print("Normalized OCR readings:", ocr_readings)

    direct_readings = _read_pmd_tables()

    final_readings = _fix_known_ocr_station_issues(
        ocr_readings=ocr_readings,
        direct_readings=direct_readings,
    )

    missing = [station for station in EXPECTED_STATIONS if station not in final_readings]
    if missing:
        print(f"⚠️ Missing stations after scraper correction: {missing}")

    result = list(final_readings.values())
    print("Final scraper readings:", result)

    return result
