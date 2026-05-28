from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import joblib
import numpy as np
import pandas as pd
import warnings
import math
import jwt
import bcrypt
import json
import os
import re
import secrets
import requests
import resend
from groq import Groq
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(title="C-Guard Backend")

# CORS: allow local Vite during development.
# For deployment, optionally set FRONTEND_ORIGINS as comma-separated URLs.
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "*")
ALLOWED_ORIGINS = [origin.strip() for origin in FRONTEND_ORIGINS.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False if ALLOWED_ORIGINS == ["*"] else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # Basic browser security headers.
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(self), microphone=(), camera=()"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"

    return response

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────
# Use DATABASE_URL in deployment. If it is not provided, fall back to local SQLite
# so the backend can still run on Hugging Face/local without PostgreSQL crashing.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cguard.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ─────────────────────────────────────────────
# DATABASE TABLES
# ─────────────────────────────────────────────
class UserDB(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String)
    email      = Column(String, unique=True, index=True)
    password   = Column(String)
    role       = Column(String, default="authority")
    created_at = Column(DateTime, default=datetime.utcnow)

class ReportDB(Base):
    __tablename__ = "reports"
    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String)
    location     = Column(String)
    description  = Column(Text)
    submitted_at = Column(DateTime, default=datetime.utcnow)

class ContactDB(Base):
    __tablename__ = "contact_messages"
    id      = Column(Integer, primary_key=True, index=True)
    name    = Column(String)
    email   = Column(String)
    subject = Column(String)
    message = Column(Text)
    sent_at = Column(DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────────
# NEW: FLOOD DATA TABLE
# Stores ML input features + prediction results
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# UNION COUNCIL TABLE
# Stores UC-level GIS/administrative mapping information
# ─────────────────────────────────────────────
class UnionCouncilDB(Base):
    __tablename__ = "union_councils"
    id         = Column(Integer, primary_key=True, index=True)
    uc_name    = Column(String, index=True)
    district   = Column(String, index=True)
    station    = Column(String, index=True)
    geometry   = Column(Text, nullable=True)      # GeoJSON/polygon text if available
    created_at = Column(DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────────
# SHELTER TABLE
# Replaces old in-memory shelters_db list with persistent SQLite storage
# ─────────────────────────────────────────────
class ShelterDB(Base):
    __tablename__ = "shelters"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String)
    location   = Column(String)
    district   = Column(String, nullable=True)
    contact_number = Column(String, nullable=True)
    capacity   = Column(Integer)
    occupied   = Column(Integer, default=0)
    status     = Column(String)
    facilities = Column(Text)                    # JSON string list
    uc_id      = Column(Integer, nullable=True)   # Logical FK to union_councils.id
    updated_at = Column(DateTime, default=datetime.utcnow)

# ─────────────────────────────────────────────
# EXPORT REPORT HISTORY TABLE
# Stores authority export/report generation history
# ─────────────────────────────────────────────
class ExportReportDB(Base):
    __tablename__ = "export_reports"
    id              = Column(Integer, primary_key=True, index=True)
    authority_id    = Column(Integer, nullable=True)  # Logical FK to users.id
    station         = Column(String, nullable=True)
    forecast_period = Column(String, nullable=True)
    report_type     = Column(String)                  # PDF / CSV
    generated_at    = Column(DateTime, default=datetime.utcnow)

class FloodDataDB(Base):
    __tablename__ = "flood_data"
    id                = Column(Integer, primary_key=True, index=True)
    uc_id             = Column(Integer, nullable=True)     # Logical FK to union_councils.id
    station           = Column(String, index=True)        # e.g. Marala, Trimmu
    location          = Column(String)                    # human-readable area name
    rainfall          = Column(Float, default=0.0)        # mm/24h
    humidity          = Column(Float, default=50.0)       # percentage
    temperature       = Column(Float, default=25.0)       # Celsius
    river_level       = Column(Float, default=5.0)        # metres
    current_discharge = Column(Float, default=5000.0)     # m³/s
    prediction_result = Column(Float, nullable=True)      # predicted discharge m³/s
    risk_percentage   = Column(Float, nullable=True)      # 0–100
    risk_level        = Column(String, nullable=True)     # Low/Medium/High/Very High/Exceptionally High
    created_at        = Column(DateTime, default=datetime.utcnow)


class AlertSubscriptionDB(Base):
    __tablename__ = "alert_subscriptions"
    id                   = Column(Integer, primary_key=True, index=True)
    uc_name              = Column(String, index=True)
    district             = Column(String, nullable=True)
    latitude             = Column(Float, nullable=True)
    longitude            = Column(Float, nullable=True)
    email                = Column(String, nullable=True)
    phone                = Column(String, nullable=True)
    email_alerts         = Column(Boolean, default=False)
    sms_alerts           = Column(Boolean, default=False)
    threshold            = Column(String, default="Very High only")
    is_active            = Column(Boolean, default=True)
    last_alert_sent_at   = Column(DateTime, nullable=True)
    created_at           = Column(DateTime, default=datetime.utcnow)


class StationReadingDB(Base):
    """
    Stores live/scraped station readings in PostgreSQL.
    These readings are used to calculate lag, rolling mean/std, and dQ features
    required by the trained 24h/48h/72h XGBoost models.
    """
    __tablename__ = "station_readings"

    id               = Column(Integer, primary_key=True, index=True)
    station          = Column(String, index=True)
    reading_time     = Column(DateTime, index=True, default=datetime.utcnow)
    discharge        = Column(Float, default=0.0)       # PMD inflow/discharge
    rainfall_mm      = Column(Float, default=0.0)
    temperature_c    = Column(Float, default=25.0)
    soil_moisture_mm = Column(Float, default=25.0)
    source           = Column(String, default="manual_or_scraper")
    created_at       = Column(DateTime, default=datetime.utcnow)



# Only these sources count as real observed/current historical readings.
# Fallback rows are for temporary prediction support only and must not be shown
# as actual scraper/database history in the authority dashboard.
REAL_READING_SOURCES = [
    "scheduled_scraper",
    "manual_scraper",
    "scraper",
    "ocr_scraper",
    "manual",
    "api_manual",
]

FALLBACK_READING_SOURCES = [
    "forecast_dashboard_fallback",
    "forecast_request_fallback",
    "location_request_fallback",
]


Base.metadata.create_all(bind=engine)

# SQLite create_all does not add new columns to existing tables.
def ensure_database_columns():
    from sqlalchemy import inspect, text
    inspector = inspect(engine)

    if "flood_data" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("flood_data")]
        if "uc_id" not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE flood_data ADD COLUMN uc_id INTEGER"))
                conn.commit()


    if "station_readings" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("station_readings")]

        new_columns = {
            "rainfall_mm": "DOUBLE PRECISION DEFAULT 0.0",
            "temperature_c": "DOUBLE PRECISION DEFAULT 25.0",
            "soil_moisture_mm": "DOUBLE PRECISION DEFAULT 25.0",
            "source": "TEXT DEFAULT 'manual_or_scraper'",
        }

        with engine.connect() as conn:
            for column_name, column_type in new_columns.items():
                if column_name not in columns:
                    conn.execute(
                        text(f"ALTER TABLE station_readings ADD COLUMN {column_name} {column_type}")
                    )
            conn.commit()

    # SQLite create_all does not add new columns to existing tables.
    # These columns support frontend alert subscriptions with email/phone.
    if "shelters" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("shelters")]
        if "contact_number" not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE shelters ADD COLUMN contact_number TEXT"))
                conn.commit()

    if "alert_subscriptions" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("alert_subscriptions")]

        new_columns = {
            "email": "TEXT",
            "phone": "TEXT",
            "is_active": "BOOLEAN DEFAULT 1",
            "last_alert_sent_at": "DATETIME",
        }

        with engine.connect() as conn:
            for column_name, column_type in new_columns.items():
                if column_name not in columns:
                    conn.execute(
                        text(f"ALTER TABLE alert_subscriptions ADD COLUMN {column_name} {column_type}")
                    )
            conn.commit()

ensure_database_columns()


# ─────────────────────────────────────────────
# SEED SAMPLE FLOOD DATA
# Runs once on startup — skips if data exists
# ─────────────────────────────────────────────
def seed_flood_data():
    db = SessionLocal()
    try:
        if db.query(FloodDataDB).count() == 0:
            samples = [
                FloodDataDB(uc_id=1, station="Marala",    location="Hafizabad UC 1", rainfall=12.5, humidity=72.0, temperature=32.0, river_level=9.8,  current_discharge=8500,  prediction_result=9200,  risk_percentage=45.2, risk_level="Moderate"),
                FloodDataDB(uc_id=2, station="Marala",    location="Hafizabad UC 2", rainfall=18.0, humidity=80.0, temperature=30.5, river_level=11.2, current_discharge=12000, prediction_result=14500, risk_percentage=67.8, risk_level="High"),
                FloodDataDB(uc_id=3, station="Khanki",    location="Chiniot UC 3",   rainfall=5.0,  humidity=60.0, temperature=35.0, river_level=7.5,  current_discharge=5500,  prediction_result=5800,  risk_percentage=28.4, risk_level="Moderate"),
                FloodDataDB(uc_id=5, station="Qadirabad", location="Qadirabad Area", rainfall=22.0, humidity=85.0, temperature=28.0, river_level=10.5, current_discharge=18000, prediction_result=21000, risk_percentage=52.1, risk_level="High"),
                FloodDataDB(uc_id=4, station="Trimmu",    location="Jhang UC 4",     rainfall=30.0, humidity=90.0, temperature=27.5, river_level=12.1, current_discharge=25000, prediction_result=31000, risk_percentage=78.6, risk_level="Critical"),
                FloodDataDB(uc_id=4, station="Trimmu",    location="Jhang District",  rainfall=8.0,  humidity=65.0, temperature=33.0, river_level=6.2,  current_discharge=4500,  prediction_result=4800,  risk_percentage=22.3, risk_level="Low"),
                FloodDataDB(uc_id=1, station="Marala",    location="Hafizabad UC 1", rainfall=45.0, humidity=92.0, temperature=26.0, river_level=13.5, current_discharge=35000, prediction_result=42000, risk_percentage=88.5, risk_level="Critical"),
                FloodDataDB(uc_id=3, station="Khanki",    location="Chiniot UC 3",   rainfall=2.0,  humidity=55.0, temperature=38.0, river_level=5.1,  current_discharge=3000,  prediction_result=3100,  risk_percentage=14.2, risk_level="Low"),
                FloodDataDB(uc_id=6, station="Panjnad",   location="Panjnad Area",  rainfall=15.0, humidity=76.0, temperature=30.5, river_level=8.4,  current_discharge=6000,  prediction_result=7200,  risk_percentage=34.0, risk_level="Medium"),
            ]
            db.add_all(samples)
            db.commit()
            print("✅ Sample flood data seeded successfully")
    finally:
        db.close()

# seed_flood_data()  # Disabled: avoid mixing fake flood data with real scraper/ML data

# ─────────────────────────────────────────────
# JWT TOKEN SETUP + AUTHORITY SECURITY
# ─────────────────────────────────────────────
# For real deployment, set this in Windows environment:
# setx CGUARD_SECRET_KEY "your-very-long-random-secret"
SECRET_KEY = os.getenv(
    "CGUARD_SECRET_KEY",
    "CHANGE-THIS-CGUARD-SECRET-KEY-BEFORE-DEPLOYMENT-2026"
)
ALGORITHM = "HS256"

# Shorter session for authority dashboard.
TOKEN_EXPIRE_HOURS = 6

# Token extractor.
security_scheme = HTTPBearer(auto_error=True)

# Brute-force protection. This is in-memory and works for FYP/local deployment.
FAILED_LOGIN_ATTEMPTS = defaultdict(list)
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

# Authority registration is restricted.
# Frontend signup must send invite_code if you want signup to work.
ALLOW_PUBLIC_AUTHORITY_SIGNUP = os.getenv("ALLOW_PUBLIC_AUTHORITY_SIGNUP", "true").lower() == "true"
AUTHORITY_INVITE_CODE = os.getenv("CGUARD_AUTHORITY_INVITE_CODE", "CGUARD-ADMIN-2026")

# ─────────────────────────────────────────────
# RESEND EMAIL ALERT CONFIG
# ─────────────────────────────────────────────
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
resend.api_key = RESEND_API_KEY

CGUARD_SENDER_EMAIL = os.getenv("CGUARD_SENDER_EMAIL", "onboarding@resend.dev")
CGUARD_SENDER_NAME = os.getenv("CGUARD_SENDER_NAME", "C Guard Alerts")

LAST_EMAIL_ERROR = ""

# ─────────────────────────────────────────────
# GROQ AI CHATBOT CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def mask_secret(value: Optional[str]):
    if not value:
        return "NOT SET"
    if len(value) <= 10:
        return "***"
    return f"{value[:8]}...{value[-4:]}"


def create_token(email: str, role: str, name: str):
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    now = datetime.utcnow()
    data = {
        "sub": email,
        "email": email,
        "role": role,
        "name": name,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])

        if not payload.get("email") or not payload.get("role"):
            raise HTTPException(status_code=401, detail="Invalid token payload.")

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please login again.")


def require_authority(token: dict = Depends(verify_token)):
    """
    Use this dependency on authority-only endpoints.
    """
    if token.get("role") != "authority":
        raise HTTPException(status_code=403, detail="Authority access required.")
    return token


def validate_password_strength(password: str):
    """
    Frontend-compatible password rule.
    The frontend currently accepts 6+ characters, so backend must not reject
    a valid frontend signup. Strength can be tightened later after UI updates.
    """
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")
    return True



def normalize_email(email: str):
    return email.lower().strip()


def is_account_locked(email: str):
    now = datetime.utcnow()

    FAILED_LOGIN_ATTEMPTS[email] = [
        attempt_time for attempt_time in FAILED_LOGIN_ATTEMPTS[email]
        if now - attempt_time < timedelta(minutes=LOCKOUT_MINUTES)
    ]

    return len(FAILED_LOGIN_ATTEMPTS[email]) >= MAX_FAILED_ATTEMPTS


def record_failed_login(email: str):
    FAILED_LOGIN_ATTEMPTS[email].append(datetime.utcnow())


def clear_failed_logins(email: str):
    FAILED_LOGIN_ATTEMPTS[email] = []


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────
# DEFAULT ADMIN USER
# ─────────────────────────────────────────────
def create_default_users():
    db = SessionLocal()
    try:
        existing = db.query(UserDB).filter(UserDB.email == "authority@cguard.pk").first()

        if not existing:
            default_password = os.getenv("CGUARD_DEFAULT_PASSWORD", "CGuard@2026")
            hashed = bcrypt.hashpw(default_password.encode(), bcrypt.gensalt(rounds=12)).decode()

            user = UserDB(
                name="Authority User",
                email="authority@cguard.pk",
                password=hashed,
                role="authority"
            )

            db.add(user)
            db.commit()

            print("✅ Default authority user created")
            print("📧 Email: authority@cguard.pk")
            print(f"🔐 Password: {default_password}")
        else:
            print("✅ Default user already exists")
    finally:
        db.close()

create_default_users()

# ─────────────────────────────────────────────
# SEED UNION COUNCILS + SHELTERS
# Runs once on startup — skips if data exists
# ─────────────────────────────────────────────
def seed_union_councils_and_shelters():
    db = SessionLocal()
    try:
        if db.query(UnionCouncilDB).count() == 0:
            union_councils = [
                UnionCouncilDB(id=1, uc_name="UC 1 - Hafizabad", district="Hafizabad", station="Marala"),
                UnionCouncilDB(id=2, uc_name="UC 2 - Hafizabad", district="Hafizabad", station="Marala"),
                UnionCouncilDB(id=3, uc_name="UC 3 - Chiniot", district="Chiniot", station="Khanki"),
                UnionCouncilDB(id=4, uc_name="UC 4 - Jhang", district="Jhang", station="Trimmu"),
                UnionCouncilDB(id=5, uc_name="Qadirabad Area", district="Mandi Bahauddin", station="Qadirabad"),
                UnionCouncilDB(id=6, uc_name="Panjnad Area", district="Bahawalpur", station="Panjnad"),
            ]
            db.add_all(union_councils)
            db.commit()
            print("✅ Sample union councils seeded successfully")

        if db.query(ShelterDB).count() == 0:
            shelters = [
                ShelterDB(name="Kot Saleem Community Shelter", location="Kot Saleem, Jhang District", district="Jhang", capacity=180, occupied=112, status="Available", facilities=json.dumps(["Drinking Water", "Medical Aid", "Food Supply"]), uc_id=4),
                ShelterDB(name="Trimmu Relief Center", location="Trimmu, Jhang District", district="Jhang", capacity=250, occupied=98, status="Available", facilities=json.dumps(["Drinking Water", "Medical Aid"]), uc_id=4),
                ShelterDB(name="Qadirabad School Shelter", location="Qadirabad, Mandi Bahauddin District", district="Mandi Bahauddin", capacity=300, occupied=300, status="Full", facilities=json.dumps(["Drinking Water", "Medical Aid", "Food Supply"]), uc_id=5),
                ShelterDB(name="Khanki Health Post Shelter", location="Khanki, Gujrat District", district="Gujrat", capacity=200, occupied=45, status="Available", facilities=json.dumps(["Drinking Water", "Medical Aid"]), uc_id=3),
            ]
            db.add_all(shelters)
            db.commit()
            print("✅ Sample shelters seeded successfully")
    finally:
        db.close()

seed_union_councils_and_shelters()

# ─────────────────────────────────────────────
# LOAD ML MODELS
# ─────────────────────────────────────────────
print("Loading C-Guard 24h/48h/72h flood prediction models...")
try:
    model_24h = joblib.load("cguard_xgb_24h_model.pkl")
    model_48h = joblib.load("cguard_xgb_48h_model.pkl")
    model_72h = joblib.load("cguard_xgb_72h_model.pkl")

    scaler = joblib.load("cguard_scaler.pkl")
    feature_cols = joblib.load("cguard_feature_cols.pkl")
    station_encoder = joblib.load("cguard_station_encoder.pkl")

    MODEL_LOADED = True
    print("✅ 24h/48h/72h models loaded successfully")
    print(f"✅ Required features: {list(feature_cols)}")
except Exception as e:
    print(f"⚠️ ML models not loaded: {e}")
    MODEL_LOADED = False
    model_24h = None
    model_48h = None
    model_72h = None
    scaler = None
    feature_cols = []
    station_encoder = None

# ─────────────────────────────────────────────
# FLOOD CONSTANTS
# ─────────────────────────────────────────────
DISCHARGE_THRESHOLDS = {
    'critical': 50000, 'high': 30000, 'moderate': 15000, 'low': 5000,
}
STATION_GAUGE_CAPACITY = {
    'Marala': 12.0,
    'Khanki': 10.0,
    'Qadirabad': 11.0,
    'Trimmu': 9.0,
    'Panjnad': 8.5,
}

STATION_DISCHARGE_CAPACITY = {
    'Marala': 25000,
    'Khanki': 20000,
    'Qadirabad': 22000,
    'Trimmu': 18000,
    'Panjnad': 16000,
}

# Default discharge values per station (used when no live data)
STATION_DEFAULT_DISCHARGE = {
    'Marala': 8500,
    'Khanki': 5500,
    'Qadirabad': 10000,
    'Trimmu': 7000,
    'Panjnad': 6000,
}



def make_json_safe(value):
    """
    Converts NumPy/Pandas values into normal Python types so FastAPI can return JSON.
    """
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value

# Model threshold classification based on ML partner's predictor API.
def classify_model_flood_risk(discharge: float):
    discharge = float(discharge)

    if discharge < 100000:
        return "Normal"
    elif discharge < 150000:
        return "Low"
    elif discharge < 200000:
        return "Medium"
    elif discharge < 400000:
        return "High"
    elif discharge < 600000:
        return "Very High"
    return "Exceptionally High"


def parse_scraper_datetime(date_value: Optional[str]):
    """
    Parses dates like '09-May-2026 06:00 PKT'.
    If parsing fails, uses current UTC time.
    """
    if not date_value:
        return datetime.utcnow()

    text_value = str(date_value).replace("PKT", "").strip()

    for fmt in ["%d-%b-%Y %H:%M", "%d-%B-%Y %H:%M", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(text_value, fmt)
        except Exception:
            pass

    return datetime.utcnow()



def parse_date_range_value(value: Optional[str], end_of_day: bool = False):
    """
    Parses frontend date range values for historical mode.
    Accepts YYYY-MM-DD or full ISO datetime.
    """
    if not value:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    try:
        if "T" in raw:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)

        parsed = datetime.strptime(raw, "%Y-%m-%d")
        if end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
        return parsed
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use YYYY-MM-DD."
        )


def serialize_station_reading(row: StationReadingDB):
    return {
        "id": row.id,
        "station": row.station,
        "reading_time": row.reading_time.isoformat() if row.reading_time else None,
        "discharge": round(float(row.discharge or 0.0), 2),
        "rainfall_mm": round(float(row.rainfall_mm or 0.0), 2),
        "temperature_c": round(float(row.temperature_c or 25.0), 2),
        "soil_moisture_mm": round(float(row.soil_moisture_mm or 25.0), 2),
        "source": row.source,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def normalize_station_name(station: str):
    if not station:
        raise HTTPException(status_code=400, detail="Station is required.")

    station = station.strip()
    aliases = {
        "trimum": "Trimmu",
        "trimmu": "Trimmu",
        "khanki": "Khanki",
        "panjnad": "Panjnad",
        "marala": "Marala",
        "qadirabad": "Qadirabad",
    }

    normalized = aliases.get(station.lower())
    if not normalized:
        allowed = ["Khanki", "Marala", "Panjnad", "Qadirabad", "Trimmu"]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid station '{station}'. Allowed stations: {allowed}"
        )

    return normalized


def get_latest_station_reading(db: Session, station: str):
    station = normalize_station_name(station)
    return (
        db.query(StationReadingDB)
        .filter(StationReadingDB.station == station)
        .order_by(StationReadingDB.reading_time.desc(), StationReadingDB.id.desc())
        .first()
    )



def get_latest_real_station_reading(db: Session, station: str):
    """
    Latest actual observed reading only.
    Excludes forecast/location fallback rows so current/historical views are not polluted.
    """
    station = normalize_station_name(station)
    return (
        db.query(StationReadingDB)
        .filter(StationReadingDB.station == station)
        .filter(StationReadingDB.source.in_(REAL_READING_SOURCES))
        .order_by(StationReadingDB.reading_time.desc(), StationReadingDB.id.desc())
        .first()
    )


def build_temporary_station_reading(
    station: str,
    discharge: float,
    rainfall_mm: float = 0.0,
    temperature_c: float = 25.0,
    soil_moisture_mm: float = 25.0,
    reading_time: Optional[datetime] = None,
    source: str = "temporary_forecast_fallback"
):
    """
    Creates an in-memory StationReadingDB-like object for prediction fallback.
    It is NOT saved to database, so historical mode stays actual-only.
    """
    return StationReadingDB(
        station=normalize_station_name(station),
        reading_time=reading_time or datetime.utcnow(),
        discharge=float(discharge or 0.0),
        rainfall_mm=float(rainfall_mm or 0.0),
        temperature_c=float(temperature_c or 25.0),
        soil_moisture_mm=float(soil_moisture_mm or 25.0),
        source=source,
    )


def get_station_history(db: Session, station: str, limit: int = 14):
    """
    History used for ML lag features.
    Uses real observed readings only; excludes fallback rows.
    """
    station = normalize_station_name(station)
    return (
        db.query(StationReadingDB)
        .filter(StationReadingDB.station == station)
        .filter(StationReadingDB.source.in_(REAL_READING_SOURCES))
        .order_by(StationReadingDB.reading_time.desc(), StationReadingDB.id.desc())
        .limit(limit)
        .all()
    )


def calculate_ml_features_from_reading(db: Session, reading: StationReadingDB):
    """
    Creates the exact feature row expected by:
    cguard_feature_cols.pkl

    Current values come from scraper/weather.
    Lag/rolling/dQ values come from historical station_readings in PostgreSQL.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="ML models are not loaded.")

    station = normalize_station_name(reading.station)

    try:
        station_encoded = int(station_encoder.transform([station])[0])
    except Exception:
        allowed = list(getattr(station_encoder, "classes_", []))
        raise HTTPException(
            status_code=400,
            detail=f"Station '{station}' not supported by encoder. Allowed stations: {allowed}"
        )

    history = get_station_history(db, station, limit=14)
    discharges = [float(row.discharge or 0.0) for row in history]

    current_discharge = float(reading.discharge or 0.0)

    # If history is not enough yet, safely backfill with current discharge.
    while len(discharges) < 14:
        discharges.append(current_discharge)

    def lag(n: int):
        # history[0] is latest/current if just inserted, so lag_1 should use index 1.
        idx = n
        if idx < len(discharges):
            return float(discharges[idx])
        return current_discharge

    def mean_last(n: int):
        values = discharges[:n] if len(discharges) >= n else discharges
        return float(np.mean(values)) if values else current_discharge

    def std_last(n: int):
        values = discharges[:n] if len(discharges) >= n else discharges
        return float(np.std(values, ddof=0)) if values else 0.0

    reading_time = reading.reading_time or datetime.utcnow()
    month = int(reading_time.month)
    day_of_year = int(reading_time.timetuple().tm_yday)
    is_flood_season = 1 if month in [7, 8, 9] else 0

    feature_dict = {
        "Station_Encoded": int(station_encoded),
        "Discharge": current_discharge,
        "Rainfall_mm": float(reading.rainfall_mm or 0.0),
        "Temperature_C": float(reading.temperature_c or 25.0),
        "Soil_Moisture_mm": float(reading.soil_moisture_mm or 25.0),
        "Month": month,
        "DayOfYear": day_of_year,
        "IsFloodSeason": is_flood_season,
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_3": lag(3),
        "lag_5": lag(5),
        "lag_7": lag(7),
        "lag_14": lag(13),
        "mean_3d": mean_last(3),
        "std_3d": std_last(3),
        "mean_7d": mean_last(7),
        "std_7d": std_last(7),
        "mean_14d": mean_last(14),
        "std_14d": std_last(14),
        "dQ_1": current_discharge - lag(1),
        "dQ_3": current_discharge - lag(3),
    }

    input_df = pd.DataFrame([feature_dict])

    missing = [col for col in feature_cols if col not in input_df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing ML feature columns: {missing}"
        )

    input_df = input_df[list(feature_cols)]
    return input_df, feature_dict


def predict_from_station_reading(db: Session, reading: StationReadingDB):
    input_df, feature_dict = calculate_ml_features_from_reading(db, reading)
    input_scaled = scaler.transform(input_df)

    pred_24h = float(model_24h.predict(input_scaled)[0])
    pred_48h = float(model_48h.predict(input_scaled)[0])
    pred_72h = float(model_72h.predict(input_scaled)[0])

    return {
        "station": reading.station,
        "reading_time": reading.reading_time.isoformat() if reading.reading_time else None,
        "current_discharge": round(float(reading.discharge or 0.0), 2),
        "input_weather": {
            "rainfall_mm": round(float(reading.rainfall_mm or 0.0), 2),
            "temperature_c": round(float(reading.temperature_c or 25.0), 2),
            "soil_moisture_mm": round(float(reading.soil_moisture_mm or 25.0), 2),
        },
        "forecast": {
            "24h": {
                "predicted_discharge": round(pred_24h, 2),
                "risk_level": classify_model_flood_risk(pred_24h),
            },
            "48h": {
                "predicted_discharge": round(pred_48h, 2),
                "risk_level": classify_model_flood_risk(pred_48h),
            },
            "72h": {
                "predicted_discharge": round(pred_72h, 2),
                "risk_level": classify_model_flood_risk(pred_72h),
            },
        },
        "features_used": make_json_safe(feature_dict),
        "model_loaded": MODEL_LOADED,
    }


def save_station_reading(
    db: Session,
    station: str,
    discharge: float,
    rainfall_mm: float = 0.0,
    temperature_c: float = 25.0,
    soil_moisture_mm: float = 25.0,
    reading_time: Optional[datetime] = None,
    source: str = "manual"
):
    station = normalize_station_name(station)
    reading_time = reading_time or datetime.utcnow()

    # Duplicate protection:
    # If scraper runs more than once for the same station/time, reuse the existing row.
    existing = (
        db.query(StationReadingDB)
        .filter(StationReadingDB.station == station)
        .filter(StationReadingDB.reading_time == reading_time)
        .first()
    )

    if existing:
        print(f"ℹ️ Duplicate reading skipped: {station} at {reading_time}")
        return existing

    reading = StationReadingDB(
        station=station,
        reading_time=reading_time,
        discharge=float(discharge or 0.0),
        rainfall_mm=float(rainfall_mm or 0.0),
        temperature_c=float(temperature_c or 25.0),
        soil_moisture_mm=float(soil_moisture_mm or 25.0),
        source=source,
    )

    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


# ─────────────────────────────────────────────
# LOAD UC DATABASE FROM CSV
# Supports all UCs from uc_river_distance_lookup(1).csv.
# If a user location/UC is not found in this database, /forecast returns 0 risk.
# ─────────────────────────────────────────────
UC_DATABASE = {}

def load_uc_database_from_csv():
    """
    Loads UC lookup data.

    Priority:
    1. Fixed GeoJSON file from frontend/ML partner:
       chenab_ucs_FIXED.geojson / chenab_ucs_FIXED(1).geojson
    2. Old CSV fallback:
       uc_river_distance_lookup.csv

    The risk calculation uses Distance_to_River_km from this lookup.
    This avoids old Unknown_* UC names and wrong distance values.
    """
    global UC_DATABASE

    def choose_station_from_distance(distance):
        # Existing distance-based station routing kept unchanged.
        if distance <= 10:
            return "Trimmu"
        elif distance <= 20:
            return "Qadirabad"
        elif distance <= 35:
            return "Khanki"
        else:
            return "Marala"

    # ─────────────────────────────────────────
    # 1) Prefer fixed GeoJSON lookup
    # ─────────────────────────────────────────
    possible_geojson_paths = [
        "chenab_ucs_FIXED.geojson",
        "chenab_ucs_FIXED(1).geojson",
        "./chenab_ucs_FIXED.geojson",
        "./chenab_ucs_FIXED(1).geojson",
    ]

    geojson_path = None
    for path in possible_geojson_paths:
        if os.path.exists(path):
            geojson_path = path
            break

    if geojson_path:
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)

            loaded_ucs = {}
            features = geojson_data.get("features", [])

            for index, feature in enumerate(features):
                props = feature.get("properties", {}) or {}

                # Use the fixed UC name first. Avoid Unknown_* unless no real name exists.
                uc_name = (
                    props.get("UC")
                    or props.get("UC_NAME")
                    or props.get("New_Name")
                    or props.get("NAME")
                    or props.get("Name")
                    or f"UC-{index + 1}"
                )

                uc_name = str(uc_name).strip()
                if not uc_name or uc_name.lower() in ["nan", "none", "null"]:
                    uc_name = f"UC-{index + 1}"

                district = str(props.get("DISTRICT") or props.get("District") or "Unknown").strip()
                tehsil = str(props.get("TEHSIL") or props.get("Tehsil") or "").strip()
                uc_code = str(props.get("UC_C") or props.get("UC_CODE") or props.get("OBJECTID") or index + 1).strip()

                try:
                    distance = float(props.get("Distance_to_River_km"))
                except Exception:
                    distance = 999.0

                station = choose_station_from_distance(distance)

                # Keep existing simple elevation/population assumptions.
                # Only the UC source and distance values are being corrected here.
                elevation = max(150.0, 300.0 - (distance * 2.0))
                population = 25000 + (index % 50000)

                loaded_ucs[f"UC-{index + 1}"] = {
                    "name": uc_name,
                    "district": district,
                    "tehsil": tehsil,
                    "uc_code": uc_code,
                    "station": station,
                    "distance_km": distance,
                    "elevation_m": elevation,
                    "population": population,
                }

            UC_DATABASE = loaded_ucs
            print(f"✅ Loaded {len(UC_DATABASE)} UCs from fixed GeoJSON: {geojson_path}")
            return

        except Exception as e:
            print(f"❌ Failed to load fixed GeoJSON lookup: {e}")
            print("⚠️ Falling back to CSV lookup...")

    # ─────────────────────────────────────────
    # 2) CSV fallback
    # ─────────────────────────────────────────
    possible_csv_paths = [
        "uc_river_distance_lookup(1).csv",
        "uc_river_distance_lookup.csv",
        "./uc_river_distance_lookup(1).csv",
        "./uc_river_distance_lookup.csv",
    ]

    csv_path = None
    for path in possible_csv_paths:
        try:
            open(path, "r", encoding="utf-8").close()
            csv_path = path
            break
        except Exception:
            continue

    if not csv_path:
        print("⚠️ UC lookup file not found. Falling back to sample UCs only.")
        UC_DATABASE = {
            "UC-1": {"name": "UC 1 - Hafizabad", "district": "Hafizabad", "station": "Marala", "distance_km": 2.5, "elevation_m": 278, "population": 45000},
            "UC-2": {"name": "UC 2 - Hafizabad", "district": "Hafizabad", "station": "Marala", "distance_km": 5.0, "elevation_m": 275, "population": 38000},
            "UC-3": {"name": "UC 3 - Chiniot", "district": "Chiniot", "station": "Khanki", "distance_km": 1.5, "elevation_m": 250, "population": 52000},
            "UC-4": {"name": "UC 4 - Jhang", "district": "Jhang", "station": "Trimmu", "distance_km": 1.0, "elevation_m": 190, "population": 61000},
        }
        return

    try:
        uc_df = pd.read_csv(csv_path)

        required_columns = ["UC", "Distance_to_River_km"]
        missing_columns = [col for col in required_columns if col not in uc_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in UC CSV: {missing_columns}")

        loaded_ucs = {}

        for index, row in uc_df.iterrows():
            uc_name = str(row["UC"]).strip()

            if not uc_name or uc_name.lower() == "nan":
                continue

            try:
                distance = float(row["Distance_to_River_km"])
            except Exception:
                distance = 999.0

            station = choose_station_from_distance(distance)

            district = str(row.get("DISTRICT", row.get("District", "Unknown"))).strip()
            if not district or district.lower() == "nan":
                district = "Unknown"

            elevation = max(150.0, 300.0 - (distance * 2.0))
            population = 25000 + (index % 50000)

            loaded_ucs[f"UC-{index + 1}"] = {
                "name": uc_name,
                "district": district,
                "station": station,
                "distance_km": distance,
                "elevation_m": elevation,
                "population": population,
            }

        UC_DATABASE = loaded_ucs
        print(f"✅ Loaded {len(UC_DATABASE)} UCs from CSV: {csv_path}")

    except Exception as e:
        print(f"❌ Failed to load UC CSV: {e}")
        UC_DATABASE = {}

load_uc_database_from_csv()

# Mock weather data per station
STATION_WEATHER = {
    'Marala': {
        'temperature': 32.0,
        'rainfall': 12.5,
        'humidity': 72.0,
        'wind_speed': 14.0
    },
    'Khanki': {
        'temperature': 33.5,
        'rainfall': 5.0,
        'humidity': 60.0,
        'wind_speed': 11.0
    },
    'Qadirabad': {
        'temperature': 31.0,
        'rainfall': 18.0,
        'humidity': 78.0,
        'wind_speed': 16.0
    },
    'Trimmu': {
        'temperature': 29.5,
        'rainfall': 22.0,
        'humidity': 82.0,
        'wind_speed': 18.0
    },
    'Panjnad': {
        'temperature': 30.5,
        'rainfall': 15.0,
        'humidity': 76.0,
        'wind_speed': 15.0
    },
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def discharge_to_percentage(discharge, distance_km=1.0, elevation_m=200):
    discharge = float(discharge)

    # ─────────────────────────────────────────
    # BASE RISK FROM DISCHARGE
    # Safe/default conditions must remain Normal (0–20%).
    # Low risk (21–40%) should only appear when discharge starts rising.
    # ─────────────────────────────────────────
    if discharge >= DISCHARGE_THRESHOLDS['critical']:
        base_risk = 95.0

    elif discharge >= DISCHARGE_THRESHOLDS['high']:
        ratio = (
            (discharge - DISCHARGE_THRESHOLDS['high']) /
            (DISCHARGE_THRESHOLDS['critical'] - DISCHARGE_THRESHOLDS['high'])
        )
        base_risk = 75.0 + ratio * 20.0

    elif discharge >= DISCHARGE_THRESHOLDS['moderate']:
        ratio = (
            (discharge - DISCHARGE_THRESHOLDS['moderate']) /
            (DISCHARGE_THRESHOLDS['high'] - DISCHARGE_THRESHOLDS['moderate'])
        )
        base_risk = 50.0 + ratio * 25.0

    elif discharge >= DISCHARGE_THRESHOLDS['low']:
        ratio = (
            (discharge - DISCHARGE_THRESHOLDS['low']) /
            (DISCHARGE_THRESHOLDS['moderate'] - DISCHARGE_THRESHOLDS['low'])
        )
        # 5,000–15,000 discharge now maps roughly 15–40%
        # so safe/default low-flow situations do not automatically become Low Risk.
        base_risk = 15.0 + ratio * 25.0

    else:
        ratio = discharge / DISCHARGE_THRESHOLDS['low']
        # Below low threshold stays safely inside Normal range.
        base_risk = ratio * 15.0

    # ─────────────────────────────────────────
    # DISTANCE EFFECT
    # Nearby UCs should stay more vulnerable,
    # but distant UCs should still reduce risk.
    # ─────────────────────────────────────────
    distance_factor = max(
        0.65,
        1.0 - (distance_km * 0.015)
    )

    # ─────────────────────────────────────────
    # ELEVATION EFFECT
    # High elevation slightly reduces risk.
    # Prevent over-reduction.
    # ─────────────────────────────────────────
    elevation_factor = max(
        0.70,
        1.0 - max(0.0, (elevation_m - 150.0) * 0.004)
    )

    # ─────────────────────────────────────────
    # FINAL RISK
    # ─────────────────────────────────────────
    final_risk = (
        base_risk *
        distance_factor *
        elevation_factor
    )

    return round(
        max(0.0, min(99.0, final_risk)),
        1
    )

def risk_color(pct):
    # Public-facing 6-level flood risk scale:
    # Normal 0-20, Low 21-40, Medium 41-60, High 61-80,
    # Very High 81-95, Exceptionally High 96-100
    pct = float(pct)

    if pct >= 96:
        return "#8B0000"   # Exceptionally High
    if pct >= 81:
        return "#FF0000"   # Very High
    if pct >= 61:
        return "#FF6600"   # High
    if pct >= 41:
        return "#FFCC00"   # Medium
    if pct >= 21:
        return "#EAB308"   # Low
    return "#00CC00"       # Normal


def risk_label(pct):
    # Public-facing 6-level flood risk scale.
    pct = float(pct)

    if pct >= 96:
        return "Exceptionally High"
    if pct >= 81:
        return "Very High"
    if pct >= 61:
        return "High"
    if pct >= 41:
        return "Medium"
    if pct >= 21:
        return "Low"
    return "Normal"

def build_features(discharge, rainfall, temperature, month, is_flood_season):
    """
    Deprecated compatibility helper.
    The final model must use calculate_ml_features_from_reading()
    so station encoding, lag, rolling mean/std, and dQ features are correct.
    """
    raise RuntimeError(
        "Old build_features() should not be used. Use StationReadingDB + predict_from_station_reading()."
    )

def get_supported_uc(data):
    """
    Finds a UC from UC_DATABASE using uc_id, uc_name, or location.
    Returns (uc_id, uc_dict) if found, otherwise None.
    """
    search_values = [
        getattr(data, "uc_id", None),
        getattr(data, "uc_name", None),
        getattr(data, "location", None),
    ]

    # Also allow station-only forecast only when no UC/location is provided.
    station_value = getattr(data, "station", None)

    search_values = [
        str(v).strip().lower()
        for v in search_values
        if v is not None and str(v).strip()
    ]

    for uc_id, uc in UC_DATABASE.items():
        uc_name = str(uc.get("name", "")).strip().lower()

        for value in search_values:
            if (
                value == uc_id.lower()
                or value == uc_name
                or value in uc_name
                or uc_name in value
            ):
                return uc_id, uc

    # Backward compatibility: if frontend sends only station, use first UC of that station.
    if station_value and not search_values:
        station_value = str(station_value).strip().lower()
        for uc_id, uc in UC_DATABASE.items():
            if str(uc.get("station", "")).strip().lower() == station_value:
                return uc_id, uc

    return None


def calculate_uc_risk_snapshot(uc_id, uc):
    """
    Shared UC risk calculation for basin overview and UC listing.

    IMPORTANT:
    Basin overview/current status should use ACTUAL current live readings,
    not future predicted discharge values.

    Forecast panels/endpoints still use ML predictions separately.
    """
    station = uc["station"]

    db = SessionLocal()
    try:
        reading = get_latest_real_station_reading(db, station)

        if reading:
            current_discharge = float(reading.discharge or 0.0)
        else:
            current_discharge = float(
                STATION_DEFAULT_DISCHARGE.get(station, 5000)
            )

        # Keep forecast generation available for frontend forecast panels,
        # but DO NOT use prediction for current basin overview risk.
        predicted_discharge = current_discharge

        if reading and MODEL_LOADED:
            try:
                forecast = predict_from_station_reading(db, reading)
                predicted_discharge = forecast["forecast"]["24h"]["predicted_discharge"]
            except Exception:
                pass

    finally:
        db.close()

    # Basin overview risk uses ACTUAL current discharge.
    risk_percentage = discharge_to_percentage(
        current_discharge,
        uc["distance_km"],
        uc["elevation_m"]
    )

    return {
        "id": uc_id,
        "name": uc["name"],
        "district": uc.get("district", "Unknown"),
        "tehsil": uc.get("tehsil"),
        "uc_code": uc.get("uc_code"),
        "station": station,
        "current_discharge": round(float(current_discharge), 2),
        "predicted_discharge": round(float(predicted_discharge), 2),
        "risk_percentage": risk_percentage,
        "risk_level": risk_label(risk_percentage),
        "risk_color": risk_color(risk_percentage),
        "population": uc.get("population", 0),
        "distance_km": round(float(uc.get("distance_km", 0)), 2),
        "tooltip": {
            "title": uc["name"],
            "risk_percentage": risk_percentage,
            "risk_level": risk_label(risk_percentage),
            "district": uc.get("district", "Unknown"),
            "station": station,
            "distance_km": round(float(uc.get("distance_km", 0)), 2),
        },
        "ml_driven": MODEL_LOADED and reading is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


def calculate_chenab_overall_risk():
    """
    Calculates one overall risk value for the whole Chenab monitoring area.
    It uses all UC snapshots so the frontend can show a basin-wide status.
    """
    snapshots = [
        calculate_uc_risk_snapshot(uc_id, uc)
        for uc_id, uc in UC_DATABASE.items()
    ]

    if not snapshots:
        return {
            "overall_percentage": 0,
            "overall_level": "Normal",
            "overall_color": risk_color(0),
            "average_percentage": 0,
            "highest_risk_uc": None,
            "total_ucs": 0,
            "counts": {
                "NORMAL": 0,
                "LOW": 0,
                "MEDIUM": 0,
                "HIGH": 0,
                "VERY_HIGH": 0,
                "EXCEPTIONALLY_HIGH": 0
            }
        }

    label_to_count = {
        "Normal": "NORMAL",
        "Low": "LOW",
        "Medium": "MEDIUM",
        "High": "HIGH",
        "Very High": "VERY_HIGH",
        "Exceptionally High": "EXCEPTIONALLY_HIGH"
    }

    counts = {
        "NORMAL": 0,
        "LOW": 0,
        "MEDIUM": 0,
        "HIGH": 0,
        "VERY_HIGH": 0,
        "EXCEPTIONALLY_HIGH": 0
    }

    for item in snapshots:
        counts[label_to_count.get(item["risk_level"], "LOW")] += 1

    highest_risk_uc = max(snapshots, key=lambda item: item["risk_percentage"])
    average_percentage = round(
        sum(item["risk_percentage"] for item in snapshots) / len(snapshots),
        1
    )

    # Basin-level status is based on highest current UC risk so warnings are not missed.
    overall_percentage = highest_risk_uc["risk_percentage"]

    return {
        "overall_percentage": overall_percentage,
        "overall_level": risk_label(overall_percentage),
        "overall_color": risk_color(overall_percentage),
        "average_percentage": average_percentage,
        "highest_risk_uc": highest_risk_uc,
        "total_ucs": len(snapshots),
        "counts": counts
    }




def send_email_alert(to_email: str, subject: str, html_content: str):
    """
    Sends transactional alert emails using Resend API.
    """
    global LAST_EMAIL_ERROR
    LAST_EMAIL_ERROR = ""

    if not to_email:
        LAST_EMAIL_ERROR = "No recipient email provided."
        print("⚠️", LAST_EMAIL_ERROR)
        return False

    if not RESEND_API_KEY:
        LAST_EMAIL_ERROR = "RESEND_API_KEY is not configured."
        print("⚠️", LAST_EMAIL_ERROR)
        return False

    try:
        params = {
            "from": f"{CGUARD_SENDER_NAME} <onboarding@resend.dev>",
            "to": [to_email],
            "subject": subject,
            "html": html_content
        }

        resend.Emails.send(params)

        print(f"✅ Email sent to {to_email}")
        return True

    except Exception as e:
        LAST_EMAIL_ERROR = f"Resend email failed: {str(e)}"
        print("❌", LAST_EMAIL_ERROR)
        return False



def build_alert_subscription_email(subscription: AlertSubscriptionDB):
    district = subscription.district or "N/A"
    threshold = subscription.threshold or "N/A"

    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #1f2937; line-height: 1.6;">
        <div style="max-width: 620px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;">
          <div style="background: #173b5f; color: white; padding: 18px 22px;">
            <h2 style="margin: 0;">C Guard Flood Alerts Activated</h2>
          </div>

          <div style="padding: 22px;">
            <p>Your flood alert subscription is now active.</p>

            <table style="width: 100%; border-collapse: collapse; margin: 18px 0;">
              <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;"><strong>Union Council / Area</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{subscription.uc_name}</td>
              </tr>
              <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;"><strong>District</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{district}</td>
              </tr>
              <tr>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;"><strong>Alert Threshold</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #e5e7eb;">{threshold}</td>
              </tr>
            </table>

            <p>You will receive email alerts when flood risk reaches your selected threshold.</p>

            <p style="font-size: 13px; color: #64748b;">
              This is an automated message from C Guard: Chenab River Flood Forecasting and Early Warning System.
            </p>
          </div>
        </div>
      </body>
    </html>
    """






def get_chatbot_context(db: Session):
    """
    Builds live C-Guard context for the AI chatbot using current backend data.
    """
    try:
        shelters = db.query(ShelterDB).order_by(ShelterDB.updated_at.desc()).limit(5).all()
    except Exception:
        shelters = []

    shelter_text = []
    for shelter in shelters:
        shelter_text.append(
            f"{shelter.name} in {shelter.location}, district {shelter.district or 'N/A'}, "
            f"capacity {shelter.capacity}, occupied {shelter.occupied}, status {shelter.status}"
        )

    try:
        basin = calculate_chenab_overall_risk()
    except Exception:
        basin = {}

    highest_risk_uc = basin.get("highest_risk_uc")
    highest_risk_name = None
    if isinstance(highest_risk_uc, dict):
        highest_risk_name = highest_risk_uc.get("name") or highest_risk_uc.get("uc_name")

    return {
        "project_name": "C-Guard",
        "basin": "Chenab River Basin",
        "basin_risk_level": basin.get("overall_level", "Unknown"),
        "basin_risk_percentage": basin.get("overall_percentage", "Unknown"),
        "highest_risk_uc": highest_risk_name or "Unknown",
        "total_ucs": basin.get("total_ucs", len(UC_DATABASE)),
        "shelters": shelter_text,
        "emergency_contacts": [
            "PDMA Punjab Helpline: 1129",
            "Rescue 1122: 1122",
            "Police Emergency: 15",
            "Punjab Flood Control Room: (042) 99203005",
            "District Administration: 1043"
        ],
        "features": [
            "24-hour, 48-hour, and 72-hour flood forecasts",
            "UC-level flood risk",
            "Chenab River Basin risk map",
            "Emergency shelter information",
            "Email flood alert subscriptions",
            "Authority dashboard for shelters and analytics"
        ]
    }


def ask_cguard_ai(user_message: str, context: dict, user_type: str = "citizen", uc_name: str = None, district: str = None, language: str = "en"):
    """
    Sends the user question plus live C-Guard context to Groq AI.
    """
    language = (language or "en").lower().strip()

    if language in ["ur", "urdu"]:
        language_instruction = (
            "Reply in Urdu using simple, clear, natural Urdu. "
            "Use Urdu script. Keep flood-safety instructions easy for citizens to understand. "
            "You may keep official terms like C-Guard, PDMA, Rescue 1122, UC, and forecast in English where helpful."
        )
    else:
        language_instruction = "Reply in English using simple, clear language."

    system_prompt = f"""
You are C-Guard Assistant, an AI chatbot for a bilingual web-based flood forecasting and disaster management system for the Chenab River Basin.

Language instruction:
{language_instruction}

Your job:
- Help citizens understand flood risk, forecasts, shelters, emergency contacts, and alert subscriptions.
- Help authority users understand dashboard features, shelter management, analytics, reports, and alerts.
- Answer general questions politely, but guide the conversation back to flood safety and C-Guard when useful.

Rules:
- Use simple, clear language.
- Do not invent exact live flood values beyond the provided backend context.
- If exact current risk is needed, tell the user to check the map or selected UC risk panel.
- If the user asks for emergency help, immediately share emergency contacts.
- If the question is unrelated to floods/C-Guard, answer briefly and then offer C-Guard help.
- Do not claim you are replacing official PDMA/Rescue instructions.
- For safety advice, recommend following local authority instructions.

Current backend context:
Project: {context.get("project_name")}
Basin: {context.get("basin")}
Overall basin risk level: {context.get("basin_risk_level")}
Overall basin risk percentage: {context.get("basin_risk_percentage")}
Highest risk UC: {context.get("highest_risk_uc")}
Total monitored UCs: {context.get("total_ucs")}
Available shelters summary: {context.get("shelters")}
Emergency contacts: {context.get("emergency_contacts")}
Available system features: {context.get("features")}

Current user context:
User type: {user_type}
Selected UC: {uc_name or "Not provided"}
Selected district: {district or "Not provided"}
"""

    if groq_client is None:
        # Safe fallback when GROQ_API_KEY is not configured.
        msg = user_message.lower()
        if any(word in msg for word in ["contact", "emergency", "1122", "pdma", "rescue"]):
            return "Emergency contacts: PDMA Punjab Helpline 1129, Rescue 1122, Police 15, Punjab Flood Control Room (042) 99203005."
        if any(word in msg for word in ["shelter", "safe place", "evacuate"]):
            return "Please open the Emergency/Shelters page in C-Guard to see available shelters, capacity, and facilities."
        if any(word in msg for word in ["risk", "flood", "forecast"]):
            return "Please open the Flood Risk Map and select your UC/location to view 24h, 48h, and 72h flood risk."
        return "I can help with flood risk, shelters, emergency contacts, and C-Guard alerts."

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.4,
        max_tokens=500
    )

    return response.choices[0].message.content



# ═══════════════════════════════════════════════
#  E N D P O I N T S
# ═══════════════════════════════════════════════

@app.get("/")
def home():
    return {"message": "C-Guard Backend Running", "model_loaded": MODEL_LOADED}



@app.get("/debug/tesseract")
def debug_tesseract():
    """
    Temporary Hugging Face diagnostic endpoint.
    Shows whether the Linux container can see the tesseract binary installed by packages.txt.
    Remove or protect this endpoint before final deployment.
    """
    import shutil
    import subprocess

    path = shutil.which("tesseract")
    version = None
    error = None

    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        version = result.stdout
        error = result.stderr
    except Exception as e:
        error = str(e)

    return {
        "tesseract_path": path,
        "version": version,
        "error": error,
        "packages_txt_expected": "tesseract-ocr and libgl1 must be in packages.txt at the Space root"
    }


@app.get("/debug/subscriptions")
def debug_subscriptions(db: Session = Depends(get_db)):
    """
    Temporary debug endpoint to view alert subscriptions saved in the active database.
    Useful on Hugging Face when the app is using cguard.db SQLite.
    Remove or protect this endpoint before final deployment.
    """
    subs = (
        db.query(AlertSubscriptionDB)
        .order_by(AlertSubscriptionDB.created_at.desc(), AlertSubscriptionDB.id.desc())
        .all()
    )

    return {
        "success": True,
        "database_url": DATABASE_URL,
        "total_subscriptions": len(subs),
        "subscriptions": [
            {
                "id": s.id,
                "uc_name": s.uc_name,
                "district": s.district,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "email": s.email,
                "phone": s.phone,
                "email_alerts": s.email_alerts,
                "sms_alerts": s.sms_alerts,
                "threshold": s.threshold,
                "is_active": s.is_active,
                "last_alert_sent_at": s.last_alert_sent_at.isoformat() if s.last_alert_sent_at else None,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in subs
        ],
    }



@app.get("/api/analytics/forecast")
def analytics_forecast(
    horizon: str = "24h",
    station: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Authority dashboard forecast mode.

    Forecast horizon selected = ML prediction data only.
    Historical date range should be reset/ignored on frontend when this endpoint is used.
    """
    horizon_key = str(horizon).lower().replace(" ", "")
    aliases = {
        "24": "24h", "24h": "24h",
        "48": "48h", "48h": "48h",
        "72": "72h", "72h": "72h",
    }

    horizon_key = aliases.get(horizon_key)
    if not horizon_key:
        raise HTTPException(status_code=400, detail="Invalid horizon. Use 24h, 48h, or 72h.")

    stations = ["Khanki", "Marala", "Panjnad", "Qadirabad", "Trimmu"]

    if station and str(station).strip().lower() not in ["all", "all stations", "overview"]:
        stations = [normalize_station_name(station)]

    results = []
    predicted_values = []

    for station_name in stations:
        reading = get_latest_real_station_reading(db, station_name)

        fallback_used = False
        if not reading:
            weather = STATION_WEATHER.get(station_name, {"rainfall": 0.0, "temperature": 25.0})
            reading = build_temporary_station_reading(
                station=station_name,
                discharge=float(STATION_DEFAULT_DISCHARGE.get(station_name, 5000)),
                rainfall_mm=float(weather.get("rainfall", 0.0)),
                temperature_c=float(weather.get("temperature", 25.0)),
                soil_moisture_mm=25.0,
                source="temporary_forecast_fallback"
            )
            fallback_used = True

        forecast = predict_from_station_reading(db, reading)
        pred = float(forecast["forecast"][horizon_key]["predicted_discharge"])
        predicted_values.append(pred)

        results.append({
            "station": station_name,
            "horizon": horizon_key,
            "reading_time": forecast.get("reading_time"),
            "current_discharge": forecast.get("current_discharge"),
            "predicted_discharge": round(pred, 2),
            "risk_level": forecast["forecast"][horizon_key]["risk_level"],
            "data_source": "ml_prediction",
            "current_reading_source": reading.source,
            "fallback_used": fallback_used,
            "data_quality": "default_input_no_scraper_reading" if fallback_used else "scraper_or_manual_reading",
        })

    summary = {
        "total_discharge": round(float(sum(predicted_values)), 2) if predicted_values else 0.0,
        "average_discharge": round(float(np.mean(predicted_values)), 2) if predicted_values else 0.0,
        "maximum_discharge": round(float(max(predicted_values)), 2) if predicted_values else 0.0,
        "minimum_discharge": round(float(min(predicted_values)), 2) if predicted_values else 0.0,
    }

    return {
        "success": True,
        "mode": "forecast",
        "data_source": "ml_prediction",
        "horizon": horizon_key,
        "date_range_behavior": "frontend should reset date range to current/default date",
        "total": len(results),
        "summary": summary,
        "forecasts": results
    }



@app.get("/api/analytics/historical")
def analytics_historical(
    start_date: str,
    end_date: str,
    station: Optional[str] = None,
    limit: int = 500,
    db: Session = Depends(get_db)
):
    """
    Authority dashboard historical mode.

    Date range selected = actual recorded database readings only.
    Forecast selection should be cleared on frontend when this endpoint is used.
    """
    start_dt = parse_date_range_value(start_date, end_of_day=False)
    end_dt = parse_date_range_value(end_date, end_of_day=True)

    query = db.query(StationReadingDB).filter(StationReadingDB.source.in_(REAL_READING_SOURCES))

    selected_station = None
    if station and str(station).strip().lower() not in ["all", "all stations", "overview"]:
        selected_station = normalize_station_name(station)
        query = query.filter(StationReadingDB.station == selected_station)

    if start_dt:
        query = query.filter(StationReadingDB.reading_time >= start_dt)
    if end_dt:
        query = query.filter(StationReadingDB.reading_time <= end_dt)

    rows = (
        query
        .order_by(StationReadingDB.reading_time.asc(), StationReadingDB.id.asc())
        .limit(limit)
        .all()
    )

    readings = [serialize_station_reading(row) for row in rows]

    by_station = {}
    for row in rows:
        by_station.setdefault(row.station, []).append(float(row.discharge or 0.0))

    station_summary = {}
    for station_name, values in by_station.items():
        station_summary[station_name] = {
            "count": len(values),
            "total_discharge": round(float(sum(values)), 2),
            "average_discharge": round(float(np.mean(values)), 2) if values else 0.0,
            "maximum_discharge": round(float(max(values)), 2) if values else 0.0,
            "minimum_discharge": round(float(min(values)), 2) if values else 0.0,
        }

    all_values = [float(row.discharge or 0.0) for row in rows]

    summary = {
        "total_records": len(rows),
        "total_discharge": round(float(sum(all_values)), 2) if all_values else 0.0,
        "average_discharge": round(float(np.mean(all_values)), 2) if all_values else 0.0,
        "maximum_discharge": round(float(max(all_values)), 2) if all_values else 0.0,
        "minimum_discharge": round(float(min(all_values)), 2) if all_values else 0.0,
        "stations": station_summary,
    }

    return {
        "success": True,
        "mode": "historical",
        "data_source": "database_station_readings",
        "station": selected_station or "All Stations",
        "start_date": start_date,
        "end_date": end_date,
        "total": len(readings),
        "message": "No data found for selected date range." if len(readings) == 0 else "Historical readings found.",
        "summary": summary,
        "readings": readings
    }



@app.delete("/debug/cleanup-fallback-readings")
def cleanup_fallback_readings(db: Session = Depends(get_db)):
    """
    Temporary cleanup endpoint.
    Removes old fallback rows that were accidentally saved as historical readings.
    Real scraper/manual readings are not deleted.
    """
    deleted = (
        db.query(StationReadingDB)
        .filter(StationReadingDB.source.in_(FALLBACK_READING_SOURCES))
        .delete(synchronize_session=False)
    )
    db.commit()

    return {
        "success": True,
        "deleted_fallback_rows": deleted,
        "message": "Fallback rows removed. Historical mode now shows real scraper/manual readings only."
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": MODEL_LOADED, "mae": 1273}


# ── LOGIN ──────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/login")
@app.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    email = normalize_email(data.email)

    if is_account_locked(email):
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed login attempts. Try again after {LOCKOUT_MINUTES} minutes."
        )

    user = db.query(UserDB).filter(UserDB.email == email).first()

    # Same error for wrong email/wrong password, so attackers cannot guess accounts.
    if not user:
        record_failed_login(email)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not bcrypt.checkpw(data.password.encode(), user.password.encode()):
        record_failed_login(email)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if user.role != "authority":
        record_failed_login(email)
        raise HTTPException(status_code=403, detail="Authority access required.")

    clear_failed_logins(email)

    token = create_token(user.email, user.role, user.name)

    return {
        "message": "Login successful",
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "expires_in_hours": TOKEN_EXPIRE_HOURS,
        "role": user.role,
        "name": user.name,
        "email": user.email
    }


# ── REGISTER ───────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    invite_code: Optional[str] = None


@app.post("/api/register")
@app.post("/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    email = normalize_email(data.email)

    # Authority accounts must not be open to public signup.
    if not ALLOW_PUBLIC_AUTHORITY_SIGNUP:
        if data.invite_code != AUTHORITY_INVITE_CODE:
            raise HTTPException(
                status_code=403,
                detail="Authority registration is restricted. Valid invite code required."
            )

    validate_password_strength(data.password)

    existing = db.query(UserDB).filter(UserDB.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    hashed = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt(rounds=12)).decode()

    new_user = UserDB(
        name=data.name.strip(),
        email=email,
        password=hashed,
        role="authority"
    )

    db.add(new_user)
    db.commit()

    return {"success": True, "message": "Authority account created successfully. You can now login."}


# ── AUTH VERIFY ─────────────────────────────────
@app.get("/api/auth/me")
@app.get("/auth/me")
def auth_me(token: dict = Depends(require_authority)):
    return {
        "authenticated": True,
        "role": token.get("role"),
        "name": token.get("name"),
        "email": token.get("email")
    }


# ── FLOOD FORECAST ─────────────────────────────
# ── FLOOD FORECAST ─────────────────────────────
class ForecastRequest(BaseModel):
    station: Optional[str] = None
    uc_id: Optional[str] = None
    uc_name: Optional[str] = None
    location: Optional[str] = None
    district: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    distance_to_river_km: Optional[float] = None
    current_discharge: Optional[float] = 5000
    rainfall_24h: Optional[float] = 0
    temperature: Optional[float] = 25


class LocationFloodRiskRequest(BaseModel):
    latitude: float
    longitude: float
    uc_name: str
    district: Optional[str] = None
    distance_to_river_km: Optional[float] = None
    current_discharge: Optional[float] = None
    rainfall_24h: Optional[float] = None
    temperature: Optional[float] = None


class AlertSubscriptionRequest(BaseModel):
    uc_name: str
    district: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    email_alerts: bool = False
    sms_alerts: bool = False

    threshold: str = "Very High only"


class ChatbotRequest(BaseModel):
    message: str
    user_type: Optional[str] = "citizen"
    uc_name: Optional[str] = None
    district: Optional[str] = None
    language: Optional[str] = "en"  # "en" for English, "ur" for Urdu


class ChatbotResponse(BaseModel):
    success: bool
    reply: str

@app.post("/forecast")
def post_forecast(data: ForecastRequest, db: Session = Depends(get_db)):
    """
    Correct ML forecast endpoint.
    Uses latest StationReadingDB row for the station, calculates lags/rolling/dQ from PostgreSQL,
    then predicts 24h/48h/72h with separate trained models.
    """
    supported_uc = get_supported_uc(data)

    station = None
    uc_id = None
    uc = None

    if supported_uc:
        uc_id, uc = supported_uc
        station = uc["station"]
    elif data.station:
        station = normalize_station_name(data.station)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide station or supported uc/location."
        )

    reading = get_latest_real_station_reading(db, station)

    if not reading:
        # Allow one manual reading from request if DB has no station history yet.
        weather = STATION_WEATHER.get(station, {"rainfall": 0.0, "temperature": 25.0})
        reading = build_temporary_station_reading(
            station=station,
            discharge=float(data.current_discharge or STATION_DEFAULT_DISCHARGE.get(station, 5000)),
            rainfall_mm=float(data.rainfall_24h if data.rainfall_24h is not None else weather.get("rainfall", 0.0)),
            temperature_c=float(data.temperature if data.temperature is not None else weather.get("temperature", 25.0)),
            soil_moisture_mm=25.0,
            source="temporary_forecast_request_fallback"
        )

    forecast = predict_from_station_reading(db, reading)

    return make_json_safe({
        "success": True,
        "supported": bool(supported_uc),
        "uc_id": uc_id,
        "uc": uc["name"] if uc else None,
        "station": station,
        **forecast,
        "timestamp": datetime.utcnow().isoformat(),
    })


@app.post("/api/flood-risk/location")
def flood_risk_by_location(data: LocationFloodRiskRequest, db: Session = Depends(get_db)):
    """
    Personal location-based flood risk endpoint for FloodMap.jsx.
    Uses latest station reading from PostgreSQL + correct 24h/48h/72h ML models.
    """
    supported_uc = get_supported_uc(data)

    if not supported_uc:
        return {
            "success": True,
            "inside_coverage": False,
            "message": "Outside Chenab Basin Coverage",
            "uc_name": data.uc_name,
            "district": data.district,
            "latitude": data.latitude,
            "longitude": data.longitude,
            "last_updated": datetime.utcnow().isoformat(),
            "risk": {
                "24h": {"percentage": 0, "level": "Normal"},
                "48h": {"percentage": 0, "level": "Normal"},
                "72h": {"percentage": 0, "level": "Normal"},
            }
        }

    uc_id, uc = supported_uc
    source_station = uc["station"]

    reading = get_latest_station_reading(db, source_station)

    if not reading:
        weather = STATION_WEATHER.get(source_station, {"rainfall": 0.0, "temperature": 25.0})
        reading = build_temporary_station_reading(
            station=source_station,
            discharge=float(data.current_discharge or STATION_DEFAULT_DISCHARGE.get(source_station, 5000)),
            rainfall_mm=float(data.rainfall_24h if data.rainfall_24h is not None else weather.get("rainfall", 0.0)),
            temperature_c=float(data.temperature if data.temperature is not None else weather.get("temperature", 25.0)),
            soil_moisture_mm=25.0,
            source="temporary_location_request_fallback"
        )

    forecast = predict_from_station_reading(db, reading)

    distance_km = float(
        data.distance_to_river_km
        if data.distance_to_river_km is not None
        else uc["distance_km"]
    )
    elevation_m = uc["elevation_m"]

    pred_24h = forecast["forecast"]["24h"]["predicted_discharge"]
    pred_48h = forecast["forecast"]["48h"]["predicted_discharge"]
    pred_72h = forecast["forecast"]["72h"]["predicted_discharge"]

    risk_24h = discharge_to_percentage(pred_24h, distance_km, elevation_m)
    risk_48h = discharge_to_percentage(pred_48h, distance_km, elevation_m)
    risk_72h = discharge_to_percentage(pred_72h, distance_km, elevation_m)

    return {
        "success": True,
        "inside_coverage": True,
        "uc_id": uc_id,
        "uc_name": uc["name"],
        "district": data.district or uc.get("district", "Unknown"),
        "latitude": data.latitude,
        "longitude": data.longitude,
        "distance_to_river_km": round(distance_km, 2),
        "source_station": source_station,
        "last_updated": datetime.utcnow().isoformat(),
        "model_loaded": MODEL_LOADED,
        "input": forecast["input_weather"] | {
            "current_discharge": forecast["current_discharge"]
        },
        "predicted_discharge": {
            "24h": pred_24h,
            "48h": pred_48h,
            "72h": pred_72h,
        },
        "risk": {
            "24h": {
                "percentage": risk_24h,
                "level": risk_label(risk_24h),
                "model_level": forecast["forecast"]["24h"]["risk_level"]
            },
            "48h": {
                "percentage": risk_48h,
                "level": risk_label(risk_48h),
                "model_level": forecast["forecast"]["48h"]["risk_level"]
            },
            "72h": {
                "percentage": risk_72h,
                "level": risk_label(risk_72h),
                "model_level": forecast["forecast"]["72h"]["risk_level"]
            }
        }
    }


@app.post("/api/alerts/subscribe")
def subscribe_location_alerts(data: AlertSubscriptionRequest, db: Session = Depends(get_db)):
    """
    Saves user's location alert preference from FloodMap.jsx.

    This endpoint currently stores the subscription in the database.
    Actual email/SMS sending can be connected next using Resend/Twilio.
    """
    if data.email_alerts and not data.email:
        raise HTTPException(
            status_code=400,
            detail="Email is required when email alerts are enabled."
        )

    if data.sms_alerts and not data.phone:
        raise HTTPException(
            status_code=400,
            detail="Phone number is required when SMS alerts are enabled."
        )

    new_subscription = AlertSubscriptionDB(
        uc_name=data.uc_name.strip(),
        district=data.district,
        latitude=data.latitude,
        longitude=data.longitude,
        email=data.email.strip() if data.email else None,
        phone=data.phone.strip() if data.phone else None,
        email_alerts=data.email_alerts,
        sms_alerts=data.sms_alerts,
        threshold=data.threshold,
        is_active=True,
        last_alert_sent_at=None,
    )

    db.add(new_subscription)
    db.commit()
    db.refresh(new_subscription)

    subscription_email_sent = False
    if new_subscription.email_alerts and new_subscription.email:
        subscription_email_sent = send_email_alert(
            to_email=new_subscription.email,
            subject="C Guard Flood Alerts Activated",
            html_content=build_alert_subscription_email(new_subscription)
        )

    return {
        "success": True,
        "message": "Location alerts enabled successfully",
        "subscription_id": new_subscription.id,
        "uc_name": new_subscription.uc_name,
        "district": new_subscription.district,
        "email": new_subscription.email,
        "phone": new_subscription.phone,
        "email_alerts": new_subscription.email_alerts,
        "sms_alerts": new_subscription.sms_alerts,
        "threshold": new_subscription.threshold,
        "is_active": new_subscription.is_active,
        "created_at": new_subscription.created_at,
        "email_sent": subscription_email_sent,
    }



class TestEmailRequest(BaseModel):
    email: str
    station: Optional[str] = "Qadirabad"
    risk_level: Optional[str] = "HIGH"


@app.get("/api/email-config-status")
def email_config_status():
    """
    Quick diagnostic endpoint to confirm backend can read Resend settings.
    It never returns the full API key.
    """
    return {
        "resend_api_key": mask_secret(RESEND_API_KEY),
        "resend_api_key_loaded": bool(RESEND_API_KEY),
        "sender_email": CGUARD_SENDER_EMAIL,
        "sender_name": CGUARD_SENDER_NAME,
        "last_email_error": LAST_EMAIL_ERROR,
    }


@app.post("/api/test-email")
def test_email(data: TestEmailRequest):
    """
    Sends a test email to verify Resend email configuration.
    This test endpoint is public for local testing.
    Remove or protect it before final deployment.
    """
    station = data.station or "Qadirabad"
    risk_level = data.risk_level or "HIGH"

    sent = send_email_alert(
        to_email=data.email,
        subject=f"C Guard Test Alert - {risk_level}",
        html_content=f"""
        <html>
          <body style="font-family: Arial, sans-serif; color: #1f2937;">
            <div style="max-width: 620px; margin: 0 auto; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;">
              <div style="background: #173b5f; color: white; padding: 18px 22px;">
                <h2 style="margin: 0;">C Guard Test Flood Alert</h2>
              </div>
              <div style="padding: 22px;">
                <p>This is a test email from the C Guard backend.</p>
                <p><strong>Station:</strong> {station}</p>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <p>If you received this email, Resend email sending is configured correctly.</p>
              </div>
            </div>
          </body>
        </html>
        """
    )

    if not sent:
        raise HTTPException(
            status_code=500,
            detail=LAST_EMAIL_ERROR or "Email could not be sent. Check RESEND_API_KEY configuration."
        )

    return {
        "success": True,
        "message": "Test email sent successfully",
        "to": data.email,
        "sender_email": CGUARD_SENDER_EMAIL,
    }



@app.post("/api/chatbot", response_model=ChatbotResponse)
def chatbot(data: ChatbotRequest, db: Session = Depends(get_db)):
    """
    AI-powered C-Guard chatbot endpoint.
    Frontend sends the user's message here and displays the returned reply.
    """
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    context = get_chatbot_context(db)

    try:
        reply = ask_cguard_ai(
            user_message=data.message.strip(),
            context=context,
            user_type=data.user_type or "citizen",
            uc_name=data.uc_name,
            district=data.district,
            language=data.language or "en"
        )

        return {
            "success": True,
            "reply": reply
        }

    except Exception as e:
        print("❌ Chatbot error:", e)
        fallback_reply = (
            "معذرت، میں اس وقت جواب نہیں دے سکا۔ براہِ کرم دوبارہ کوشش کریں یا سیلابی نقشہ، پناہ گاہوں کا صفحہ، اور ہنگامی رابطے چیک کریں۔"
            if (data.language or "en").lower().strip() in ["ur", "urdu"]
            else "Sorry, I could not answer right now. Please try again later or check the flood map, shelters page, and emergency contacts."
        )
        return {
            "success": False,
            "reply": fallback_reply
        }


@app.get("/api/basin-overview")
def basin_overview():
    """
    Basin overview endpoint for FloodMap.jsx right-side summary card.
    Returns total monitored UCs and count of UCs in each risk category.
    """
    summary = {
        "normal": 0,
        "low": 0,
        "medium": 0,
        "high": 0,
        "very_high": 0,
        "exceptionally_high": 0
    }

    label_to_key = {
        "Normal": "normal",
        "Low": "low",
        "Medium": "medium",
        "High": "high",
        "Very High": "very_high",
        "Exceptionally High": "exceptionally_high"
    }

    highest_risk = None

    for uc_id, uc in UC_DATABASE.items():
        snapshot = calculate_uc_risk_snapshot(uc_id, uc)
        label = snapshot["risk_level"]
        key = label_to_key.get(label, "low")
        summary[key] += 1

        if highest_risk is None or snapshot["risk_percentage"] > highest_risk["risk_percentage"]:
            highest_risk = snapshot

    return {
        "success": True,
        "live": True,
        "total_ucs": len(UC_DATABASE),
        "ucs_monitored": len(UC_DATABASE),
        "summary": summary,
        "counts": {
            "LOW": summary["low"],
            "MEDIUM": summary["medium"],
            "HIGH": summary["high"],
            "VERY_HIGH": summary["very_high"],
            "EXCEPTIONALLY_HIGH": summary["exceptionally_high"]
        },
        "highest_risk_uc": highest_risk,
        "last_updated": datetime.utcnow().isoformat()
    }


@app.get("/api/chenab-risk")
def chenab_risk():
    """
    Public API for the whole Chenab basin risk card.
    Frontend can use this for landing page, map overview, and dashboard status.
    """
    overall = calculate_chenab_overall_risk()

    return {
        "success": True,
        "basin": "Chenab River Basin",
        "overall_risk_percentage": overall["overall_percentage"],
        "overall_risk_level": overall["overall_level"],
        "overall_risk_color": overall["overall_color"],
        "average_risk_percentage": overall["average_percentage"],
        "highest_risk_uc": overall["highest_risk_uc"],
        "total_ucs": overall["total_ucs"],
        "counts": overall["counts"],
        "last_updated": datetime.utcnow().isoformat()
    }


@app.get("/api/map-risk")
def map_risk():
    """
    Public API for map hover/tooltips.
    Returns UC risk percentage and level for every UC.
    """
    ucs = []
    for uc_id, uc in UC_DATABASE.items():
        snapshot = calculate_uc_risk_snapshot(uc_id, uc)
        ucs.append({
            "id": snapshot["id"],
            "uc_name": snapshot["name"],
            "station": snapshot["station"],
            "risk_percentage": snapshot["risk_percentage"],
            "risk_level": snapshot["risk_level"],
            "risk_color": snapshot["risk_color"],
            "distance_km": snapshot["distance_km"],
            "tooltip": snapshot["tooltip"],
            "last_updated": snapshot["timestamp"],
        })

    return {
        "success": True,
        "union_councils": ucs,
        "total": len(ucs),
        "last_updated": datetime.utcnow().isoformat()
    }


@app.get("/forecast")
def get_forecast(db: Session = Depends(get_db)):
    """
    ML-driven GET /forecast.
    Uses latest PostgreSQL station readings + correct 24h/48h/72h models.
    """
    results = []

    for uc_id, uc in UC_DATABASE.items():
        station = uc["station"]
        reading = get_latest_real_station_reading(db, station)

        if reading and MODEL_LOADED:
            forecast = predict_from_station_reading(db, reading)
            pred_24h = forecast["forecast"]["24h"]["predicted_discharge"]
            pred_48h = forecast["forecast"]["48h"]["predicted_discharge"]
            pred_72h = forecast["forecast"]["72h"]["predicted_discharge"]
            input_discharge = forecast["current_discharge"]
        else:
            input_discharge = STATION_DEFAULT_DISCHARGE.get(station, 5000)
            pred_24h = pred_48h = pred_72h = input_discharge

        pct = discharge_to_percentage(pred_24h, uc["distance_km"], uc["elevation_m"])

        results.append({
            "id": uc_id,
            "uc": uc["name"],
            "station": station,
            "input_discharge": round(float(input_discharge), 2),
            "predicted_discharge": round(float(pred_24h), 2),
            "predicted_discharge_24h": round(float(pred_24h), 2),
            "predicted_discharge_48h": round(float(pred_48h), 2),
            "predicted_discharge_72h": round(float(pred_72h), 2),
            "risk_percentage": pct,
            "risk_level": risk_label(pct),
            "risk_color": risk_color(pct),
            "population": uc["population"],
            "ml_driven": MODEL_LOADED and reading is not None,
            "timestamp": datetime.utcnow().isoformat(),
        })

    return results


@app.get("/all-ucs")
def get_all_ucs():
    """Returns all UC risk snapshots for map/dashboard use."""
    ucs = []
    for uc_id, uc in UC_DATABASE.items():
        snapshot = calculate_uc_risk_snapshot(uc_id, uc)
        snapshot["color"] = snapshot["risk_color"]
        ucs.append(snapshot)

    return {
        "success": True,
        "union_councils": ucs,
        "total": len(ucs),
        "last_updated": datetime.utcnow().isoformat()
    }


# ── ANALYTICS ──────────────────────────────────
@app.get("/analytics/{station}")
def get_analytics(station: str, hours: int = 48, token: dict = Depends(require_authority)):
    if station not in STATION_GAUGE_CAPACITY:
        raise HTTPException(status_code=404, detail=f"Station not found. Valid: {list(STATION_GAUGE_CAPACITY.keys())}")

    gauge_cap     = STATION_GAUGE_CAPACITY[station]
    discharge_cap = STATION_DISCHARGE_CAPACITY[station]
    time_points   = list(range(0, hours + 1, 6))
    peak_t        = hours * 0.4

    def wave(t, base, amp):
        return base + amp * math.exp(-((t - peak_t) ** 2) / (2 * (hours / 3) ** 2))

    gauge_data, discharge_data, risk_data = [], [], []
    for t in time_points:
        g = round(wave(t, gauge_cap * 0.75,     gauge_cap * 0.2),    2)
        d = round(wave(t, discharge_cap * 0.45, discharge_cap * 0.4), 0)
        r = discharge_to_percentage(d)
        gauge_data.append(    {"time": f"{t:02d}:00", "value": g})
        discharge_data.append({"time": f"{t:02d}:00", "value": d})
        risk_data.append({
            "time":     f"h{t}" if t > 0 else "h1",
            "critical": round(max(0, r - 30), 1),
            "high":     round(min(r, 30), 1),
            "moderate": round(max(0, 70 - r), 1),
            "low":      round(max(0, 30 - r * 0.3), 1),
        })

    current_risk = discharge_to_percentage(discharge_data[0]["value"])
    return {
        "station": station, "forecast_hours": hours,
        "gauge_capacity_m": gauge_cap, "discharge_capacity": discharge_cap,
        "current_risk": current_risk, "risk_level": risk_label(current_risk),
        "gauge_level": gauge_data, "discharge": discharge_data, "risk_progression": risk_data,
    }


# ═══════════════════════════════════════════════
#  NEW ML ENDPOINTS
# ═══════════════════════════════════════════════

# ── FLOOD LATEST ───────────────────────────────
@app.get("/flood/latest")
def flood_latest(db: Session = Depends(get_db)):
    """
    Returns latest stored station readings from PostgreSQL.
    """
    result = {}

    for station in ["Marala", "Khanki", "Qadirabad", "Trimmu", "Panjnad"]:
        reading = get_latest_real_station_reading(db, station)

        if reading:
            result[station] = {
                "station": station,
                "date": reading.reading_time.isoformat() if reading.reading_time else None,
                "discharge": reading.discharge,
                "rainfall_mm": reading.rainfall_mm,
                "temperature_c": reading.temperature_c,
                "soil_moisture_mm": reading.soil_moisture_mm,
                "source": reading.source,
                "timestamp": reading.created_at.isoformat() if reading.created_at else None,
            }
        else:
            result[station] = {
                "station": station,
                "date": None,
                "discharge": STATION_DEFAULT_DISCHARGE.get(station, 0),
                "rainfall_mm": 0,
                "temperature_c": 25,
                "soil_moisture_mm": 25,
                "source": "default_no_db_reading",
                "timestamp": datetime.utcnow().isoformat(),
            }

    return result


# ── WEATHER CURRENT ────────────────────────────
@app.get("/weather/current")
def weather_current():
    """
    Returns current weather for all stations.
    Uses weather.py if available, otherwise falls back to static values.
    """
    try:
        from weather import get_all_weather_data
        return get_all_weather_data()
    except Exception as e:
        print(f"⚠️ weather.py unavailable, using fallback weather: {e}")
        return STATION_WEATHER


# ── WEATHER BY STATION ─────────────────────────
@app.get("/weather/station/{station_name}")
def weather_by_station(station_name: str):
    """
    Returns weather for a specific station.
    Uses weather.py if available, otherwise falls back to static values.
    """
    station_name = normalize_station_name(station_name)

    try:
        from weather import get_weather_for_station
        return get_weather_for_station(station_name)
    except Exception as e:
        print(f"⚠️ weather.py unavailable for {station_name}, using fallback weather: {e}")
        weather = STATION_WEATHER.get(station_name, {"temperature": 25.0, "rainfall": 0.0, "humidity": 50.0, "wind_speed": 0.0})
        return {
            "station": station_name,
            "temperature": weather.get("temperature", 25.0),
            "rainfall": weather.get("rainfall", 0.0),
            "soil_moisture": 25.0,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ── FEATURES LIVE ──────────────────────────────
@app.get("/features/live")
def features_live(db: Session = Depends(get_db)):
    """
    Shows the exact ML features currently being sent into the model for each station.
    This is the main debugging endpoint for wrong prediction values.
    """
    results = []

    for station in ["Marala", "Khanki", "Qadirabad", "Trimmu", "Panjnad"]:
        reading = get_latest_real_station_reading(db, station)

        if not reading:
            continue

        try:
            _, feature_dict = calculate_ml_features_from_reading(db, reading)
            results.append({
                "station": station,
                "reading_id": reading.id,
                "reading_time": reading.reading_time.isoformat() if reading.reading_time else None,
                "feature_order": list(feature_cols),
                "features": feature_dict,
            })
        except Exception as e:
            results.append({
                "station": station,
                "error": str(e)
            })

    return make_json_safe({
        "success": True,
        "total": len(results),
        "results": results
    })



# ─────────────────────────────────────────────
# RAILWAY-SAFE DIRECT PMD SCRAPER FALLBACK
# ─────────────────────────────────────────────
PMD_DISCHARGE_URL = "https://ffd.pmd.gov.pk/staff/discharge-report-carousel"
EXPECTED_SCRAPER_STATIONS = ["Khanki", "Marala", "Panjnad", "Qadirabad", "Trimmu"]


def _scraper_station_name(value):
    if value is None:
        return None

    text_value = str(value).strip().lower()
    text_value = re.sub(r"\s+", " ", text_value)

    aliases = {
        "khanki": "Khanki",
        "khankl": "Khanki",
        "marala": "Marala",
        "panjnad": "Panjnad",
        "punjnad": "Panjnad",
        "panj nad": "Panjnad",
        "qadirabad": "Qadirabad",
        "qadir abad": "Qadirabad",
        "qadirbad": "Qadirabad",
        "trimmu": "Trimmu",
        "trimu": "Trimmu",
        "trimum": "Trimmu",
    }

    if text_value in aliases:
        return aliases[text_value]

    for key, station in aliases.items():
        if key in text_value:
            return station

    for station in EXPECTED_SCRAPER_STATIONS:
        if station.lower() in text_value:
            return station

    return None


def _scraper_float(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    cleaned = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    try:
        return float(match.group(0))
    except Exception:
        return None


def direct_pmd_discharge_scraper():
    """
    Railway-safe PMD scraper.
    Does not need Tesseract, Chrome, Selenium, or OCR.
    Used when OCR scraper fails on Railway.
    """
    readings = {}

    response = requests.get(
        PMD_DISCHARGE_URL,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 C-Guard Railway scraper"}
    )
    response.raise_for_status()
    html = response.text

    # First try pandas HTML table parsing.
    try:
        tables = pd.read_html(html)

        for table in tables:
            table = table.fillna("")
            columns = [str(col).strip().lower() for col in table.columns]

            for _, row in table.iterrows():
                row_values = [str(v).strip() for v in row.tolist()]
                row_text = " ".join(row_values)
                station = _scraper_station_name(row_text)

                if not station:
                    continue

                discharge = None
                preferred_columns = ["inflow", "discharge", "current", "flow", "value"]

                for index, column_name in enumerate(columns):
                    if any(keyword in column_name for keyword in preferred_columns):
                        discharge = _scraper_float(row_values[index])
                        if discharge is not None:
                            break

                if discharge is None:
                    numbers = [_scraper_float(v) for v in row_values]
                    numbers = [v for v in numbers if v is not None]
                    if numbers:
                        # Station discharge/inflow is usually the largest number in its row.
                        discharge = max(numbers)

                if discharge is not None and discharge > 0:
                    readings[station] = {
                        "station": station,
                        "discharge": float(discharge),
                        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "pmd_direct_railway_fallback",
                    }

    except Exception as table_error:
        print(f"⚠️ Direct PMD table parse failed, trying regex fallback: {table_error}")

    # Regex fallback if pandas finds no station rows.
    if not readings:
        plain_text = re.sub(r"<[^>]+>", " ", html)
        plain_text = re.sub(r"\s+", " ", plain_text)

        for station in EXPECTED_SCRAPER_STATIONS:
            pattern = rf"({station}).{{0,500}}?(\d[\d,]{{3,}}(?:\.\d+)?)"
            match = re.search(pattern, plain_text, flags=re.IGNORECASE)
            if match:
                discharge = _scraper_float(match.group(2))
                if discharge is not None and discharge > 0:
                    readings[station] = {
                        "station": station,
                        "discharge": float(discharge),
                        "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "pmd_direct_regex_fallback",
                    }

    return list(readings.values())


def normalize_scraper_rows(scraped_rows):
    """
    Converts scraper response into a clean list of station rows.
    Accepts dict/list/string-ish scraper outputs.
    """
    if not scraped_rows:
        return []

    if isinstance(scraped_rows, dict):
        for key in ["data", "readings", "stations", "results"]:
            if isinstance(scraped_rows.get(key), list):
                scraped_rows = scraped_rows.get(key)
                break
        else:
            scraped_rows = list(scraped_rows.values())

    if not isinstance(scraped_rows, list):
        scraped_rows = [scraped_rows]

    clean_rows = []
    for row in scraped_rows:
        if not isinstance(row, dict):
            continue

        station = _scraper_station_name(
            row.get("station")
            or row.get("Station")
            or row.get("name")
            or row.get("Name")
        )

        if not station:
            # Support shape like {"Khanki": 9967}
            for key, value in row.items():
                possible_station = _scraper_station_name(key)
                if possible_station:
                    station = possible_station
                    row = {"station": station, "discharge": value}
                    break

        if not station:
            continue

        discharge = None
        for key in ["discharge", "Discharge", "inflow", "Inflow", "flow", "Flow", "value", "Value", "reading", "Reading"]:
            if key in row:
                discharge = _scraper_float(row.get(key))
                if discharge is not None:
                    break

        if discharge is None or discharge <= 0 or discharge > 1000000:
            continue

        clean_rows.append({
            "station": station,
            "discharge": float(discharge),
            "rainfall_mm": _scraper_float(row.get("rainfall_mm")) or 0.0,
            "temperature_c": _scraper_float(row.get("temperature_c")) or 25.0,
            "soil_moisture_mm": _scraper_float(row.get("soil_moisture_mm")) or 25.0,
            "date": row.get("date") or row.get("reading_time") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "source": row.get("source") or "scraper",
        })

    return clean_rows


class StationReadingInput(BaseModel):
    station: str
    date: Optional[str] = None
    discharge: float
    rainfall_mm: Optional[float] = 0.0
    temperature_c: Optional[float] = 25.0
    soil_moisture_mm: Optional[float] = 25.0
    source: Optional[str] = "manual"


@app.post("/station-readings/add")
def add_station_reading(data: StationReadingInput, db: Session = Depends(get_db)):
    """
    Manually add one station reading into PostgreSQL.
    Use this to seed test history or insert scraper values.
    """
    reading = save_station_reading(
        db=db,
        station=data.station,
        discharge=data.discharge,
        rainfall_mm=data.rainfall_mm,
        temperature_c=data.temperature_c,
        soil_moisture_mm=data.soil_moisture_mm,
        reading_time=parse_scraper_datetime(data.date),
        source=data.source or "manual"
    )

    forecast = None
    if MODEL_LOADED:
        forecast = predict_from_station_reading(db, reading)

    return make_json_safe({
        "success": True,
        "message": "Station reading saved successfully",
        "reading_id": reading.id,
        "forecast": forecast,
    })


@app.post("/station-readings/scrape-now")
def scrape_now(db: Session = Depends(get_db)):
    """
    Railway-safe scraper endpoint.

    1. Tries the normal scraper.py flow first.
    2. If OCR/Tesseract/Chrome/Selenium fails on Railway, it does NOT crash.
    3. Falls back to direct PMD HTML table parsing.
    4. Saves only valid station readings into station_readings.
    """
    scraper_errors = []

    try:
        from scraper import get_flood_data
        scraped_rows = get_flood_data()
    except Exception as scraper_error:
        error_text = str(scraper_error)
        scraper_errors.append(error_text)
        print(f"⚠️ Normal scraper failed, using Railway-safe direct PMD fallback: {error_text}")
        scraped_rows = []

    rows_to_process = normalize_scraper_rows(scraped_rows)

    if not rows_to_process:
        try:
            fallback_rows = direct_pmd_discharge_scraper()
            rows_to_process = normalize_scraper_rows(fallback_rows)
        except Exception as fallback_error:
            scraper_errors.append(str(fallback_error))
            print(f"❌ Direct PMD fallback failed: {fallback_error}")

    if not rows_to_process:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Scraper returned no usable station readings.",
                "errors": scraper_errors,
                "note": "Railway has no Tesseract by default, so direct PMD fallback was attempted."
            }
        )

    saved_results = []
    skipped_rows = []

    for row in rows_to_process:
        try:
            station = normalize_station_name(row.get("station"))
            discharge = float(row.get("discharge") or 0.0)

            if discharge <= 0:
                skipped_rows.append({"row": row, "reason": "discharge <= 0"})
                continue

            if discharge > 1000000:
                skipped_rows.append({"row": row, "reason": "unrealistic discharge"})
                continue

            reading = save_station_reading(
                db=db,
                station=station,
                discharge=discharge,
                rainfall_mm=float(row.get("rainfall_mm", 0) or 0),
                temperature_c=float(row.get("temperature_c", 25) or 25),
                soil_moisture_mm=float(row.get("soil_moisture_mm", 25) or 25),
                reading_time=parse_scraper_datetime(row.get("date")),
                source=row.get("source") or "scraper"
            )

            forecast = None
            forecast_error = None
            if MODEL_LOADED:
                try:
                    forecast = predict_from_station_reading(db, reading)
                except Exception as e:
                    forecast_error = str(e)
                    print(f"⚠️ Forecast failed for {station}, but reading was saved: {e}")

            saved_results.append({
                "station": station,
                "reading_id": int(reading.id),
                "reading_time": reading.reading_time.isoformat() if reading.reading_time else None,
                "discharge": float(discharge),
                "source": reading.source,
                "forecast": make_json_safe(forecast),
                "forecast_error": forecast_error,
            })

            print(f"✅ Saved {station}: {discharge}")

        except Exception as row_error:
            print(f"❌ Error processing scraper row: {row_error}")
            skipped_rows.append({"row": make_json_safe(row), "reason": str(row_error)})
            continue

    return make_json_safe({
        "success": True,
        "saved_count": len(saved_results),
        "skipped_count": len(skipped_rows),
        "scraper_errors": scraper_errors,
        "results": saved_results,
        "skipped_rows": skipped_rows,
    })

@app.get("/station-readings/history/{station_name}")
def station_history(
    station_name: str,
    limit: int = 20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Historical mode endpoint.

    Use this when the authority selects a custom date range.
    It returns ACTUAL recorded station_readings from the database only.
    It does not return ML predictions.
    """
    station = normalize_station_name(station_name)
    start_dt = parse_date_range_value(start_date, end_of_day=False)
    end_dt = parse_date_range_value(end_date, end_of_day=True)

    query = (
        db.query(StationReadingDB)
        .filter(StationReadingDB.station == station)
        .filter(StationReadingDB.source.in_(REAL_READING_SOURCES))
    )

    if start_dt:
        query = query.filter(StationReadingDB.reading_time >= start_dt)
    if end_dt:
        query = query.filter(StationReadingDB.reading_time <= end_dt)

    rows = (
        query
        .order_by(StationReadingDB.reading_time.desc(), StationReadingDB.id.desc())
        .limit(limit)
        .all()
    )

    readings = [serialize_station_reading(row) for row in rows]

    return {
        "success": True,
        "mode": "historical",
        "data_source": "database_station_readings",
        "station": station,
        "start_date": start_date,
        "end_date": end_date,
        "total": len(readings),
        "message": "No data found for selected date range." if len(readings) == 0 else "Historical readings found.",
        "readings": readings
    }

@app.get("/predict/station/{station_name}")
def predict_station(station_name: str, db: Session = Depends(get_db)):
    """
    Predict 24h/48h/72h forecast for one station from latest PostgreSQL reading.
    """
    station_name = normalize_station_name(station_name)
    reading = get_latest_station_reading(db, station_name)

    if not reading:
        raise HTTPException(
            status_code=404,
            detail=f"No station reading found for {station_name}. Add reading or run /station-readings/scrape-now first."
        )

    return make_json_safe({
        "success": True,
        **predict_from_station_reading(db, reading)
    })


# ═══════════════════════════════════════════════
#  FLOOD DATA TABLE ENDPOINTS
# ═══════════════════════════════════════════════

class FloodDataInput(BaseModel):
    uc_id:             Optional[int] = None
    station:           str
    location:          str
    rainfall:          float = 0.0
    humidity:          float = 50.0
    temperature:       float = 25.0
    river_level:       float = 5.0
    current_discharge: float = 5000.0

@app.post("/flood-data/add")
def add_flood_data(data: FloodDataInput, db: Session = Depends(get_db)):
    """
    Insert a new flood data record and run ML prediction on it.
    Stores prediction result in the database.
    """
    month           = datetime.now().month
    is_flood_season = 1 if month in [7, 8, 9] else 0

    prediction_result = None
    risk_pct          = None
    risk_lbl          = None

    if MODEL_LOADED:
        prediction_result = float(data.current_discharge)
        risk_pct          = discharge_to_percentage(prediction_result)
        risk_lbl          = risk_label(risk_pct)

    new_record = FloodDataDB(
        uc_id             = data.uc_id,
        station           = data.station,
        location          = data.location,
        rainfall          = data.rainfall,
        humidity          = data.humidity,
        temperature       = data.temperature,
        river_level       = data.river_level,
        current_discharge = data.current_discharge,
        prediction_result = prediction_result,
        risk_percentage   = risk_pct,
        risk_level        = risk_lbl,
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    return {
        "message":          "Flood data record added and prediction run successfully",
        "id":               new_record.id,
        "uc_id":            new_record.uc_id,
        "station":          new_record.station,
        "prediction_result": round(prediction_result, 2) if prediction_result else None,
        "risk_percentage":  risk_pct,
        "risk_level":       risk_lbl,
        "created_at":       new_record.created_at,
    }


@app.get("/flood-data")
def get_flood_data(station: Optional[str] = None, db: Session = Depends(get_db)):
    """
    Fetch all flood data records. Optionally filter by station.
    Example: /flood-data?station=Marala
    """
    query = db.query(FloodDataDB)
    if station:
        query = query.filter(FloodDataDB.station == station)
    records = query.order_by(FloodDataDB.created_at.desc()).all()

    return [
        {
            "id":                r.id,
            "uc_id":             r.uc_id,
            "station":           r.station,
            "location":          r.location,
            "rainfall":          r.rainfall,
            "humidity":          r.humidity,
            "temperature":       r.temperature,
            "river_level":       r.river_level,
            "current_discharge": r.current_discharge,
            "prediction_result": r.prediction_result,
            "risk_percentage":   r.risk_percentage,
            "risk_level":        r.risk_level,
            "created_at":        r.created_at,
        }
        for r in records
    ]


@app.post("/flood-data/predict/{record_id}")
def predict_for_record(record_id: int, db: Session = Depends(get_db)):
    """
    Run ML prediction on an existing flood data record and update it.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    record = db.query(FloodDataDB).filter(FloodDataDB.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")

    month           = record.created_at.month
    is_flood_season = 1 if month in [7, 8, 9] else 0

    prediction_result = float(record.current_discharge)
    risk_pct          = discharge_to_percentage(prediction_result)
    risk_lbl          = risk_label(risk_pct)

    record.prediction_result = prediction_result
    record.risk_percentage   = risk_pct
    record.risk_level        = risk_lbl
    db.commit()

    return {
        "message":           "Prediction updated successfully",
        "id":                record.id,
        "station":           record.station,
        "prediction_result": round(prediction_result, 2),
        "risk_percentage":   risk_pct,
        "risk_level":        risk_lbl,
    }


# ── EMERGENCY CONTACTS ─────────────────────────
@app.get("/api/emergency-contacts")
@app.get("/emergency-contacts")
def get_contacts():
    return [
        {"department": "PDMA Punjab Helpline",      "number": "1129"},
        {"department": "Rescue 1122",               "number": "1122"},
        {"department": "Police Emergency",          "number": "15"},
        {"department": "Punjab Flood Control Room", "number": "(042) 99203005"},
        {"department": "District Administration",   "number": "1043"},
    ]


@app.get("/api/emergency-resources")
def emergency_resources(db: Session = Depends(get_db)):
    """
    Public endpoint for all emergency pages/cards.
    It returns shelters directly from the database, so authority shelter changes
    are reflected everywhere after frontend refresh/re-fetch.
    """
    shelters = db.query(ShelterDB).order_by(ShelterDB.updated_at.desc()).all()

    return {
        "success": True,
        "shelters": [serialize_shelter(s) for s in shelters],
        "contacts": get_contacts(),
        "total_shelters": len(shelters),
        "last_updated": datetime.utcnow().isoformat()
    }


# ── UNION COUNCILS ──────────────────────────────
def serialize_union_council(uc: UnionCouncilDB):
    return {
        "id": uc.id,
        "uc_name": uc.uc_name,
        "name": uc.uc_name,          # frontend-friendly alias
        "district": uc.district,
        "station": uc.station,
        "geometry": uc.geometry,
        "created_at": uc.created_at,
    }

@app.get("/union-councils")
def get_union_councils(db: Session = Depends(get_db)):
    records = db.query(UnionCouncilDB).order_by(UnionCouncilDB.id.asc()).all()
    return {
        "success": True,
        "union_councils": [serialize_union_council(uc) for uc in records],
        "total": len(records),
    }


# ── SHELTERS ───────────────────────────────────
class ShelterModel(BaseModel):
    name:       str
    location:   str
    capacity:   int
    occupied:   int = 0
    status:     str
    facilities: List[str]
    district:   Optional[str] = None
    contact_number: Optional[str] = None
    uc_id:      Optional[int] = None

def serialize_shelter(shelter: ShelterDB):
    try:
        facilities = json.loads(shelter.facilities) if shelter.facilities else []
    except Exception:
        facilities = []

    return {
        "id": shelter.id,
        "name": shelter.name,
        "location": shelter.location,
        "district": shelter.district,
        "contact_number": shelter.contact_number,
        "capacity": shelter.capacity,
        "occupied": shelter.occupied,
        "status": shelter.status,
        "facilities": facilities,
        "uc_id": shelter.uc_id,
        "updated_at": shelter.updated_at,
    }

@app.get("/api/shelters")
@app.get("/shelters")
def get_shelters(db: Session = Depends(get_db)):
    shelters = db.query(ShelterDB).order_by(ShelterDB.id.asc()).all()
    return [serialize_shelter(s) for s in shelters]


@app.get("/api/public/shelters")
def get_public_shelters(db: Session = Depends(get_db)):
    """
    Public shelter endpoint for landing page, emergency page, flood map,
    and risk-card emergency resources button.
    """
    shelters = db.query(ShelterDB).order_by(ShelterDB.updated_at.desc()).all()
    return {
        "success": True,
        "shelters": [serialize_shelter(s) for s in shelters],
        "total": len(shelters),
        "last_updated": datetime.utcnow().isoformat()
    }

@app.post("/api/shelters/add")
@app.post("/shelters/add")
def add_shelter(shelter: ShelterModel, db: Session = Depends(get_db), token: dict = Depends(require_authority)):
    new_shelter = ShelterDB(
        name=shelter.name,
        location=shelter.location,
        district=shelter.district,
        contact_number=shelter.contact_number,
        capacity=shelter.capacity,
        occupied=shelter.occupied,
        status=shelter.status,
        facilities=json.dumps(shelter.facilities),
        uc_id=shelter.uc_id,
        updated_at=datetime.utcnow(),
    )
    db.add(new_shelter)
    db.commit()
    db.refresh(new_shelter)
    return {"message": "Shelter added successfully", "shelter": serialize_shelter(new_shelter)}

@app.put("/api/shelters/{shelter_id}")
@app.put("/shelters/{shelter_id}")
def update_shelter(shelter_id: int, shelter: ShelterModel, db: Session = Depends(get_db), token: dict = Depends(require_authority)):
    existing = db.query(ShelterDB).filter(ShelterDB.id == shelter_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Shelter not found")

    existing.name = shelter.name
    existing.location = shelter.location
    existing.district = shelter.district
    existing.contact_number = shelter.contact_number
    existing.capacity = shelter.capacity
    existing.occupied = shelter.occupied
    existing.status = shelter.status
    existing.facilities = json.dumps(shelter.facilities)
    existing.uc_id = shelter.uc_id
    existing.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(existing)
    return {"message": "Shelter updated successfully", "shelter": serialize_shelter(existing)}

@app.delete("/api/shelters/{shelter_id}")
@app.delete("/shelters/{shelter_id}")
def delete_shelter(shelter_id: int, db: Session = Depends(get_db), token: dict = Depends(require_authority)):
    existing = db.query(ShelterDB).filter(ShelterDB.id == shelter_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Shelter not found")

    deleted_name = existing.name
    db.delete(existing)
    db.commit()
    return {"message": f"Shelter '{deleted_name}' deleted successfully"}


# ── EXPORT REPORT HISTORY ──────────────────────
class ExportReportInput(BaseModel):
    station: Optional[str] = None
    forecast_period: Optional[str] = None
    report_type: str = "CSV"

@app.post("/api/export-reports/add")
@app.post("/export-reports/add")
def add_export_report(data: ExportReportInput, db: Session = Depends(get_db), token: dict = Depends(require_authority)):
    user = db.query(UserDB).filter(UserDB.email == token.get("email")).first()
    new_report = ExportReportDB(
        authority_id=user.id if user else None,
        station=data.station,
        forecast_period=data.forecast_period,
        report_type=data.report_type,
        generated_at=datetime.utcnow(),
    )
    db.add(new_report)
    db.commit()
    db.refresh(new_report)
    return {
        "message": "Export report history saved successfully",
        "report": {
            "id": new_report.id,
            "authority_id": new_report.authority_id,
            "station": new_report.station,
            "forecast_period": new_report.forecast_period,
            "report_type": new_report.report_type,
            "generated_at": new_report.generated_at,
        }
    }

@app.get("/export-reports")
def get_export_reports(db: Session = Depends(get_db), token: dict = Depends(require_authority)):
    records = db.query(ExportReportDB).order_by(ExportReportDB.generated_at.desc()).all()
    return [
        {
            "id": r.id,
            "authority_id": r.authority_id,
            "station": r.station,
            "forecast_period": r.forecast_period,
            "report_type": r.report_type,
            "generated_at": r.generated_at,
        }
        for r in records
    ]

# ── CONTACT FORM ───────────────────────────────
class ContactMessage(BaseModel):
    name:    str
    email:   str
    subject: str
    message: str

@app.post("/api/contact")
@app.post("/contact")
def submit_contact(msg: ContactMessage, db: Session = Depends(get_db)):
    new_msg = ContactDB(name=msg.name, email=msg.email, subject=msg.subject, message=msg.message)
    db.add(new_msg)
    db.commit()
    return {"success": True, "message": "Your message has been received. We'll get back to you shortly."}

@app.get("/contact-messages")
def get_messages(db: Session = Depends(get_db), token: dict = Depends(require_authority)):
    messages = db.query(ContactDB).order_by(ContactDB.sent_at.desc()).all()
    return [{"id": m.id, "name": m.name, "email": m.email,
             "subject": m.subject, "message": m.message, "sent_at": m.sent_at} for m in messages]


# ── CITIZEN REPORTS ─────────────────────────────
class Report(BaseModel):
    name:        str
    location:    str
    description: str

@app.post("/submit-report")
def submit_report(report: Report, db: Session = Depends(get_db)):
    new_report = ReportDB(name=report.name, location=report.location, description=report.description)
    db.add(new_report)
    db.commit()
    return {"message": "Report submitted successfully"}

@app.get("/reports")
def get_reports(db: Session = Depends(get_db), token: dict = Depends(require_authority)):
    reports = db.query(ReportDB).order_by(ReportDB.submitted_at.desc()).all()
    return [{"id": r.id, "name": r.name, "location": r.location,
             "description": r.description, "submitted_at": r.submitted_at} for r in reports]

# ─────────────────────────────────────────────
# RUN APP FOR HUGGING FACE
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# AUTOMATIC SCRAPER SCHEDULER
# Runs scraper 3 times a day to build station history for lag/rolling features.
# ─────────────────────────────────────────────
def scheduled_scraper_job():
    db = SessionLocal()
    try:
        print("⏰ Running scheduled scraper job...")

        from scraper import get_flood_data
        scraped_rows = get_flood_data()

        if not scraped_rows:
            print("⚠️ Scheduled scraper returned no data.")
            return

        rows_to_process = scraped_rows.values() if isinstance(scraped_rows, dict) else scraped_rows

        for row in rows_to_process:
            try:
                station = normalize_station_name(str(row.get("station", "")).strip())

                discharge_value = row.get("discharge") or row.get("inflow") or 0
                discharge = float(discharge_value)

                if discharge <= 0 or discharge > 1000000:
                    print(f"⚠️ Skipping invalid scraper value for {station}: {discharge}")
                    continue

                reading = save_station_reading(
                    db=db,
                    station=station,
                    discharge=discharge,
                    rainfall_mm=float(row.get("rainfall_mm", 0) or 0),
                    temperature_c=float(row.get("temperature_c", 25) or 25),
                    soil_moisture_mm=float(row.get("soil_moisture_mm", 25) or 25),
                    reading_time=parse_scraper_datetime(row.get("date")),
                    source="scheduled_scraper"
                )

                print(f"✅ Scheduled reading saved: {station} = {discharge}, row id = {reading.id}")

            except Exception as row_error:
                print(f"❌ Scheduled row skipped: {row_error}")

    except Exception as e:
        print(f"❌ Scheduled scraper failed: {e}")

    finally:
        db.close()


scheduler = BackgroundScheduler(timezone="Asia/Karachi")
scheduler.add_job(scheduled_scraper_job, "cron", hour=6, minute=0)
scheduler.add_job(scheduled_scraper_job, "cron", hour=14, minute=0)
scheduler.add_job(scheduled_scraper_job, "cron", hour=22, minute=0)
scheduler.start()
print("✅ Scraper scheduler started: 6 AM, 2 PM, 10 PM")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False
    )