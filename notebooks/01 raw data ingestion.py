# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Download Open-Meteo weather to a Unity Catalog Volume
# MAGIC Writes one CSV per city per year into the Volume folder. Resumable: complete files are skipped,
# MAGIC so the job can simply be rerun after a quota stop. Run as a Databricks Job (or interactively).

# COMMAND ----------

import csv
import os
import re
import time
from datetime import date, timedelta

import requests

dbutils.widgets.text("volume_path", "/Volumes/weather_medallion_dlt/bronze/raw_weather")
dbutils.widgets.text("start_date", "2020-01-01")
dbutils.widgets.text("end_date", "2026-10-31")

VOLUME_PATH = dbutils.widgets.get("volume_path").rstrip("/")
START = date.fromisoformat(dbutils.widgets.get("start_date"))
ARCHIVE_LAG_DAYS = 5  # the archive trails real time by a few days
END = min(
    date.fromisoformat(dbutils.widgets.get("end_date")),
    date.today() - timedelta(days=ARCHIVE_LAG_DAYS),
)

API_URL = "https://archive-api.open-meteo.com/v1/archive"
PAUSE_BETWEEN_CALLS = 2   # seconds
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
HEADER = ["city", "state", "latitude", "longitude", "time", "temperature_max", "precipitation"]

LOCATIONS = [
    {"city": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    {"city": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"city": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946},
    {"city": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639},
    {"city": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707},
    {"city": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867},
    {"city": "Ahmedabad", "state": "Gujarat", "lat": 23.0258, "lon": 72.5873},
    {"city": "Pune", "state": "Maharashtra", "lat": 18.5133, "lon": 73.8499},
    {"city": "Amaravati", "state": "Andhra Pradesh", "lat": 16.5149, "lon": 80.5163},
    {"city": "Itanagar", "state": "Arunachal Pradesh", "lat": 27.0844, "lon": 93.6053},
    {"city": "Dispur", "state": "Assam", "lat": 26.1408, "lon": 91.7898},
    {"city": "Patna", "state": "Bihar", "lat": 25.5941, "lon": 85.1376},
    {"city": "Raipur", "state": "Chhattisgarh", "lat": 21.2514, "lon": 81.6296},
    {"city": "Panaji", "state": "Goa", "lat": 15.4909, "lon": 73.8278},
    {"city": "Gandhinagar", "state": "Gujarat", "lat": 23.2156, "lon": 72.6369},
    {"city": "Chandigarh", "state": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    {"city": "Shimla", "state": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734},
    {"city": "Ranchi", "state": "Jharkhand", "lat": 23.3441, "lon": 85.3096},
    {"city": "Thiruvananthapuram", "state": "Kerala", "lat": 8.5241, "lon": 76.9366},
    {"city": "Bhopal", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126},
    {"city": "Imphal", "state": "Manipur", "lat": 24.8170, "lon": 93.9368},
    {"city": "Shillong", "state": "Meghalaya", "lat": 25.5788, "lon": 91.8933},
    {"city": "Aizawl", "state": "Mizoram", "lat": 23.7271, "lon": 92.7176},
    {"city": "Kohima", "state": "Nagaland", "lat": 25.6751, "lon": 94.1086},
    {"city": "Bhubaneswar", "state": "Odisha", "lat": 20.2961, "lon": 85.8245},
    {"city": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873},
    {"city": "Gangtok", "state": "Sikkim", "lat": 27.3314, "lon": 88.6138},
    {"city": "Agartala", "state": "Tripura", "lat": 23.8315, "lon": 91.2868},
    {"city": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
    {"city": "Dehradun", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322},
    {"city": "Port Blair", "state": "Andaman and Nicobar Islands", "lat": 11.6234, "lon": 92.7265},
    {"city": "Daman", "state": "Dadra and Nagar Haveli and Daman and Diu", "lat": 20.3974, "lon": 72.8328},
    {"city": "Srinagar", "state": "Jammu and Kashmir", "lat": 34.0837, "lon": 74.7973},
    {"city": "Jammu", "state": "Jammu and Kashmir", "lat": 32.7266, "lon": 74.8570},
    {"city": "Leh", "state": "Ladakh", "lat": 34.1526, "lon": 77.5770},
    {"city": "Kavaratti", "state": "Lakshadweep", "lat": 10.5593, "lon": 72.6358},
    {"city": "Puducherry", "state": "Puducherry", "lat": 11.9416, "lon": 79.8083},
]

# COMMAND ----------

class QuotaExceeded(Exception):
    """Hourly or daily Open-Meteo limit reached; retrying now cannot help."""


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def year_chunks(start, end):
    for y in range(start.year, end.year + 1):
        yield max(start, date(y, 1, 1)), min(end, date(y, 12, 31))


def file_is_complete(path, chunk_end):
    """Complete if the last data row is on/after the chunk end date."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return len(rows) > 1 and rows[-1][4] >= chunk_end.isoformat()
    except Exception:
        return False


def fetch_chunk(loc, c_start, c_end):
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": c_start.isoformat(),
        "end_date": c_end.isoformat(),
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto",
    }
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                body = resp.text.lower()
                if "daily" in body or "hourly" in body:
                    raise QuotaExceeded(resp.text[:200])
                last_error = f"429 (minutely): {resp.text[:200]}"
                time.sleep(60)
                continue
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            daily = resp.json().get("daily")
            if not daily or not daily.get("time"):
                raise ValueError(f"No daily data: {resp.text[:200]}")
            return [
                [loc["city"], loc["state"], loc["lat"], loc["lon"], t,
                 "" if tmax is None else tmax, "" if prcp is None else prcp]
                for t, tmax, prcp in zip(
                    daily["time"], daily["temperature_2m_max"], daily["precipitation_sum"]
                )
            ]
        except QuotaExceeded:
            raise
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{loc['city']} {c_start}..{c_end}: failed after {MAX_RETRIES} attempts: {last_error}")

# COMMAND ----------

# Preflight: one tiny request. If this compute's IP is already over quota, stop now.
try:
    fetch_chunk(LOCATIONS[0], END - timedelta(days=2), END)
    print("Preflight OK: Open-Meteo is reachable and quota is available from this compute.")
except QuotaExceeded as exc:
    raise RuntimeError(
        "Open-Meteo quota is already exhausted for this compute's IP address "
        f"({exc}). Try classic (non-serverless) compute, run the laptop script instead, "
        "or rerun tomorrow."
    )

# COMMAND ----------

os.makedirs(VOLUME_PATH, exist_ok=True)

todo = [(loc, cs, ce) for loc in LOCATIONS for cs, ce in year_chunks(START, END)]
print(f"Range {START} .. {END}: {len(todo)} city-year files to check in {VOLUME_PATH}")

downloaded = skipped = 0
stopped_reason = None

for loc, cs, ce in todo:
    path = f"{VOLUME_PATH}/{slug(loc['city'])}_{cs.year}.csv"
    if file_is_complete(path, ce):
        skipped += 1
        continue
    try:
        rows = fetch_chunk(loc, cs, ce)
    except QuotaExceeded as exc:
        stopped_reason = str(exc)
        break
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    downloaded += 1
    print(f"[{downloaded + skipped}/{len(todo)}] {loc['city']} {cs.year}: {len(rows)} rows")
    time.sleep(PAUSE_BETWEEN_CALLS)

print(f"Downloaded {downloaded}, already complete {skipped}, total {len(todo)}.")

# Fail the task if we stopped early so the job shows as failed and can be retried/rerun.
if stopped_reason:
    raise RuntimeError(
        f"Stopped early on quota limit: {stopped_reason}. Progress is saved; "
        f"{len(todo) - downloaded - skipped} files remain. Rerun the job later."
    )
print("All files are complete.")