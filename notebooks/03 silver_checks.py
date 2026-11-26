# Databricks notebook source
# MAGIC %md
# MAGIC # Silver checks: validate `dim_location` and `weather_daily`
# MAGIC Runs after `silver_transform` in the job. Raises an error (so the task fails and downstream
# MAGIC tasks are skipped) if any check does not pass.
# MAGIC
# MAGIC Optional: set `restore_on_failure = true` to roll both silver tables back to the versions they had
# MAGIC before the transform ran. This only works when the versions were passed by the transform task
# MAGIC (i.e. when running in the job). It is off by default.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "weather_medallion_dlt")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.dropdown("restore_on_failure", "false", ["false", "true"])
dbutils.widgets.text("transform_task_key", "silver_transform")  # task key of the transform task in the job

CATALOG = dbutils.widgets.get("catalog")
BRONZE_SCHEMA = dbutils.widgets.get("bronze_schema")
SILVER_SCHEMA = dbutils.widgets.get("silver_schema")
RESTORE_ON_FAILURE = dbutils.widgets.get("restore_on_failure") == "true"
TRANSFORM_TASK_KEY = dbutils.widgets.get("transform_task_key")

BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.weather_bronze"
DIM_LOCATION = f"{CATALOG}.{SILVER_SCHEMA}.dim_location"
WEATHER_DAILY = f"{CATALOG}.{SILVER_SCHEMA}.weather_daily"

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)
silver = spark.table(WEATHER_DAILY)
dim = spark.table(DIM_LOCATION)
failures = []

# 1) Row count equals the distinct (city, day) pairs in bronze
bronze_rows = bronze.count()
expected_rows = (
    bronze.where(F.col("city").isNotNull() & F.col("time").isNotNull())
    .select("city", "time").distinct().count()
)
silver_rows = silver.count()
if silver_rows != expected_rows:
    failures.append(f"Row count mismatch: silver={silver_rows}, expected={expected_rows}")

# 2) No duplicate keys
dupes = silver.groupBy("location_id", "weather_date").count().where("count > 1").count()
if dupes:
    failures.append(f"{dupes} duplicate (location_id, weather_date) keys")

# 3) Every fact row has a matching location
orphans = silver.join(dim, "location_id", "left_anti").count()
if orphans:
    failures.append(f"{orphans} rows with no matching dim_location row")

# 4) dim_location is clean: one row per location_id, no null coordinates or state
dim_rows = dim.count()
dim_keys = dim.select("location_id").distinct().count()
if dim_rows != dim_keys:
    failures.append(f"dim_location has duplicate location_id values ({dim_rows} rows, {dim_keys} keys)")
null_coords = dim.where("latitude IS NULL OR longitude IS NULL OR state IS NULL").count()
if null_coords:
    failures.append(f"{null_coords} dim_location rows with null state/latitude/longitude")

# 5) Each city has exactly one (state, lat, lon) in bronze, otherwise dim_location would hide a conflict
conflicts = (
    bronze.where(F.col("city").isNotNull())
    .groupBy("city")
    .agg(F.countDistinct("state", "latitude", "longitude").alias("variants"))
    .where("variants > 1")
    .count()
)
if conflicts:
    failures.append(f"{conflicts} cities have conflicting state/coordinates in bronze")

# 6) Each city should cover the same days
days_per_city = silver.groupBy("location_id").agg(F.countDistinct("weather_date").alias("days"))
spread = days_per_city.agg(F.min("days").alias("lo"), F.max("days").alias("hi")).first()
if spread["lo"] != spread["hi"]:
    failures.append(f"Cities have different day counts (min={spread['lo']}, max={spread['hi']})")

# 7) Value ranges
bad_values = silver.where(
    "(temperature_max IS NOT NULL AND temperature_max NOT BETWEEN -30 AND 55) "
    "OR (precipitation IS NOT NULL AND precipitation < 0)"
).count()
if bad_values:
    failures.append(f"{bad_values} rows with out-of-range temperature or negative precipitation")

# Summary (shown in the notebook / job output)
print(f"dim_location rows: {dim_rows}")
print(f"weather_daily rows: {silver_rows} (bronze rows: {bronze_rows})")
print(f"Date range: {silver.agg(F.min('weather_date')).first()[0]} .. {silver.agg(F.max('weather_date')).first()[0]}")
print(f"Missing temperature rows: {silver.where('is_temperature_missing').count()}")
print(f"Missing precipitation rows: {silver.where('is_precipitation_missing').count()}")

# COMMAND ----------

def restore_previous_versions():
    """Roll both silver tables back to the versions recorded by the transform task."""
    try:
        prev = dbutils.jobs.taskValues.get(
            taskKey=TRANSFORM_TASK_KEY, key="prev_versions", default=None, debugValue=None
        )
    except Exception:
        prev = None

    if not prev:
        return "Restore skipped: previous versions are not available (not running after the transform task in a job)."

    messages = []
    for key, table in (("dim_location", DIM_LOCATION), ("weather_daily", WEATHER_DAILY)):
        version = prev.get(key, -1)
        if version is None or version < 0:
            messages.append(f"{table}: no earlier version existed (first run), left as is")
            continue
        spark.sql(f"RESTORE TABLE {table} TO VERSION AS OF {int(version)}")
        messages.append(f"{table}: restored to version {int(version)}")
    return "Restore result: " + "; ".join(messages)


if failures:
    restore_note = ""
    if RESTORE_ON_FAILURE:
        restore_note = "\n" + restore_previous_versions()
    raise RuntimeError("Silver quality checks failed:\n- " + "\n- ".join(failures) + restore_note)

print("All silver quality checks passed.")
