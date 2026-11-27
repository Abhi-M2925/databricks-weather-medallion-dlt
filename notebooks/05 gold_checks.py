# Databricks notebook source
# MAGIC %md
# MAGIC # Gold checks: validate the four gold tables
# MAGIC Runs after `gold_transform` in the job. Raises an error (so the task fails) if any check does not pass.
# MAGIC
# MAGIC Optional: set `restore_on_failure = true` to roll all four gold tables back to the versions they had before
# MAGIC the transform ran. This only works when the versions were passed by the transform task (i.e. in the job).
# MAGIC It is off by default.

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "weather_medallion_dlt")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.dropdown("restore_on_failure", "false", ["false", "true"])
dbutils.widgets.text("transform_task_key", "gold_transform")  # task key of the transform task in the job

CATALOG = dbutils.widgets.get("catalog")
SILVER_SCHEMA = dbutils.widgets.get("silver_schema")
GOLD_SCHEMA = dbutils.widgets.get("gold_schema")
RESTORE_ON_FAILURE = dbutils.widgets.get("restore_on_failure") == "true"
TRANSFORM_TASK_KEY = dbutils.widgets.get("transform_task_key")

GOLD_TABLES = {
    "weather_monthly_city": f"{CATALOG}.{GOLD_SCHEMA}.weather_monthly_city",
    "weather_yearly_city": f"{CATALOG}.{GOLD_SCHEMA}.weather_yearly_city",
    "weather_monthly_state": f"{CATALOG}.{GOLD_SCHEMA}.weather_monthly_state",
    "weather_monthly_anomaly": f"{CATALOG}.{GOLD_SCHEMA}.weather_monthly_anomaly",
}

silver = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.weather_daily")
dim = spark.table(f"{CATALOG}.{SILVER_SCHEMA}.dim_location")
monthly_city = spark.table(GOLD_TABLES["weather_monthly_city"])
yearly_city = spark.table(GOLD_TABLES["weather_yearly_city"])
monthly_state = spark.table(GOLD_TABLES["weather_monthly_state"])
anomaly = spark.table(GOLD_TABLES["weather_monthly_anomaly"])

failures = []

# COMMAND ----------

# 1) Nothing lost: monthly day counts add up to the silver row count
silver_rows = silver.count()
monthly_days = monthly_city.agg(F.sum("days_with_data")).first()[0] or 0
if monthly_days != silver_rows:
    failures.append(f"weather_monthly_city covers {monthly_days} days but silver has {silver_rows} rows")

# 2) Row counts equal the distinct groups in silver / city table
expected_monthly = silver.select("location_id", "year", "month").distinct().count()
if monthly_city.count() != expected_monthly:
    failures.append(f"weather_monthly_city rows={monthly_city.count()}, expected={expected_monthly}")

expected_yearly = silver.select("location_id", "year").distinct().count()
if yearly_city.count() != expected_yearly:
    failures.append(f"weather_yearly_city rows={yearly_city.count()}, expected={expected_yearly}")

expected_state = monthly_city.select("state", "year", "month").distinct().count()
if monthly_state.count() != expected_state:
    failures.append(f"weather_monthly_state rows={monthly_state.count()}, expected={expected_state}")

if anomaly.count() != expected_monthly:
    failures.append(f"weather_monthly_anomaly rows={anomaly.count()}, expected={expected_monthly}")

# 3) No duplicate keys
key_sets = {
    "weather_monthly_city": (monthly_city, ["location_id", "year", "month"]),
    "weather_yearly_city": (yearly_city, ["location_id", "year"]),
    "weather_monthly_state": (monthly_state, ["state", "year", "month"]),
    "weather_monthly_anomaly": (anomaly, ["location_id", "year", "month"]),
}
for name, (df, keys) in key_sets.items():
    dupes = df.groupBy(*keys).count().where("count > 1").count()
    if dupes:
        failures.append(f"{name}: {dupes} duplicate keys on {keys}")

# 4) Monthly figures add up to the yearly figures
reconcile = (
    monthly_city.groupBy("location_id", "year")
    .agg(
        F.sum("days_with_data").alias("m_days"),
        F.sum("total_precipitation").alias("m_precip"),
        F.sum("rainy_days").alias("m_rainy"),
    )
    .join(yearly_city.select("location_id", "year", "days_with_data", "total_precipitation", "rainy_days"),
          ["location_id", "year"])
    .where(
        (F.col("m_days") != F.col("days_with_data"))
        | (F.abs(F.col("m_precip") - F.col("total_precipitation")) > 0.05)
        | (F.col("m_rainy") != F.col("rainy_days"))
    )
    .count()
)
if reconcile:
    failures.append(f"{reconcile} city-years where monthly totals do not match the yearly table")

# 5) Sanity of counts
bad_counts = monthly_city.where(
    "days_with_data > days_in_month OR rainy_days > days_with_data OR days_with_data < 1"
).count()
if bad_counts:
    failures.append(f"{bad_counts} monthly rows with impossible day counts")

# 6) State rollup uses all of the state's cities
dim_cities = dim.groupBy("state").agg(F.countDistinct("location_id").alias("dim_cities"))
wrong_cities = (
    monthly_state.join(dim_cities, "state", "left")
    .where(F.col("cities_in_state") != F.col("dim_cities"))
    .count()
)
if wrong_cities:
    failures.append(f"{wrong_cities} state-months where cities_in_state differs from dim_location")

# 7) Anomalies average out to about zero for each city and calendar month (they are measured against that average)
bad_anomaly = (
    anomaly.where("is_complete_month")
    .groupBy("location_id", "month")
    .agg(F.avg("temperature_anomaly").alias("t"), F.avg("precipitation_anomaly").alias("p"))
    .where((F.abs("t") > 0.02) | (F.abs("p") > 0.02))
    .count()
)
if bad_anomaly:
    failures.append(f"{bad_anomaly} city-months where anomalies do not average to about zero")

incomplete_with_anomaly = anomaly.where(
    "NOT is_complete_month AND (temperature_anomaly IS NOT NULL OR precipitation_anomaly IS NOT NULL)"
).count()
if incomplete_with_anomaly:
    failures.append(f"{incomplete_with_anomaly} incomplete months have an anomaly value")

# 8) Value ranges
bad_ranges = monthly_city.where(
    "(highest_temperature_max IS NOT NULL AND highest_temperature_max NOT BETWEEN -30 AND 55) "
    "OR total_precipitation < 0"
).count()
if bad_ranges:
    failures.append(f"{bad_ranges} monthly rows with out-of-range temperature or negative precipitation")

# Summary
print(f"weather_monthly_city: {monthly_city.count()} rows")
print(f"weather_yearly_city: {yearly_city.count()} rows "
      f"({yearly_city.where('is_complete_year').count()} complete city-years)")
print(f"weather_monthly_state: {monthly_state.count()} rows")
print(f"weather_monthly_anomaly: {anomaly.count()} rows")
last_year = yearly_city.agg(F.max("year")).first()[0]
print(f"Latest year in data: {last_year} "
      f"(complete for {yearly_city.where(f'year = {last_year} AND is_complete_year').count()} of "
      f"{yearly_city.where(f'year = {last_year}').count()} cities)")

# COMMAND ----------

def restore_previous_versions():
    """Roll all gold tables back to the versions recorded by the transform task."""
    try:
        prev = dbutils.jobs.taskValues.get(
            taskKey=TRANSFORM_TASK_KEY, key="prev_versions", default=None, debugValue=None
        )
    except Exception:
        prev = None

    if not prev:
        return "Restore skipped: previous versions are not available (not running after the transform task in a job)."

    messages = []
    for name, table in GOLD_TABLES.items():
        version = prev.get(name, -1)
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
    raise RuntimeError("Gold quality checks failed:\n- " + "\n- ".join(failures) + restore_note)

print("All gold quality checks passed.")
