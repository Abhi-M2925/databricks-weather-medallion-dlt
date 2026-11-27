# Databricks notebook source
# MAGIC %md
# MAGIC # Gold transform: analytics-ready weather tables
# MAGIC Reads `silver.weather_daily` + `silver.dim_location` and writes four Delta tables in the `gold` schema:
# MAGIC * `weather_monthly_city`    - one row per city per month
# MAGIC * `weather_yearly_city`     - one row per city per year, with hottest/wettest day and year-over-year change
# MAGIC * `weather_monthly_state`   - one row per state per month (rollup of the city rows)
# MAGIC * `weather_monthly_anomaly` - each city-month compared with that city's own average for the same calendar month
# MAGIC
# MAGIC Partial periods (e.g. 2026, which ends on 31 Oct) are kept but flagged with `is_complete_month` / `is_complete_year`.
# MAGIC Filter on those flags before comparing years. Each run rebuilds all four tables, so reruns are safe.
# MAGIC Validation lives in the separate `gold_checks` notebook.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "weather_medallion_dlt")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")

CATALOG = dbutils.widgets.get("catalog")
SILVER_SCHEMA = dbutils.widgets.get("silver_schema")
GOLD_SCHEMA = dbutils.widgets.get("gold_schema")

WEATHER_DAILY = f"{CATALOG}.{SILVER_SCHEMA}.weather_daily"
DIM_LOCATION = f"{CATALOG}.{SILVER_SCHEMA}.dim_location"

GOLD_TABLES = {
    "weather_monthly_city": f"{CATALOG}.{GOLD_SCHEMA}.weather_monthly_city",
    "weather_yearly_city": f"{CATALOG}.{GOLD_SCHEMA}.weather_yearly_city",
    "weather_monthly_state": f"{CATALOG}.{GOLD_SCHEMA}.weather_monthly_state",
    "weather_monthly_anomaly": f"{CATALOG}.{GOLD_SCHEMA}.weather_monthly_anomaly",
}

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}")

# COMMAND ----------

def current_version(table_name):
    """Latest Delta version of a table, or -1 if the table does not exist yet."""
    if not spark.catalog.tableExists(table_name):
        return -1
    return int(spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1").first()["version"])


# Record the versions we may want to go back to, BEFORE anything is overwritten.
prev_versions = {name: current_version(full_name) for name, full_name in GOLD_TABLES.items()}
print(f"Versions before this run: {prev_versions}")

try:
    dbutils.jobs.taskValues.set(key="prev_versions", value=prev_versions)
except Exception as exc:  # not running inside a job: harmless
    print(f"Task value not saved (not running in a job): {exc}")


def write_table(df, name):
    (
        df.withColumn("gold_loaded_at", F.current_timestamp())
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(GOLD_TABLES[name])
    )
    print(f"Wrote {GOLD_TABLES[name]}")

# COMMAND ----------

dim = spark.table(DIM_LOCATION).select("location_id", "city", "state")
daily = spark.table(WEATHER_DAILY).join(dim, "location_id", "inner")
print(f"Silver daily rows used: {daily.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## weather_monthly_city

# COMMAND ----------

monthly_city = (
    daily.groupBy("location_id", "city", "state", "year", "month")
    .agg(
        F.count("*").alias("days_with_data"),
        F.round(F.avg("temperature_max"), 2).alias("avg_temperature_max"),
        F.max("temperature_max").alias("highest_temperature_max"),
        F.min("temperature_max").alias("lowest_temperature_max"),
        F.round(F.sum("precipitation"), 2).alias("total_precipitation"),
        F.max("precipitation").alias("max_daily_precipitation"),
        F.sum(F.when(F.col("is_rainy_day"), 1).otherwise(0)).alias("rainy_days"),
    )
    .withColumn(
        "days_in_month",
        F.dayofmonth(F.last_day(F.expr("make_date(year, month, 1)"))),
    )
    .withColumn("is_complete_month", F.col("days_with_data") == F.col("days_in_month"))
)

write_table(monthly_city, "weather_monthly_city")

# COMMAND ----------

# MAGIC %md
# MAGIC ## weather_yearly_city

# COMMAND ----------

hottest_window = Window.partitionBy("location_id", "year").orderBy(
    F.col("temperature_max").desc(), F.col("weather_date").asc()
)
wettest_window = Window.partitionBy("location_id", "year").orderBy(
    F.col("precipitation").desc(), F.col("weather_date").asc()
)

hottest_day = (
    daily.where(F.col("temperature_max").isNotNull())
    .withColumn("_rn", F.row_number().over(hottest_window))
    .where("_rn = 1")
    .select(
        "location_id", "year",
        F.col("temperature_max").alias("hottest_day_temperature"),
        F.col("weather_date").alias("hottest_day"),
    )
)

wettest_day = (
    daily.where(F.col("precipitation").isNotNull())
    .withColumn("_rn", F.row_number().over(wettest_window))
    .where("_rn = 1")
    .select(
        "location_id", "year",
        F.col("precipitation").alias("wettest_day_precipitation"),
        F.col("weather_date").alias("wettest_day"),
    )
)

yearly_base = (
    daily.groupBy("location_id", "city", "state", "year")
    .agg(
        F.count("*").alias("days_with_data"),
        F.round(F.avg("temperature_max"), 2).alias("avg_temperature_max"),
        F.round(F.sum("precipitation"), 2).alias("total_precipitation"),
        F.sum(F.when(F.col("is_rainy_day"), 1).otherwise(0)).alias("rainy_days"),
    )
    .withColumn(
        "days_in_year",
        F.expr("datediff(make_date(year, 12, 31), make_date(year, 1, 1)) + 1"),
    )
    .withColumn("is_complete_year", F.col("days_with_data") == F.col("days_in_year"))
)

by_year = Window.partitionBy("location_id").orderBy("year")

yearly_city = (
    yearly_base.join(hottest_day, ["location_id", "year"], "left")
    .join(wettest_day, ["location_id", "year"], "left")
    .withColumn("_prev_temp", F.lag("avg_temperature_max").over(by_year))
    .withColumn("_prev_precip", F.lag("total_precipitation").over(by_year))
    .withColumn("_prev_complete", F.lag("is_complete_year").over(by_year))
    # Year-over-year change only when both years are complete, otherwise null
    .withColumn(
        "yoy_avg_temperature_change",
        F.when(
            F.col("is_complete_year") & F.col("_prev_complete"),
            F.round(F.col("avg_temperature_max") - F.col("_prev_temp"), 2),
        ),
    )
    .withColumn(
        "yoy_total_precipitation_change",
        F.when(
            F.col("is_complete_year") & F.col("_prev_complete"),
            F.round(F.col("total_precipitation") - F.col("_prev_precip"), 2),
        ),
    )
    .drop("_prev_temp", "_prev_precip", "_prev_complete")
)

write_table(yearly_city, "weather_yearly_city")

# COMMAND ----------

# MAGIC %md
# MAGIC ## weather_monthly_state
# MAGIC A state value is the plain average across its cities' monthly values (every city counts equally).
# MAGIC `avg_city_precipitation_total` is the average of each city's monthly total, not the sum across cities.

# COMMAND ----------

monthly_state = (
    monthly_city.groupBy("state", "year", "month")
    .agg(
        F.countDistinct("location_id").alias("cities_in_state"),
        F.round(F.avg("avg_temperature_max"), 2).alias("avg_temperature_max"),
        F.max("highest_temperature_max").alias("highest_temperature_max"),
        F.round(F.avg("total_precipitation"), 2).alias("avg_city_precipitation_total"),
        F.round(F.avg("rainy_days"), 2).alias("avg_city_rainy_days"),
        (F.min(F.col("is_complete_month").cast("int")) == 1).alias("is_complete_month"),
    )
)

write_table(monthly_state, "weather_monthly_state")

# COMMAND ----------

# MAGIC %md
# MAGIC ## weather_monthly_anomaly
# MAGIC The baseline is the city's average for the same calendar month across all complete months in the data.
# MAGIC Incomplete months get null anomalies and are excluded from the baseline.

# COMMAND ----------

calendar_month = Window.partitionBy("location_id", "month")

anomaly = (
    monthly_city.select(
        "location_id", "city", "state", "year", "month",
        "avg_temperature_max", "total_precipitation", "is_complete_month",
    )
    .withColumn(
        "baseline_avg_temperature_max",
        F.round(F.avg(F.when(F.col("is_complete_month"), F.col("avg_temperature_max"))).over(calendar_month), 2),
    )
    .withColumn(
        "baseline_total_precipitation",
        F.round(F.avg(F.when(F.col("is_complete_month"), F.col("total_precipitation"))).over(calendar_month), 2),
    )
    .withColumn(
        "temperature_anomaly",
        F.when(
            F.col("is_complete_month"),
            F.round(F.col("avg_temperature_max") - F.col("baseline_avg_temperature_max"), 2),
        ),
    )
    .withColumn(
        "precipitation_anomaly",
        F.when(
            F.col("is_complete_month"),
            F.round(F.col("total_precipitation") - F.col("baseline_total_precipitation"), 2),
        ),
    )
)

write_table(anomaly, "weather_monthly_anomaly")
print("Gold transform finished. Run gold_checks next.")
