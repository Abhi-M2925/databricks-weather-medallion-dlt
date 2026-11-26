# Databricks notebook source
# MAGIC %md
# MAGIC # Silver transform: cleaned daily weather + location dimension
# MAGIC Reads `bronze.weather_bronze` and writes two Delta tables:
# MAGIC * `silver.dim_location`  - one row per city (key: `location_id`)
# MAGIC * `silver.weather_daily` - one row per city per day (key: `location_id` + `weather_date`)
# MAGIC
# MAGIC Each run rebuilds both tables from bronze (about 92k rows), so reruns are safe.
# MAGIC Validation lives in the separate `silver_checks` notebook, which runs as the next job task.
# MAGIC Before writing, this notebook records the current table versions as a task value so the
# MAGIC checks notebook can optionally restore them if validation fails.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

dbutils.widgets.text("catalog", "weather_medallion_dlt")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")

CATALOG = dbutils.widgets.get("catalog")
BRONZE_SCHEMA = dbutils.widgets.get("bronze_schema")
SILVER_SCHEMA = dbutils.widgets.get("silver_schema")

BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.weather_bronze"
DIM_LOCATION = f"{CATALOG}.{SILVER_SCHEMA}.dim_location"
WEATHER_DAILY = f"{CATALOG}.{SILVER_SCHEMA}.weather_daily"

# Business rules (edit here if you want different thresholds)
RAINY_DAY_MM = 2.5  # a day counts as "rainy" at or above this precipitation (mm)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}")

# COMMAND ----------

def current_version(table_name):
    """Latest Delta version of a table, or -1 if the table does not exist yet."""
    if not spark.catalog.tableExists(table_name):
        return -1
    return int(spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1").first()["version"])


# Record the versions we may want to go back to, BEFORE anything is overwritten.
prev_versions = {
    "dim_location": current_version(DIM_LOCATION),
    "weather_daily": current_version(WEATHER_DAILY),
}
print(f"Versions before this run: {prev_versions}")

try:
    dbutils.jobs.taskValues.set(key="prev_versions", value=prev_versions)
except Exception as exc:  # not running inside a job: harmless
    print(f"Task value not saved (not running in a job): {exc}")

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)
print(f"Bronze rows: {bronze.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## dim_location

# COMMAND ----------

# Stable, readable key: lower-case city name with non-alphanumerics replaced by "_"
def location_key(city_col):
    return F.regexp_replace(F.lower(F.trim(city_col)), r"[^a-z0-9]+", "_")


dim_location = (
    bronze.where(F.col("city").isNotNull())
    .groupBy(F.trim(F.col("city")).alias("city"))
    .agg(
        F.first(F.trim(F.col("state")), ignorenulls=True).alias("state"),
        F.first("latitude", ignorenulls=True).alias("latitude"),
        F.first("longitude", ignorenulls=True).alias("longitude"),
    )
    .withColumn("location_id", location_key(F.col("city")))
    .select("location_id", "city", "state", "latitude", "longitude")
)

(
    dim_location.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(DIM_LOCATION)
)
print(f"Wrote {DIM_LOCATION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## weather_daily

# COMMAND ----------

# 1) Remove duplicates on (city, day), keeping the most recently ingested row
latest_first = Window.partitionBy("city", "time").orderBy(F.col("ingested_at").desc())

deduped = (
    bronze.where(F.col("city").isNotNull() & F.col("time").isNotNull())
    .withColumn("_rn", F.row_number().over(latest_first))
    .where(F.col("_rn") == 1)
    .drop("_rn")
)

# 2) Shape, flag missing values and add derived columns
weather_daily = (
    deduped.select(
        location_key(F.col("city")).alias("location_id"),
        F.col("time").alias("weather_date"),
        F.col("temperature_max").cast("double").alias("temperature_max"),
        F.col("precipitation").cast("double").alias("precipitation"),
    )
    .withColumn("year", F.year("weather_date"))
    .withColumn("month", F.month("weather_date"))
    # Missing-value flags: nulls are kept, never silently dropped
    .withColumn("is_temperature_missing", F.col("temperature_max").isNull())
    .withColumn("is_precipitation_missing", F.col("precipitation").isNull())
    # Indian meteorological seasons
    .withColumn(
        "season",
        F.when(F.col("month").isin(1, 2), "Winter")
        .when(F.col("month").isin(3, 4, 5), "Pre-monsoon")
        .when(F.col("month").isin(6, 7, 8, 9), "Monsoon")
        .otherwise("Post-monsoon"),  # Oct-Dec
    )
    # Null precipitation stays null (unknown), not "not rainy"
    .withColumn(
        "is_rainy_day",
        F.when(F.col("precipitation").isNull(), F.lit(None).cast("boolean"))
        .otherwise(F.col("precipitation") >= F.lit(RAINY_DAY_MM)),
    )
    .withColumn(
        "temperature_band",
        F.when(F.col("temperature_max").isNull(), F.lit(None).cast("string"))
        .when(F.col("temperature_max") < 20, "Cold (<20C)")
        .when(F.col("temperature_max") < 30, "Pleasant (20-30C)")
        .when(F.col("temperature_max") < 40, "Hot (30-40C)")
        .otherwise("Extreme heat (40C+)"),
    )
    .withColumn("silver_loaded_at", F.current_timestamp())
)

(
    weather_daily.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(WEATHER_DAILY)
)
print(f"Wrote {WEATHER_DAILY}")
print("Silver transform finished. Run silver_checks next.")
