import dlt
from pyspark.sql.functions import col, current_timestamp, month, to_date, year
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

# ---------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------
# Pipeline settings (set in the pipeline UI, not in this file):
#   Default catalog : weather_medallion_dlt
#   Default schema  : bronze
#   -> the table below is created as weather_medallion_dlt.bronze.weather_bronze
#
# Pipeline configuration key (Settings -> Configuration):
#   raw_weather_path = /Volumes/weather_medallion_dlt/bronze/raw_weather
# If the key is missing, the default below is used.
RAW_PATH = spark.conf.get(
    "raw_weather_path",
    "/Volumes/weather_medallion_dlt/bronze/raw_weather",
)

RAW_SCHEMA = StructType([
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("time", StringType(), True),  # cast to DateType below
    StructField("temperature_max", DoubleType(), True),
    StructField("precipitation", DoubleType(), True),
])


# ---------------------------------------------------------
# BRONZE LAYER: Raw Data Ingestion (from CSV files in the Volume, no API calls)
# ---------------------------------------------------------
@dlt.table(
    name="weather_bronze",
    comment="Raw daily weather data (max temperature, precipitation) from Open-Meteo, loaded from CSV files in the raw_weather Volume",
    table_properties={"quality": "bronze"},
)
@dlt.expect_all_or_drop({
    "valid_date": "time IS NOT NULL",
    "valid_city": "city IS NOT NULL",
})
@dlt.expect_all({
    "plausible_temperature": "temperature_max IS NULL OR temperature_max BETWEEN -30 AND 55",
    "non_negative_precipitation": "precipitation IS NULL OR precipitation >= 0",
})
def weather_bronze():
    return (
        spark.read.format("csv")
        .option("header", True)
        .option("pathGlobFilter", "*.csv")
        .option("recursiveFileLookup", True)
        .schema(RAW_SCHEMA)
        .load(RAW_PATH)
        .withColumn("time", to_date(col("time")))
        .withColumn("year", year(col("time")))
        .withColumn("month", month(col("time")))
        .withColumn("source_file", col("_metadata.file_name"))
        .withColumn("ingested_at", current_timestamp())
    )
