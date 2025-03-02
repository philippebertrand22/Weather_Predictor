#%%
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

spark = SparkSession.builder.appName("MergedData").getOrCreate()

folder_data = "data/"
csv_files = glob.glob(folder_data + "*.csv")

#print(csv_files)

dfs = [spark.read.option("header", "true").csv(f) for f in csv_files]

# Get all unique columns
all_columns = set()
for df in dfs:
    all_columns.update(df.columns)

# Convert to list
all_columns = list(all_columns)

def align_schema(df, all_columns):
    for col_name in all_columns:
        if col_name not in df.columns:
            df = df.withColumn(col_name, lit(None))  # Add missing columns with NULL
    return df.select(all_columns)  # Ensure consistent column order

# Apply schema alignment
dfs = [align_schema(df, all_columns) for df in dfs]

# Merge all DataFrames
final_df = dfs[0]
for df in dfs[1:]:
    final_df = final_df.union(df)

keep_columns = ["STATION_NAME",
                "LOCAL_YEAR",
                "LOCAL_MONTH",
                "MEAN_TEMPERATURE"]

final_df = final_df.select(keep_columns)

#final_df.show()

final_df.createOrReplaceTempView("weather_data")

query = f"""
    SELECT DISTINCT STATION_NAME
    FROM weather_data
"""

        
sql = spark.sql(query)
#sql = spark.sql("SELECT COUNT(*) AS total_rows FROM weather_data")

sql.show()
#%%