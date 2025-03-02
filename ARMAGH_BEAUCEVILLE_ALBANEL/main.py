#%%
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

for j in range(1,13):
    data = []

    for i in range(1916, 2026):
        query = f"""
            SELECT {i} AS LOCAL_YEAR, AVG(MEAN_TEMPERATURE) AS `AVERAGE TEMPERATURE` 
            FROM weather_data
            WHERE LOCAL_YEAR = {i}
            AND LOCAL_MONTH = {j}
        """

        #sql = spark.sql(query)
        #sql = spark.sql("SELECT COUNT(*) AS total_rows FROM weather_data")

        #sql.show()
        
        results = spark.sql(query).toPandas()
        data.append(results)
        
    df = pd.concat(data)

    # Ensure there are no missing values
    df = df.dropna()

    plt.plot(df["LOCAL_YEAR"], df["AVERAGE TEMPERATURE"], label='Average Temperature')

    # Fit a linear trendline (1st degree polynomial)
    slope, intercept = np.polyfit(df["LOCAL_YEAR"], df["AVERAGE TEMPERATURE"], 1)
    trendline = slope * df["LOCAL_YEAR"] + intercept
    plt.plot(df["LOCAL_YEAR"], trendline, color='red', label='Trendline')

    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.title(f"Average Temperature for month {j}")
    plt.legend()

    plt.show()  
#%%