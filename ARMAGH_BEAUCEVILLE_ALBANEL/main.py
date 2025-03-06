#%%
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import calendar

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
                "LOCAL_DAY",
                "MEAN_TEMPERATURE",
                "TOTAL_PRECIPITATION"]

final_df = final_df.select(keep_columns)

#final_df.show()

# Convert columns to appropriate data types
final_df = final_df.withColumn("LOCAL_YEAR", final_df["LOCAL_YEAR"].cast("int"))
final_df = final_df.withColumn("LOCAL_MONTH", final_df["LOCAL_MONTH"].cast("int"))
final_df = final_df.withColumn("LOCAL_DAY", final_df["LOCAL_DAY"].cast("int"))
final_df = final_df.withColumn("MEAN_TEMPERATURE", final_df["MEAN_TEMPERATURE"].cast("float"))
final_df = final_df.withColumn("TOTAL_PRECIPITATION", final_df["TOTAL_PRECIPITATION"].cast("float"))

#remove row with missing values
final_df = final_df.na.drop()

# Sort the data by date
final_df = final_df.orderBy("LOCAL_YEAR", "LOCAL_MONTH", "LOCAL_DAY")

# Save the final data to a CSV file
pandas_df = final_df.toPandas()

#make sure the file does not exist
file_path = "final_data.csv"
if os.path.exists(file_path):
    pass
else:
    pandas_df.to_csv("final_data.csv", index=False)


final_df.createOrReplaceTempView("weather_data")

for j in range(3, 5):
    data = []

    for i in range(1916, 2026):
        query = f"""
            SELECT {i} AS LOCAL_YEAR, AVG(MEAN_TEMPERATURE) AS `AVERAGE TEMPERATURE`, AVG(TOTAL_PRECIPITATION) AS `AVERAGE PRECIPITATION` 
            FROM weather_data
            WHERE LOCAL_YEAR = {i}
            AND LOCAL_MONTH = {j}
        """

        # Execute the query
        results = spark.sql(query).toPandas()
        data.append(results)
        
    # Concatenate data into a single DataFrame
    df = pd.concat(data)

    # Ensure there are no missing values
    df = df.dropna()
    
    # Skip months with insufficient data
    if len(df) < 2:
        print(f"Skipping month {j} due to insufficient data")
        continue

    # Convert data for regression
    x = df["LOCAL_YEAR"].values.reshape(-1, 1)
    y1 = df["AVERAGE TEMPERATURE"].values
    y2 = df["AVERAGE PRECIPITATION"].values
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(x, y1)
        
    # Generate trendline predictions
    trendline = model.predict(x)

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(df["LOCAL_YEAR"], df["AVERAGE TEMPERATURE"], '-', alpha=0.5, label="Average Temperature")
    plt.plot(df["LOCAL_YEAR"], trendline, 'r-', linewidth=2, label=f"Slope: {model.coef_[0]:.4f}°/year)")
    plt.plot(df["LOCAL_YEAR"], df["AVERAGE PRECIPITATION"], '-', alpha=0.5, label="Average Precipitation")
   
    # Get the month name
    month = calendar.month_name[j]
    
    # Adding labels and title
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Average Temperature for {month}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
#%%