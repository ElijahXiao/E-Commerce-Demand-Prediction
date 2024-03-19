import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import calendar

def data_preprocessing(df:pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.rename(columns={
    "TransactionNo":"Transaction_id",
    "ProductName":"Product_name",
    "ProductNo":"Product_id",
    "CustomerNo":"Customer_id"}, inplace=True)
    
    # Casting data in the Date column into datetime type
    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Day_name"] = df["Date"].dt.day_name()
    df["Transaction_profit"] = df["Price"] * df["Quantity"]
    df["Week"] = df["Date"].dt.isocalendar().week
    
    # Group the data by product ID and week and aggregate the data using sum()
    df_product_weekly = df.groupby(["Product_id", "Week"]).agg({"Quantity": "sum"}).reset_index()
    # print(df_product_weekly.head())
    # Sort the data in descending order of quantity and select the top 50 products
    top_products = df_product_weekly.groupby("Product_id").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False).head(50).index.values
    # print(top_products)
    # Filter the data to only include the top 50 products
    df_product_weekly = df_product_weekly[df_product_weekly["Product_id"].isin(top_products)].copy()
    
    # Pivot the data to have product IDs as columns and weeks as rows
    df_pivot = df_product_weekly.pivot(index="Week", columns="Product_id", values="Quantity").abs()
    
    # print(df_pivot.head())
    # Replace NaN values in the product columns with mean
    product_cols = df_pivot.columns[:]
    # print('product_cols\n', product_cols)
    df_pivot[product_cols] = df_pivot[product_cols].fillna(df_pivot[product_cols].mean(axis=0))
    # print(df_pivot.head())
    
    # calculate the Pearson correlation coefficients between product 22197 and all other features
    corr = df_pivot.corr()['22197'].sort_values(ascending=False)
    # keep the top 10 values
    corr_top_10 = corr.head(11)
    print(corr_top_10)
    
    df_product_weekly = df_product_weekly[df_product_weekly["Product_id"].isin(corr_top_10.index.values)].copy()
    df_pivot = df_product_weekly.pivot(index="Week", columns="Product_id", values="Quantity").abs()
    
    # print(df_pivot.head())
    product_cols = df_pivot.columns
    # print(product_cols)
    df_pivot[product_cols] = df_pivot[product_cols].fillna(df_pivot[product_cols].mean(axis=0))
    # print(df_pivot.head())
    
    df_pivot['quantity_mean'] = df_pivot[product_cols].mean(axis=1)
    # print(df_pivot.head())
    df_pivot['quantity_sum'] = df_pivot[product_cols[:-1]].sum(axis=1)    
    df_pivot['quantity_std'] = df_pivot[product_cols[:-2]].std(axis=1)    
    # print(df_pivot.head())
    
    # Get the top 2 products sold
    top_products = df.groupby("Product_id").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False).head(5).index.tolist()
    print(top_products)
    # Filter the DataFrame to include only the top 2 products
  #  df = df[df["Product_id"].isin(top_products)].copy()
    df = df[df["Product_id"].isin(['22197'])].copy()
    # print(df)
    # Split the data into cancelled and successful transactions
    cancelled = df[df["Quantity"]<0].copy()
    successful = df[df["Quantity"] >= 0].copy()
    
    # Create a new DataFrame with one row for each week and one column for each attribute
    week_range = range(df["Week"].min(), df["Week"].max() + 1)
    attribute_cols = ["Units_auctioned", "Amount_auctioned", "Items_auctioned", "Bidders_participating", "Units_sold", "Amount_sold", "Items_sold", "Buyers"]
    df_summary = pd.DataFrame(index=week_range, columns=attribute_cols)
    
    # Fill in the DataFrame with the appropriate values for each attribute in each week
    for week in week_range:
        # Get the cancelled and successful transactions in the current week
        cancelled_week = cancelled[cancelled["Week"] == week]
        successful_week = successful[successful["Week"] == week]
    
        # Calculate the values for the "Units_auctioned" and "Bidders_participating" attributes
        df_summary.at[week, "Units_auctioned"] = len(cancelled_week) + len(successful_week)
        df_summary.at[week, "Bidders_participating"] = len(cancelled_week[cancelled_week["Week"] == week]["Customer_id"].unique()) + len(successful_week[successful_week["Week"] == week]["Customer_id"].unique())
        # df_summary.at[week, "conversion_rate"] = len(successful_week) / 
    
        # Calculate the values for the "Amount_auctioned" attribute
        df_summary.at[week, "Amount_auctioned"] = cancelled_week["Price"].sum() + successful_week["Price"].sum()
    
        # Calculate the values for the "Items_auctioned" attribute
        df_summary.at[week, "Items_auctioned"] = -cancelled_week["Quantity"].sum() + successful_week["Quantity"].sum()
    
        # Calculate the values for the "Units_sold", "Amount_sold", "Items_sold", and "Buyers" attributes
        df_summary.at[week, "Units_sold"] = len(successful_week)
        df_summary.at[week, "Amount_sold"] = successful_week["Price"].sum()
        df_summary.at[week, "Items_sold"] = successful_week["Quantity"].sum()
        df_summary.at[week, "Buyers"] = len(successful_week["Customer_id"].unique())
    
    # Reorder the columns
    df_final = df_summary[attribute_cols]
    df_final['feature_sum'] = df_final.sum(axis=1)
    df_final['feature_avg'] = df_final.mean(axis=1)
    
    # print(df_final.head())
    
    merged = pd.merge(df_pivot, df_final, left_index=True, right_index=True)

    # Convert object type columns to numeric types
    df = merged.apply(pd.to_numeric, errors='coerce')
    # print(df.info())
    # print(df.head())
    
    # Create a dataframe with lagged values
    lagged_cols = []
    for i in range(1, 4):
        for col in df.columns:
            lagged_cols.append(df[col].shift(i).rename(f'{col}_lag{i}'))
    df_lagged = pd.concat(lagged_cols, axis=1)
    # print(df_lagged.head())
    # print(df_lagged.info())
    # print(df_lagged.describe())
    # Combine the original and lagged dataframes
    df = pd.concat([df, df_lagged], axis=1)
            
    # Drop the rows with missing values
    df = df.dropna()
    # print(df.head())
    # print(df.describe())
    # print(df.info())
    
    return df
  
df = pd.read_csv('file name')
df = data_preprocessing(df, 0)
