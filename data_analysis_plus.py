import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import calendar

# Read and Check Data
df = pd.read_csv('file name')
def data_report(dataframe):
    """ Function to get a dataframe review """
    print(dataframe.info())
    print("\n")
    print(dataframe.describe())
    print("\n")
    print(df.isnull().sum())

#data_report(df)

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# data_report(df)

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
#df["Week"] = df["Date"].dt.week    
df["Week"] = df["Date"].dt.isocalendar().week

#data_report(df)
# print("df", df.head())

# Group the data by product ID and week and aggregate the data using sum()
df_product_weekly = df.groupby(["Product_id", "Week"]).agg({"Quantity": "sum"}).reset_index()

# Sort the data in descending order of quantity and select the top 50 products
top_products = df_product_weekly.groupby("Product_id").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False).head(2).index

# Filter the data to only include the top 50 products
df_product_weekly = df_product_weekly[df_product_weekly["Product_id"].isin(top_products)].copy()

# Pivot the data to have product IDs as columns and weeks as rows
df_pivot = df_product_weekly.pivot(index="Week", columns="Product_id", values="Quantity")

# Replace NaN values in the product columns with 0
product_cols = df_pivot.columns[1:]  # Assumes the product columns start at index 4
df_pivot[product_cols] = df_pivot[product_cols].fillna(0)

# # Compare the total number of products sold with the quantity
# df_final["Products_sold_equal_quantity"] = df_final["Products_sold"] == df_final["Quantity"]
# print("df_pivot\n", df_pivot.head())

# Get the top 50 products sold

top_products = df.groupby("Product_id").agg({"Quantity": "sum"}).sort_values(by="Quantity", ascending=False).head(2).index.tolist()

# Filter the DataFrame to include only the top 50 products
df = df[df["Product_id"].isin(top_products)].copy()

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
# print(df_final.head())

merged = pd.merge(df_pivot, df_final, left_index=True, right_index=True)
print(merged.head())
print(merged.describe())
print(merged.info())

#print("number of unique products: ",df["Product_id"].nunique()) 3688

price_ranges = df.groupby(by=["Product_id"]).agg({"Price":lambda x: x.max()-x.min()}).rename(columns={"Price":"Price_range"})

def show_profit_every_month():
  # Variable to Store
  listMonth = []
  listTotalMoney = []
  for i in df['Month'].unique():
      monthName = dt.datetime.strptime(str(i), "%m")
      monthName = monthName.strftime("%B")
      listMonth.append(monthName)
  for i in df['Month'].unique():
      totalMoney = round(df['Transaction_profit'].loc[(df['Month']==i)&(df['Year']==2022)].sum(),2)
      listTotalMoney.append(totalMoney)
      
  # Dictionary for DataFrame
  dictMonth = {
      'MonthName' : listMonth,
      'TotalMoney' : listTotalMoney
  }
  
  # Adjust Data Frame
  dfMonth = pd.DataFrame(dictMonth)
  dfMonth = dfMonth.iloc[::-1]
  
  
  # Create Figure
  plt.figure(figsize = (12,8))
  plt.plot(dfMonth['MonthName'], dfMonth['TotalMoney'], color = 'Red', marker = 's',alpha = 0.8)
  plt.title('Total Revenue Every Month in 2022')
  plt.yticks(rotation = 45)
  plt.xticks(rotation = 45)
  plt.xlabel('Month')
  plt.ylabel('Total Revenue')
  for i in dfMonth['MonthName']: 
      text = str(dfMonth['TotalMoney'].loc[dfMonth['MonthName'] == i].values[0])
      y = dfMonth['TotalMoney'].loc[dfMonth['MonthName'] == i]+(dfMonth['TotalMoney'].min()*0.1)
      plt.text(i,y,text, ha = 'center', rotation = 45) 
  plt.ylim(0,dfMonth['TotalMoney'].max()*1.3)
  plt.grid(axis = 'y')
  plt.show()
  
def show_profit_every_week():
  # Variable to Store
    listWeek = []
    listTotalMoney = []
    print(df['Week'].unique())
    for i in df['Week'].unique():
        weekName = "{:02d}".format(i)
        listWeek.append(weekName)
        
    print(listWeek)
    for i in df['Week'].unique():
        totalMoney = round(df['Transaction_profit'].loc[(df['Week']==i)&(df['Year']==2022)].sum(),2)
        listTotalMoney.append(totalMoney)

    print(listTotalMoney)
    # Dictionary for DataFrame
    dictWeek = {
        'WeekNumber' : listWeek,
        'TotalMoney' : listTotalMoney
    }

    # Adjust Data Frame
    dfWeek = pd.DataFrame(dictWeek)
    dfWeek = dfWeek.iloc[::-1]
    
    dfWeek.to_excel('total_revenue_every_week.xlsx', index=False)
    
    # Create Figure
    plt.figure(figsize = (12,8))
    plt.plot(dfWeek['WeekNumber'], dfWeek['TotalMoney'], color = 'Red', marker = 's',alpha = 0.8)
    plt.title('Total Revenue Every Week in 2022')
    plt.yticks(rotation = 45)
    plt.xticks(rotation = 45)
    plt.xlabel('Week Number')
    plt.ylabel('Total Revenue')
    # for i in dfWeek['WeekNumber']: 
    #     # text = str(dfWeek['TotalMoney'].loc[dfWeek['WeekNumber'] == i].values[0])
    #     y = dfWeek['TotalMoney'].loc[dfWeek['WeekNumber'] == i]+(dfWeek['TotalMoney'].min()*0.1)
    #     plt.text(i,y, ha = 'center', rotation = 45) 
    plt.ylim(0,dfWeek['TotalMoney'].max()*1.3)
    plt.grid(axis = 'y')
    plt.show()
  
def show_transactions_cancellation():
    # Iterate over unique months in df
    for month in df["Month"].unique():
        # Filter transactions for current month
        month_transactions = df[df["Month"] == month]

        # Calculate the number of unique transactions per week
        weekly_transactions = month_transactions.groupby(by=["Week"]).agg({"Transaction_id":"nunique"})
        weekly_transactions.rename(columns={"Transaction_id":"Total_unique_transactions"}, inplace=True)

        # Filter cancellations for current month
        month_cancellations = cancelled[cancelled["Month"] == month]

        # Calculate the number of unique cancellations per week
        weekly_cancellations = month_cancellations.groupby(by=["Week"]).agg({"Transaction_id":"nunique"}).rename(columns={"Transaction_id":"Total_unique_cancellations"})

        # Merge transactions and cancellations dataframes for the current month
        merged = pd.merge(weekly_transactions, weekly_cancellations, how="outer", on="Week").fillna(0)

        # Plot the data
        merged.plot(kind="bar", figsize=[16,5], rot=0, title="Transactions and Cancellations for {} in 2022".format(calendar.month_name[month]), color=["steelblue","violet"])
        plt.ylabel("Number of Transactions")
        plt.grid(alpha=0.35)
        plt.show()

def show_conversion_rate():

    product_sucessful = sucessful.groupby(["Product_name"]).agg({"Quantity":"sum"})
    product_cancelled = cancelled.groupby(["Product_name"]).agg({"Quantity":"sum"})

    merged = pd.merge(product_sucessful, product_cancelled, how="outer", on="Product_name").fillna(0)
    
    merged["Total_transactions"] = merged["Quantity_x"] - merged["Quantity_y"]
    merged["Successful_transactions"] = merged["Quantity_x"]
    merged["Cancelled_transactions"] = merged["Quantity_y"]
    merged["Conversion_rate"] = merged["Successful_transactions"] / merged["Total_transactions"]
    merged.sort_values(by="Conversion_rate", ascending=True, inplace=True)

    # merged.to_csv('conversion_rate.csv', index=True)

    # plot
    plt.figure(figsize=(16,5))
    weights = np.ones_like(merged[merged["Conversion_rate"] < 1.0]["Conversion_rate"]) / np.sum(merged[merged["Conversion_rate"] < 1.0]["Conversion_rate"])
    plt.hist(merged[merged["Conversion_rate"] < 1.0]["Conversion_rate"], bins=100, color="steelblue", weights=weights)
    # sns.kdeplot(merged[merged["Conversion_rate"] < 1.0]["Conversion_rate"], color="red", label="Best fit curve (kde)", common_norm=True)
    # sns.histplot(, color="steelblue", stat="density", kde=True)
    # sns.kdeplot(merged["Conversion_rate"], color="red", label="Best fit curve (kde)")
    plt.title("Distribution of Conversion Rate(less than 1)")
    plt.xlabel("Conversion Rate")
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.grid(alpha=0.35)
    plt.legend()
    plt.show()
    return 
    
     # Group transactions by product and count the number of total transactions
    product_transactions = df.groupby(["Product_id"]).agg({"Transaction_id":"count"})

    # Group cancelled transactions by product and count the number of cancelled transactions
    product_cancellations = cancelled.groupby(["Product_id"]).agg({"Transaction_id":"count"})

    # Merge the total transactions and cancelled transactions by product
    merged = pd.merge(product_transactions, product_cancellations, how="outer", on="Product_id").fillna(0)

    # Calculate the number of successful transactions for each product
    merged["Successful_transactions"] = merged["Total_transactions"] - merged["Cancelled_transactions"]

    # Calculate the conversion rate for each product
    merged["Conversion_rate"] = merged["Successful_transactions"] / merged["Total_transactions"]

    # Rename the columns
    merged.rename(columns={"Total_transactions":"Total_transactions", "Cancelled_transactions":"Cancelled_transactions"}, inplace=True)

    # Sort the DataFrame by conversion rate in descending order
    merged.sort_values(by="Conversion_rate", ascending=False, inplace=True)

    # Display the resulting DataFrame
    print(merged)
    
def show_Distribution_values_products():

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16,5])

    ax1 = df["Price"].plot.density(ax=ax1)
    ax1.set_xlim(0,df["Price"].max()+1)
    ax1.set_title("Product Prices Density Plot")
    ax1.set_xlabel("Product Price")
    ax1.grid(alpha=0.35)

    ax2 = sns.histplot(df["Price"], bins=50, color="steelblue", stat="density", ax=ax2)
    ax2.set_title("Product Prices Distribution")
    ax2.set_xlabel("Product Price")
    ax2.set_yscale("log")
    ax2.set_ylabel("Density (log scale)")
    ax2.grid(alpha=0.35)

    plt.show()

def show_price_difference():
    plt.figure(figsize=(16,5))
    sns.histplot(price_ranges["Price_range"], color="steelblue", stat="density")
    sns.kdeplot(price_ranges["Price_range"], color="red", label="Best fit curve (kde)")
    plt.title("Distribution of Price Differences")
    plt.xlabel("Price Differences")
    plt.grid(alpha=0.35)
    plt.legend()
    plt.show()
  
def show_customer_country():
    demographics =  df.groupby(by=["Country"])["Customer_id"].nunique()
    demographics.sort_values(inplace=True)
    
    # Calculating region specific demographics
    #total_non_my = demographics[demographics.index != "Malaysia"].values.sum()
    total_non_my = np.array(demographics[demographics.index != "Malaysia"].values).flatten()
    # total_my = np.array(demographics[demographics.index == "Malaysia"].values).flatten()
    total_my = np.array(demographics[demographics.index == ["Malaysia"]].values).flatten()
    print(total_my.shape)
    print(total_non_my.shape)
    
    total_global = total_non_my + total_my
    # Plotting the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16,10])
    
    # Horizontal bar plot
    ax1 = demographics.plot.barh(color="steelblue", ax=ax1)
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Unique Customers")
    ax1.set_title("Unique Customers from each Country")
    
    # Draw the pie chart
    data = np.ravel([total_my,total_non_my])
    ax2 = plt.pie(data,labels=["Malaysia Customers", "International Customers"], startangle=90, counterclock=False, wedgeprops=dict(width=0.3, edgecolor="w"))
    plt.title("Number of Malaysia vs International Customers")
    
    # Add a legend with percentages on the pie chart
    plt.legend([f"{np.round(total_my/total_global *100,2)}%", f"{np.round(total_non_my/total_global * 100,2)}%"],
    bbox_to_anchor=(0.2, 0.1))
    
    plt.show() 

def show_domestic_and_international_monthly():
    
    month2revenue_domestic = {}
    month2revenue_international = {}

    transactions_domestic = df[df["Country"] == "Malaysia"]
    transactions_international = df[df["Country"] != "Malaysia"]
    for month in df["Month"].unique():

        # domestic
        month_transactions_domestic = transactions_domestic[transactions_domestic["Month"] == month].copy()

        # intertional
        month_transactions_international = transactions_international[transactions_international["Month"] == month].copy()

        # Revenue of domestic
        reveune = month_transactions_domestic["Price"] * month_transactions_domestic["Quantity"]
        month_transactions_domestic["Revenue"] = reveune
        month2revenue_domestic[month] =  month_transactions_domestic["Revenue"].sum()
        
        # Revenue of interional
        reveune = month_transactions_international["Price"] * month_transactions_international["Quantity"]
        month_transactions_international["Revenue"] = reveune
        month2revenue_international[month] =  month_transactions_international["Revenue"].sum()

    plt.figure(figsize=(10,6))
    plt.plot(list(month2revenue_domestic.keys()), list(month2revenue_domestic.values()), color="blue", label="Domestic")

    plt.plot(list(month2revenue_international.keys()), list(month2revenue_international.values()), color="red", label="International")
    for k, v in month2revenue_domestic.items():
        if k == 11:
            plt.text(k - 0.5, v + 1e5, '{:.2e}'.format(v), color="blue")
            continue
        if k % 2 == 0:
            plt.text(k  - 0.5, v - 2e5, '{:.2e}'.format(v), color="blue")
        else:
            plt.text(k  - 0.5, v + 2e5, '{:.2e}'.format(v), color="blue")
    for k, v in month2revenue_international.items():
        if k % 2 == 0:
            plt.text(k  - 0.5, v - 3e5, '{:.2e}'.format(v), color="red")
        else:
            plt.text(k  - 0.5, v + 3e5, '{:.2e}'.format(v), color="red")
    plt.xlabel("Month")
    plt.ylabel("Reveune")
    plt.title("Monthly Revenue from Domestic and International Transactions for 2022")
    plt.xticks(np.arange(1,13))
    plt.grid(alpha=0.35)
    plt.legend()
    plt.show()



#show_profit_every_month()
#show_profit_every_week()
#show_transactions_cancellation()
#show_conversion_rate()
#show_Distribution_values_products()
#show_price_difference()
#show_customer_country()
#show_domestic_and_international_monthly()
    
