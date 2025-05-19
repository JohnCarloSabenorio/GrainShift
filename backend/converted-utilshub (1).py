import pandas as pd
pd.set_option('display.max_columns', 500)
food_prices_data = pd.read_csv("wfp_food_prices_phl.csv", skiprows=[1])

# Number of data and features 
print(food_prices_data.shape)
print(len(str(food_prices_data.shape))*'-')
# Check how many types of data are in the dataset
print(food_prices_data.dtypes.value_counts())
print(len(str(food_prices_data.shape))*'-')
# Check the first 16 columns
food_prices_data.head(16)


# Check The amount of regions recorded
for region in food_prices_data['admin1'].unique():
    print(region)
print(f"TOTAL REGION: {food_prices_data['admin1'].nunique()}")

food_prices_data.isnull().sum()

commodity_counts = food_prices_data["commodity"].value_counts().reset_index()
commodity_counts.columns = ["commodity", "count"]
print(commodity_counts.to_string(index=False))

food_prices_data["currency"].value_counts()

food_prices_data = food_prices_data.drop(columns=["category", "currency", "latitude", "longitude"])
food_prices_data.head()

food_prices_data["price"] = pd.to_numeric(food_prices_data["price"], errors="coerce")
# food_prices_data["latitude"] = pd.to_numeric(food_prices_data["latitude"], errors="coerce")
# food_prices_data["longitude"] = pd.to_numeric(food_prices_data["longitude"], errors="coerce")
food_prices_data["date"] = pd.to_datetime(food_prices_data["date"], errors="coerce")
print(food_prices_data.dtypes.value_counts())

all_rice_df = food_prices_data[food_prices_data["commodity"].str.startswith("Rice", na=False)]
all_rice_df["commodity"].value_counts()


all_rice_df = all_rice_df[(all_rice_df['date'] >= '2015-01-01') & (all_rice_df['date'] <= '2025-12-31')]
all_rice_df.shape

all_rice_df["admin1"].value_counts()


all_rice_df["admin2"].value_counts()


all_rice_df["market"].value_counts()

all_rice_df["pricetype"].value_counts()

all_rice_df["unit"].value_counts()

all_rice_df["priceflag"].value_counts()

all_rice_df.head(30)

df_train = all_rice_df.copy()
df_train['date'] = pd.to_datetime(df_train['date'])
df_train = df_train.sort_values(by=['date'])
# Aggregate to a single time series
monthly_price = df_train.set_index('date')['price'].resample('M').mean()
price_ts = df_train.groupby('date')['price'].mean()
print(monthly_price.head(10))
# Compute the returns
returns = 100 * monthly_price.pct_change().dropna()
print(returns.head(10))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import matplotlib.cm as cm
from arch import arch_model
import seaborn as sns

avg_price_by_region = all_rice_df.groupby(['date', 'admin1'])['price'].mean().reset_index()

pivot_df = avg_price_by_region.pivot(index='date', columns='admin1', values='price')

regions = pivot_df.columns
n = len(regions)

colors = cm.get_cmap('tab20', n)(np.arange(n))

luzon = ['National Capital region', 'Region I', 'Region II', 'Region III', 'Region IV-A', 'Region IV-B', 'Region V', 'Cordillera Administrative region']
visayas = ['Region VI', 'Region VII', 'Region VIII']
mindanao = ['Region IX', 'Region X', 'Region XI', 'Region XII', 'Region XIII', 'Autonomous region in Muslim Mindanao']

clusters = {'Luzon': luzon, 'Visayas': visayas, 'Mindanao': mindanao}