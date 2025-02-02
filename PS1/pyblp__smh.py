import pyblp
import pandas as pd
import numpy as np

data = pd.read_csv('./cleaned_data/data.csv')

# demo = pd.read_csv('./cleaned_data/OTCDemographics.csv')

ivs = pd.read_csv('./cleaned_data/OTCDataInstruments.csv')


# # nodes = {c: 'nodes' + str(int(c[8:]) -1) for c in demo.columns if c[:8] == "hhincome"}
# # demo = demo.rename(columns={"store": "market_ids", **nodes})
# # demo['weights'] = 0.05
# # data = pd.merge(data, ivs, on=['store', 'brand', 'week', 'cost_'])
# # data = data.rename(columns={"price_per_50": "prices", "brand": "product_ids", "ms_by_store_week": "shares", "store": "market_ids"})



# # Reshape from wide to long format
# df_long = demo.melt(id_vars=["store", "week"], var_name="draw", value_name="income")

# # Extract draw number from "hhincome#" column
# df_long["draw"] = df_long["draw"].str.extract("(\d+)").astype(int)

# # Create a market identifier (store-week)
# df_long["market_ids"] = df_long["store"].astype(str) + "_" + df_long["week"].astype(str)

# # Rename columns to match pyblp expectations
# df_long = df_long.rename(columns={"draw": "draw_ids"})

# # Select necessary columns
# df_long = df_long[["market_ids", "draw_ids", "income"]]

# # Save for pyblp
# df_long.to_csv("./cleaned_data/formatted_demographics.csv", index=False)





# # Create a unique market-product identifier
# data["join"] = data["store"].astype(str) + "_" + data["week"].astype(str) + "_" + data["brand"].astype(str)
# ivs["join"] = ivs["store"].astype(str) + "_" + ivs["week"].astype(str) + "_" + ivs["brand"].astype(str)

# # # Merge on market-product identifier
# data = data.merge(ivs.drop(columns=["store", "week", "brand", "cost_"]), on="join", how="left")
# data['market_ids'] = data["store"].astype(str) + "_" + data["week"].astype(str)

# # # Save merged dataset
# data.to_csv("formatted_data.csv", index=False)



demographics = pd.read_csv("./cleaned_data/formatted_demographics.csv")
data = pd.read_csv("./formatted_data.csv")
instrument_columns = ['cost_', 'avoutprice'] + [f'pricestore{i}' for i in range(1, 31)]

data['demand_instruments33'] = data[instrument_columns].transform(np.size)

data = data.rename(columns={'price_': 'prices', 'ms_by_store_week': 'shares'})

# Linear product characteristics (including price)
X1_formulation = pyblp.Formulation('1 + prices + ' + ' + '.join([f'brand_{i}' for i in range(2,11)]))  # Includes a constant, price, and brand dummies

# Nonlinear product characteristics (random coefficients)
X2_formulation = pyblp.Formulation('1 + prices')

# Instrument formulation (exclude price to avoid redundancy)
Z_formulation = pyblp.Formulation('0 + cost_ + avoutprice + ' + ' + '.join([f'pricestore{i}' for i in range(1, 31)]))

integration = pyblp.Integration('monte_carlo', size=20)

pyblp.build_blp_instruments(Z_formulation, data)

problem = pyblp.Problem(
    product_formulations=(X1_formulation, X2_formulation),
    product_data=data,
    agent_data=demographics,
    integration=integration,
)



instruments = data[instrument_columns].to_numpy()
W = np.linalg.inv(np.matmul(np.transpose(instruments), instruments))
print(W.shape)

results = problem.solve(
    sigma=np.ones([2,2]) + np.eye(2),
    method='2s',  # Choose an optimization method
    # W=W
    # instruments=Z_formulation
)
print(results)
