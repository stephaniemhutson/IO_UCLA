import pyblp
import pandas as pd
import numpy as np
import pickle

# Data setup
data = pd.read_csv('./PS1/cleaned_data/data.csv')
demo = pd.read_csv('./PS1/cleaned_data/OTCDemographics.csv')
demo = demo.drop(columns=['Unnamed: 0'])
ivs = pd.read_csv('./PS1/cleaned_data/OTCDataInstruments.csv')


# Define B_jt in problem 2
data['branded'] = data['brand'] < 10
data['branded'] = data['branded'].astype(int)

# Markets are store * week
data['market_ids'] = data['store'].astype(str) + '_' + data['week'].astype(str)
demo['market_ids'] = demo['store'].astype(str) + '_' + demo['week'].astype(str)

# Get all variables
data = pd.merge(data, ivs, on=['store', 'brand', 'week', 'cost_'])
data = data.rename(columns={"price_per_50": "prices", "brand": "product_ids", "ms_by_store_week": "shares"})

# Get income
demo = pd.wide_to_long(demo,stubnames='hhincome',i=['market_ids'],j='agent')
demo['weights'] = 0.05
demo = demo.reset_index()
demo = demo.drop(columns=['store','week','agent'])

# Nodes are random utility draws
demo['nodes0'] = np.random.normal(size=len(demo))


rename_dict = {'cost_': 'demand_instruments0', 'avoutprice': 'demand_instruments1'}
for i in range(1, 31):
    rename_dict['pricestore' + str(i)] = 'demand_instruments' + str(i+1)


data = data.rename(columns=rename_dict)
for i in range(32):
    data['demand_instruments' + str(i)] = np.where(((((data['product_ids'] == 1) | (data['product_ids'] == 4)) | (data['product_ids'] == 7))),2*data['demand_instruments' + str(i)],data['demand_instruments' + str(i)])
    data['demand_instruments' + str(i)] = np.where((((data['product_ids'] == 3) | (data['product_ids'] == 6)) | ((data['product_ids'] == 7) | (data['product_ids'] == 11))), data['demand_instruments' + str(i)]/2,data['demand_instruments' + str(i)])

data['price_'] = np.where(((((data['product_ids'] == 1) | (data['product_ids'] == 4)) | (data['product_ids'] == 7))),2*data['price_'],data['price_'])
data['price_'] = np.where((((data['product_ids'] == 3) | (data['product_ids'] == 6)) | ((data['product_ids'] == 7) | (data['product_ids'] == 11))), data['price_']/2,data['price_'])

X1 = pyblp.Formulation('1 + prices + prom_ + C(product_ids) ')
X2 = pyblp.Formulation('0 + prices + branded')
agent_formulation = pyblp.Formulation('0 + hhincome')

problem = pyblp.Problem(
    product_formulations=(X1, X2, ),
    agent_formulation=agent_formulation,
    product_data=data,
    agent_data=demo,
)


initial_sigma = np.diag([0,1])
initial_pi = np.array([1,0])
bfgs = pyblp.Optimization('bfgs',{'gtol':1e-4})
results = problem.solve(
    initial_sigma,
    initial_pi,
    optimization=bfgs,
    method='2s')

elasticities = results.compute_elasticities()
single_market = data['market_ids'] == '9_10'
print(elasticities[single_market])
single_market_data = data[single_market]
