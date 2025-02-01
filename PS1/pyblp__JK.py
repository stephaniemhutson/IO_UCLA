import pyblp
import pandas as pd
import numpy as np

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


# Linear (X1) and nonlinear (X2) variables
X1 = pyblp.Formulation('1 + prices + prom_ + C(product_ids) ')
X2 = pyblp.Formulation('0 + prices + branded')
# Demographic that interacts: income
agent_formulation = pyblp.Formulation('0 + hhincome')

problem = pyblp.Problem(
    product_formulations=(X1, X2, ),
    agent_formulation=agent_formulation,
    product_data=data,
    agent_data=demo,
    )

# Restrict parameters: no random interaction with price, no inc interaction with brand
initial_sigma = np.diag([0,1])
initial_pi = np.array([1,0])
bfgs = pyblp.Optimization('bfgs',{'gtol':1e-4})
results = problem.solve(initial_sigma,initial_pi,optimization=bfgs,method='1s')


# 2.2: Compute elasticities for store 9 week 10
elasticities = results.compute_elasticities()
single_market = data['market_ids'] == '9_10'
elasticities[single_market]


# 2.3 Back out marginal costs (INCOMPLETE)

# Define Same-brand matrix
Omega = np.zeros((11,11))
for i in range(0,3):
    for j in range(0,3):
        Omega[i][j] = 1
for i in range(3,6):
    for j in range(3,6):
        Omega[i][j] = 1
for i in range(6,9):
    for j in range(6,9):
        Omega[i][j] = 1
for i in range(9,11):
    for j in range(9,11):
        Omega[i][j] = 1

data = pd.concat([data,pd.DataFrame(elasticities)])
rename_dict = {}
for i in range(0,11):
    rename_dict[i] = 'e_'+str(i+1)
data = data.rename(columns=rename_dict)
