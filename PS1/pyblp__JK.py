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
data.loc[:, 'firm_ids'] = ((data.loc[:, 'product_ids']) -1) // 3 +1
data.loc[data['firm_ids'] == 4, 'firm_ids'] = data.loc[data['firm_ids'] == 4]['product_ids']

# Get income
demo = pd.wide_to_long(demo,stubnames='hhincome',i=['market_ids'],j='agent')
demo['weights'] = 0.05
demo = demo.reset_index()
demo = demo.drop(columns=['store','week','agent'])


# Nodes are random utility draws
demo['nodes0'] = np.random.normal(size=len(demo))

try:
    with open('./PS1/blp_pickle', 'rb') as f:

        results = pickle.load(f)
        f.close()
except Exception as e:
    print(str(e))

    # Set up Instruments
    instrument_columns = ['cost_', 'avoutprice'] + [f'pricestore{i}' for i in range(1, 31)]
    IV_formulation = pyblp.Formulation('0 + ' + ' + '.join(instrument_columns))
    local_instruments = pyblp.build_blp_instruments(
        IV_formulation,
        data
    )

    for i, column in enumerate(local_instruments.T):
        data[f'demand_instruments{i}'] = column

    # Normalize instruments
    for i in range(64):
        data['demand_instruments' + str(i)] = np.where(((((data['product_ids'] == 1) | (data['product_ids'] == 4)) | (data['product_ids'] == 7))),2*data['demand_instruments' + str(i)],data['demand_instruments' + str(i)])
        data['demand_instruments' + str(i)] = np.where((((data['product_ids'] == 3) | (data['product_ids'] == 6)) | ((data['product_ids'] == 7) | (data['product_ids'] == 11))), data['demand_instruments' + str(i)]/2,data['demand_instruments' + str(i)])

    # Linear (X1) and nonlinear (X2) variables
    X1 = pyblp.Formulation('1 + prices + prom_ + C(product_ids) ')
    X2 = pyblp.Formulation('0 + prices + C(firm_ids)')
    # Demographic that interacts: income
    agent_formulation = pyblp.Formulation('0 + hhincome')

    problem = pyblp.Problem(
        product_formulations=(X1, X2, ),
        agent_formulation=agent_formulation,
        product_data=data,
        agent_data=demo,
        )

    # Restrict parameters: no random interaction with price, no inc interaction with brand
    initial_sigma = np.diag([0,1,0,0,0,0])
    initial_pi = np.array([1,0,0,0,0,0])
    bfgs = pyblp.Optimization('bfgs',{'gtol':1e-4})
    results = problem.solve(
        initial_sigma,
        initial_pi,
        optimization=bfgs,
        method='2s')

    file = open('./PS1/blp_pickle', 'wb')

    # dump information to that file
    pickle.dump(results, file)

    # close the file
    file.close()

print(results)



# 2.2: Compute elasticities for store 9 week 10
elasticities = results.compute_elasticities()
single_market = data['market_ids'] == '9_10'
print("Market elasticities:")
print(elasticities[single_market])
single_market_data = data[single_market]


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



# pyblp
## Note: these result in the same answer
costs = results.compute_costs(market_id='9_10')
print("Computed Costs:")
print(costs)

# manual
## Note: these result in the same answer
el_data = pd.concat([data,pd.DataFrame(elasticities)])
rename_dict = {}
for i in range(11):
    rename_dict[i] = f'e_{i+1}'
el_data = el_data.rename(columns=rename_dict)

single_market_data = el_data[(el_data['store'] == 9) & (el_data['week'] == 10)]
mc = single_market_data['prices'].to_numpy() * (np.eye(11) + np.linalg.inv(np.multiply(Omega, elasticities[single_market])))
manual_costs = np.matrix([[mc[i][i]] for i, _ in enumerate(mc)])



### Question 3 Mergers



single_market_data['merger_ids'] = single_market_data['firm_ids'].replace(3, 1)
single_market_data['merger_ids'] = single_market_data['firm_ids'].replace(2, 1)
single_market_data['merger_ids'] = single_market_data['firm_ids'].replace(1, 1)



all_costs = results.compute_costs()

changed_prices = results.compute_prices(
    firm_ids=single_market_data['merger_ids'],
    costs=costs,
    market_id='9_10'
)

original_prices = results.compute_prices(market_id='9_10', costs=costs)
print("New prices after merger:")
print(changed_prices)
print("Verify the originial prices:")
print(" - Predicted:")
print(original_prices)
print(" - Raw:")
print(single_market_data['prices'])
