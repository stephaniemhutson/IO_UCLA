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

brands = [
    'Tylenol 25', 'Tylenol 50', 'Tylenol 100',
    'Advil 25', 'Advil 50', 'Advil 100',
    'Bayer 25', 'Bayer 50', 'Bayer 100',
    'Generic 50', 'Generic 100',
]


def get_blp_results(is_logit=False):
    file_name = './PS1/blp_pickle'
    if is_logit:
        file_name = file_name + '_logit'
    try:
        with open(file_name, 'rb') as f:
            results = pickle.load(f)
            f.close()
            return results
    except Exception as e:
        print(str(e))


        # Set up Instruments
        instrument_columns = ['cost_per_50', 'avoutprice'] + [f'pricestore{i}' for i in range(1, 31)]
        IV_formulation = pyblp.Formulation('0 + ' + ' + '.join(instrument_columns))
        local_instruments = pyblp.build_blp_instruments(
            IV_formulation,
            data
        )

        # Define instruments: cost and prices
        rename_dict = {'cost_': 'demand_instruments0', 'avoutprice': 'demand_instruments1'}
        for i in range(1, 31):
            rename_dict['pricestore' + str(i)] = 'demand_instruments' + str(i+1)

        # Normalize instruments
        data = data.rename(columns=rename_dict)
        for i in range(32):
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
        if is_logit:
            initial_sigma = np.diag([0,0,0,0,0,0])
            initial_pi = np.array([0,0,0,0,0,0])
        else:
            initial_sigma = np.diag([0,1, 1, 1, 1, 1])
            initial_pi = np.array([1,0, 0,0 ,0,0])
        bfgs = pyblp.Optimization('bfgs',{'gtol':1e-4})
        results = problem.solve(
            initial_sigma,
            initial_pi,
            optimization=bfgs,
            method='2s')

        file = open(file_name, 'wb')

        # dump information to that file
        pickle.dump(results, file)

        # close the file
        file.close()
        return results

results = get_blp_results()

print("Question 3.1")


print(" & coeficient \\\\")
print("$\\alpha$ & " + f"{round(results.beta.item(0), 4)}" +" \\\\")
for i, brand in enumerate(brands):
    print(f"{brand} & {round(results.beta.item(i + 2), 4)} "+"\\\\")

print(" $\\sigma_{b1}$: Tylenol & " + f"{round(results.sigma.item((1,1)), 4)}"+" \\\\")
print(" $\\sigma_{b2}$: Advil & " +f"{round(results.sigma.item((2,2)), 4)} "+"\\\\")
print(" $\\sigma_{b3}$: Bayer & " +f"{round(results.sigma.item((3,3)), 4)} "+"\\\\")
print(" $\\sigma_{I}$ & "+f"{round(results.pi.item(0), 4)}")

logit_results = get_blp_results(True)


def get_estasticities(results):
    # 2.2: Compute elasticities for store 9 week 10
    elasticities = results.compute_elasticities()
    return elasticities

elasticities = get_estasticities(results)
logit_elasticities = get_estasticities(logit_results)


single_market = data['market_ids'] == '9_10'
print("Market elasticities:")


print(f"& " +" & ".join(brands) + " \\\\")
for i, row in enumerate(elasticities[single_market]):
    row = [str(round(i, 4)) for i in row]
    print(f'{brands[i]} & ' + ' & '.join(row) + ' \\\\')

single_market_data = data[single_market]


# 2.3 Back out marginal costs (INCOMPLETE)

Omega_separate_firms = np.eye(11)

costs_separate_firms = results.compute_costs(market_id='9_10', ownership=Omega_separate_firms)
print("2.3: costs under the assumption that each brand is owned by a single company")
print(costs_separate_firms)


### Question 3 Mergers

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
    Omega[i][i] = 1

for row in Omega:
    print(" & ".join([str(int(i.item())) for i in row]))

Omega_merged = np.eye(11)
for i in range(9):
    for j in range(9):
        Omega_merged[i][j] = 1


# manual
## Note: these result in the same answer
el_data = pd.concat([data,pd.DataFrame(logit_elasticities)])
rename_dict = {}
for i in range(11):
    rename_dict[i] = f'e_{i+1}'
el_data = el_data.rename(columns=rename_dict)



# manual prices (logit)

wholesale_costs = single_market_data['cost_per_50'].to_numpy()


# manual prices (blp, seperate firms)
mc = single_market_data['prices'].to_numpy() * (np.eye(11) + np.linalg.inv(np.multiply(Omega_separate_firms, elasticities[single_market])))
manual_costs = [mc[i][i] for i, _ in enumerate(mc)]

mp = mc * np.linalg.inv(np.eye(11) + np.linalg.inv(np.multiply(Omega_merged, elasticities[single_market])))
print("2.3 Marginal Costs")
print(" & wholesa1le & computed MC \\\\")
for i, brand in enumerate(brands):
    print(f"{brand} & {wholesale_costs[i]} & {round(manual_costs[i], 4)} \\\\")

manual_prices = [[mp[i][i]] for i, _ in enumerate(mp)]



# 3.1 Predict prices from logit

# manual
single_market_data = el_data[el_data['market_ids'] == '9_10']
mc = single_market_data['prices'].to_numpy() * (np.eye(11) + np.linalg.inv(np.multiply(Omega, logit_elasticities[single_market])))
manual_costs = [mc[i][i] for i, _ in enumerate(mc)]
mp_logit = mc * np.linalg.inv(np.eye(11) + np.linalg.inv(np.multiply(Omega_merged, logit_elasticities[single_market])))
manual_prices_logit = [[mp_logit[i][i]] for i, _ in enumerate(mp_logit)]

# pyblp
logit_costs = logit_results.compute_costs(market_id='9_10', ownership=Omega)
changed_prices_logit = logit_results.compute_prices(
    ownership=Omega_merged,
    costs=logit_costs,
    market_id='9_10'
)
original_prices_logit = logit_results.compute_prices(market_id='9_10', costs=logit_costs)

# 3.3 Predict prices from blp
costs = results.compute_costs(market_id='9_10', ownership=Omega)

single_market_data.loc[single_market_data['firm_ids'] < 5, 'merger_ids'] = 1
changed_prices = results.compute_prices(
    # firm_ids=single_market_data['merger_ids'],
    ownership=Omega_merged,
    costs=costs,
    market_id='9_10'
)

original_prices = results.compute_prices(market_id='9_10', costs=costs)
original_prices = single_market_data['prices'].to_numpy()

print("Question 3")
print( "& org. & original predicted &  logit & rnd. coef \\\\")
for i, brand in enumerate(brands):
    print(f"{brand} &   {round(original_prices[i], 4)} & {round(original_prices_logit[i][0], 4)}& {round(changed_prices_logit[i][0], 4)} & {round(changed_prices[i][0], 4)} \\\\")
