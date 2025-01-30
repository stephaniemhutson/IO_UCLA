import pyblp
import pandas as pd
import numpy as np

data = pd.read_csv('./cleaned_data/data.csv')

demo = pd.read_csv('./cleaned_data/OTCDemographics.csv')

ivs = pd.read_csv('./cleaned_data/OTCDataInstruments.csv')


nodes = {c: 'nodes' + str(int(c[8:]) -1) for c in demo.columns if c[:8] == "hhincome"}
demo = demo.rename(columns={"store": "market_ids", **nodes})
demo['weights'] = 0.05
data = pd.merge(data, ivs, on=['store', 'brand', 'week', 'cost_'])
data = data.rename(columns={"price_per_50": "prices", "brand": "product_ids", "ms_by_store_week": "shares", "store": "market_ids"})
# demo['market_ids'] = demo['store'].astype(str) + '_' + demo['week'].astype(str)
# data['market_ids'] = demo['store'].astype(str) + '_' + demo['week'].astype(str)
# data.set_index('market_ids')
# demo.set_index('market_ids')

# demo_vert = pd.DataFrame({'market_ids': [], 'week': [], 'weights': [], 'income': []})
# for i in range(20):
#     temp_vert = demo[['market_ids', 'week', f'hhincome{i+1}']].rename(columns={f'hhincome{i+1}': 'income'})
#     temp_vert['weights'] = 0.05
#     demo_vert = pd.concat([demo_vert, temp_vert])

# # agent_income_str = " + ".join([c for c in demo.columns if c[:5] == "nodes"])
# price_store_str = " + ".join([c for c in data.columns if c[:10] == 'pricestore'])
# initial_sigma = np.eye(11)


# X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
# X2_formulation = pyblp.Formulation('1 + prices + prom_')

# product_formulations = (X1_formulation, X2_formulation)


# mc_integration = pyblp.Integration('monte_carlo', size=50, specification_options={'seed': 0})
# # pr_integration = pyblp.Integration('product', size=11)
# mc_problem = pyblp.Problem(
#     product_formulations=product_formulations,
#     agent_formulation=agent_formulation,
#     product_data=data,
#     agent_data=demo_vert,
#     integration=mc_integration
# )


# # pr_problem = pyblp.Problem(product_formulations, data, integration=pr_integration)
# bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-4})
# results1 = mc_problem.solve(sigma=np.ones((3, 3)), optimization=bfgs)
# print(results1)

# # results2 = pr_problem.solve(sigma=np.ones((3, 3)), optimization=bfgs)
# # print(results2)


# without Demo data to:

# market_ids, product_ids, firm_ids, shares, prices
X1 = pyblp.Formulation('1 + C(product_ids) ')
X2 = pyblp.Formulation('1 + prices')

problem = pyblp.Problem(
    product_formulations=(X1, X2, ),
    # agent_formulation=agent_formulation,
    product_data=data,
    agent_data=demo,
    )

problem.solve(sigma=np.ones([2,2]))
