import pandas as pd
import statsmodels.api as sm
import statsmodels.sandbox.regression.gmm as gmm
import numpy as np
import sys

sys.setrecursionlimit(2000)
data = pd.read_csv('./cleaned_data/data.csv')

# 1. Using OLS with price and promotion as product characteristics.
Y = np.log(data['ms_by_store_week'] / data['ms_naught'])
X1 = data[['price_', 'prom_']]
X1 = sm.add_constant(X1)
model = sm.OLS(Y,X1)
results1 = model.fit()



print("***** Q 1 *****")
print(results1.params)

# 2. Using OLS with price and promotion as product characteristics and
# brand dummies.

brand_dummies = [col for col in data.columns if col[:6] == 'brand_' and len(col) < 8 and col != 'brand_1']

X2 = data[['price_', 'prom_', *brand_dummies]]
X2 = sm.add_constant(X2)
model = sm.OLS(Y,X2)
results2 = model.fit()
print("***** Q 2 *****")
print(results2.params)

# 3. Using OLS with price and promotion as product characteristics and
# store-brand (the interaction of brand and store) dummies.
store_brand = [col for col in data.columns if col[:6] == 'brand_' and len(col) > 8 and col != 'brand_5_store_2']


X3 = data[['price_', 'prom_', *store_brand]]
X3 = sm.add_constant(X3)
model = sm.OLS(Y,X3)
results3 = model.fit()
print("***** Q 3 *****")
print(results3.params)


# 4. Estimate the models of 1, 2 and 3 using wholesale cost as an instrument.
print("***** Q 4 *****")
IV = data[['cost_', 'prom_']]


stage1 = sm.OLS(X1, IV)
res1 = stage1.fit()
# print(res1.fittedvalues)
# data['fitted_iv4a'] = res1.fittedvalues

model = sm.OLS(Y, res1.fittedvalues)
results4a = model.fit()
print("***** part 1 *****")
print(results4a.params)
IV = data[['cost_', 'prom_', *brand_dummies]]

model = gmm.IV2SLS(Y, X2, instrument=IV)
results4b = model.fit()
print("***** part 2 *****")
print(results4b.params)

IV = data[['cost_', 'prom_', *store_brand]]

model = gmm.IV2SLS(Y, X3, instrument=IV)
results4c = model.fit()
print("***** part 3 *****")
# print("Skip because of singular matrix?")
print(results4c.params)

# 5. Estimate the models of 1, 2 and 3 using the Hausman instrument (average price in other markets).

print("***** Q 5 *****")
model = gmm.IV2SLS(Y, X1, instrument=IV)
model = gmm.IV2SLS(Y, X1, instrument=IV)
results5a = model.fit()
print("***** part 1 *****")
print(results5a.params)

IV = data[['hausman', 'prom_', *brand_dummies]]
model = gmm.IV2SLS(Y, X2, instrument=IV)
results5b = model.fit()
print("***** part 2 *****")
print(results5b.params)

IV = data[['hausman', 'prom_', *store_brand]]

model = gmm.IV2SLS(Y, X3, instrument=IV)
results5c = model.fit()
print("***** part 3 *****")
print(results5c.params)
# print("Skip because of singular matrix?")

# # 6. Using the analytic formula for elasticity of the logit model, compute
# the mean own-price elasticities for all brand in the market using the
# estimates in 1, 2 and 3. Do these results make sense? (Discuss)


print("***** Q 6 *****")

data['ratio']  = (1-data['ms_by_store_week'])/data['ms_by_store_week']

data['elasticity_est_1'] = results1.params['price_']*data['price_']*(1-data['ms_by_store_week'])
data['elasticity_est_2'] = results2.params['price_']*data['price_']*(1-data['ms_by_store_week'])
data['elasticity_est_3'] = results3.params['price_']*data['price_']*(1-data['ms_by_store_week'])

elasticity = data.groupby(['brand'])[['elasticity_est_1', 'elasticity_est_2', 'elasticity_est_3']].aggregate('mean')
print(elasticity)
