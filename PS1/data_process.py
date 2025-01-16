import pandas as pd
from scipy.special import logit

data = pd.read_csv('./PS1_Data/OTC_Data.csv', sep='\t')
instruments = pd.read_csv('./PS1_Data/OTCDataInstruments.csv', sep='\t')
dems = pd.read_csv('./PS1_Data/OTCDemographics.csv', sep='\t')


# 1. Using OLS with price and promotion as product characteristics.
data['rev'] = data['sales_'] * data['price_']
data['total_cost'] = data['cost_'] * data['sales_']

sws = data.groupby(['store', 'week'])['sales_'].sum()
total_sales = data['sales_'].sum()
total_sales = data['rev'].sum()

data.loc[:, 'ms_by_store_week'] = data.loc[:, 'sales_'] / sws[(data.loc[:, 'store'][1], data.loc[:, 'week'][1])]

# Validate that the data looks similar to the data presented in table i
# grouped = data.groupby('brand')[['sales_', 'rev', 'total_cost']].aggregate('sum')
# grouped.loc[:, 'ms'] = grouped.loc[:, 'sales_'] / (total_sales/ 0.62)
# print(grouped)

data.to_csv('./cleaned_data/data.csv')


# 2. Using OLS with price and promotion as product characteristics and
# brand dummies.
# 3. Using OLS with price and promotion as product characteristics and
# store-brand (the interaction of brand and store) dummies.
# 4. Estimate the models of 1, 2 and 3 using wholesale cost as an instrument.
# 5. Estimate the models of 1, 2 and 3 using the Hausman instrument (average price in other markets).
# 6. Using the analytic formula for elasticity of the logit model, compute
# the mean own-price elasticities for all brand in the market using the
# estimates in 1, 2 and 3. Do these results make sense? (Discuss)
