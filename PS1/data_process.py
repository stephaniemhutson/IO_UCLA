import pandas as pd
from scipy.special import logit

data = pd.read_csv('./PS1_Data/OTC_Data.csv', sep='\t')
instruments = pd.read_csv('./PS1_Data/OTCDataInstruments.csv', sep='\t')
dems = pd.read_csv('./PS1_Data/OTCDemographics.csv', sep='\t')


data['rev'] = data['sales_'] * data['price_']
data['total_cost'] = data['cost_'] * data['sales_']

sws = data.groupby(['store', 'week'])['sales_'].sum()
# total_sales = data['sales_'].sum()
# total_sales = data['rev'].sum()

# data.loc[:, 'ms_by_store_week'] = data.loc[:, 'sales_'] / sws[(data.loc[:, 'store'][1], data.loc[:, 'week'][1])]
data.loc[:, 'ms_by_store_week'] = data.loc[:, 'sales_'] / data.loc[:, 'count']

# Validate that the data looks similar to the data presented in table i
# grouped = data.groupby('brand')[['sales_', 'rev', 'total_cost']].aggregate('sum')
# grouped.loc[:, 'ms'] = grouped.loc[:, 'sales_'] / (total_sales/ 0.62)

# data.set_index(['store', 'week', 'brand'])

# add brand dummies
brands = data['brand'].unique()
stores = data['store'].unique()
weeks = data['week'].unique()


def store_brand(s, store, brand):
    if (s['brand'] == brand) and (s['store'] == store):
        return 1
    else:
        return 0

data['hausman'] = 0

# data.loc[:, 'hausman'] = data.loc[(data['week'] == data.loc[:, 'week']) & (data['store'] != data.loc[:, 'store']) & (data['brand'] == 1), 'cost_'].mean()
# print(data)


for brand in brands:
    data[f'brand_{brand}'] = (data['brand'] == brand).astype(int)
    for store in stores:
        col_name = f'brand_{brand}_store_{store}'
        data[col_name] = 0
        data.loc[(data['store'] == store) & (data['brand'] == brand), col_name] = 1

        for week in weeks:
            data.loc[(data['store'] == store) & (data['week'] == week) & (data['brand'] == brand), 'hausman'] = data[(data['store'] != store) & (data['week'] == week) & (data['brand'] == brand)]['price_'].mean()

print(data)
data.to_csv('./cleaned_data/data.csv')
