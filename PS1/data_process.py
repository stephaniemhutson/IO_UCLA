import pandas as pd
from scipy.special import logit

data = pd.read_csv('./PS1_Data/OTC_Data.csv', sep='\t')
instruments = pd.read_csv('./PS1_Data/OTCDataInstruments.csv', sep='\t')
dems = pd.read_csv('./PS1_Data/OTCDemographics.csv', sep='\t')


data['rev'] = data['sales_'] * data['price_']
data['total_cost'] = data['cost_'] * data['sales_']

sws = data.groupby(['store', 'week'])['sales_'].sum()


# data.loc[:, 'ms_by_store_week'] = data.loc[:, 'sales_'] / sws[(data.loc[:, 'store'][1], data.loc[:, 'week'][1])]
data.loc[:, 'ms_by_store_week'] = data.loc[:, 'sales_'] / data.loc[:, 'count']

# Validate that the data looks similar to the data presented in table i

# add brand dummies
brands = data['brand'].unique()
stores = data['store'].unique()
weeks = data['week'].unique()

ms_naught =  1 - data.groupby(['week','store'])['ms_by_store_week'].aggregate('sum')

for brand in brands:
    data[f'brand_{brand}'] = (data['brand'] == brand).astype(int)
    for store in stores:
        col_name = f'brand_{brand}_store_{store}'
        data[col_name] = 0
        data.loc[(data['store'] == store) & (data['brand'] == brand), col_name] = 1

        for week in weeks:
            data.loc[(data['store'] == store) & (data['week'] == week) & (data['brand'] == brand), 'hausman'] = data[(data['store'] != store) & (data['week'] == week) & (data['brand'] == brand)]['price_'].mean()
            data.loc[(data['store'] == store) & (data['week'] == week) & (data['brand'] == brand), 'ms_naught'] = ms_naught[(week, store)]


print(data)
data.to_csv('./cleaned_data/data.csv')
