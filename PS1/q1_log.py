import pandas as pd
import statsmodels.api as sm
# import statsmodels.sandbox.regression.gmm as gmm
import numpy as np
import statsmodels.formula.api as smf
import sys
import re
sys.setrecursionlimit(2000)



data = pd.read_csv('./cleaned_data/data.csv')
coefs = []
std_devs = []
pvals = []
# 1. Using OLS with price and promotion as product characteristics.

data['Y'] = data['ms_by_store_week'] / data['ms_naught']
X1 = data[['price_', 'prom_']]

brand_dummies = [col for col in data.columns if col[:6] == 'brand_' and len(col) < 8 and col != 'brand_1']
X2 = data[['price_', 'prom_', *brand_dummies]]

store_brand = [col for col in data.columns if col[:6] == 'brand_' and len(col) > 8 and col != 'brand_1_store_2']
X3 = data[['price_', 'prom_', *store_brand]]


print("***** Q 1 *****")
col1 = " + ".join(X1.columns)
formula1 = "Y ~ " + col1
log_reg_1 = smf.logit(formula1, data=data).fit()
coefs.append([log_reg_1.params['Intercept'], log_reg_1.params['price_'], log_reg_1.params['prom_']])
pvals.append([log_reg_1.pvalues['Intercept'], log_reg_1.pvalues['price_'], log_reg_1.pvalues['prom_']])
std_devs.append([
    log_reg_1.cov_params()['Intercept']['Intercept'],
    log_reg_1.cov_params()['price_']['price_'],
    log_reg_1.cov_params()['prom_']['prom_']
])



print("***** Q 2 *****")
col2 = " + ".join(X2.columns)
formula2 = "Y ~ " + col2
log_reg_2 = smf.logit(formula2, data=data).fit()
coefs.append([log_reg_2.params['Intercept'], log_reg_2.params['price_'], log_reg_2.params['prom_']])
pvals.append([log_reg_2.pvalues['Intercept'], log_reg_2.pvalues['price_'], log_reg_2.pvalues['prom_']])
std_devs.append([
    log_reg_2.cov_params()['Intercept']['Intercept'],
    log_reg_2.cov_params()['price_']['price_'],
    log_reg_2.cov_params()['prom_']['prom_']
])

print("***** Q 3 *****")
col3 = " + ".join(X3.columns)
formula3 = "Y ~ " + col3
log_reg_3 = smf.logit(formula3, data=data).fit()
coefs.append([log_reg_3.params['Intercept'], log_reg_3.params['price_'], log_reg_3.params['prom_']])
pvals.append([log_reg_3.pvalues['Intercept'], log_reg_3.pvalues['price_'], log_reg_3.pvalues['prom_']])
std_devs.append([
    log_reg_3.cov_params()['Intercept']['Intercept'],
    log_reg_3.cov_params()['price_']['price_'],
    log_reg_3.cov_params()['prom_']['prom_']
])


# 4. Estimate the models of 1, 2 and 3 using wholesale cost as an instrument.
print("***** Q 4 *****")
model = sm.OLS(data['price_'],data['cost_'])
results = model.fit()
data['price_iv4'] = results.fittedvalues

formula4a = re.sub(r'price_ ', r'price_iv4 ', formula1)
formula4b = re.sub(r'price_ ', r'price_iv4 ', formula2)
formula4c = re.sub(r'price_ ', r'price_iv4 ', formula3)

log_reg_4a = smf.logit(formula4a, data=data).fit()
coefs.append([log_reg_4a.params['Intercept'], log_reg_4a.params['price_iv4'], log_reg_4a.params['prom_']])
pvals.append([log_reg_4a.pvalues['Intercept'], log_reg_4a.pvalues['price_iv4'], log_reg_4a.pvalues['prom_']])
std_devs.append([
    log_reg_4a.cov_params()['Intercept']['Intercept'],
    log_reg_4a.cov_params()['price_iv4']['price_iv4'],
    log_reg_4a.cov_params()['prom_']['prom_']
])

log_reg_4b = smf.logit(formula4b, data=data).fit()
coefs.append([log_reg_4b.params['Intercept'], log_reg_4b.params['price_iv4'], log_reg_4b.params['prom_']])
pvals.append([log_reg_4b.pvalues['Intercept'], log_reg_4b.pvalues['price_iv4'], log_reg_4b.pvalues['prom_']])
std_devs.append([
    log_reg_4b.cov_params()['Intercept']['Intercept'],
    log_reg_4b.cov_params()['price_iv4']['price_iv4'],
    log_reg_4b.cov_params()['prom_']['prom_']
])

log_reg_4c = smf.logit(formula4c, data=data).fit()
coefs.append([log_reg_4b.params['Intercept'], log_reg_4b.params['price_iv4'], log_reg_4b.params['prom_']])
pvals.append([log_reg_4b.pvalues['Intercept'], log_reg_4b.pvalues['price_iv4'], log_reg_4b.pvalues['prom_']])
std_devs.append([
    log_reg_4c.cov_params()['Intercept']['Intercept'],
    log_reg_4c.cov_params()['price_iv4']['price_iv4'],
    log_reg_4c.cov_params()['prom_']['prom_']
])
print(log_reg_4b.pvalues)



# 5. Estimate the models of 1, 2 and 3 using the Hausman instrument (average price in other markets).
print("***** Q 5 *****")
model = sm.OLS(data['price_'],data['hausman'])
results = model.fit()
data['price_iv5'] = results.fittedvalues

formula5a = re.sub(r'price_ ', r'price_iv5 ', formula1)
formula5b = re.sub(r'price_ ', r'price_iv5 ', formula2)
formula5c = re.sub(r'price_ ', r'price_iv5 ', formula3)

log_reg_5a = smf.logit(formula5a, data=data).fit()
coefs.append([log_reg_5a.params['Intercept'], log_reg_5a.params['price_iv5'], log_reg_5a.params['prom_']])
pvals.append([log_reg_5a.pvalues['Intercept'], log_reg_5a.pvalues['price_iv5'], log_reg_5a.pvalues['prom_']])
std_devs.append([
    log_reg_5a.cov_params()['Intercept']['Intercept'],
    log_reg_5a.cov_params()['price_iv5']['price_iv5'],
    log_reg_5a.cov_params()['prom_']['prom_']
])

log_reg_5b = smf.logit(formula5b, data=data).fit()
coefs.append([log_reg_5b.params['Intercept'], log_reg_5b.params['price_iv5'], log_reg_5b.params['prom_']])
pvals.append([log_reg_5b.pvalues['Intercept'], log_reg_5b.pvalues['price_iv5'], log_reg_5b.pvalues['prom_']])
std_devs.append([
    log_reg_5b.cov_params()['Intercept']['Intercept'],
    log_reg_5b.cov_params()['price_iv5']['price_iv5'],
    log_reg_5b.cov_params()['prom_']['prom_']
])

log_reg_5c = smf.logit(formula5c, data=data).fit()
coefs.append([log_reg_5c.params['Intercept'], log_reg_5c.params['price_iv5'], log_reg_5c.params['prom_']])
pvals.append([log_reg_5c.pvalues['Intercept'], log_reg_5c.pvalues['price_iv5'], log_reg_5c.pvalues['prom_']])
std_devs.append([
    log_reg_5c.cov_params()['Intercept']['Intercept'],
    log_reg_5c.cov_params()['price_iv5']['price_iv5'],
    log_reg_5c.cov_params()['prom_']['prom_']
])

def get_stars(p):

    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.1:
        return "*"
    else:
        return " "

stars = [[get_stars(p) for p in row] for row in pvals]
names = ['Intercept', "Price\t", "Promotion"]

print("\t\t" + "\t\t".join(["Q1", "Q2", "Q3", "Q4a", "Q4b", "Q4c", "Q5a", "Q5b", "Q5c"]))
for j, _ in enumerate(coefs[0]):
    print(names[j] + "\t" +"\t\t".join([f"{round(row[j], 3)}{stars[i][j]}" for i, row in enumerate(coefs)]))
    print("\t\t" + "\t\t".join([f"{round(st[j], 3)}" for st in std_devs]))


# for i, row in enumerate(coefs):
#     print("\t\t".join([f"{round(r, 3)}{stars[i][j]}" for j, r in enumerate(row)]))
#     print("\t\t".join([f"({round(st, 3)})" for st in std_devs[i]]))




# # 6. Using the analytic formula for elasticity of the logit model, compute
# # the mean own-price elasticities for all brand in the market using the
# # estimates in 1, 2 and 3. Do these results make sense? (Discuss)

# # E = (β * X * (1 - P)) / P
print("***** Q 6 *****")
data['ratio']  = (1-data['ms_by_store_week'])/data['ms_by_store_week']
data['fitted1'] = log_reg_1.fittedvalues
print(data['fitted1'])
data['elasticity_est_1'] = data['fitted1'] * data['ratio']
data['fitted2'] = log_reg_2.fittedvalues
data['elasticity_est_2'] = data['fitted2'] * data['ratio']
data['fitted3'] = log_reg_3.fittedvalues
data['elasticity_est_3'] = data['fitted3'] * data['ratio']
elasticity = data.groupby(['brand'])[['elasticity_est_1', 'elasticity_est_2', 'elasticity_est_3']].aggregate('mean')
print(elasticity)

