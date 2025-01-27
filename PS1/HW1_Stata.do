global root "."
global data "$root/PS1_Data"
import delimited "$data/OTC_Data.csv", clear

list in 1/4

sum store week brand

// Get data into store * week level to construct shares
reshape wide sales_ price_ prom_ cost_, i(store week count) j(brand)

// Construct shares: tally outside option and compute log shares
gen buy = 0
forvalues i = 1/11 {
    gen lshare`i' = log(sales_`i'/count)
    qui replace buy = buy + sales_`i'
}
gen nobuy = count-buy
gen lshare0 = log(nobuy/count)

// Subtract logs to get LHS
forvalues i = 1/11 {
    gen y`i' = lshare`i' - lshare0
}

// Return to store-week * brand level
drop sales* lshare* *buy*
reshape long price_ prom_ cost_ y, i(store week count) j(brand)

list in 1/4

// Construct store-week variable for brand FE regressions
tostring store, gen(store_id)
tostring week, gen(week_id)
gen store_week_id = store_id + " " + week_id
encode store_week_id, gen(t)

// Construct store-brand variable for store-brand FE regressions
tostring brand, gen(brand_id)
gen store_brand_id = store_id + " " + brand_id
encode store_brand_id, gen(sb)

// 1. Use price and promotion as product characteristics
qui reg y price prom_
estimates store model1ols

// 2. Price and promotion plus brand dummies
qui xtset brand t
qui xtreg y price prom_, fe vce(robust)
estimates store model2ols

// 3. Price and promotion plus store-brand dummies
qui xtset sb week
qui xtreg y price prom_, fe vce(robust)
estimates store model3ols

// 4. Now we use wholesale cost as an instrument for price, and estimate the same models as 1-3
qui ivreg y prom_ (price=cost_), robust
estimates store model1costiv
qui xtset brand t
qui xtivreg y prom_ (price=cost_), fe vce(robust)
estimates store model2costiv
qui xtset sb week
qui xtivreg y prom_ (price=cost_), fe vce(robust)
estimates store model3costiv


save "$data/reg_data", replace

// 5. Next, we use a Hausman instrument for price, and estimate the models from 1-3
// First, construct the instrument:

import delimited "$data/OTCDataInstruments.csv", clear
forvalues i = 1/30 {
    qui replace pricestore`i' = 0 if store == `i'
}

// Compute average of all other prices
gen hausman = 0
forvalues i = 1/30 {
    qui replace hausman = hausman + pricestore`i'
}
replace hausman = hausman/29
keep store week brand hausman

// Add average price to our regressors
qui merge 1:1 store week brand using "$data/reg_data.dta"
drop _merge

// 5. Estimate models
qui ivreg y prom_ (price=hausman)
estimates store model1hausmaniv
qui xtset brand t
qui xtivreg y prom_ (price=hausman), fe vce(robust)
estimates store model2hausmaniv
qui xtset sb week
qui xtivreg y prom_ (price=hausman), fe vce(robust)
estimates store model3hausmaniv

esttab model1ols model1costiv model1hausmaniv model2ols model2costiv model2hausmaniv model3ols model3costiv model3hausmaniv using "$data/estimates.tex",  se replace


