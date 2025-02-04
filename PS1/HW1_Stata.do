global root "."
global data "$root/PS1_Data"
import delimited "$data/OTC_Data.csv", clear

list in 1/4

sum store week brand

// Normalize prices to per 50 tablets
replace price_ = price_*2 if (brand == 1) | (brand == 4) | (brand == 7)
replace price_ = price_/2 if brand == 3 | brand == 6 | brand == 9 | brand == 11

// Get data into store * week level to construct shares
reshape wide sales_ price_ prom_ cost_, i(store week count) j(brand)

// Construct shares: tally outside option and compute log shares
gen buy = 0
forvalues i = 1/11 {
    gen share`i' = sales_`i'/count
    gen lshare`i' = log(share`i')
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
reshape long price_ prom_ cost_ share y, i(store week count) j(brand)

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

// Label variables for table
label variable price_ "Price"
label variable prom_ "Promotion"

// 1. Use price and promotion as product characteristics
qui reg y price prom_
estimates store model1ols
qui estadd local spec "OLS"
qui estadd local FE "No"

// 2. Price and promotion plus brand dummies
qui xtset brand t
qui xtreg y price prom_, fe vce(robust)
estimates store model2ols
qui estadd local spec "OLS"
qui estadd local FE "Brand"

// 3. Price and promotion plus store-brand dummies
qui xtset sb week
qui xtreg y price prom_, fe vce(robust)
estimates store model3ols
qui estadd local spec "OLS"
qui estadd local FE "Store-Brand"

// 4. Now we use wholesale cost as an instrument for price, and estimate the same models as 1-3
qui ivreg y prom_ (price=cost_), robust
estimates store model1costiv
qui estadd local spec "Cost IV"
qui estadd local FE "No"

qui xtset brand t
qui xtivreg y prom_ (price=cost_), fe vce(robust)
estimates store model2costiv
qui estadd local spec "Cost IV"
qui estadd local FE "Brand"

qui xtset sb week
qui xtivreg y prom_ (price=cost_), fe vce(robust)
estimates store model3costiv
qui estadd local spec "Cost IV"
qui estadd local FE "Store-Brand"



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
qui estadd local spec "Hausman IV"
qui estadd local FE "No"

qui xtset brand t
qui xtivreg y prom_ (price=hausman), fe vce(robust)
estimates store model2hausmaniv
qui estadd local spec "Hausman IV"
qui estadd local FE "Brand"

qui xtset sb week
qui xtivreg y prom_ (price=hausman), fe vce(robust)
estimates store model3hausmaniv
qui estadd local spec "Hausman IV"
qui estadd local FE "Store-Brand"


esttab model1ols model1costiv model1hausmaniv model2ols model2costiv model2hausmaniv model3ols model3costiv model3hausmaniv using "$data/OLS_estimates.tex",  scalars("spec Model" "FE FE") label nomtitles se replace

keep store week brand price share
reshape wide price share, i(store week) j(brand)
merge 1:m store week using "$data/reg_data"

keep store week brand price* share*

save "$data/tmp", replace

// Save elasticities for all store-weeks
local allmodels "model1ols model1costiv model1hausmaniv model2ols model2costiv model2hausmaniv model3ols model3costiv model3hausmaniv"

foreach m in `allmodels' {
    use "$data/tmp", clear
    estimates restore `m'

    // Calculate elasticities using logit formula
    forvalues i = 1/11 {
        local alpha=_b[price_]
        qui gen e_`i' = `alpha'*price_`i'*share`i'
        qui replace e_`i' = -`alpha'*price_*(1-share) if brand == `i'
    }

    // Calculate mean
    sort store week brand
    collapse e*
    forvalues i = 1/11 {
        format e_`i' %6.5f
    }

    // Make each row a brand
    qui gen ct = 1
    qui reshape long e_, i(ct) j(brand)
    drop ct
    rename e_ e
    qui export delimited "./elasticity_`m'.csv", replace datafmt

    // Save as dta to merge later
    qui save "$data/elasticity_`m'", replace
}

erase "$data/tmp.dta"

// Get OLS elasticities to merge into one (LaTeX) table
local ols "model1ols model2ols model3ols"
import delimited "./elasticity_model1ols.csv", clear
rename e one
merge 1:1 brand using "$data/elasticity_model2ols"
drop _merge
rename e two
merge 1:1 brand using "$data/elasticity_model3ols"
drop _merge
rename e three
export delimited "$data/elasticity_ols.csv", replace datafmt


