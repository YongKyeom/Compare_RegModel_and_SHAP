# Compare Regression Model and SHAP(XGBoost)

## Description
* Regression problem
* According to various variable combinations, Compare prediction error with (Lasso) Linear Regression, Random Forest and XGBoost
* I used 4 variable combinations

## Script: LM_RF_XGB_Compare
### Description
  * If you adjust X(independent variable) and Y(dependent variable), you revise 'VAR_LS' & 'TARGET_VAR'
  * 'VISCOSITY' in the script means that I want to know property variable, main target variable.
  So If you do not have vairable like 'VISCOSITY', you need to remove that
  * LM with caret package, Random Forest with h2o package and XGBoost with xgboost package
  * In case of XGBoost, there is parameter tunning with grid search. So if you adjust parameter, revie 'PARAGRID_LS'

### Output
  * Error_Result: RMSE & MAE according to variable combinations & VISCOSITY level
  * VAR_Import_Result : Variable importance 
  * GLM_Use_Var: In lasso model, check used variables
  * baseLine_ERROR: base line of error
 
## Script: SHAP_XGB
### Description
  * It is same with LM_RF_XGB_Compare
  * To interprete XGBoost Model, I use SHAP value with 'SHAPforxgboost' package in R
  * Especially, I want to know 'VISCOSITY's impact on Y(dependent variable). 
  So I check dependency plot with VISCOSITY and another variable to Y(dependent variable).

## Script: RFE_XGB
### Description
  * I want to RFE(Recursive Feature Elimination) with XGBoost Model, but there is no package this mehthod.
  * So I write 'User define function' with RFE with XGBoost
