import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tqdm as tqdm

nsim = 10 ## number of simulations
nreg = 50 ## number of regressors

nvec = np.arange(60,660,step=60) # (nreg < minimum of nvec)
nmax = max(nvec) 
prop_mat = np.zeros((nsim,len(nvec)))

for j in tqdm.tqdm(range(nsim)):
    ## data generation steps
    X = np.random.uniform(2,10,size=(nmax,nreg))
    columns=[f'X{i+1}' for i in range(nreg)]
    X_df = pd.DataFrame(X, columns=columns)
    beta = np.random.uniform(-5,5,size=nreg)
    notsig = 20 ## number of not significant variables
    beta[-notsig:] = 0
    true_zero_vars = set(columns[-notsig:])
    ep = np.random.normal(0,1,size=nmax)
    y = X_df @ beta + ep
    prop = []
    for n in nvec:
        X_df_sub = X_df.iloc[:n,:]
        y_sub = y[:n]
        model = sm.OLS(y_sub, sm.add_constant(X_df_sub)).fit()
        ranked_vars = model.pvalues.drop("const").sort_values().index.tolist() ## drop intercept
        bic_values = [] ## store bic value for each combination
        for k in range(1,len(ranked_vars)+1):
            k_vars = ranked_vars[:k]
            x_k = X_df_sub[k_vars] ## get data on only selected regressors
            x_k = sm.add_constant(x_k)
            model_k = sm.OLS(y_sub, x_k).fit() ## fit model
            bic_values.append(model_k.bic) ## store the BIC value

        min_idx = np.argmin(bic_values) ## index of minimum BIC value
        selected_vars = ranked_vars[:min_idx+1] 
        excluded_vars = set(columns) - set(selected_vars)
        correct = excluded_vars.intersection(true_zero_vars)
        prop.append(len(correct)/notsig) ## proportion of correctly excluded variables
    prop_mat[j,:] = prop 


## Compute average proportions across simulations
avg_prop = prop_mat.mean(axis=0)

plt.plot(nvec, avg_prop, marker='o', linewidth=2)
plt.xlabel('Sample Size (n)')
plt.ylabel('Proportion of Correctly Excluded Variables')
plt.title('BIC Consistency Simulation')

plt.grid(True)
plt.show()