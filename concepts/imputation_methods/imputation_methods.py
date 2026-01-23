import numpy as np 
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

SEED = 220126
np.random.seed(SEED)

n = 100 ## number of rows
p=6 ## number of columns

data = np.zeros((n,p))

data[:,0] = np.random.uniform(30,50,size=n) ## generate first column randomly

coeffs = np.random.uniform(-3,3,size=p-1)
for i in range(1,p):
    data[:, i] = data[:, :i] @ coeffs[:i] + np.random.normal(0,1,n) ## generate each column as a linear combination of previous columns plus noise

df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(p)])

missing_prop = np.random.uniform(low=0.1, high=0.25, size=p) ## proportion of missing data for each column

missing_data = df.copy()

missing_idx = np.random.binomial(n=1,p=missing_prop[i], size=(n,p))

for i in range(p):
    missing_data.loc[missing_idx[:,i]==1, f"Var{i+1}"] = np.nan

## mean imputation
mean_imputed_data = missing_data.copy()
for i in range(p):
    mean_imputed_data[f"Var{i+1}"] = mean_imputed_data[f"Var{i+1}"].fillna(mean_imputed_data[f"Var{i+1}"].mean())

chained_imputation_data = mean_imputed_data.copy()

import tqdm as tqdm

max_iter = 500
iter_no = 0

diff = 1.0
diff_list = []
diff_list.append(diff)
old_df = chained_imputation_data.copy()
with tqdm.tqdm(desc="Chained Imputation", unit="iter") as pbar:
    while diff > 0.001 and iter_no < max_iter:
        iter_no += 1
        for i in range(p):
            subset_data = chained_imputation_data.loc[(missing_idx[:,i]==0), :]
            y = subset_data[f"Var{i+1}"]
            reg = subset_data.drop(columns=[f"Var{i+1}"])

            fit = sm.OLS(y,sm.add_constant(reg)).fit()
            new_reg = chained_imputation_data.drop(columns=[f"Var{i+1}"]).loc[(missing_idx[:,i]==1), :]
            pred = fit.predict(sm.add_constant(new_reg))

            chained_imputation_data.loc[(missing_idx[:,i]==1), f"Var{i+1}"] = pred

        
        diff = np.max(np.abs(old_df - chained_imputation_data))
        diff_list.append(diff)
        old_df = chained_imputation_data.copy()

        pbar.update(1)
        pbar.set_postfix(diff=f"{diff:.6f}")


## plot convergence

plt.figure()
plt.plot(diff_list, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Max absolute difference")
plt.title("Convergence of Chained Imputation")
plt.yscale("log")   # highly recommended
plt.grid(True)
plt.show()

## plot original vs imputed 

fig, axes = plt.subplots(3, 2, figsize=(14, 18),constrained_layout=True)
axes = axes.flatten()

variables = [f"Var{i+1}" for i in range(p)]

for i, col in enumerate(variables):
    df[col].plot.density(ax=axes[i], color='blue', lw=2, alpha=0.7, label='Original')
    mean_imputed_data[col].plot.density(ax=axes[i], color='orange', lw=2, alpha=0.7, label='Mean Imputed')
    chained_imputation_data[col].plot.density(ax=axes[i], color='green', lw=2, alpha=0.7, label='Chained Imputed')

    axes[i].set_title(f'Distribution of {col}', fontsize=8, pad=5)
    axes[i].set_xticks([])

    if i % 2 != 0:
        axes[i].set_ylabel("")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)

plt.show()