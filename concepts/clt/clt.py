import numpy as np
from scipy import stats

np.random.seed(10123)
dist = ['exponential','poisson','geometric']
nvec = np.arange(30,170,step=30)
nmax = max(nvec)
nsim = 300
p_value_mat = np.zeros((len(dist),len(nvec)))

for d in dist:
    if d == 'exponential':
        mu = 1
        sample_matrix = np.random.exponential(scale=mu,size=(nsim,nmax))
        p_value = []

        for n in nvec:
            sample = sample_matrix[:,:n]
            z_n = (np.sqrt(n)*(sample.mean(axis=1)-mu))/mu
            p_value.append(stats.shapiro(z_n).pvalue)
        
    elif d == 'poisson':
        mu = 1
        sample_matrix = np.random.poisson(lam=mu,size=(nsim,nmax))
        p_value = []

        for n in nvec:
            sample = sample_matrix[:,:n]
            z_n = (np.sqrt(n)*(sample.mean(axis=1)-mu))/np.sqrt(mu)

            p_value.append(stats.shapiro(z_n).pvalue)
    elif d == 'geometric':
        p = 0.5
        mu = (1 - p) / p
        sample_matrix = np.random.geometric(p=p,size=(nsim,nmax))
        sample_matrix = sample_matrix - 1
        p_value = []

        for n in nvec:
            sample = sample_matrix[:,:n]
            z_n = (np.sqrt(n)*(sample.mean(axis=1)-mu))/np.sqrt((1-p)/p**2)

            p_value.append(stats.shapiro(z_n).pvalue)

    p_value_mat[dist.index(d),:] = p_value


alpha = 0.05

mask = p_value_mat > alpha 

idx = mask.argmax(axis=1)           # first True along nvec
n_min = nvec[idx]

for i in range(len(dist)):
    print(f"For {dist[i]} distribution, minimum sample size = {n_min[i]}")


print("------------------------")

print("Case where distibution has finite mean and finite variance: Exponential distribution")


nsim = 1000
nvec = np.arange(50,270,step=30)
nmax = max(nvec)

l = 0.5
mu = 1/l
sample_matrix = np.random.exponential(scale=1/l, size=(nsim, max(nvec)))

prop = []

for n in nvec:
    sample = sample_matrix[:, :n]
    sample_mean = sample.mean(axis=1)
    prop.append((sample_mean <= mu).sum()/nsim)


for i in range(len(nvec)):
    print(f"For n = {nvec[i]}, proportion of sample means <= {mu} is {prop[i]:.3f}")


 
import matplotlib.pyplot as plt

plt.plot(nvec, prop, marker='o')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()

print("------------------------")

## case where distibution has finite mean and infinite variance

print("Case where distibution has finite mean and infinite variance:")

np.random.seed(30226)
nsim = 500
nvec = np.arange(start=1000,stop = 50000, step = 5000)

## for random sample use probability integral transform

th = 1.5
def cdf_inverse(u):
    return th/((1 - u) ** (1 / 2))

nmax = max(nvec)
sample_matrix = cdf_inverse(np.random.uniform(0, 1, size=(nsim, nmax)))

pop_mean = 2*th

prop = []

for n in nvec:
    sample = sample_matrix[:, :n]
    sample_mean = sample.mean(axis=1)
    prop.append((sample_mean <= pop_mean).sum()/nsim)


for i in range(len(nvec)):
    print(f"For n = {nvec[i]}, proportion of sample means <= {pop_mean} is {prop[i]:.3f}")


print("------------------------")
print("\nThe difference in the two cases illustrates the role of finite variance. The sample size required for convergence is much larger when the variance is infinite.")

print("------------------------")

print("Lindeberg-Feller CLT")


nvec = np.arange(10,150,step = 20)

nsim = 300

from scipy import stats
pvalues = np.zeros(len(nvec))
v = np.zeros(len(nvec))
y = np.zeros((nsim, len(nvec)))
for i in range(len(nvec)):
    
    n = nvec[i]
    for j in range(nsim):
        np.random.seed(j)
        x = np.zeros(n)
        for k in range(1,n):
            x[k] = np.random.uniform(-k, k)
        v[i] = np.sqrt(n*(n+1)*(2*n+1)/(18)) 
        y[j,i] = (np.sum(x))/v[i]

    pvalues[i] = stats.shapiro(y[:,i]).pvalue

import pandas as pd
df = pd.DataFrame({'nvec': nvec, 'pvalues': pvalues, 'v': v}).round(4)

print(df)