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