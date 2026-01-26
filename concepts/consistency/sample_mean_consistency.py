import numpy as np
import matplotlib.pyplot as plt

np.random.seed(260126)
nsim = 500
nvec = np.arange(500,5500,step=500)
nmax = max(nvec)
ep = 0.05
dist = ['exponential','geometric','poisson']
prop_mat = np.zeros((len(dist),len(nvec)))  ## to store proportions
exp_scale = 1
geometric_p = 0.5
pois_lambda = 1

for i in dist:
    prop = []
    if i == 'exponential':
        sample_matrix = np.random.exponential(scale=exp_scale, size=(nsim,nmax))
        pop_mean = exp_scale
    elif i == 'geometric':
        sample_matrix = np.random.geometric(p=geometric_p, size=(nsim,nmax))
        sample_matrix -= 1  ## to make it start from 0
        pop_mean = (1 - geometric_p) / geometric_p
    elif i == 'poisson':
        sample_matrix = np.random.poisson(lam=pois_lambda, size=(nsim,nmax))
        pop_mean = pois_lambda

    for n in nvec:
        sample = sample_matrix[:,:n]
        sample_mean = sample.mean(axis=1) ## row mean
        sample_mean = np.array(sample_mean)
        prop.append(((np.abs(sample_mean - pop_mean))< ep).sum()/nsim)

    prop_mat[dist.index(i),:] = np.array(prop)



plt.plot(nvec,prop_mat[0,:],label='Exponential',marker='o')
plt.plot(nvec,prop_mat[1,:],label='Geometric',marker='o')
plt.plot(nvec,prop_mat[2,:],label='Poisson',marker='o')
plt.xlabel('Sample Size (n)')
plt.ylabel('Proportion within epsilon of Population Mean')
plt.title('Convergence in Probability of Sample Mean to Population Mean')
plt.legend()
plt.grid()
plt.show()