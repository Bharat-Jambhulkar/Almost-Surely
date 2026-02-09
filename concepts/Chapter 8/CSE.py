import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
nvec = np.arange(100,2700,step=300)
th = 2.5
a = 2
b = 4
nsim = 400
ep = 0.05
sample_matrix = np.random.normal(loc=th,scale=1,size=(nsim,max(nvec)))

rn = []
mn = []
vn = []

for n in nvec:
    sample = sample_matrix[:,:n]
    sample_mean = sample.mean(axis=1)
    th_hat = np.zeros(nsim)
    for i in range(nsim):
        if sample_mean[i] < a:
            th_hat[i] = a
        elif sample_mean[i] > b:
            th_hat[i] = b
        else:
            th_hat[i] = sample_mean[i]

    rn.append((np.abs(th_hat - th) < ep).sum() / nsim)
    mn.append(np.mean(th_hat))
    vn.append(np.mean((th_hat - th)**2))


d = pd.DataFrame({'n':nvec,'rn':rn,'mn':mn,'vn':vn})
d = np.round(d,4)
print(d)

plt.plot(nvec, rn, marker='o',color='red')
plt.xlabel('sample size (n) ')
plt.title('Convergence in Probability')
plt.show()

plt.plot(nvec, vn, marker='o',color='blue')
plt.xlabel('sample size (n) ')
plt.title('Convergence to Zero')
plt.show()