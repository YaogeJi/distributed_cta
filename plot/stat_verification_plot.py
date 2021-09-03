import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


lmda_group = [0.01, 0.02, 0.025, 0.027, 0.03, 0.031, 0.032, 0.033, 0.04, 0.05, 0.07]
N = 220
d = 360
s = 5
m = 5
p = 1
rho = 0
solver_mode = "centralized"
iter_type = "lagrangian"
gamma = 0.175
total = 30
loss = []
for num_exp in range(total):
    loss.append([])
    for lmda in lmda_group:
        filename = "../output/stat_veri/N{}_rho{}_exp{}/{}_lambda{}.output".format(N, rho, num_exp, solver_mode, lmda)
        _, logloss = pickle.load(open(filename, "rb"))
        loss[num_exp].append(logloss[-1])

x = np.array(lmda_group)
loss = np.array(loss)

plt.plot(x, np.mean(loss, axis=0))
plt.show()