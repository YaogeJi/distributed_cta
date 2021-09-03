import os
os.chdir("../")

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
for num_exp in range(total):
    print("Monte Carlo: {}/{}".format(num_exp+1, total))
    for lmda in lmda_group:
        command = "python main.py -N {} -d {} -s {} -m {} -p {} -rho {} --data_index {} --solver_mode {} --iter_type {} --gamma {} --lmda {} --verbose --storing_filepath ./output/stat_veri/N{}_rho{}_exp{}/ --storing_filename {}_lambda{}.output".format(
            N, d, s, m, p, rho, num_exp, solver_mode, iter_type, gamma, lmda, N, rho, num_exp, solver_mode, lmda)
        # print(command)
        os.system(command)