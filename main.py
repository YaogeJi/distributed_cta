from solver import *
import argparse
import pickle
from generator import Generator
from network import ErodoRenyi


# configuration
parser = argparse.ArgumentParser(description='distributed optimization')
parser.add_argument('--storing_file', default='', type=str, help='storing_file_name')
## data
parser.add_argument("-N", "--num_samples", type=int)
parser.add_argument("-d", "--num_dimension", type=int)
parser.add_argument("-s", "--sparsity", type=int)
parser.add_argument("-k", type=float, default=0.25)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--data_index", type=int, default=0)

## network
parser.add_argument("-m", "--num_nodes", type=int)
parser.add_argument("-p", "--probability", type=float)
parser.add_argument("-rho", "--connectivity", type=float)
## solver
parser.add_argument("-r", "--project_radius", type=float)
parser.add_argument("--solver_mode", choices=("centralized", "distributed", "localized"))
parser.add_argument("--max_iter", type=int, default=1e6)
parser.add_argument("--tol", type=float, default=1e-10)
parser.add_argument("--iter_type", choices=("lagrangian", "projected"))
parser.add_argument("--gamma", type=float)
parser.add_argument("--lmda", type=float)
## others
parser.add_argument("--verbose", action="storing_true")
args = parser.parse_args()


def main():
    # preprocessing data
    data_path = "./data/N{}_d{}_s{}_k{}_sigma{}/exp{}.data".format(
        args.N, args.d, args.s, args.k, args.sigma, args.data_index)
    network_path = "./network/m{}_p{}_rho{}.network".format(args.m, args.p, args.rho)
    try:
        w = pickle.load(open(network_path, "rb"))
    except:
        w = ErodoRenyi(m=args.m, rho=args.rho, p=args.p)
        pickle.dump(w, open(network_path, "wb"))
    ## processing data
    try:
        X, Y, ground_truth, optimal_lambda, min_stat_error = pickle.load(open(data_path, "rb"))
    except FileNotFoundError:
        generator = Generator(args.N, args.d, args.s, args.k, args.sigma)
        X, Y, ground_truth, optimal_lambda, min_stat_error = generator.generate()
        pickle.dump([X, Y, ground_truth, optimal_lambda, min_stat_error], open(data_path, "wb"))
    ## processing network


    # solver run
    if args.solver_mode == 'centralized':
        solver = Lasso(args.max_iter, args.gamma, args.tol, args.iter_type, args.lmda)
    elif args.solver_mode == 'distributed':
        solver = DistributedLasso(args.max_iter, args.gamma, args.tol, args.iter_type, args.lmda, w)
    # elif args.solver_mode == 'localized':
    #     solver = LocalizedLasso()
    else:
        raise NotImplementedError("solver mode currently only support centralized or distributed")
    outputs = solver.fit(X, Y, args.r, ground_truth, verbose=args.verbose)
    output_filename = "./output/N{}_m{}_rho{}_{}_exp{}".format(
        args.N, args.m, args.rho, args.solver_mode, args.data_index)
    pickle.dump(outputs, open(output_filename, "w"))


if __name__ == "__main__":
    main()
