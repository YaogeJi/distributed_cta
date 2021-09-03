from solver import *
import argparse
import pickle
import os
from generator import Generator
from network import ErodoRenyi


# configuration
parser = argparse.ArgumentParser(description='distributed optimization')
parser.add_argument('--storing_filepath', default='', type=str, help='storing_file_path')
parser.add_argument('--storing_filename', default='', type=str, help='storing_file_name')
## data
parser.add_argument("-N", "--num_samples", type=int)
parser.add_argument("-d", "--num_dimensions", type=int)
parser.add_argument("-s", "--sparsity", type=int)
parser.add_argument("-k", type=float, default=0.25)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument("--data_index", type=int, default=0)
## network
parser.add_argument("-m", "--num_nodes", type=int)
parser.add_argument("-p", "--probability", default=1, type=float)
parser.add_argument("-rho", "--connectivity", default=0, type=float)
## solver
parser.add_argument("--solver_mode", choices=("centralized", "distributed", "localized"))
parser.add_argument("--projecting", action="store_true")
parser.add_argument("--max_iter", type=int, default=1e6)
parser.add_argument("--tol", type=float, default=1e-10)
parser.add_argument("--iter_type", choices=("lagrangian", "projected"))
parser.add_argument("--gamma", type=float)
parser.add_argument("--lmda", type=float)
## others
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

def main():
    # preprocessing data
    data_path = "./data/N{}_d{}_s{}_k{}_sigma{}/".format(
        args.num_samples, args.num_dimensions, args.sparsity, args.k, args.sigma)
    data_file = data_path + "exp{}.data".format(args.data_index)
    network_path = "./network/"
    network_file = network_path + "m{}_p{}_rho{}.network".format(args.num_nodes, args.probability, args.connectivity)

    ## processing data
    try:
        X, Y, ground_truth, optimal_lambda, min_stat_error = pickle.load(open(data_file, "rb"))
    except FileNotFoundError:
        os.makedirs(data_path, exist_ok=True)
        generator = Generator(args.num_samples, args.num_dimensions, args.sparsity, args.k, args.sigma)
        X, Y, ground_truth, optimal_lambda, min_stat_error = generator.generate()
        pickle.dump([X, Y, ground_truth, optimal_lambda, min_stat_error], open(data_file, "wb"))

    ## processing network
    try:
        w = pickle.load(open(network_file, "rb"))
    except:
        w = ErodoRenyi(m=args.num_nodes, rho=args.connectivity, p=args.probability).generate()
        os.makedirs(network_path, exist_ok=True)
        pickle.dump(w, open(network_file, "wb"))

    # solver run
    if args.solver_mode == 'centralized':
        solver = Lasso(args.max_iter, args.gamma, args.tol, args.iter_type, args.lmda, args.projecting)
    elif args.solver_mode == 'distributed':
        solver = DistributedLasso(args.max_iter, args.gamma, args.tol, args.iter_type, args.lmda, args.projecting, w)
    # elif args.solver_mode == 'localized':
    #     solver = LocalizedLasso()
    else:
        raise NotImplementedError("solver mode currently only support centralized or distributed")
    outputs = solver.fit(X, Y, ground_truth, verbose=args.verbose)
    output_filepath = args.storing_filepath
    output_filename = args.storing_filename
    os.makedirs(output_filepath, exist_ok=True)
    pickle.dump(outputs, open(output_filepath + output_filename, "wb"))


if __name__ == "__main__":
    main()
