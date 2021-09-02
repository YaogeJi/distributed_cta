import numpy as np
from networkx import *


class FullyConnectedNetwork:
    def __init__(self, m):
        self.m = m
        self.w = 1/m * np.ones((m, m))

    def generate(self):
        return self.w

class ErodoRenyi:
    def __init__(self, m, rho, p):
        self.node = m
        self.rho = rho
        self.probability = p

    def generate(self):
        connected = False
        connectivity = 1
        while True:
            G = erdos_renyi_graph(self.node, self.probability)
            connected = is_connected(G)
            if not connected:
                continue
            adjacent_matrix = to_numpy_matrix(G)
            matrix = np.zeros((self.node, self.node))
            for i in G.edges:
                degree = max(G.degree(i[0]), G.degree(i[1]))
                G.add_edge(*i, weight=1/(degree + 1))
            adjacent_matrix = to_numpy_array(G)
            weighted_matrix = np.eye(self.node) - np.diag(sum(adjacent_matrix)) + adjacent_matrix
            eigenvalue, _ = np.linalg.eig(weighted_matrix)
            sorted_eigenvalue = np.sort(np.abs(eigenvalue))
            connectivity = sorted_eigenvalue[-2]
            print(connectivity)
            if np.abs(connectivity - self.rho) < 0.001:
                break
        return weighted_matrix
