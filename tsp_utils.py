
import itertools

from collections import defaultdict

from dwave_networkx.utils import binary_quadratic_model_sampler
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
import networkx as nx
import numpy as np

def traveling_salesperson(
    G, sampler=None, lagrange=None, weight="weight", start=None, **sampler_args
):
    Q = traveling_salesperson_qubo(G, lagrange, weight)
    #h, J = traveling_salesperson_ising(G)
    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)
    #response = sampler.sample_ising(h, J, **sampler_args)
    sample = response.first.sample
    route = [None] * len(G)
    # for (city, time), val in sample.items():
    #     if val:
    #         route[time] = city

    # if start is not None and route[0] != start:
    #     # rotate to put the start in front
    #     idx = route.index(start)
    #     route = route[-idx:] + route[:-idx]

    return sample


traveling_salesman = traveling_salesperson

def get_default_time_window(num_nodes):
    time_window = defaultdict(bool)
    for node in range(num_nodes):
        for pos in range(num_nodes):
            time_window[(node, pos)] = True
    return time_window

def traveling_salesperson_qubo(G, lagrange=None, time_window=None, weight="weight"):
    N = G.number_of_nodes()

    if lagrange is None:
        if G.number_of_edges() > 0:
            lagrange = G.size(weight=weight) * G.number_of_nodes() / G.number_of_edges()
        else:
            lagrange = 2

    if N in (1, 2) or len(G.edges) != N * (N - 1) // 2:
        msg = "graph must be a complete graph with at least 3 nodes or empty"
        raise ValueError(msg)

    Q = defaultdict(float)
    
    if(time_window == None):
        time_window = get_default_time_window(N)

    for node in G:
        for pos_1 in range(N):
            if(time_window[(node,pos_1)]):
                Q[((node, pos_1), (node, pos_1))] -= lagrange
                for pos_2 in range(pos_1 + 1, N):
                    if(time_window[(node,pos_2)]):
                        Q[((node, pos_1), (node, pos_2))] += 2.0 * lagrange
            else:
                Q[((node, pos_1), (node, pos_1))] += lagrange

    for pos in range(N):
        for node_1 in G:
            if(time_window[(node_1,pos)]):
                Q[((node_1, pos), (node_1, pos))] -= lagrange
                for node_2 in set(G) - {node_1}:
                    if(time_window[(node_2,pos)]):
                        Q[((node_1, pos), (node_2, pos))] += lagrange
                    
            else:
                Q[((node_1, pos), (node_1, pos))] += lagrange
            

    for u, v in itertools.combinations(G.nodes, 2):
        for pos in range(N):
            nextpos = (pos + 1) % N
            ##Add the logic to penalise infeasible solutions 
            Q[((u, pos), (v, nextpos))] += G[u][v][weight]
            Q[((v, pos), (u, nextpos))] += G[u][v][weight]

    return Q

def traveling_salesperson_edge_qubo(G, lagrange=None, weight="weight"):
    N = G.number_of_nodes()

    if lagrange is None:
        if G.number_of_edges() > 0:
            lagrange = G.size(weight=weight) * G.number_of_nodes() / G.number_of_edges()
        else:
            lagrange = 2

    Q = defaultdict(float)
    for u in range(N):
        neighbors = list(G.neighbors(u))
        for v in G.neighbors(u):
            Q[((u,v), (u,v))] += -lagrange
            for v2 in set(neighbors) - {v}:
                Q[((u,v), (u,v2))] += 2*lagrange
            neighbors.remove(v)
    
    for u in range(N):
        neighbors = list(G.neighbors(u))
        for v in G.neighbors(u):
            Q[((v,u), (v,u))] += -lagrange
            for v2 in set(neighbors) - {v}:
                Q[((v,u), (v2,u))] += 2*lagrange
            neighbors.remove(v)  
            
    for u in range(N):
        for v in G.neighbors(u):
            Q[(u,v), (v,u)] += 2*lagrange
            Q[(u,v), (u,v)] += -lagrange + G[u][v][weight]
            Q[(v,u), (v,u)] += -lagrange + G[v][u][weight]        
        
    return Q


def travelling_salesperson_pauli(G, lagrange=None, time_window=None, weight="weight"):
 
    num_nodes = G.number_of_nodes()
    num_qubits = num_nodes ** 2
    zero = np.zeros(num_qubits, dtype=np.bool)
    pauli_list = []
    shift = 0
    if lagrange is None:
        if G.number_of_edges() > 0:
            lagrange = G.size(weight=weight) * G.number_of_nodes() / G.number_of_edges()
        else:
            lagrange = 100000
            
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            for p in range(num_nodes):
                q = (p + 1) % num_nodes
                shift += G[i][j][weight] / 4

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[i * num_nodes + p] = True
                pauli_list.append([-G[i][j][weight] / 4, Pauli(z_p, zero)])

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[j * num_nodes + q] = True
                pauli_list.append([-G[i][j][weight] / 4, Pauli(z_p, zero)])

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[i * num_nodes + p] = True
                z_p[j * num_nodes + q] = True
                pauli_list.append([G[i][j][weight] / 4, Pauli(z_p, zero)])

    for i in range(num_nodes):
        for p in range(num_nodes):
            z_p = np.zeros(num_qubits, dtype=np.bool)
            z_p[i * num_nodes + p] = True
            pauli_list.append([lagrange, Pauli(z_p, zero)])
            shift += -lagrange

    for p in range(num_nodes):
        for i in range(num_nodes):
            for j in range(i):
                shift += lagrange / 2

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[i * num_nodes + p] = True
                pauli_list.append([-lagrange / 2, Pauli(z_p, zero)])

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[j * num_nodes + p] = True
                pauli_list.append([-lagrange / 2, Pauli(z_p, zero)])

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[i * num_nodes + p] = True
                z_p[j * num_nodes + p] = True
                pauli_list.append([lagrange / 2, Pauli(z_p, zero)])

    for i in range(num_nodes):
        for p in range(num_nodes):
            for q in range(p):
                shift += lagrange / 2

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[i * num_nodes + p] = True
                pauli_list.append([-lagrange / 2, Pauli(z_p, zero)])

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[i * num_nodes + q] = True
                pauli_list.append([-lagrange / 2, Pauli(z_p, zero)])

                z_p = np.zeros(num_qubits, dtype=np.bool)
                z_p[i * num_nodes + p] = True
                z_p[i * num_nodes + q] = True
                pauli_list.append([lagrange / 2, Pauli(z_p, zero)])
    shift += 2 * lagrange * num_nodes
    return WeightedPauliOperator(paulis=pauli_list), shift


def is_hamiltonian_path(G, route):

    return set(route) == set(G)


def calculate_route_cost(cost_matrix, solution):
    cost = 0
    for i in range(len(solution)):
        a = i % len(solution)
        b = (i + 1) % len(solution)
        cost += cost_matrix[solution[a]][solution[b]]

    return cost


