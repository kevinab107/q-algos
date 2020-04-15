#%%
from tsp_utils import (
    traveling_salesperson,
    traveling_salesperson_qubo,
    travelling_salesperson_pauli,
    traveling_salesperson_edge_qubo,
)
from q_utils import get_compute_backend
from max_cut import get_maxcut_qubo
import networkx as nx
import numpy as np
from dwave.system.samplers import (
    DWaveSampler,
    LeapHybridSampler,
)
from dwave.system.composites import EmbeddingComposite, AutoEmbeddingComposite

from neal import SimulatedAnnealingSampler
from dwave_qbsolv import QBSolv
from benchmark import solve_tsp_gort
from data_models import create_tsp_data_model
import time
from dimod import BinaryQuadraticModel

# from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP, L_BFGS_B
# from qiskit.aqua.algorithms.adaptive import QAOA


#%%
def benchmark_tsp():

    # print("solving with dimod ")
    # route_classical = []
    # route_classical = solve_tsp_sa(G)

    # sampler_q = AutoEmbeddingComposite(DWaveSampler())
    # sampler_h = solve_tsp_hybrid(G)
    # route_quantum = traveling_salesperson(G, sampler_q)
    #
    # sample_hybrid = traveling_salesperson(G, LeapHybridSampler())
    # Q = traveling_salesperson_qubo(G)

    # use the sampler to find low energy states
    # response = QBSolv().sample_qubo(Q)
    # route_classical = traveling_salesperson(G, SimulatedAnnealingSampler())
    # route_classical = traveling_salesperson(G, sampler_c)
    # print("solving with Brute force ")
    # route_brute = solve_tsp_brute_force(distance_matrix)
    google_cost = {}
    google_time = {}
    qbsolv_cost = {}
    qbsolve_time = {}
    qbsolve_qubo_time = {}
    hybrid_cost = {}
    hybrid_time = {}

    for node_len in [6]:
        data = create_tsp_data_model(node_len)
        G = nx.from_numpy_matrix(data["distance_matrix_numpy"])

        print("solving with google or")
        start = time.time()
        result_google_or = solve_tsp_gort(data)
        end = time.time()
        google_cost[node_len] = result_google_or
        google_time[node_len] = end - start
        print("solved with google or time:", end - start)

        # print("solving with hybrid")
        # start = time.time()
        # Q = traveling_salesperson_qubo(G, lagrange=1200)
        # mid = time.time()
        # result_qbsolv_h = solve_tsp_hybrid(Q, G)
        # end = time.time()
        # hybrid_cost[node_len] = result_qbsolv_h[1]
        # hybrid_time[node_len] = end - start
        # print("solved with hybrid", end - start)

        print("solving with qbsolve ")
        start = time.time()
        Q = traveling_salesperson_edge_qubo(G, lagrange=1000)
        mid = time.time()
        result_qbsolv_c = solve_tsp_Qbsolv_c(Q, G)
        end = time.time()
        qbsolv_cost[node_len] = result_qbsolv_c[1]
        qbsolve_time[node_len] = end - start
        qbsolve_qubo_time[node_len] = mid - start
        print("solved with qbsolve time", end - start)

        # print("solving with qbsolve ")
        # start = time.time()
        # Q = get_maxcut_qubo(G, lagrange=1200)
        # mid = time.time()
        # result_qbsolv_c = solve_maxcut_Qbsolv_c(Q, G)
        # end = time.time()
        # qbsolv_cost[node_len] = result_qbsolv_c[1]
        # qbsolve_time[node_len] = end - start
        # qbsolve_qubo_time[node_len] = mid - start
        # print("solved with qbsolve time", end - start)

        # print("solving with qbsolve ")
        # start = time.time()
        # result = solve_tsp_qaoa(G, solve_type="simulation")
        # end = time.time()
        # print("solved with qbsolve time", end - start)

    return (
        G,
        google_cost,
        google_time,
        qbsolv_cost,
        qbsolve_time,
        hybrid_cost,
        hybrid_time,
        result_qbsolv_c,
    )


def calculate_cost(cost_matrix, route=None, adj_mat=None):
    cost = 0
    if route != None:
        for i in range(len(route)):
            a = i % len(route)
            b = (i + 1) % len(route)
            cost += cost_matrix[route[a]][route[b]]
    elif adj_mat != 0:
        for i in range(len(adj_mat)):
            cost += cost_matrix[i][adj_mat[i]]
    return cost


def solve_tsp_Qbsolv_c(Q, G):
    response = QBSolv().sample_qubo(Q)
    sample = response.first.sample
    adj_mat = [None] * len(G)
    for (u, v), val in sample.items():
        if val:
            adj_mat[u] = v
    cost = calculate_cost(nx.to_numpy_array(G), adj_mat=adj_mat)
    return (adj_mat, cost)


def solve_maxcut_Qbsolv_c(Q, G):
    response = QBSolv().sample_qubo(Q)
    sample = response.first.sample
    return sample


def solve_tsp_hybrid(Q, G):
    bqm = BinaryQuadraticModel.from_qubo(Q)
    response = LeapHybridSampler().sample(bqm, time_limit=40)
    sample = response.first.sample
    route = [None] * len(G)
    for (city, time), val in sample.items():
        if val:
            route[time] = city
    cost = calculate_cost(nx.to_numpy_array(G), route)
    return (route, cost)


# def solve_tsp_qaoa(graph, solve_type):
#     backend = get_compute_backend(solve_type)
#     print("initialized backend")
#     operator, offset = travelling_salesperson_pauli(graph)
#     print("initialized operator")
#     qaoa = QAOA(operator, p=1, optimizer=L_BFGS_B(maxfun=1000))
#     print("running the experinment")
#     result = qaoa.run(backend)
#     print("got the results")
#     return result


# %%
import dwavebinarycsp
import itertools

data = create_tsp_data_model(5)
G = nx.from_numpy_matrix(data["distance_matrix_numpy"])
# Qubo = traveling_salesperson_qubo(G)
csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)


def sigma_is_one_constraints(num_variables):
    configuration = [0] * num_variables
    configuration[0] = 1
    configuration = set(itertools.permutations(configuration))
    return configuration


config = sigma_is_one_constraints(G.number_of_nodes())
for node in G.nodes():
    variables = [str(node) + str(pos) for pos in G.nodes()]
    csp.add_constraint(config, variables)

for pos in G.nodes():
    variables = [str(node) + str(pos) for node in G.nodes()]
    csp.add_constraint(config, variables)

bqm = dwavebinarycsp.stitch(csp, max_graph_size=15)

#%%

sampler = LeapHybridSampler()
response = sampler.sample(bqm)


# %%
from collections import defaultdict

data = create_tsp_data_model(5)
G = nx.from_numpy_matrix(data["distance_matrix_numpy"])

time_window = defaultdict(bool)
i = 0
for node in range(5):
    i += 1
    for pos in range(5):
        if pos < i:
            time_window[(4 - node, pos)] = True
        else:
            time_window[(4 - node, pos)] = True


Q = traveling_salesperson_qubo(G, time_window=time_window)
res = solve_tsp_Qbsolv_c(Q, G)

# %%
