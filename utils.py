#%%
import networkx as nx
from qiskit import Aer, IBMQ
from qiskit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    execute,
    transpile,
    Opera,
)
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_circuit_layout, plot_histogram
import numpy as np
from scipy.optimize import minimize

##VQE
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.aqua.algorithms import VQE, ExactEigensolver


def get_configs():
    # TODO make the shots proportional to number of nodes
    shots_simulation = 100000
    shots_quantum = 2048
    p_repetitions = 4
    return {
        "p": p_repetitions,
        "shots_quantum": shots_quantum,
        "shots_simulation": shots_simulation,
    }


def get_test_graph():
    n = 5
    V = np.arange(0, n, 1)
    E = [(0, 1, 3.0), (1, 2, 2.0), (2, 3, 2.0), (3, 4, 3.0), (4, 0, 1.0), (0, 3, 3.0)]

    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G


# def get maxcut_brute_force():
#     max_cost = 0
#     for b in range(2**n):
#         x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
#         cost = 0
#         for i in range(n):
#             for j in range(n):
#                 cost = cost + w[i,j]*x[i]*(1-x[j])
#         if best_cost_brute < cost:
#             best_cost_brute = cost
#             xbest_brute = x
#         print('case = ' + str(x)+ ' cost = ' + str(cost))

#     colors = ['r' if xbest_brute[i] == 0 else 'b' for i in range(n)]
#     nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, pos=pos)
#     print('\nBest solution = ' + str(xbest_brute) + ' cost = ' + str(best_cost_brute))
#     return ()


def build_cost_operator_circuit(qaoa_circuit, edges, gamma):
    for edge in edges:
        k = edge[0]
        l = edge[1]
        qaoa_circuit.cu1(-2 * gamma, k, l)
        qaoa_circuit.u1(gamma, k)
        qaoa_circuit.u1(gamma, l)
    return


def get_quantum_circuit(graph, gammas, betas):
    num_nodes = graph.number_of_nodes()
    edges = graph.edges()
    approx_level = get_configs()["p"]

    qaoa_circuit = QuantumCircuit(num_nodes)
    qaoa_circuit.h(range(num_nodes))
    qaoa_circuit.barrier()

    for depth in range(approx_level):
        build_cost_operator_circuit(qaoa_circuit, edges, gammas[depth])
        qaoa_circuit.barrier()
        qaoa_circuit.rx(2 * betas[depth], range(num_nodes))

    qaoa_circuit.measure_all()

    return qaoa_circuit


def get_eigen_values(state_bit):
    eigen_value_map = {0: -1, 1: 1}
    return eigen_value_map[state_bit]


def get_configuration_cost(state_bitstring, graph):

    edges = graph.edges()
    C = 0
    for edge in edges:
        k = edge[0]
        l = edge[1]
        k_eigen_val = get_eigen_values(int(state_bitstring[k]))
        l_eigen_val = get_eigen_values(int(state_bitstring[l]))

        w = graph[k][l]["weight"]
        C = C + w * (1 - k_eigen_val * l_eigen_val) / 2
    return C


def compute_quasm_simulation(quantum_circuit):
    backend = Aer.get_backend("qasm_simulator")
    shots = get_configs()["shots_simulation"]

    simulate = execute(quantum_circuit, backend=backend, shots=shots)
    results = simulate.result()
    return results


def compute_quantum(quantum_circuit):
    provider = IBMQ.load_account()
    backend = provider.get_backend("ibmq_essex")
    shots = get_configs()["shots_quantum"]

    compute_job = execute(quantum_circuit, backend=backend, shots=shots)
    job_monitor(compute_job)
    return compute_job.result()


def get_cicuit_layout(circuit, backend, optimization_level):
    provider = IBMQ.get_provider(group="open")
    backend = provider.get_backend(backend)
    transpiled_circ = transpile(
        circuit, backend=backend, optimization_level=optimization_level
    )
    return plot_circuit_layout(transpiled_circ, backend)


def analyse_results(graph, compute_result):
    counts = compute_result.get_counts()
    expectation_cost = 0
    max_cost = 0
    cost_dict = {}
    total_count = 0
    for state, count in counts.items():

        state_cost = get_configuration_cost(state, graph)
        cost_dict[state] = state_cost
        expectation_cost += count * state_cost
        total_count += count

    expectation_cost = expectation_cost / total_count
    return expectation_cost, cost_dict


def get_expectation_value(graph, compute_result):

    expectation_value, cost_dict = analyse_results(graph, compute_result)
    return expectation_value


def get_max_cost_state(graph, compute_result):

    expectation_value, cost_dict = analyse_results(graph, compute_result)
    max_cost_state = max(cost_dict, key=cost_dict.get)
    return (max_cost_state, cost_dict[max_cost_state])


def get_cost_dict(graph, compute_result):
    expectation_value, cost_dict = analyse_results(graph, compute_result)
    return cost_dict


# finding intial angles


def objective_function(x):

    backend = Aer.get_backend("qasm_simulator")
    shots = 512

    approx_level = get_configs()["p"]
    graph = get_test_graph()
    gammas = x[:approx_level]
    betas = x[approx_level:]
    circ = get_quantum_circuit(graph, gammas, betas)
    result = execute(circ, backend, shots=shots).result()
    cost = get_expectation_value(graph, result)

    return cost


def get_initial_angles():

    # optimizer = COBYLA(maxiter=500, tol=0.0001)
    approx_level = get_configs()["p"]
    params = np.random.rand(2 * approx_level)
    # initial_angles = optimizer.optimize(num_vars=2*approx_level, objective_function=objective_function, initial_point=params)
    res = minimize(objective_function, params, method="COBYLA", tol=1e-6)
    return res


# %%
angle = get_initial_angles()
# %%
gamma = [1.9]
beta = [0.2]
test_graph = get_test_graph()
test_cic = get_quantum_circuit(test_graph, gamma, beta)

#%%
result_sim = compute_quasm_simulation(test_cic)
result_quantum = compute_quantum(test_cic)


# %%
