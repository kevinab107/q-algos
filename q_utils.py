#%%
import networkx as nx
from qiskit import Aer, IBMQ
from qiskit import (
    QuantumRegister,
    ClassicalRegister,
    QuantumCircuit,
    execute,
    transpile,
)
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_circuit_layout, plot_histogram
import numpy as np
from scipy.optimize import minimize


def compute_quasm_simulation(quantum_circuit):
    backend = Aer.get_backend("qasm_simulator")
    shots = get_configs()["shots_simulation"]

    simulate = execute(quantum_circuit, backend=backend, shots=shots)
    results = simulate.result()
    return results


def get_compute_backend(solve_type):
    if solve_type == "simulation":
        backend = Aer.get_backend("qasm_simulator")

    elif solve_type == "quantum":
        provider = IBMQ.load_account()
        backend = provider.get_backend("ibmq_16_melbourne")
    return backend


def compute_quantum(quantum_circuit):
    provider = IBMQ.load_account()
    backend = provider.get_backend("ibmqx2")
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
