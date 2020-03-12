#%%
import numpy as np
from qiskit import *
from qiskit.visualization import plot_state_city, plot_histogram
from qiskit import Aer
from qiskit import IBMQ
from qiskit.tools.monitor import job_monitor
#%%
IBMQ.load_account()
provider = IBMQ.get_provider(group='open')
#provider.backends()
#provider.backends(filters = lambda x : x.configuration().n_qubits >=10 and not x.configuration().simulator)
# %%
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0,1)
circ.cx(1,2)
circ.draw()

simulator = Aer.get_backend('statevector_simulator')
job = execute(circ, simulator)
result = job.result()
outputstate = result.get_statevector(circ, decimals = 3 )

plot_state_city(outputstate)

meas = QuantumCircuit(3, 3)
meas.barrier(range(3))
qc = meas + circ
qc.measure(range(2),[0,2])
qc.draw()

simulator = Aer.get_backend('qasm_simulator')
backend = provider.get_backend('ibmq_16_melbourne')
meas_job = execute(qc, simulator, shots = 1024)
result_sim = meas_job.result()
counts_sim = result_sim.get_counts()

quantum_job = execute(qc, backend)
job_monitor(quantum_job)
quantum_result = quantum_job.result()
counts_quantum = quantum_result.get_counts()

plot_histogram([counts_sim, counts_quantum], legend= ['simulator','quantum_hardware'])
# %%
