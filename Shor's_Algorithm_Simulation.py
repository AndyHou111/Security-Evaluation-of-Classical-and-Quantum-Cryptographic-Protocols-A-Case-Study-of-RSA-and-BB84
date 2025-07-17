from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor

backend = Aer.get_backend("aer_simulator")
qi = QuantumInstance(backend)
shor = Shor(quantum_instance=qi)
res = shor.factor(N=15)
print(res.factors)