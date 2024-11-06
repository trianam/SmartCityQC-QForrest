import pennylane as qml
from pennylane.templates import AngleEmbedding, AmplitudeEmbedding
import numpy as np
import matplotlib.pyplot as plt

# Space-time embedding  


n_qubits = 4

dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
projector[0, 0] = 1


def embedding (x, t):
    AngleEmbedding(x, wires=range(n_qubits))
    qml.Hadamard(n_qubits-1)
    for q in range(n_qubits-1):
        qml.CRY(t[q], wires=[n_qubits-1, q])
    

@qml.qnode(dev_kernel)
def quantum_kernel(x1, x2, t1, t2):
    """temporal quantum kernel"""
    embedding(x1, t1)
    qml.Barrier(only_visual=True)
    qml.adjoint(embedding)(x2, t2)
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

x = np.random.random((10, n_qubits))
x, t = x[:, :3], x[:, -1]
qml.draw_mpl(quantum_kernel)(x, x, t, t)
plt.savefig('temporal_kernel.png')