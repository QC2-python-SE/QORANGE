import numpy as np
import numpy.testing as npt
from qorange.circuits import QuantumCircuit
from qorange.gates import *

def test_circuit_bell():
    '''
    Test the identity operator acting on qubit 1.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())
    circuit.apply_gate(1, CNOT())
    
    npt.assert_allclose(
        circuit.state, 
        np.array([1, 0, 0, 1])/np.sqrt(2)
    )
