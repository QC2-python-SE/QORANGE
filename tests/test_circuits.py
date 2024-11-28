import numpy as np
import numpy.testing as npt
from qorange.circuits import QuantumCircuit
from qorange.gates import *

def test_circuit_bell():
    '''
    Test the Bell circuit for creating entanglement.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())
    circuit.apply_gate(1, CNOT())
    
    npt.assert_allclose(
        circuit.state, 
        np.array([1, 0, 0, 1])/np.sqrt(2)
    )

def test_circuit_1():
    '''
    Test the circuit with a Pauli-X gate on the first qubit and a SWAP gate.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, PauliX())  # Apply X gate to the first qubit
    circuit.apply_gate(1, SWAP())  # Apply SWAP gate between qubits 1 and 2

    # Expected state: |01> -> [0, 1, 0, 0]
    expected_state = np.array([0, 1, 0, 0])

    # Verify the output state matches the expected state
    npt.assert_allclose(
        circuit.state,
        expected_state,
        rtol=1e-5,
        atol=1e-8
    )
