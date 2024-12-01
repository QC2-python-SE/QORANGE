import numpy as np
import numpy.testing as npt
from qorange.circuits import QuantumCircuit
from qorange.gates import *

def test_circuit_meas_1():
    '''
    Test the circuit with a Pauli-X gate on the first qubit and a SWAP gate.
    '''
    # Apply the gates
    circuit = QuantumCircuit()
    circuit.apply_gate(Hadamard(), 1)  # Apply Hadamard on qubit 1
    circuit.apply_gate(CNOT(), (1, 2))  # Apply CNOT with qubit 1 as control, qubit 2 as target

    expected_outcome_1 = [0.5, 0.5]
    expected_outcome_2 = [0.5, 0.5]
    # Measure qubit 1
    outcome_1 = circuit.measure_qubit_computational(1)
    # Measure qubit 2
    outcome_2 = circuit.measure_qubit_computational(2)
    npt.assert_allclose(outcome_1, expected_outcome_1, rtol=1e-5, atol=1e-8)
    npt.assert_allclose(outcome_2, expected_outcome_2, rtol=1e-5, atol=1e-8)

