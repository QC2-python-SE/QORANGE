import numpy as np
import numpy.testing as npt
from qorange.circuits import QuantumCircuit
from qorange.gates import *

###########################
# 1-QUBIT GATES
###########################

def test_apply_gate_identity_q1():
    '''
    Test the identity operator acting on qubit 1.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Identity())

    npt.assert_array_equal(
        circuit.state, 
        np.kron(
            np.array([1, 0]), 
            np.array([1, 0]),
        )
    )

def test_apply_gate_identity_q2():
    '''
    Test the identity operator acting on qubit 2.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, Identity())

    npt.assert_array_equal(
        circuit.state, 
        np.kron(
            np.array([1, 0]), 
            np.array([1, 0]),
        )
    )

def test_apply_gate_pauli_x_q1():
    '''
    Test the Pauli X operator acting on qubit 1.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, PauliX())

    npt.assert_array_equal(
        circuit.state, 
        np.kron(
            np.array([0, 1]), 
            np.array([1, 0]),
        )
    )

def test_apply_gate_pauli_x_q2():
    '''
    Test the Pauli X operator acting on qubit 2.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, PauliX())

    npt.assert_array_equal(
        circuit.state, 
        np.kron(
            np.array([1, 0]),
            np.array([0, 1]), 
        )
    )

def test_apply_gate_pauli_y_q1():
    '''
    Test the Pauli Y operator acting on qubit 1.
    '''
    pass

def test_apply_gate_pauli_y_q2():
    '''
    Test the Pauli Y operator acting on qubit 2.
    '''
    pass

def test_apply_gate_pauli_z_q1():
    '''
    Test the Pauli Z operator acting on qubit 1.
    '''
    pass

def test_apply_gate_pauli_z_q2():
    '''
    Test the Pauli Z operator acting on qubit 2.
    '''
    pass

def test_apply_gate_hadamard_q1():
    '''
    Test the Hadamard operator acting on qubit 1.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())

    # Applying Hadamard to the first qubit of |00> results in (1/sqrt(2)) * (|00> + |10>)
    expected_state = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0])  # |00> + |10> / sqrt(2)
    
    npt.assert_allclose(circuit.state, expected_state)

def test_apply_gate_hadamard_q2():
    '''
    Test the Hadamard operator acting on qubit 2.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, Hadamard())

    # Applying Hadamard to the second qubit of |00> results in (1/sqrt(2)) * (|00> + |01>)
    expected_state = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])  # |00> + |01> / sqrt(2)

    npt.assert_allclose(circuit.state, expected_state)

def test_apply_gate_hadamard_q1_q2():
    '''
    Test the Hadamard operator acting on qubits 1 and 2.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())
    circuit.apply_gate(2, Hadamard())

    # Applying Hadamard to both qubits of |00> results in (1/2) * (|00> + |01> + |10> + |11>)
    expected_state = np.array([1/2, 1/2, 1/2, 1/2])  # (|00> + |01> + |10> + |11>) / 2

    npt.assert_allclose(circuit.state, expected_state)

def test_apply_gate_s():
    '''
    TODO
    '''
    pass

def test_apply_gate_t():
    '''
    TODO
    '''
    pass

###########################
# CONTROLLED GATES
###########################

def test_apply_gate_cnot():
    '''
    TODO
    '''
    pass

def test_apply_gate_cz():
    '''
    TODO
    '''
    pass

###########################
# 2-QUBIT GATES
###########################

def test_apply_gate_swap():
    '''
    TODO
    '''
    pass
