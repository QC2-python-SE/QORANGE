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

def test_circuit_2():
    '''
    Test the circuit with Hadamard, X, and Hadamard gates on the first qubit.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())  # Apply Hadamard to the first qubit
    circuit.apply_gate(1, PauliX())   # Apply X gate to the first qubit
    circuit.apply_gate(1, Hadamard())  # Apply Hadamard again to the first qubit

    # Expected state: |00> -> [1, 0, 0, 0]
    expected_state = np.array([1, 0, 0, 0])

    # Verify the output state matches the expected state
    npt.assert_allclose(
        circuit.state,
        expected_state,
        rtol=1e-5,
        atol=1e-8
    )

def test_circuit_3():
    '''
    Test the circuit with X, Hadamard, and CNOT gates.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, PauliX())  # Apply X gate to the first qubit
    circuit.apply_gate(1, Hadamard())  # Apply Hadamard to the first qubit
    circuit.apply_gate(1, CNOT())  # Apply CNOT with qubit 1 as control, qubit 2 as target

    # Expected state: (|00> - |11>)/sqrt(2) -> [1/sqrt(2), 0, 0, -1/sqrt(2)]
    expected_state = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)])

    # Verify the output state matches the expected state
    npt.assert_allclose(
        circuit.state,
        expected_state,
        rtol=1e-5,
        atol=1e-8
    )

def test_circuit_4():
    '''
    Test the circuit with a Pauli-Y gate on the first qubit and a SWAP gate.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, PauliY())  # Apply Y gate to the first qubit
    circuit.apply_gate(1, SWAP())  # Apply SWAP gate between qubits 1 and 2

    # Expected state: [0, 0, i, 0]
    expected_state = np.array([0, 1j, 0, 0])

    npt.assert_allclose(
        circuit.state,
        expected_state
    )

def test_circuit_5():
    '''
    Test the circuit with Hadamard, Pauli-Z, and Hadamard gates on the first qubit.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())  # Apply Hadamard to the first qubit
    circuit.apply_gate(1, PauliZ())   # Apply Z gate to the first qubit
    circuit.apply_gate(1, Hadamard())  # Apply Hadamard again to the first qubit

    # Expected state: |10> -> [0, 1, 0, 0]
    expected_state = np.array([0, 0, 1, 0])

    npt.assert_allclose(
        circuit.state,
        expected_state
    )

def test_circuit_6():
    '''
    Test the Bell circuit for creating entanglement.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())
    circuit.apply_gate(1, CZ())
    
    npt.assert_allclose(
        circuit.state, 
        np.array([1, 0, 1, 0])/np.sqrt(2)
    )
