import numpy as np
import numpy.testing as npt
from qorange.circuits import QuantumCircuit
from qorange.gates import *

STATE_00 = np.array([1, 0, 0, 0])
STATE_01 = np.array([0, 1, 0, 0])
STATE_10 = np.array([0, 0, 1, 0])
STATE_11 = np.array([0, 0, 0, 1])

###########################
# 1-QUBIT GATES
###########################

def test_identity_q1_state_00():
    '''
    Test the identity operator acting on qubit 1 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Identity())

    npt.assert_array_equal(circuit.state, STATE_00)

def test_identity_q1_state_01():
    '''
    Test the identity operator acting on qubit 1 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(1, Identity())

    npt.assert_array_equal(circuit.state, STATE_01)

def test_identity_q1_state_10():
    '''
    Test the identity operator acting on qubit 1 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(1, Identity())

    npt.assert_array_equal(circuit.state, STATE_10)

def test_identity_q1_state_11():
    '''
    Test the identity operator acting on qubit 1 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(1, Identity())

    npt.assert_array_equal(circuit.state, STATE_11)

def test_identity_q2_state_00():
    '''
    Test the identity operator acting on qubit 2 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, Identity())

    npt.assert_array_equal(circuit.state, STATE_00)

def test_identity_q2_state_01():
    '''
    Test the identity operator acting on qubit 2 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(2, Identity())

    npt.assert_array_equal(circuit.state, STATE_01)

def test_identity_q2_state_10():
    '''
    Test the identity operator acting on qubit 2 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(2, Identity())

    npt.assert_array_equal(circuit.state, STATE_10)

def test_identity_q2_state_11():
    '''
    Test the identity operator acting on qubit 2 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(2, Identity())

    npt.assert_array_equal(circuit.state, STATE_11)

def test_pauli_x_q1_state_00():
    '''
    Test the Pauli X operator acting on qubit 1 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, PauliX())

    npt.assert_array_equal(circuit.state, STATE_10)

def test_pauli_x_q1_state_01():
    '''
    Test the Pauli X operator acting on qubit 1 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(1, PauliX())

    npt.assert_array_equal(circuit.state, STATE_11)

def test_pauli_x_q1_state_10():
    '''
    Test the Pauli X operator acting on qubit 1 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(1, PauliX())

    npt.assert_array_equal(circuit.state, STATE_00)

def test_pauli_x_q1_state_11():
    '''
    Test the Pauli X operator acting on qubit 1 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(1, PauliX())

    npt.assert_array_equal(circuit.state, STATE_01)

def test_pauli_x_q2_state_00():
    '''
    Test the Pauli X operator acting on qubit 2 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, PauliX())

    expected = np.array([0, 1, 0, 0]) # |01>

    npt.assert_array_equal(circuit.state, expected)

def test_pauli_x_q2_state_01():
    '''
    Test the Pauli X operator acting on qubit 2 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(2, PauliX())

    npt.assert_array_equal(circuit.state, STATE_00)

def test_pauli_x_q2_state_10():
    '''
    Test the Pauli X operator acting on qubit 2 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(2, PauliX())

    npt.assert_array_equal(circuit.state, STATE_11)

def test_pauli_x_q2_state_11():
    '''
    Test the Pauli X operator acting on qubit 2 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(2, PauliX())

    npt.assert_array_equal(circuit.state, STATE_10)

def test_pauli_y_q1_state_00():
    '''
    Test the Pauli Y operator acting on qubit 1 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, PauliY())

    npt.assert_array_equal(circuit.state, 1j * STATE_10)

def test_pauli_y_q1_state_01():
    '''
    Test the Pauli Y operator acting on qubit 1 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(1, PauliY())

    npt.assert_array_equal(circuit.state, 1j * STATE_11)

def test_pauli_y_q1_state_10():
    '''
    Test the Pauli Y operator acting on qubit 1 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(1, PauliY())

    npt.assert_array_equal(circuit.state, -1j * STATE_00)

def test_pauli_y_q1_state_11():
    '''
    Test the Pauli Y operator acting on qubit 1 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(1, PauliY())

    npt.assert_array_equal(circuit.state, -1j * STATE_01)

def test_pauli_y_q2_state_00():
    '''
    Test the Pauli Y operator acting on qubit 2 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, PauliY())

    npt.assert_array_equal(circuit.state, 1j * STATE_01)

def test_pauli_y_q2_state_01():
    '''
    Test the Pauli Y operator acting on qubit 2 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(2, PauliY())

    npt.assert_array_equal(circuit.state, -1j * STATE_00)

def test_pauli_y_q2_state_10():
    '''
    Test the Pauli Y operator acting on qubit 2 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(2, PauliY())

    npt.assert_array_equal(circuit.state, 1j * STATE_11)

def test_pauli_y_q2_state_11():
    '''
    Test the Pauli Y operator acting on qubit 2 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(2, PauliY())

    npt.assert_array_equal(circuit.state, -1j * STATE_10)

def test_pauli_z_q1_state_00():
    '''
    Test the Pauli Z operator acting on qubit 1 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, PauliZ())

    npt.assert_array_equal(circuit.state, STATE_00)

def test_pauli_z_q1_state_01():
    '''
    Test the Pauli Z operator acting on qubit 1 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(1, PauliZ())

    npt.assert_array_equal(circuit.state, STATE_01)

def test_pauli_z_q1_state_10():
    '''
    Test the Pauli Z operator acting on qubit 1 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(1, PauliZ())

    npt.assert_array_equal(circuit.state, -STATE_10)

def test_pauli_z_q1_state_11():
    '''
    Test the Pauli Z operator acting on qubit 1 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(1, PauliZ())

    npt.assert_array_equal(circuit.state, -STATE_11)

def test_pauli_z_q2_state_00():
    '''
    Test the Pauli Z operator acting on qubit 2 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, PauliZ())

    npt.assert_array_equal(circuit.state, STATE_00)

def test_pauli_z_q2_state_01():
    '''
    Test the Pauli Z operator acting on qubit 2 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(2, PauliZ())

    npt.assert_array_equal(circuit.state, -STATE_01)

def test_pauli_z_q2_state_10():
    '''
    Test the Pauli Z operator acting on qubit 2 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(2, PauliZ())

    npt.assert_array_equal(circuit.state, STATE_10)

def test_pauli_z_q2_state_11():
    '''
    Test the Pauli Z operator acting on qubit 2 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(2, PauliZ())

    npt.assert_array_equal(circuit.state, -STATE_11)

def test_hadamard_q1_state_00():
    '''
    Test the Hadamard operator acting on qubit 1 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, Hadamard())
    
    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_00 + STATE_10))

def test_hadamard_q1_state_01():
    '''
    Test the Hadamard operator acting on qubit 1 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(1, Hadamard())
    
    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_01 + STATE_11))

def test_hadamard_q1_state_10():
    '''
    Test the Hadamard operator acting on qubit 1 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(1, Hadamard())
    
    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_00 - STATE_10))

def test_hadamard_q1_state_11():
    '''
    Test the Hadamard operator acting on qubit 1 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(1, Hadamard())
    
    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_01 - STATE_11))

def test_hadamard_q2_state_00():
    '''
    Test the Hadamard operator acting on qubit 2 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, Hadamard())

    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_00 + STATE_01))

def test_hadamard_q2_state_01():
    '''
    Test the Hadamard operator acting on qubit 2 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(2, Hadamard())
    
    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_00 - STATE_01))

def test_hadamard_q2_state_10():
    '''
    Test the Hadamard operator acting on qubit 2 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(2, Hadamard())
    
    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_10 + STATE_11))

def test_hadamard_q2_state_11():
    '''
    Test the Hadamard operator acting on qubit 2 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(2, Hadamard())
    
    npt.assert_allclose(circuit.state, 1 / np.sqrt(2) * (STATE_10 - STATE_11))

def test_s_q1_state_00():
    '''
    Test the S operator acting on qubit 1 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, S())
    
    npt.assert_array_equal(circuit.state, STATE_00)

def test_s_q1_state_01():
    '''
    Test the S operator acting on qubit 1 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(1, S())
    
    npt.assert_array_equal(circuit.state, STATE_01)

def test_s_q1_state_10():
    '''
    Test the S operator acting on qubit 1 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(1, S())
    
    npt.assert_array_equal(circuit.state, 1j * STATE_10)

def test_s_q1_state_11():
    '''
    Test the S operator acting on qubit 1 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(1, S())
    
    npt.assert_array_equal(circuit.state, 1j * STATE_11)

def test_s_q2_state_00():
    '''
    Test the S operator acting on qubit 2 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, S())
    
    npt.assert_array_equal(circuit.state, STATE_00)

def test_s_q2_state_01():
    '''
    Test the S operator acting on qubit 2 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(2, S())
    
    npt.assert_array_equal(circuit.state, 1j * STATE_01)

def test_s_q2_state_10():
    '''
    Test the S operator acting on qubit 2 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(2, S())
    
    npt.assert_array_equal(circuit.state, STATE_10)

def test_s_q2_state_11():
    '''
    Test the S operator acting on qubit 2 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(2, S())
    
    npt.assert_array_equal(circuit.state, 1j * STATE_11)

def test_t_q1_state_00():
    '''
    Test the T operator acting on qubit 1 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(1, T())
    
    npt.assert_array_equal(circuit.state, STATE_00)

def test_t_q1_state_01():
    '''
    Test the T operator acting on qubit 1 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(1, T())
    
    npt.assert_array_equal(circuit.state, STATE_01)

def test_t_q1_state_10():
    '''
    Test the T operator acting on qubit 1 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(1, T())
    
    npt.assert_allclose(circuit.state, np.exp(1j * np.pi / 4) * STATE_10)

def test_t_q1_state_11():
    '''
    Test the T operator acting on qubit 1 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(1, T())
    
    npt.assert_allclose(circuit.state, np.exp(1j * np.pi / 4) * STATE_11)

def test_t_q2_state_00():
    '''
    Test the T operator acting on qubit 2 of state |00>.
    '''
    circuit = QuantumCircuit()
    circuit.apply_gate(2, T())
    
    npt.assert_array_equal(circuit.state, STATE_00)

def test_t_q2_state_01():
    '''
    Test the T operator acting on qubit 2 of state |01>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_01
    circuit.apply_gate(2, T())
    
    npt.assert_allclose(circuit.state, np.exp(1j * np.pi / 4) * STATE_01)

def test_t_q2_state_10():
    '''
    Test the T operator acting on qubit 2 of state |10>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_10
    circuit.apply_gate(2, T())
    
    npt.assert_array_equal(circuit.state, STATE_10)

def test_t_q2_state_11():
    '''
    Test the T operator acting on qubit 2 of state |11>.
    '''
    circuit = QuantumCircuit()
    circuit.state = STATE_11
    circuit.apply_gate(2, T())
    
    npt.assert_allclose(circuit.state, np.exp(1j * np.pi / 4) * STATE_11)

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
