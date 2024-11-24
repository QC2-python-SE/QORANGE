import numpy as np

class GateClass:
    """
    A base class for quantum gates, ensuring that the matrix representation
    of the gate is unitary.

    Attributes:
        _matrix (numpy.ndarray): The unitary matrix representing the quantum gate.

    Methods:
        __init__(matrix): Initializes the gate with the given unitary matrix.
    """
    def __init__(self, matrix):
        """
        Initializes the quantum gate with the given matrix.
        Checks whether the matrix is unitary.

        Args:
            matrix (numpy.ndarray): The matrix representation of the quantum gate.

        Raises:
            Exception: If the matrix is not unitary.
        """
        self._matrix = matrix
        # Check if the matrix is unitary by verifying that matrix * matrix† = Identity.
        if not np.allclose(np.eye(np.shape(matrix)[0]), np.dot(matrix, matrix.conj().T)):
            raise Exception("Matrix is not unitary!")

class Identity(GateClass):
    """
    Represents the Identity quantum gate (I), which leaves the quantum state unchanged.
    """
    def __init__(self):
        """
        Initializes the Identity gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([[1, 0], 
                                           [0, 1]]))

class PauliX(GateClass):
    """
    Represents the Pauli-X quantum gate, also known as the NOT gate.
    It flips the state of a qubit (|0⟩ ↔ |1⟩).
    """
    def __init__(self):
        """
        Initializes the Pauli-X gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([[0, 1], 
                                           [1, 0]]))

class PauliY(GateClass):
    """
    Represents the Pauli-Y quantum gate. 
    It applies a 180° rotation around the Y-axis on the Bloch sphere.
    """
    def __init__(self):
        """
        Initializes the Pauli-Y gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([[0, -1j], 
                                           [1j, 0]]))

class PauliZ(GateClass):
    """
    Represents the Pauli-Z quantum gate.
    It applies a 180° rotation around the Z-axis on the Bloch sphere.
    """
    def __init__(self):
        """
        Initializes the Pauli-Z gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([[1, 0], 
                                           [0, -1]]))  # Corrected the second element

class Hadamard(GateClass):
    """
    Represents the Hadamard quantum gate (H).
    It creates superpositions by transforming |0⟩ to (|0⟩ + |1⟩)/√2 
    and |1⟩ to (|0⟩ - |1⟩)/√2.
    """
    def __init__(self):
        """
        Initializes the Hadamard gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([[1, 1], 
                                           [1, -1]])/np.sqrt(2))

class S(GateClass):
    """
    Represents the S gate (phase gate).
    It applies a 90° phase shift to the |1⟩ state.
    """
    def __init__(self):
        """
        Initializes the S gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([[1, 0], 
                                           [0, 1j]]))

class T(GateClass):
    """
    Represents the T gate (π/8 gate).
    It applies a π/4 phase shift to the |1⟩ state.
    """
    def __init__(self):
        """
        Initializes the T gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([[1, 0],
                                           [0, 1/np.sqrt(2) + 1j/np.sqrt(2)]]))

class CNOT(GateClass):
    """
    Represents the Controlled-NOT (CNOT) gate.
    It flips the target qubit if the control qubit is |1⟩.
    """
    def __init__(self):
        """
        Initializes the CNOT gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([
            [1, 0, 0, 0],  # |00⟩ → |00⟩
            [0, 1, 0, 0],  # |01⟩ → |01⟩
            [0, 0, 0, 1],  # |10⟩ → |11⟩
            [0, 0, 1, 0]   # |11⟩ → |10⟩
        ]))

class CZ(GateClass):
    """
    Represents the Controlled-Z (CZ) gate.
    It applies a Z gate to the target qubit if the control qubit is |1⟩.
    """
    def __init__(self):
        """
        Initializes the CZ gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([
            [1, 0, 0, 0],  # |00⟩ → |00⟩
            [0, 1, 0, 0],  # |01⟩ → |01⟩
            [0, 0, 1, 0],  # |10⟩ → |10⟩
            [0, 0, 0, -1]  # |11⟩ → -|11⟩
        ]))

class SWAP(GateClass):
    """
    Represents the SWAP gate.
    It swaps the states of two qubits.
    """
    def __init__(self):
        """
        Initializes the SWAP gate with its matrix representation.
        """
        GateClass.__init__(self, np.array([
            [1, 0, 0, 0],  # |00⟩ → |00⟩
            [0, 0, 1, 0],  # |01⟩ → |10⟩
            [0, 1, 0, 0],  # |10⟩ → |01⟩
            [0, 0, 0, 1]   # |11⟩ → |11⟩
        ]))
