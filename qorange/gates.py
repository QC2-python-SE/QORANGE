import numpy as np

class Gate:
    """
    A base class for quantum gates, ensuring that the matrix representation
    of the gate is unitary.

    Attributes:
        _matrix (numpy.ndarray): The unitary matrix representing the quantum gate.

    Methods:
        __init__(matrix): Initializes the gate with the given unitary matrix.
    """
    def __init__(self, matrix, span = 2):
        """
        Initializes the quantum gate with the given matrix.
        Checks whether the matrix is unitary.

        Args:
            matrix (numpy.ndarray): The matrix representation of the quantum gate.

        Raises:
            ValueError: If the matrix is not unitary.
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be a numpy array.")
        if len(matrix.shape) != span or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")
        
        self.matrix = matrix
        # Check if the matrix is unitary by verifying that matrix * matrix† = Identity.
        if not np.allclose(np.eye(matrix.shape[0]), np.dot(matrix, matrix.conj().T)):
            raise ValueError("Matrix is not unitary!")

    def __repr__(self):
        """
        Returns a string representation of the Gate object.
        """
        return f"Gate({self._matrix})"

class Identity(Gate):
    """
    Represents the Identity quantum gate (I), which leaves the quantum state unchanged.
    """
    def __init__(self):
        """
        Initializes the Identity gate with its matrix representation.
        """
        Gate.__init__(self, np.array([[1, 0], 
                                           [0, 1]]))

class PauliX(Gate):
    """
    Represents the Pauli-X quantum gate, also known as the NOT gate.
    It flips the state of a qubit (|0⟩ ↔ |1⟩).
    """
    def __init__(self):
        """
        Initializes the Pauli-X gate with its matrix representation.
        """
        Gate.__init__(self, np.array([[0, 1], 
                                           [1, 0]]))

class PauliY(Gate):
    """
    Represents the Pauli-Y quantum gate. 
    It applies a 180° rotation around the Y-axis on the Bloch sphere.
    """
    def __init__(self):
        """
        Initializes the Pauli-Y gate with its matrix representation.
        """
        Gate.__init__(self, np.array([[0, -1j], 
                                           [1j, 0]]))

class PauliZ(Gate):
    """
    Represents the Pauli-Z quantum gate.
    It applies a 180° rotation around the Z-axis on the Bloch sphere.
    """
    def __init__(self):
        """
        Initializes the Pauli-Z gate with its matrix representation.
        """
        Gate.__init__(self, np.array([[1, 0], 
                                           [0, -1]]))  # Corrected the second element

class Hadamard(Gate):
    """
    Represents the Hadamard quantum gate (H).
    It creates superpositions by transforming |0⟩ to (|0⟩ + |1⟩)/√2 
    and |1⟩ to (|0⟩ - |1⟩)/√2.
    """
    def __init__(self):
        """
        Initializes the Hadamard gate with its matrix representation.
        """
        Gate.__init__(self, np.array([[1, 1], 
                                           [1, -1]])/np.sqrt(2))

class S(Gate):
    """
    Represents the S gate (phase gate).
    It applies a 90° phase shift to the |1⟩ state.
    """
    def __init__(self):
        """
        Initializes the S gate with its matrix representation.
        """
        Gate.__init__(self, np.array([[1, 0], 
                                           [0, 1j]]))

class T(Gate):
    """
    Represents the T gate (π/8 gate).
    It applies a π/4 phase shift to the |1⟩ state.
    """
    def __init__(self):
        """
        Initializes the T gate with its matrix representation.
        """
        Gate.__init__(self, np.array([[1, 0],
                                           [0, 1/np.sqrt(2) + 1j/np.sqrt(2)]]))

class ControlledGate():
    def __init__(self, gate):
        if isinstance(gate, Gate):
            self.gate = gate
        else:
            raise Exception("Specified gate object is invalid, use Gate class")
        
    def get_matrix(self):
        return self.gate.matrix

class CNOT(ControlledGate):
    """
    Represents the Controlled-NOT (CNOT) gate.
    It flips the target qubit if the control qubit is |1⟩.
    """
    def __init__(self):
        super().__init__(PauliX())

class CZ(ControlledGate):
    """
    Represents the Controlled-Z (CZ) gate.
    It applies a Z gate to the target qubit if the control qubit is |1⟩.
    """
    def __init__(self):
        super().__init__(PauliZ())


class TwoQubitGate(Gate):
    def __init__(self, matrix):
        super().__init__(matrix, span=4)

class SWAP(TwoQubitGate):
    """
    Represents the SWAP gate.
    It swaps the states of two qubits.
    """
    def __init__(self):
        """
        Initializes the SWAP gate with its matrix representation.
        """
        Gate.__init__(self, np.array([
            [1, 0, 0, 0],  # |00⟩ → |00⟩
            [0, 0, 1, 0],  # |01⟩ → |10⟩
            [0, 1, 0, 0],  # |10⟩ → |01⟩
            [0, 0, 0, 1]   # |11⟩ → |11⟩
        ]))