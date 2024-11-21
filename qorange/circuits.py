import numpy as np

class QuantumCircuit:
    """
    A simple object-oriented quantum circuit class.

    Attributes:
        num_qubits (int): The number of qubits in the circuit.
        gates (list): A list of gates applied to the circuit, stored as tuples (gate_name, target_qubits, parameters).
        state (np.ndarray): A numpy array representing the state vector of the circuit.
    """

    def __init__(self, num_qubits):
        """
        Initializes a quantum circuit with a given number of qubits.

        Args:
            num_qubits (int): The number of qubits in the circuit.
        """
        self.num_qubits = num_qubits
        self.gates = []  # Stores the gates applied to the circuit
        self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        self.state[0] = 1  # Initialize state vector to |0...0>

    def apply_gate(self, gate_matrix, target):
        """
        Applies a gate matrix to the circuit's state vector.

        Args:
            gate_matrix (np.ndarray): The matrix representation of the gate.
            target (int): The qubit the gate acts on.
        """
        # Build the full operator for the entire system
        full_matrix = 1
        for qubit in range(self.num_qubits):
            if qubit == target:
                full_matrix = np.kron(full_matrix, gate_matrix)
            else:
                full_matrix = np.kron(full_matrix, np.eye(2))

        # Update the state vector by applying the full matrix
        self.state = np.dot(full_matrix, self.state)

    def x(self, target):
        """
        Applies an X (NOT) gate to a qubit.

        Args:
            target (int): The qubit to apply the X gate on.
        """
        x_matrix = np.array([[0, 1],
                             [1, 0]])
        self.apply_gate(x_matrix, target)

    def h(self, target):
        """
        Applies a Hadamard (H) gate to a qubit.

        Args:
            target (int): The qubit to apply the H gate on.
        """
        h_matrix = np.array([[1,  1],
                             [1, -1]]) / np.sqrt(2)
        self.apply_gate(h_matrix, target)

    def measure(self):
        """
        Measures the state of the circuit.

        Returns:
            list: A list representing the probabilities of each basis state.
        """
        probabilities = np.abs(self.state) ** 2
        return probabilities.tolist()

   

# Example usage
circuit = QuantumCircuit(1)  # Create a 1-qubit circuit
circuit.h(0)                 # Apply H gate to qubit 0

print(circuit)               # Print the applied gates
print("Measurement probabilities:", circuit.measure())  # Print measurement probabilities
