import numpy as np
from gateClass import singleQubitGate, twoQubitGate

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

        self.q1 = np.array([1, 0])
        self.q2 = np.array([1, 0])
        self.state = np.kron(self.q1, self.q2)
        self._gates = []  # Stores the gates applied to the circuit
        # self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        # self.state[0] = 1

        

   
       

    def apply_gate(self, gate_matrix, target):
        """
        Applies a gate matrix to the circuit's state vector.

        Args:
            gate_matrix (np.ndarray): The matrix representation of the gate.
            target (int): The qubit the gate acts on.
        """
        # gate_matrix acts on the output of state() 

        # Build the full operator for the entire system
    
        if gate_matrix.controlled:

        else:
            if self.q1 == target
                self.q1 = np.dot(gate_matrix, self.q1)
            if self.q2 == target
                self.q2 = np.dot(gate_matrix, self.q2)
            


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

   

if __name__ == "__main__":
    from gateClass import *

    x = PauliX()
    y = PauliY()
    z = PauliZ()
    i = Identity()
    cnot = CNOT()

    circuit = QuantumCircuit(3)

    gate_array = [
        [x, i, i],
        [i, [cnot, (1,2)], [cnot, (1,2)]],
        [i, y, z],
        [[cnot, (2,0)], i, [cnot, (2,0)]]
    ]

    circuit.load_gates(gate_array)


    a = new Gate([[]])
    """
    # Example usage
    circuit = QuantumCircuit(1)  # Create a 1-qubit circuit
    circuit.h(0)                 # Apply H gate to qubit 0

    print(circuit)               # Print the applied gates
    print("Measurement probabilities:", circuit.measure())  # Print measurement probabilities
    """


c.apply_gate(X, 1)