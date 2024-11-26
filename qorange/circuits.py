import numpy as np

class QuantumCircuit:
    """
    A simple object-oriented quantum circuit class.

    Attributes:
        num_qubits (int): The number of qubits in the circuit.
        gates (list): A list of gates applied to the circuit, stored as tuples (gate_name, target_qubits, parameters).
        state (np.ndarray): A numpy array representing the state vector of the circuit.
    """

    def __init__(self):
        """
        Initializes a quantum circuit with a given number of qubits.

        Args:
            num_qubits (int): The number of qubits in the circuit.
        """

        self.q1 = np.array([1, 0])
        self.q2 = np.array([1, 0])
        #self.state = np.kron(self.q1, self.q2)
        self._gates = []  # Stores the gates applied to the circuit
        # self.state = np.zeros((2 ** num_qubits,), dtype=complex)
        # self.state[0] = 1

        

   
       

    def apply_gate(self, q_index, gate):
        """
        Applies a gate matrix to the circuit's state vector.

        Args:
            q_index: int if single qubit operation (1 for q1 and 2 for q2)
            gate: gate being applied
        """
        
        if isinstance(q_index, int):
            if q_index == 1:
                self.q1 = np.matmul(gate.matrix, self.q1)
            elif q_index == 2:
                self.q2 = np.matmul(gate.matrix, self.q2)
            else:
                raise Exception("Invalid indexing of qubits")
            
        elif isinstance(q_index, tuple):
            pass
            # controlled unitary operations here.

        
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

    circuit = QuantumCircuit()
    circuit.apply_gate(1, x)

    